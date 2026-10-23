"""CMMS events -> life intervals -> MTBF and Weibull per class -> PM decision -> 2026 schedule -> validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .policy import cost_rate, optimal_interval, recommend
from .prep import build_intervals, clean_events
from .reliability import beta_profile_ci, mtbf_with_bounds, weibull_mean, weibull_mle

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT, DOCS = ROOT / "data", ROOT / "outputs", ROOT / "docs"
OBS_END = "2025-12-31"
PLAN_START, PLAN_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")
DAYS_PER_MONTH = 365.25 / 12


def practical_interval(T_days: float) -> tuple[int, float]:
    """Round the optimum DOWN to whole months, so the plan never runs later than the optimum."""
    months = max(int(T_days // DAYS_PER_MONTH), 1)
    return months, months * DAYS_PER_MONTH


def analyse(ev_raw: pd.DataFrame, classes: pd.DataFrame):
    ev, clean_log = clean_events(ev_raw)
    iv = build_intervals(ev, OBS_END)
    rows = []
    for c in classes.itertuples():
        g = iv[iv.equipment_class == c.equipment_class]
        t, f = g.days.values, g.failed.values
        r = int(f.sum())
        if r < 3:  # too few failures to say anything about wear-out
            rows.append(dict(equipment_class=c.equipment_class, assets=g.asset_id.nunique(), failures=r,
                             strategy="Insufficient failure data", reason=f"only {r} failures",
                             x_T_plan=np.inf, x_cur=np.inf, beta_90ci="nan-nan"))
            continue
        mtbf, mlo, mhi = mtbf_with_bounds(t.sum(), r)
        b, e = weibull_mle(t, f)
        blo, bhi = beta_profile_ci(t, f)
        rec = recommend(b, (blo, bhi), e, c.planned_cost_usd, c.failure_cost_usd)
        cur = c.current_pm_days if pd.notna(c.current_pm_days) else np.inf
        row = dict(
            equipment_class=c.equipment_class, assets=g.asset_id.nunique(), life_intervals=len(g), failures=r,
            suspensions=len(g) - r, operating_years=round(t.sum() / 365.25, 1),
            crude_mtbf_days=round(mtbf), crude_mtbf_90ci=f"{mlo:.0f}-{mhi:.0f}",
            weibull_beta=round(b, 2), beta_90ci=f"{blo:.2f}-{bhi:.2f}", weibull_eta_days=round(e),
            mttf_days=round(weibull_mean(b, e)), current_pm_days=c.current_pm_days,
            strategy=rec["strategy"], reason=rec["reason"],
            optimal_interval_days=round(rec["interval_days"]) if np.isfinite(rec["interval_days"]) else np.nan,
        )
        if np.isfinite(rec["interval_days"]):
            months, T_prac = practical_interval(rec["interval_days"])
            row["recommended_interval_months"] = months
            row["x_T_plan"] = T_prac
        else:
            row["recommended_interval_months"] = np.nan
            row["x_T_plan"] = np.inf
        row["x_beta"], row["x_eta"], row["x_cur"] = b, e, cur
        row["fitted_cost_per_year_current_usd"] = round(365.25 * cost_rate(cur, b, e, c.planned_cost_usd, c.failure_cost_usd))
        row["fitted_cost_per_year_recommended_usd"] = round(
            365.25 * cost_rate(row["x_T_plan"], b, e, c.planned_cost_usd, c.failure_cost_usd))
        rows.append(row)
    return pd.DataFrame(rows), iv, ev, clean_log


def schedule(summary: pd.DataFrame, ev: pd.DataFrame, catch_up_weeks: int = 12) -> pd.DataFrame:
    """Planned replacements in 2026 for classes where time-based replacement pays.
    Due date = last renewal + interval. Assets already past due on 1 January (the new policy is
    stricter than past practice) are spread one a week over the first quarter, oldest first,
    instead of all landing in one week."""
    last = ev.groupby("asset_id").agg(equipment_class=("equipment_class", "first"), last_renewal=("event_date", "max"))
    plan_T = summary.set_index("equipment_class")
    plan_T = plan_T[np.isfinite(plan_T.x_T_plan)]
    items = []
    for asset, a in last[last.equipment_class.isin(plan_T.index)].iterrows():
        s = plan_T.loc[a.equipment_class]
        items.append(dict(asset_id=asset, equipment_class=a.equipment_class, last_renewal=a.last_renewal,
                          T=s.x_T_plan, months=int(s.recommended_interval_months),
                          due=a.last_renewal + pd.Timedelta(days=s.x_T_plan)))
    items = pd.DataFrame(items)
    overdue = items[items.due < PLAN_START].sort_values("due")
    slots = [PLAN_START + pd.Timedelta(days=14 + 7 * (i % catch_up_weeks)) for i in range(len(overdue))]
    items.loc[overdue.index, "first"] = slots
    items["first"] = items["first"].fillna(items.due)
    rows = []
    for it in items.itertuples():
        d, late = it.first, it.due < PLAN_START
        while d <= PLAN_END:
            rows.append(dict(asset_id=it.asset_id, equipment_class=it.equipment_class,
                             last_renewal=it.last_renewal.date(), interval_months=it.months, planned_date=d.date(),
                             note=f"catch-up: was due {it.due.date()}" if late else ""))
            d, late = d + pd.Timedelta(days=it.T), False
    return pd.DataFrame(rows).sort_values(["planned_date", "asset_id"]).reset_index(drop=True)


def validate(summary: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Judge every policy with the TRUE Weibull parameters (known only because the data is synthetic)."""
    out = []
    for s in summary.itertuples():
        t = truth.set_index("equipment_class").loc[s.equipment_class]
        cp, cf, tb, te = t.planned_cost_usd, t.failure_cost_usd, t.true_beta, t.true_eta_days
        T_opt, c_opt = optimal_interval(tb, te, cp, cf)
        c_rtf = cost_rate(np.inf, tb, te, cp, cf)
        c_cur = cost_rate(s.x_cur, tb, te, cp, cf)
        c_rec = cost_rate(s.x_T_plan, tb, te, cp, cf)
        lo, hi = map(float, s.beta_90ci.split("-"))
        possible = c_cur - c_opt
        out.append(dict(
            equipment_class=s.equipment_class, true_beta=tb, beta_90ci=s.beta_90ci, true_beta_inside_ci=lo <= tb <= hi,
            true_optimal_interval_days=round(T_opt) if np.isfinite(T_opt) else None,
            recommended_interval_days=round(s.x_T_plan) if np.isfinite(s.x_T_plan) else None,
            decision_matches_truth=bool(np.isfinite(T_opt) == np.isfinite(s.x_T_plan)),
            true_cost_per_year_run_to_failure=round(365.25 * c_rtf),
            true_cost_per_year_current=round(365.25 * c_cur),
            true_cost_per_year_recommended=round(365.25 * c_rec),
            true_cost_per_year_optimal=round(365.25 * c_opt),
            share_of_possible_saving_captured=round((c_cur - c_rec) / possible, 3) if possible > 1e-9 else None,
        ))
    v = pd.DataFrame(out)
    per_asset = summary.set_index("equipment_class").assets
    w = v.set_index("equipment_class")
    total = {k: int((w[k] * per_asset).sum()) for k in
             ["true_cost_per_year_run_to_failure", "true_cost_per_year_current", "true_cost_per_year_recommended",
              "true_cost_per_year_optimal"]}
    return dict(per_class=out, fleet_cost_per_year_usd=total)


def run():
    OUT.mkdir(exist_ok=True)
    ev_raw = pd.read_csv(DATA / "cmms_events.csv")
    classes = pd.read_csv(DATA / "equipment_classes.csv")
    truth = pd.read_csv(DATA / "ground_truth" / "true_weibull.csv")
    summary, iv, ev, clean_log = analyse(ev_raw, classes)
    plan = schedule(summary, ev)
    val = validate(summary, truth)
    val["cleaning"] = clean_log
    summary.drop(columns=[c for c in summary if c.startswith("x_")]).to_csv(OUT / "class_summary.csv", index=False)
    iv.to_csv(OUT / "life_intervals.csv", index=False)
    plan.to_csv(OUT / "pm_schedule_2026.csv", index=False)
    (OUT / "validation.json").write_text(json.dumps(val, indent=2, default=str))
    return summary, iv, plan, val, truth
