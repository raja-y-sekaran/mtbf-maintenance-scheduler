"""Weibull probability plots, cost-rate curves and the 2026 schedule."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import DOCS, PLAN_END, PLAN_START
from .policy import cost_rate
from .reliability import johnson_ranks

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
CLASS_COLOR = {"Pump mechanical seal": BLUE, "Pump bearing": ORANGE, "Motor bearing": AQUA, "Compressor valve": VIOLET}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def draw(summary, iv, plan, truth):
    DOCS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK})
    tr = truth.set_index("equipment_class")
    fitted = summary[summary.strategy != "Insufficient failure data"]

    # 1. Weibull probability plots: straight line = Weibull; slope = beta.
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), dpi=150)
    for ax, s in zip(axes.flat, fitted.itertuples()):
        _style(ax)
        g = iv[iv.equipment_class == s.equipment_class]
        x, F = johnson_ranks(g.days.values, g.failed.values)
        ax.scatter(np.log(x), np.log(-np.log(1 - F)), s=16, color=BLUE, zorder=3, label="failures (adjusted ranks)")
        tt = np.geomspace(max(g.days.min(), 5), g.days.max() * 1.5, 100)
        ax.plot(np.log(tt), s.x_beta * (np.log(tt) - np.log(s.x_eta)), color=ORANGE, linewidth=1.8, label="fitted Weibull")
        t0 = tr.loc[s.equipment_class]
        ax.plot(np.log(tt), t0.true_beta * (np.log(tt) - np.log(t0.true_eta_days)), color=MUTED, linewidth=1,
                linestyle="--", label="true (synthetic)")
        ticks = [t for t in (30, 100, 300, 1000, 3000) if tt[0] <= t <= tt[-1]]
        ax.set_xticks(np.log(ticks), [f"{t:g}" for t in ticks])
        pticks = [0.01, 0.05, 0.2, 0.5, 0.9, 0.99]
        ax.set_yticks(np.log(-np.log(1 - np.array(pticks))), [f"{p:.0%}" for p in pticks])
        ax.set_ylim(np.log(-np.log(1 - 0.005)), np.log(-np.log(1 - 0.995)))
        ax.set_title(f"{s.equipment_class}\nbeta {s.weibull_beta:.2f} (90% CI {s.beta_90ci}), true {t0.true_beta}",
                     fontsize=9, loc="left")
        ax.set_xlabel("Days in service (log)", fontsize=8)
        ax.set_ylabel("Share failed", fontsize=8)
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Weibull fits per equipment class: slope above 1 means wear-out", fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Synthetic CMMS history, 60 assets, 2016-2025. Suspensions (PM replacements, end of record) "
             "are included in the fit and the ranks.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(DOCS / "weibull_fits.png")
    plt.close(fig)

    # 2. Cost per asset-year against replacement interval.
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), dpi=150)
    for ax, s in zip(axes.flat, fitted.itertuples()):
        _style(ax)
        t0 = tr.loc[s.equipment_class]
        cp, cf = t0.planned_cost_usd, t0.failure_cost_usd
        T = np.linspace(30, 4 * s.x_eta, 300)
        c = [365.25 * cost_rate(x, s.x_beta, s.x_eta, cp, cf) for x in T]
        rtf = 365.25 * cost_rate(np.inf, s.x_beta, s.x_eta, cp, cf)
        ax.plot(T / 30.44, c, color=BLUE, linewidth=1.8)
        ax.axhline(rtf, color=MUTED, linestyle="--", linewidth=1)
        ax.text(T[-1] / 30.44, rtf, "run to failure", fontsize=7, color=MUTED, ha="right", va="bottom")
        if np.isfinite(s.x_cur):
            ax.scatter([s.x_cur / 30.44], [365.25 * cost_rate(s.x_cur, s.x_beta, s.x_eta, cp, cf)], s=40,
                       color=MUTED, marker="s", zorder=3)
            ax.annotate("current PM", (s.x_cur / 30.44, 365.25 * cost_rate(s.x_cur, s.x_beta, s.x_eta, cp, cf)),
                        xytext=(6, 6), textcoords="offset points", fontsize=7.5)
        if np.isfinite(s.x_T_plan):
            y = 365.25 * cost_rate(s.x_T_plan, s.x_beta, s.x_eta, cp, cf)
            ax.scatter([s.x_T_plan / 30.44], [y], s=60, color=ORANGE, marker="D", zorder=4, edgecolor="white")
            ax.annotate(f"recommended: {int(s.recommended_interval_months)} months", (s.x_T_plan / 30.44, y),
                        xytext=(6, -12), textcoords="offset points", fontsize=7.5)
        else:
            ax.text(0.98, 0.95, "no time-based replacement", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=INK)
        ax.set_ylim(0, max(c[len(c) // 10:] + [rtf]) * 1.25)
        ax.set_title(s.equipment_class, fontsize=9, loc="left")
        ax.set_xlabel("Replace every ... months", fontsize=8)
        ax.set_ylabel("Cost per asset-year (USD)", fontsize=8)
    fig.suptitle("Maintenance cost against replacement interval (fitted parameters)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(DOCS / "cost_curves.png")
    plt.close(fig)

    # 3. The 2026 plan.
    p = plan.copy()
    p["planned_date"] = pd.to_datetime(p.planned_date)
    order = (p.groupby("asset_id").agg(c=("equipment_class", "first"), d=("planned_date", "min"))
             .sort_values(["c", "d"]).index.tolist())
    y = {a: i for i, a in enumerate(order)}
    fig, ax = plt.subplots(figsize=(10, 0.24 * len(order) + 1.6), dpi=150)
    _style(ax)
    for cls, g in p.groupby("equipment_class"):
        late = g.note != ""
        ax.scatter(g.planned_date[~late], g.asset_id[~late].map(y), s=46, color=CLASS_COLOR[cls], marker="D",
                   edgecolor="white", linewidth=0.8, label=cls, zorder=3)
        ax.scatter(g.planned_date[late], g.asset_id[late].map(y), s=46, facecolor="white",
                   edgecolor=CLASS_COLOR[cls], linewidth=1.6, marker="D", zorder=3)
    ax.axvspan(PLAN_START, PLAN_START + pd.Timedelta(days=90), color="#f4f3f0", zorder=0)
    ax.text(PLAN_START + pd.Timedelta(days=3), len(order) - 0.3, "Q1: catch-up of overdue assets (open markers)",
            fontsize=7.5, color=MUTED, va="bottom")
    ax.set_yticks(range(len(order)), order, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(PLAN_START - pd.Timedelta(days=5), PLAN_END + pd.Timedelta(days=5))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(f"Planned replacements, 2026 ({len(p)} jobs on {p.asset_id.nunique()} assets)", fontsize=11.5,
                 loc="left", pad=24)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(DOCS / "pm_schedule_2026.png", bbox_inches="tight")
    plt.close(fig)
