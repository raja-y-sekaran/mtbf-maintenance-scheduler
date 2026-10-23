"""Clean the CMMS extract and turn it into life intervals (time between renewals)."""
from __future__ import annotations

import pandas as pd

RENEWALS = {"commissioned", "failure", "pm_replacement"}


def clean_events(ev: pd.DataFrame, dup_window_days: int = 3) -> tuple[pd.DataFrame, dict]:
    """Drop entries that renew nothing (inspections) and duplicate failure work orders for the same event."""
    ev = ev.copy()
    ev["event_date"] = pd.to_datetime(ev.event_date)
    log = {"rows_in": len(ev), "inspections_dropped": int((ev.event_type == "inspection").sum())}
    ev = ev[ev.event_type.isin(RENEWALS)].sort_values(["asset_id", "event_date"])
    gap = ev.groupby("asset_id").event_date.diff().dt.days
    prev_type = ev.groupby("asset_id").event_type.shift()
    dup = (ev.event_type == "failure") & (prev_type == "failure") & (gap <= dup_window_days)
    log["duplicate_failures_dropped"] = int(dup.sum())
    ev = ev[~dup]
    log["rows_out"] = len(ev)
    return ev.reset_index(drop=True), log


def build_intervals(ev: pd.DataFrame, obs_end: str) -> pd.DataFrame:
    """One row per life interval. Ends in a failure (event), a PM replacement or the end of the
    record (both right-censored suspensions)."""
    rows = []
    end_all = pd.Timestamp(obs_end)
    for (asset, cls), g in ev.groupby(["asset_id", "equipment_class"]):
        g = g.sort_values("event_date")
        dates, types = list(g.event_date), list(g.event_type)
        for i in range(len(dates)):
            start = dates[i]
            end, kind = (dates[i + 1], types[i + 1]) if i + 1 < len(dates) else (end_all, "end_of_record")
            rows.append(dict(asset_id=asset, equipment_class=cls, start=start, end=end,
                             days=max((end - start).days, 1), ended_by=kind, failed=kind == "failure"))
    return pd.DataFrame(rows)
