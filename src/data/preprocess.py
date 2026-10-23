"""Clean the CMMS extract: drop entries that renew nothing and duplicate failure work orders."""
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
