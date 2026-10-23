"""Turn cleaned CMMS events into life intervals (time between renewals): the feature table for the Weibull fit."""
from __future__ import annotations

import pandas as pd


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
