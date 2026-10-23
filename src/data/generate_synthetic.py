"""Generate a synthetic CMMS failure / replacement history for 60 assets in six equipment classes.

All values are invented. Each class has a TRUE Weibull time-to-failure (shape beta, scale eta, in days)
saved to data/raw/ground_truth/, so the analysis can be checked against the answer.

  * Every asset's history starts at commissioning (2016-2019), so no interval is left-truncated.
  * Repair or replacement restores the item to as-good-as-new (renewal), the usual assumption for
    replaceable components such as seals, bearings and positioners.
  * Some classes already have a time-based replacement in the plan ("current PM"). A PM replacement
    ends an interval without a failure: it is a suspension (right-censored), not a failure.
  * The extract is a little dirty, as real CMMS exports are: duplicate failure work orders and
    inspection-only entries that do not renew anything.

Usage:  python -m src.data.generate_synthetic [--seed 42]
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.utils.paths import DATA_RAW, DATA_SAMPLE, GROUND_TRUTH

OBS_END = date(2025, 12, 31)

# class, assets, beta, eta (days), current PM replacement interval (days or None),
# planned replacement cost, failure cost (repair + lost production), both in USD
CLASSES = [
    ("Pump mechanical seal", 12, 1.6, 900, None, 4_000, 22_000),
    ("Pump bearing", 12, 2.8, 1600, 1460, 3_000, 30_000),
    ("Motor bearing", 10, 2.2, 2200, None, 2_500, 18_000),
    ("Control valve positioner", 10, 0.8, 2600, 730, 1_500, 6_000),
    ("Pressure transmitter", 10, 1.0, 3500, None, 1_200, 5_000),
    ("Compressor valve", 6, 3.2, 500, 540, 6_000, 45_000),
]


def generate(seed: int = 42):
    rng = np.random.default_rng(seed)
    events, truth = [], []
    for cls, n, beta, eta, pm, cp, cf in CLASSES:
        truth.append(dict(equipment_class=cls, true_beta=beta, true_eta_days=eta, current_pm_days=pm,
                          planned_cost_usd=cp, failure_cost_usd=cf))
        code = "".join(w[0] for w in cls.split()).upper()
        for k in range(1, n + 1):
            asset = f"{code}-{100 + k}"
            t = date(2016, 1, 1) + timedelta(days=int(rng.integers(0, 4 * 365)))
            events.append(dict(asset_id=asset, equipment_class=cls, event_date=t, event_type="commissioned"))
            while True:
                ttf = eta * rng.weibull(beta)
                pm_at = None if pm is None else pm + rng.integers(-30, 31)  # PM done within a month of due
                if pm_at is not None and pm_at < ttf:
                    end, kind = t + timedelta(days=int(pm_at)), "pm_replacement"
                else:
                    end, kind = t + timedelta(days=max(int(ttf), 1)), "failure"
                if end > OBS_END:
                    break
                events.append(dict(asset_id=asset, equipment_class=cls, event_date=end, event_type=kind))
                if kind == "failure" and rng.random() < 0.05:  # same failure booked twice
                    events.append(dict(asset_id=asset, equipment_class=cls,
                                       event_date=end + timedelta(days=int(rng.integers(0, 3))), event_type="failure"))
                t = end
            for _ in range(rng.poisson(2)):  # inspections: recorded, but they renew nothing
                d = date(2016, 6, 1) + timedelta(days=int(rng.integers(0, 3400)))
                if d < OBS_END:
                    events.append(dict(asset_id=asset, equipment_class=cls, event_date=d, event_type="inspection"))
    ev = pd.DataFrame(events).sort_values(["asset_id", "event_date"]).reset_index(drop=True)
    ev.insert(0, "work_order", [f"WO{100000 + i}" for i in rng.permutation(len(ev))])
    return ev, pd.DataFrame(truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ev, truth = generate(ap.parse_args().seed)
    ev.to_csv(DATA_RAW / "cmms_events.csv", index=False)
    truth.drop(columns=["true_beta", "true_eta_days"]).to_csv(DATA_RAW / "equipment_classes.csv", index=False)
    truth.to_csv(GROUND_TRUTH / "true_weibull.csv", index=False)
    ev[ev.asset_id.isin(ev.asset_id.drop_duplicates().iloc[::6])].to_csv(DATA_SAMPLE / "cmms_events_sample.csv", index=False)
    print(f"{ev.asset_id.nunique()} assets, {len(ev)} events, "
          f"{(ev.event_type == 'failure').sum()} failure work orders, {(ev.event_type == 'pm_replacement').sum()} PM replacements")


if __name__ == "__main__":
    main()
