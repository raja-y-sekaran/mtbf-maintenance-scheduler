"""Repeat the whole analysis on many synthetic plants (seeds) and judge it with the true parameters:
does the 90% interval for beta cover the truth, is the replace / don't-replace decision right, and how
much of the possible saving does the recommended interval capture?   python scripts/robustness.py [n]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

import generate_synthetic as gen  # noqa: E402
from mtbf.pipeline import analyse, validate  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rows, fleet = [], []
for seed in range(1, n + 1):
    ev, truth = gen.generate(seed)
    classes = truth.drop(columns=["true_beta", "true_eta_days"])
    summary, *_ = analyse(ev, classes)
    v = validate(summary, truth)
    for r in v["per_class"]:
        rows.append(dict(seed=seed, **r))
    fleet.append(v["fleet_cost_per_year_usd"])
d = pd.DataFrame(rows)
fl = pd.DataFrame(fleet)
print(f"{n} synthetic plants\n")
print("| Class | 90% CI covers true beta | Decision matches truth | Median share of possible saving captured |")
print("|---|---|---|---|")
for c, g in d.groupby("equipment_class", sort=False):
    cap = g.share_of_possible_saving_captured.dropna()
    print(f"| {c} | {g.true_beta_inside_ci.mean():.0%} | {g.decision_matches_truth.mean():.0%} | "
          f"{'n/a' if cap.empty else f'{cap.median():.0%}'} |")
ratio = fl.true_cost_per_year_recommended / fl.true_cost_per_year_current
gap = fl.true_cost_per_year_recommended / fl.true_cost_per_year_optimal - 1
print(f"\nFleet cost, recommended vs current practice: median {1 - ratio.median():.0%} lower "
      f"(5th-95th percentile {1 - ratio.quantile(0.95):.0%} to {1 - ratio.quantile(0.05):.0%})")
print(f"Fleet cost, recommended vs the true optimum: median {gap.median():.1%} above "
      f"(95th percentile {gap.quantile(0.95):.1%})")
