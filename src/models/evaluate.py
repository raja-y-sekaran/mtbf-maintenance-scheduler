"""Robustness: repeat the whole analysis on many synthetic plants and judge it with the true parameters.
Does the 90% interval for beta cover the truth, is the replace / don't-replace decision right, and how much
of the possible saving does the recommended interval capture?"""
import pandas as pd

from src.data import generate_synthetic as gen
from src.pipeline import analyse, validate


def robustness(n=100):
    rows, fleet = [], []
    for seed in range(1, n + 1):
        ev, truth = gen.generate(seed)
        classes = truth.drop(columns=["true_beta", "true_eta_days"])
        summary, *_ = analyse(ev, classes)
        v = validate(summary, truth)
        for r in v["per_class"]:
            rows.append(dict(seed=seed, **r))
        fleet.append(v["fleet_cost_per_year_usd"])
    d, fl = pd.DataFrame(rows), pd.DataFrame(fleet)
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
    return d, fl
