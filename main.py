"""Run the project end to end (course layout: data -> features -> model -> maintenance policy -> evaluation).

    python main.py                         # all stages on the data in data/raw/
    python main.py --regenerate --seed 7   # make a new synthetic plant first
    python main.py --robustness 100        # judge the method on 100 synthetic plants (data/raw untouched)
"""
import argparse
import json
import sys

from src import pipeline
from src.data import generate_synthetic
from src.models import train
from src.models.evaluate import robustness
from src.utils import plots


def run_all(quiet=False):
    summary, iv, plan, val, truth = pipeline.run()
    plots.draw(summary, iv, plan, truth)
    path = train.save_fits(summary)
    if not quiet:
        cols = ["equipment_class", "failures", "crude_mtbf_days", "weibull_beta", "beta_90ci", "strategy",
                "recommended_interval_months"]
        print(summary[cols].to_string(index=False))
        print(f"\n{len(plan)} planned replacements in 2026")
        print(json.dumps(val["fleet_cost_per_year_usd"], indent=2))
        print(f"fitted model saved to {path.relative_to(path.parents[1])}")
    return summary, val


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--robustness", type=int, nargs="?", const=100, default=None, metavar="N")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.robustness:
        robustness(a.robustness)
    else:
        if a.regenerate:
            sys.argv = ["generate_synthetic", "--seed", str(a.seed)]
            generate_synthetic.main()
        run_all(a.quiet)
