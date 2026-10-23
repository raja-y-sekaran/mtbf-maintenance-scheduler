"""Run the MTBF / Weibull analysis, the 2026 schedule, the validation against the true parameters, and the figures."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mtbf import figures, pipeline  # noqa: E402

if __name__ == "__main__":
    summary, iv, plan, val, truth = pipeline.run()
    figures.draw(summary, iv, plan, truth)
    cols = ["equipment_class", "failures", "crude_mtbf_days", "weibull_beta", "beta_90ci", "strategy",
            "recommended_interval_months"]
    print(summary[cols].to_string(index=False))
    print(f"\n{len(plan)} planned replacements in 2026")
    print(json.dumps(val["fleet_cost_per_year_usd"], indent=2))
