"""Save the fitted Weibull model per equipment class (the 'model' of this project) with a model card."""
import json

import numpy as np

from src.utils.paths import MODELS


def save_fits(summary):
    fits = {}
    for s in summary.itertuples():
        if s.strategy == "Insufficient failure data":
            fits[s.equipment_class] = dict(failures=int(s.failures), strategy=s.strategy)
            continue
        fits[s.equipment_class] = dict(
            failures=int(s.failures), suspensions=int(s.suspensions), weibull_beta=float(s.x_beta),
            beta_90ci=s.beta_90ci, weibull_eta_days=float(s.x_eta), strategy=s.strategy, reason=s.reason,
            recommended_interval_months=None if np.isnan(s.recommended_interval_months)
            else int(s.recommended_interval_months))
    (MODELS / "weibull_fits.json").write_text(json.dumps(fits, indent=2))
    (MODELS / "model_card.json").write_text(json.dumps(dict(
        model="Two-parameter Weibull per equipment class, maximum likelihood with right-censored suspensions",
        beta_interval="90% profile-likelihood interval",
        decision_rule="time-based replacement only if the beta lower bound > 1.1 and the cost-optimal interval "
                      "saves >= 10% vs run-to-failure; interval rounded down to whole months",
        training_data="synthetic CMMS history (src/data/generate_synthetic.py), observed to 2025-12-31",
        intended_use="demonstration on synthetic data; refit and re-check on a plant's own history before any use",
    ), indent=2))
    return MODELS / "weibull_fits.json"
