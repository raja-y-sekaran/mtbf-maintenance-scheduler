"""Turn a fitted Weibull into a maintenance decision: time-based replacement interval or not."""
from __future__ import annotations

import numpy as np
from scipy import optimize, special

from .reliability import weibull_mean


def reliability(t, beta, eta):
    return np.exp(-(np.asarray(t, float) / eta) ** beta)


def integral_reliability(T, beta, eta):
    """Expected operating time per cycle under age replacement at T: integral_0^T R(t) dt (closed form)."""
    return eta / beta * special.gamma(1 / beta) * special.gammainc(1 / beta, (np.asarray(T, float) / eta) ** beta)


def cost_rate(T, beta, eta, cp, cf):
    """Long-run cost per day of replacing at age T or on failure, whichever comes first (age replacement).
    C(T) = [cp R(T) + cf (1 - R(T))] / integral_0^T R(t) dt.  T = inf gives run-to-failure: cf / MTTF."""
    if np.isinf(T):
        return cf / weibull_mean(beta, eta)
    R = reliability(T, beta, eta)
    return (cp * R + cf * (1 - R)) / integral_reliability(T, beta, eta)


def optimal_interval(beta, eta, cp, cf):
    """Interval that minimises the cost rate. Returns (T*, cost rate) or (inf, run-to-failure rate)
    when replacing early never pays (beta <= 1, or failure barely costs more than planned work)."""
    rtf = cost_rate(np.inf, beta, eta, cp, cf)
    if beta <= 1:
        return np.inf, rtf
    res = optimize.minimize_scalar(lambda x: cost_rate(np.exp(x), beta, eta, cp, cf),
                                   bounds=(np.log(eta * 0.01), np.log(eta * 5)), method="bounded")
    T = float(np.exp(res.x))
    return (T, float(res.fun)) if res.fun < rtf else (np.inf, rtf)


def recommend(beta, beta_ci, eta, cp, cf, min_saving=0.10):
    """Decision rule, written to be cautious with small failure counts:
      * if the lower bound of beta is not clearly above 1 (<= 1.1), the data do not prove wear-out,
        so no time-based replacement: run to failure, or use condition monitoring;
      * otherwise use the cost-optimal interval, but only if it saves at least `min_saving` against
        run-to-failure (small savings are not worth the extra planned work and its own risks)."""
    T, c = optimal_interval(beta, eta, cp, cf)
    rtf = cost_rate(np.inf, beta, eta, cp, cf)
    saving = 1 - c / rtf
    if beta_ci[0] <= 1.1:
        return dict(strategy="No time-based replacement (no proven wear-out)", interval_days=np.nan,
                    saving_vs_rtf=0.0, reason=f"beta lower bound {beta_ci[0]:.2f} <= 1.1")
    if not np.isfinite(T) or saving < min_saving:
        return dict(strategy="Run to failure / condition-based", interval_days=np.nan, saving_vs_rtf=0.0,
                    reason=f"optimal interval saves only {saving:.0%}")
    return dict(strategy="Time-based replacement", interval_days=T, saving_vs_rtf=saving,
                reason=f"wear-out (beta {beta:.2f}), saves {saving:.0%} vs run-to-failure")
