"""MTBF with chi-square bounds, Weibull fitting with right-censored data, and probability-plot ranks."""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats


def mtbf_with_bounds(total_time: float, failures: int, confidence: float = 0.90) -> tuple[float, float, float]:
    """Point MTBF and two-sided chi-square bounds for a time-terminated record (constant-rate assumption).
    lower = 2T / chi2(1 - a/2, 2r + 2),  upper = 2T / chi2(a/2, 2r)."""
    a = 1 - confidence
    point = total_time / failures if failures else np.inf
    lower = 2 * total_time / stats.chi2.ppf(1 - a / 2, 2 * failures + 2)
    upper = 2 * total_time / stats.chi2.ppf(a / 2, 2 * failures) if failures else np.inf
    return point, lower, upper


def _profile_loglik(b, t, failed):
    """Weibull log-likelihood maximised over eta for a fixed beta. eta has a closed form:
    eta(b) = (sum over all items of t^b / r)^(1/b), r = number of failures."""
    r = failed.sum()
    eta_b = (np.sum(t ** b) / r) ** (1 / b)
    return r * np.log(b) - r * b * np.log(eta_b) + (b - 1) * np.sum(np.log(t[failed])) - r, eta_b


def weibull_mle(t, failed) -> tuple[float, float]:
    """Two-parameter Weibull maximum likelihood with right-censored (suspended) items.
    log L = sum_failures [ln b - b ln eta + (b - 1) ln t] - sum_all (t / eta)^b.
    Solved as a 1-D search over beta on the profile likelihood. Times are scaled to [0, 1]
    first so t^beta stays finite; eta is scaled back at the end."""
    t, failed = np.asarray(t, float), np.asarray(failed, bool)
    if failed.sum() < 2:
        raise ValueError("need at least two failures")
    scale = t.max()
    ts = t / scale
    res = optimize.minimize_scalar(lambda lb: -_profile_loglik(np.exp(lb), ts, failed)[0],
                                   bounds=(np.log(0.05), np.log(30)), method="bounded",
                                   options=dict(xatol=1e-10))
    b = float(np.exp(res.x))
    return b, float(_profile_loglik(b, ts, failed)[1] * scale)


def beta_profile_ci(t, failed, confidence: float = 0.90) -> tuple[float, float]:
    """Likelihood-ratio (profile) confidence interval for beta: every beta with
    2 [lnL(best) - lnL(beta)] <= chi2(confidence, 1). Better behaved than the bootstrap with few failures."""
    t, failed = np.asarray(t, float), np.asarray(failed, bool)
    b_hat, _ = weibull_mle(t, failed)
    ts = t / t.max()  # beta does not depend on the time unit
    ll_max = _profile_loglik(b_hat, ts, failed)[0]
    crit = stats.chi2.ppf(confidence, 1) / 2
    f = lambda b: ll_max - _profile_loglik(b, ts, failed)[0] - crit
    lo = optimize.brentq(f, max(b_hat / 50, 0.01), b_hat)
    hi = optimize.brentq(f, b_hat, max(b_hat * 20, 60))
    return float(lo), float(hi)


def weibull_bootstrap(t, failed, n_boot: int = 500, seed: int = 0, confidence: float = 0.90):
    """Percentile bootstrap intervals for beta and eta (resampling whole life intervals).
    Kept for comparison: with 12-20 failures it under-covers (see README)."""
    rng = np.random.default_rng(seed)
    t, failed = np.asarray(t, float), np.asarray(failed, bool)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(t), len(t))
        if failed[i].sum() >= 2:
            out.append(weibull_mle(t[i], failed[i]))
    out = np.array(out)
    a = (1 - confidence) / 2 * 100
    return np.percentile(out[:, 0], [a, 100 - a]), np.percentile(out[:, 1], [a, 100 - a])


def weibull_mean(beta: float, eta: float) -> float:
    """Mean time to failure of a Weibull: eta * Gamma(1 + 1/beta)."""
    return eta * special.gamma(1 + 1 / beta)


def johnson_ranks(t, failed):
    """Johnson's adjusted ranks with Benard's median-rank approximation, for a probability plot
    that accounts for suspensions. Returns (failure times, unreliability estimates)."""
    order = np.argsort(t, kind="stable")
    t, failed = np.asarray(t)[order], np.asarray(failed)[order]
    n, prev, xs, fs = len(t), 0.0, [], []
    for i, (ti, fi) in enumerate(zip(t, failed)):
        if fi:
            reverse_rank = n - i
            prev = prev + (n + 1 - prev) / (1 + reverse_rank)
            xs.append(ti)
            fs.append((prev - 0.3) / (n + 0.4))
    return np.array(xs), np.array(fs)
