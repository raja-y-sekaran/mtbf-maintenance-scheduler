import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.maintenance.policy import cost_rate, integral_reliability, optimal_interval, recommend  # noqa: E402
from src.data.preprocess import clean_events  # noqa: E402
from src.features.life_intervals import build_intervals  # noqa: E402
from src.models.reliability import (beta_profile_ci, johnson_ranks, mtbf_with_bounds, weibull_mean,  # noqa: E402
                              weibull_mle)


def test_mtbf_chi_square_bounds_textbook_case():
    # 10,000 h, 5 failures, 90% two-sided: 2T/chi2(0.95, 12) and 2T/chi2(0.05, 10)
    point, lo, hi = mtbf_with_bounds(10_000, 5, 0.90)
    assert point == 2000
    assert lo == pytest.approx(20_000 / 21.0261, rel=1e-4)
    assert hi == pytest.approx(20_000 / 3.9403, rel=1e-4)


def _sample(beta, eta, n, censor_at, seed=0):
    rng = np.random.default_rng(seed)
    t = eta * rng.weibull(beta, n)
    failed = t < censor_at
    return np.minimum(t, censor_at), failed


def test_weibull_mle_recovers_parameters_with_censoring():
    t, f = _sample(2.5, 1000, 5000, censor_at=1200)
    b, e = weibull_mle(t, f)
    assert b == pytest.approx(2.5, rel=0.05) and e == pytest.approx(1000, rel=0.03)


def test_weibull_mle_matches_scipy_censored_fit():
    t, f = _sample(1.7, 800, 60, censor_at=900, seed=3)
    b, e = weibull_mle(t, f)
    sb, _, se = stats.weibull_min.fit(stats.CensoredData(uncensored=t[f], right=t[~f]), floc=0)
    assert b == pytest.approx(sb, rel=1e-4) and e == pytest.approx(se, rel=1e-4)


def test_profile_ci_contains_estimate_and_narrows_with_data():
    t, f = _sample(2.0, 500, 40, censor_at=700, seed=1)
    lo, hi = beta_profile_ci(t, f)
    assert lo < weibull_mle(t, f)[0] < hi
    t2, f2 = _sample(2.0, 500, 400, censor_at=700, seed=1)
    lo2, hi2 = beta_profile_ci(t2, f2)
    assert hi2 - lo2 < (hi - lo) / 2


def test_weibull_mean_exponential_case():
    assert weibull_mean(1.0, 700) == pytest.approx(700)


def test_integral_reliability_closed_form_matches_quadrature():
    num, _ = integrate.quad(lambda x: np.exp(-(x / 900) ** 1.8), 0, 600)
    assert integral_reliability(600, 1.8, 900) == pytest.approx(num, rel=1e-8)


def test_cost_rate_tends_to_run_to_failure():
    assert cost_rate(1e7, 2.0, 1000, 1, 10) == pytest.approx(cost_rate(np.inf, 2.0, 1000, 1, 10), rel=1e-6)


def test_no_optimal_interval_without_wear_out():
    assert np.isinf(optimal_interval(1.0, 1000, 1, 10)[0])
    assert np.isinf(optimal_interval(0.7, 1000, 1, 10)[0])


def test_optimal_interval_matches_brute_force():
    T, c = optimal_interval(3.0, 1000, 1, 20)
    grid = np.linspace(50, 3000, 20000)
    best = grid[np.argmin([cost_rate(x, 3.0, 1000, 1, 20) for x in grid])]
    assert T == pytest.approx(best, rel=2e-3)


def test_recommend_refuses_time_based_plan_without_proven_wear_out():
    assert recommend(1.6, (0.9, 2.4), 1000, 1, 20)["strategy"].startswith("No time-based")
    assert recommend(2.5, (1.8, 3.4), 1000, 1, 20)["strategy"] == "Time-based replacement"


def test_johnson_ranks_without_suspensions_equal_benard():
    x, F = johnson_ranks(np.array([10, 20, 30, 40]), np.array([True] * 4))
    assert F == pytest.approx([(i - 0.3) / 4.4 for i in range(1, 5)])


def test_cleaning_and_intervals():
    ev = pd.DataFrame([
        dict(work_order="W1", asset_id="A", equipment_class="X", event_date="2020-01-01", event_type="commissioned"),
        dict(work_order="W2", asset_id="A", equipment_class="X", event_date="2020-06-01", event_type="inspection"),
        dict(work_order="W3", asset_id="A", equipment_class="X", event_date="2021-01-01", event_type="failure"),
        dict(work_order="W4", asset_id="A", equipment_class="X", event_date="2021-01-02", event_type="failure"),
        dict(work_order="W5", asset_id="A", equipment_class="X", event_date="2022-01-01", event_type="pm_replacement"),
    ])
    clean, log = clean_events(ev)
    assert log["inspections_dropped"] == 1 and log["duplicate_failures_dropped"] == 1
    iv = build_intervals(clean, "2022-12-31")
    assert iv.ended_by.tolist() == ["failure", "pm_replacement", "end_of_record"]
    assert iv.failed.tolist() == [True, False, False]
    assert iv.days.tolist() == [366, 365, 364]
