import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def _history(beta, eta, n=80, censor=1500, seed=1):
    rng = np.random.default_rng(seed)
    t = eta * rng.weibull(beta, n)
    return [dict(days=float(min(x, censor)), failed=bool(x < censor)) for x in t]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_wear_out_class_gets_an_interval():
    r = client.post("/fit", json=dict(intervals=_history(3.0, 1000), planned_cost_usd=3000, failure_cost_usd=30000)).json()
    assert r["strategy"] == "Time-based replacement" and r["recommended_interval_months"] >= 1
    assert r["beta_90ci"][0] < 3.0 < r["beta_90ci"][1]
    assert r["fitted_cost_per_year_recommended_usd"] < r["fitted_cost_per_year_current_usd"]


def test_random_failures_get_no_time_based_replacement():
    r = client.post("/fit", json=dict(intervals=_history(1.0, 1000), planned_cost_usd=3000, failure_cost_usd=30000)).json()
    assert r["strategy"] != "Time-based replacement" and r["recommended_interval_months"] is None


def test_too_few_failures():
    iv = [dict(days=500, failed=False)] * 5 + [dict(days=300, failed=True)] * 2
    r = client.post("/fit", json=dict(intervals=iv, planned_cost_usd=1, failure_cost_usd=2)).json()
    assert r["strategy"] == "Insufficient failure data"


def test_rejects_bad_input():
    assert client.post("/fit", json=dict(intervals=[dict(days=-1, failed=True)], planned_cost_usd=1,
                                         failure_cost_usd=2)).status_code == 422
