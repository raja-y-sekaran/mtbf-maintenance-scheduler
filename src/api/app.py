"""FastAPI service: fit a Weibull to one equipment class's life intervals and return the PM recommendation.

    uvicorn src.api.app:app --reload      # then open http://127.0.0.1:8000/docs
"""
import json
import math
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.maintenance.policy import cost_rate, recommend
from src.models.reliability import beta_profile_ci, mtbf_with_bounds, weibull_mean, weibull_mle
from src.pipeline import practical_interval
from src.utils.paths import MODELS

app = FastAPI(title="MTBF maintenance scheduler", version="1.0",
                description="Weibull with suspensions and an age-replacement decision. Synthetic-data demo.")


class Interval(BaseModel):
    days: float = Field(gt=0, description="Run time from renewal (install / repair / replacement) to the end")
    failed: bool = Field(description="True if the interval ended in a failure; False for a PM replacement or "
                                     "still running (a suspension)")


class ClassHistory(BaseModel):
    intervals: List[Interval]
    planned_cost_usd: float = Field(gt=0)
    failure_cost_usd: float = Field(gt=0)
    current_pm_days: Optional[float] = Field(None, gt=0, description="Current replacement interval, if any")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def classes():
    """The fits saved by the last `python main.py` run."""
    path = MODELS / "weibull_fits.json"
    if not path.exists():
        raise HTTPException(404, "run python main.py first")
    return json.loads(path.read_text())


@app.post("/fit")
def fit(h: ClassHistory):
    t = np.array([i.days for i in h.intervals])
    f = np.array([i.failed for i in h.intervals])
    r = int(f.sum())
    if r < 3:
        return dict(failures=r, strategy="Insufficient failure data", reason=f"only {r} failures")
    mtbf, lo, hi = mtbf_with_bounds(t.sum(), r)
    b, e = weibull_mle(t, f)
    blo, bhi = beta_profile_ci(t, f)
    rec = recommend(b, (blo, bhi), e, h.planned_cost_usd, h.failure_cost_usd)
    months, T = practical_interval(rec["interval_days"]) if np.isfinite(rec["interval_days"]) else (None, math.inf)
    cur = h.current_pm_days or math.inf
    per_year = lambda T_: round(365.25 * cost_rate(T_, b, e, h.planned_cost_usd, h.failure_cost_usd))  # noqa: E731
    return dict(failures=r, suspensions=len(t) - r, crude_mtbf_days=round(mtbf), crude_mtbf_90ci=[round(lo), round(hi)],
                weibull_beta=round(b, 2), beta_90ci=[round(blo, 2), round(bhi, 2)], weibull_eta_days=round(e),
                mttf_days=round(weibull_mean(b, e)), strategy=rec["strategy"], reason=rec["reason"],
                recommended_interval_months=months,
                fitted_cost_per_year_current_usd=per_year(cur), fitted_cost_per_year_recommended_usd=per_year(T))
