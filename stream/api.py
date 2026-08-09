"""FastAPI scoring endpoint.

This exists to be measured against the inline consumer path, not because the
pipeline needs it. Same model, same features, same machine -- the only
difference is an HTTP hop, so the gap between the two latency distributions is
a clean measurement of what the network layer costs.

    uvicorn stream.api:app --port 8000
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import model as model_mod
from .events import FEATURES

app = FastAPI(title="drift-stream scorer")
_state = {"model": None, "version": None}


class Transaction(BaseModel):
    amount: float
    hour: int = Field(ge=0, le=23)
    distance_from_home_km: float
    txn_velocity_1h: int
    merchant_risk: float
    card_not_present: int
    foreign_country: int
    account_age_days: int

    def vector(self) -> list:
        return [getattr(self, f) for f in FEATURES]


@app.on_event("startup")
def _load():
    try:
        _state["model"], _state["version"] = model_mod.load()
    except Exception:
        _state["model"] = None  # /score reports this rather than 500-ing


@app.get("/health")
def health():
    return {"ok": _state["model"] is not None, "model_version": _state["version"]}


@app.post("/score")
def score(txn: Transaction):
    if _state["model"] is None:
        return {"error": "no model registered -- run `python run.py train`"}
    t0 = time.perf_counter_ns()
    probability = model_mod.score_one(_state["model"], txn.vector())
    return {
        "fraud_probability": probability,
        "model_version": _state["version"],
        "inference_us": (time.perf_counter_ns() - t0) / 1000.0,
    }


@app.post("/reload")
def reload_model():
    """Pick up a newly registered version without restarting the process."""
    _load()
    return {"model_version": _state["version"]}
