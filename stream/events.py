"""Synthetic card-transaction stream with controllable distribution shift.

Real fraud data cannot be redistributed, and the public sets are static, which
makes them useless for the thing this project is about: what happens to a
deployed model when the world moves underneath it.

So the generator owns the ground truth. Fraud is produced by an explicit
scoring function, which means the drift can be turned on deliberately and in one
of two genuinely different flavours:

- **Covariate shift** -- the inputs move, the rule stays. Amounts inflate, more
  transactions come from abroad. A monitor watching feature distributions sees
  this immediately.
- **Concept drift** -- the inputs look identical, the rule changes. Fraudsters
  switch from large single hits to small card-not-present probes. Feature
  monitoring is *blind* to this, and only the label-dependent metrics catch it.

Distinguishing those two is the entire point of the drift chapter, and a monitor
that only computes PSI on features will confidently report "all clear" through
the second one.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

FEATURES = [
    "amount",
    "hour",
    "distance_from_home_km",
    "txn_velocity_1h",
    "merchant_risk",
    "card_not_present",
    "foreign_country",
    "account_age_days",
]


@dataclass
class Event:
    event_id: int
    produced_ns: int
    amount: float
    hour: int
    distance_from_home_km: float
    txn_velocity_1h: int
    merchant_risk: float
    card_not_present: int
    foreign_country: int
    account_age_days: int
    is_fraud: int

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @staticmethod
    def from_json(raw: bytes) -> "Event":
        return Event(**json.loads(raw))

    def vector(self) -> list:
        return [getattr(self, f) for f in FEATURES]


class Generator:
    """Emits transactions. `regime` selects the data-generating process."""

    def __init__(self, seed: int = 0, regime: str = "baseline", fraud_rate: float = 0.02,
                 signal_strength: float = 2.2):
        self.rng = np.random.default_rng(seed)
        self.regime = regime
        self.fraud_rate = fraud_rate
        # How sharply the features separate fraud from legitimate. Calibrating
        # prevalence alone compresses the logit range until even the riskiest
        # transaction is only ~30% likely to be fraud, which makes the labels
        # close to noise and caps average precision near 0.12 no matter how good
        # the model is. This widens the spread; calibration then re-centres it.
        self.signal_strength = signal_strength
        self.counter = 0
        self._offset = 0.0
        self._offset = self._calibrate()

    def _calibrate(self, draws: int = 8000) -> float:
        """Shift the intercept so the realised fraud rate matches `fraud_rate`.

        Two reasons this is not cosmetic. Hand-picked coefficients gave a 47%
        fraud rate, which makes average precision meaningless and the imbalance
        argument a fiction. And holding prevalence *equal across regimes* is what
        keeps the concept-drift experiment clean: without it, the drifted stream
        differs in base rate as well as in rule, and any metric change could be
        explained by either.
        """
        calibrator = np.random.default_rng(12345)
        saved, self.rng = self.rng, calibrator
        try:
            samples = [self._draw_features() for _ in range(draws)]
        finally:
            self.rng = saved

        logits = self.signal_strength * np.array([self._raw_logit(f) for f in samples])
        low, high = -60.0, 60.0
        for _ in range(80):
            mid = (low + high) / 2
            rate = float(np.mean(1 / (1 + np.exp(-(logits + mid)))))
            if rate > self.fraud_rate:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def _draw_features(self) -> dict:
        rng = self.rng
        covariate = self.regime == "covariate_shift"

        # Amounts are lognormal in every regime; covariate shift inflates them.
        amount = float(rng.lognormal(4.4 if covariate else 3.6, 1.1))
        distance = float(abs(rng.normal(120 if covariate else 25, 90 if covariate else 40)))
        foreign = int(rng.random() < (0.28 if covariate else 0.06))

        return {
            "amount": round(amount, 2),
            "hour": int(rng.integers(0, 24)),
            "distance_from_home_km": round(distance, 1),
            "txn_velocity_1h": int(rng.poisson(2.4 if covariate else 1.2)),
            "merchant_risk": round(float(rng.beta(2, 5)), 4),
            "card_not_present": int(rng.random() < 0.45),
            "foreign_country": foreign,
            "account_age_days": int(rng.integers(5, 3650)),
        }

    def _fraud_probability(self, f: dict) -> float:
        logit = self.signal_strength * self._raw_logit(f) + self._offset
        return float(1 / (1 + np.exp(-logit)))

    def _raw_logit(self, f: dict) -> float:
        """The rule, before calibration. `concept_drift` swaps it for a different one."""
        if self.regime == "concept_drift":
            # Fraud moves to small, fast, card-not-present probing. Note that
            # none of the feature *distributions* change -- only what they mean.
            score = (
                -3.6
                + 1.9 * (f["amount"] < 40)
                + 1.6 * f["card_not_present"]
                + 0.55 * min(f["txn_velocity_1h"], 8)
                + 1.1 * (f["account_age_days"] < 90)
                - 0.4 * (f["distance_from_home_km"] > 200)
            )
        else:
            score = (
                -4.2
                + 1.35 * np.log1p(f["amount"]) / 2.0
                + 2.4 * f["merchant_risk"]
                + 1.5 * f["foreign_country"]
                + 0.9 * f["card_not_present"]
                + 0.45 * min(f["txn_velocity_1h"], 8)
                + 1.2 * (f["hour"] < 5)
                - 0.5 * (f["account_age_days"] > 1000)
            )
        return float(score)

    def next(self) -> Event:
        f = self._draw_features()
        p = self._fraud_probability(f)
        self.counter += 1
        return Event(
            event_id=self.counter,
            produced_ns=time.time_ns(),
            is_fraud=int(self.rng.random() < p),
            **f,
        )

    def batch(self, n: int) -> list:
        return [self.next() for _ in range(n)]


def to_arrays(events) -> tuple:
    """(X, y) for training."""
    X = np.array([e.vector() for e in events], dtype=float)
    y = np.array([e.is_fraud for e in events], dtype=int)
    return X, y
