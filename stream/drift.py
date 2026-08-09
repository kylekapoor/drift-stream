"""Drift detection: PSI, KS, and the prediction-side checks PSI cannot do.

PSI is implemented here rather than imported. It is about thirty lines, and the
two details that actually decide whether it works in production are both
invisible when you `pip install` it:

1. **Bin edges come from the reference window and are then frozen.** Re-binning
   on the current window compares each distribution to itself and PSI collapses
   toward zero exactly when drift is worst.
2. **Empty bins.** A bin with no current observations makes the ratio zero and
   the logarithm infinite. Everyone clamps; where you clamp sets the ceiling of
   the statistic.

Conventional reading, from credit scorecard practice:

    PSI < 0.10   no meaningful shift
    0.10 - 0.25  moderate, worth investigating
    PSI > 0.25   material shift, act

The limitation worth stating plainly, because it is measured in this repo rather
than asserted: **no unsupervised drift monitor detects pure concept drift.**

Concept drift changes the mapping from inputs to labels while leaving the input
distribution untouched. Feature PSI is blind to it by construction. So is
prediction PSI -- if the covariates are unchanged and the model is unchanged, the
score distribution is *identical*, and PSI on scores is ~0.00 while average
precision collapses from 0.451 to 0.026. Both numbers come from `run.py drift`.

Prediction PSI earns its place for a different reason: it catches shifts that
move the score distribution before any label arrives, including covariate shift
and upstream feature-pipeline breakage. It is an early warning, not a complete
one. Detecting concept drift requires labels, and labels arrive late -- in card
fraud, weeks late, after the chargebacks. That lag is the real operational
problem, and no amount of input monitoring removes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

PSI_MODERATE = 0.10
PSI_MATERIAL = 0.25
# Floor on a bin's share. Without it an empty bin sends PSI to infinity; with it
# the per-bin contribution is capped near log(share/1e-4), which is finite and
# large enough to still dominate the sum.
EPS = 1e-4


@dataclass
class DriftReport:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float

    @property
    def severity(self) -> str:
        if self.psi >= PSI_MATERIAL:
            return "material"
        if self.psi >= PSI_MODERATE:
            return "moderate"
        return "stable"

    @property
    def drifted(self) -> bool:
        return self.psi >= PSI_MATERIAL


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample."""
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = np.asarray(current, dtype=float)
    current = current[np.isfinite(current)]
    if len(reference) < 2 or len(current) < 2:
        return 0.0

    # Quantile edges from the reference only, then frozen. Deduplicated because
    # a feature that is mostly one value (a 0/1 flag) yields repeated edges,
    # which numpy would turn into zero-width bins.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Degenerate, e.g. a binary flag: compare category shares directly.
        values = np.unique(np.concatenate([reference, current]))
        ref_share = np.array([(reference == v).mean() for v in values])
        cur_share = np.array([(current == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
        cur_share = np.histogram(current, bins=edges)[0] / len(current)

    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def compare(reference: np.ndarray, current: np.ndarray, feature_names) -> list:
    """Per-feature PSI and a two-sample KS test."""
    out = []
    for i, name in enumerate(feature_names):
        ref_col, cur_col = reference[:, i], current[:, i]
        try:
            ks = stats.ks_2samp(ref_col, cur_col)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        except ValueError:
            ks_stat, ks_p = 0.0, 1.0
        out.append(DriftReport(name, psi(ref_col, cur_col), ks_stat, ks_p))
    return sorted(out, key=lambda r: -r.psi)


def prediction_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> float:
    """PSI on the model's own output.

    Useful because it needs no labels and fires on anything that moves the score
    distribution: covariate shift, a broken upstream feature, a bad deploy.

    It does NOT detect pure concept drift. Unchanged inputs through an unchanged
    model produce an unchanged score distribution, so this returns ~0.00 in
    exactly the scenario where the model has become worthless. `performance_drop`
    is the only check that catches that, and it needs labels.
    """
    return psi(reference_scores, current_scores, bins=10)


MIN_POSITIVES_FOR_AP = 40


def performance_drop(reference_ap: float, current_ap: float, positives: int,
                     tolerance: float = 0.6,
                     min_positives: int = MIN_POSITIVES_FOR_AP) -> bool:
    """True when windowed average precision falls below `tolerance` of baseline.

    The only detector in this module that sees concept drift, and the only one
    that cannot run until labels arrive.

    The `positives` guard is not defensive padding. At a 2% fraud rate a
    1,000-event window holds about 20 positives, and average precision estimated
    from 20 points is so noisy it swung between 0.14 and 0.55 on *identically
    distributed* baseline windows -- enough to trip a 60%-of-baseline threshold
    repeatedly and trigger retraining on nothing at all. Below `min_positives`
    the estimate is not trustworthy enough to act on, so this abstains.
    """
    if reference_ap <= 0 or positives < min_positives:
        return False
    return current_ap < reference_ap * tolerance


def summarise(reports: list, pred_psi: float | None = None) -> dict:
    drifted = [r for r in reports if r.drifted]
    return {
        "max_psi": max((r.psi for r in reports), default=0.0),
        "mean_psi": float(np.mean([r.psi for r in reports])) if reports else 0.0,
        "n_material": len(drifted),
        "drifted_features": [r.feature for r in drifted],
        "prediction_psi": pred_psi,
        "verdict": (
            "material" if drifted or (pred_psi or 0) >= PSI_MATERIAL
            else "moderate" if any(r.severity == "moderate" for r in reports)
            else "stable"
        ),
        "per_feature": [
            {"feature": r.feature, "psi": round(r.psi, 4),
             "ks": round(r.ks_statistic, 4), "ks_p": r.ks_pvalue,
             "severity": r.severity}
            for r in reports
        ],
    }
