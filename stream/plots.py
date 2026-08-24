"""The one chart worth looking at: what each drift regime does to the monitors.

The result table says feature PSI reads 0.002 while average precision collapses
to 0.026. Side by side the point lands harder, because the bars that should
alarm are flat exactly where the model is broken.

    python -m stream.plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from . import drift, model as model_mod
from .events import FEATURES, Generator, to_arrays

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

REGIMES = ("baseline", "covariate_shift", "concept_drift")
LABEL = {"baseline": "baseline", "covariate_shift": "inputs change",
         "concept_drift": "rule changes"}
GRID = {"color": "#999999", "alpha": 0.25, "linewidth": 0.6}
BLUE, GREY, RED = "#3d7ebf", "#b8b8b8", "#e8412c"


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def measure(seed: int = 0, n: int = 8000) -> dict:
    """PSI and average precision per regime, against one fixed reference."""
    reference = Generator(seed=seed, regime="baseline").batch(n)
    X_ref, y_ref = to_arrays(reference)
    model, _ = model_mod.load()
    ref_scores = model.predict_proba(X_ref)[:, 1]

    out = {}
    for regime in REGIMES:
        X, y = to_arrays(Generator(seed=seed + 99, regime=regime).batch(n))
        reports = drift.compare(X_ref, X, FEATURES)
        summary = drift.summarise(reports, drift.prediction_drift(
            ref_scores, model.predict_proba(X)[:, 1]))
        out[regime] = {
            "feature_psi": summary["max_psi"],
            "prediction_psi": summary["prediction_psi"],
            "average_precision": model_mod.evaluate(model, X, y).average_precision,
        }
    return out


def blind_spot(results: dict | None = None, path: Path = DOCS / "drift.png"):
    """Drift signal beside model health, per regime.

    Two panels rather than one axis. PSI runs 0 to 2 and average precision 0 to
    0.5, so sharing an axis would flatten whichever one you actually wanted to
    read.
    """
    results = results or measure()
    DOCS.mkdir(exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    x = np.arange(len(REGIMES))
    feature = [results[r]["feature_psi"] for r in REGIMES]
    prediction = [results[r]["prediction_psi"] for r in REGIMES]

    left.bar(x - 0.2, feature, 0.38, label="feature PSI", color=BLUE)
    left.bar(x + 0.2, prediction, 0.38, label="prediction PSI", color=GREY)
    left.axhline(drift.PSI_MATERIAL, color=RED, linestyle="--", linewidth=1.2,
                 label=f"material ({drift.PSI_MATERIAL})")
    left.set_xticks(x, [LABEL[r] for r in REGIMES], fontsize=9)
    left.legend(fontsize=8, frameon=False)
    _style(left, "What the monitors see (no labels needed)", "PSI")

    ap = [results[r]["average_precision"] for r in REGIMES]
    colours = [GREY, GREY, RED]
    right.bar(x, ap, 0.5, color=colours)
    for i, value in enumerate(ap):
        right.text(i, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    right.set_xticks(x, [LABEL[r] for r in REGIMES], fontsize=9)
    right.set_ylim(0, max(ap) * 1.25)
    _style(right, "Whether the model still works (labels required)",
           "average precision")

    fig.suptitle("The third regime is the blind spot: PSI flat, model dead",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"wrote {blind_spot()}")
