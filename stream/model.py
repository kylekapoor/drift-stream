"""Model training, evaluation, and the MLflow registry the consumer swaps from.

MLflow runs against a local file store. No server, no database, no cost -- and
the registry semantics that matter here (versioned models, promote a version,
load "the current one" by name) all work exactly the same way against files as
against a hosted deployment.

The metric that matters is average precision, not accuracy or ROC-AUC. Fraud is
around 2% of the stream, so a model that predicts "legitimate" for everything
scores 98% accurate and is worth nothing. ROC-AUC is less absurd but still
flattered by the enormous true-negative pile; average precision summarises the
precision-recall curve, which is the curve an actual fraud team operates on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .events import FEATURES

ROOT = Path(__file__).resolve().parent.parent
# SQLite, not the './mlruns' file store. MLflow put the filesystem backend into
# maintenance mode and refuses it outright, and the model registry -- versions,
# promotion, `models:/name/version` loading -- needs a real database anyway.
# Still a single local file, still no server, still free.
# Path is overridable, and the default lives under data/ so a container can
# mount one directory instead of bind-mounting a single file. Docker creates a
# *directory* when you bind-mount a file that does not exist yet, which breaks
# SQLite in a way that only shows up on a fresh clone.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'data' / 'mlflow.db'}")
EXPERIMENT = "drift-stream"
MODEL_NAME = "fraud-scorer"

# Where the model files themselves go, as opposed to the metadata database.
# MLflow otherwise picks ./mlruns relative to the working directory, which in a
# container is not the mounted volume -- so the database persists, the artifacts
# do not, and loading a registered model fails with "No such artifact" against a
# row that looks perfectly valid. Pinning it under data/ keeps the store and its
# artifacts in the single directory the container mounts.
ARTIFACT_ROOT = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", ROOT / "data" / "mlartifacts"))


def _mlflow_ready():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(EXPERIMENT)


@dataclass
class Evaluation:
    average_precision: float
    roc_auc: float
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float
    n: int
    positives: int

    def as_dict(self) -> dict:
        return {
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "precision": self.precision_at_threshold,
            "recall": self.recall_at_threshold,
            "threshold": self.threshold,
            "n": self.n,
            "positives": self.positives,
        }


def new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=0,
    )


def evaluate(model, X, y, threshold: float = 0.5) -> Evaluation:
    scores = model.predict_proba(X)[:, 1]
    predicted = scores >= threshold
    true_positive = int(((predicted == 1) & (y == 1)).sum())
    predicted_positive = int(predicted.sum())
    actual_positive = int(y.sum())
    return Evaluation(
        average_precision=float(average_precision_score(y, scores)) if actual_positive else 0.0,
        roc_auc=float(roc_auc_score(y, scores)) if 0 < actual_positive < len(y) else 0.5,
        precision_at_threshold=true_positive / predicted_positive if predicted_positive else 0.0,
        recall_at_threshold=true_positive / actual_positive if actual_positive else 0.0,
        threshold=threshold,
        n=len(y),
        positives=actual_positive,
    )


def train(X, y, tag: str = "initial", register: bool = True, log: bool = True):
    """Fit, evaluate on a holdout, and register a new model version."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y if y.sum() > 1 else None
    )
    model = new_model().fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)

    if not log:
        return model, result, None

    _mlflow_ready()
    with mlflow.start_run(run_name=tag) as run:
        mlflow.log_params({
            "max_iter": 250, "learning_rate": 0.08, "reason": tag,
            "n_train": len(X_train), "n_features": len(FEATURES),
        })
        mlflow.log_metrics(result.as_dict())
        if register:
            mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)
        else:
            mlflow.sklearn.log_model(model, name="model")
        run_id = run.info.run_id
    return model, result, run_id


def latest_version() -> int | None:
    _mlflow_ready()
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return max((int(v.version) for v in versions), default=None)


def load(version: int | None = None):
    """Load a registered version, defaulting to the newest."""
    _mlflow_ready()
    version = version or latest_version()
    if version is None:
        raise RuntimeError("no registered model -- run `python run.py train` first")
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}"), version


def score_one(model, vector) -> float:
    """Single-event probability. Reshaped because sklearn wants 2-D."""
    return float(model.predict_proba(np.asarray(vector, dtype=float).reshape(1, -1))[0, 1])
