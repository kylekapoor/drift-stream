"""PyTorch fraud classifier, TorchScript-compiled for the serving path.

Two reasons this exists next to the sklearn model rather than replacing it.

**Class imbalance is handled in the loss, not by resampling.** Fraud is 2% of the
stream. Oversampling the minority class distorts the base rate the model sees and
leaves its probabilities miscalibrated, which matters when the output feeds a
threshold. `pos_weight` on BCE reweights the gradient instead, leaving the
prevalence intact.

**TorchScript is the point of the exercise.** A single-row `predict_proba` through
sklearn costs about 2 ms, almost all of it input validation rather than arithmetic,
and that dominates the tick-to-decision budget in `pipeline.py`. A scripted module
skips the Python dispatch entirely. `run.py bench` measures the difference rather
than assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .events import FEATURES

ROOT = Path(__file__).resolve().parent.parent
SCRIPTED_PATH = ROOT / "data" / "fraud_mlp.ts"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FraudMLP(nn.Module):
    """Standardisation is baked into the module.

    Keeping the mean and scale as buffers rather than applying them in the
    caller means the exported artefact is self-contained: whoever loads the
    TorchScript file cannot forget to normalise, which is a classic way for
    training and serving to quietly disagree.
    """

    def __init__(self, n_features: int, mean: torch.Tensor, scale: torch.Tensor,
                 hidden: int = 64):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.scale).squeeze(-1)


@dataclass
class TorchEvaluation:
    average_precision: float
    roc_auc: float
    n: int
    positives: int
    device: str
    epochs: int

    def as_dict(self) -> dict:
        return {"average_precision": self.average_precision, "roc_auc": self.roc_auc,
                "n": self.n, "positives": self.positives,
                "device": self.device, "epochs": self.epochs}


def train(X, y, epochs: int = 40, batch_size: int = 512, lr: float = 2e-3,
          verbose: bool = False):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y if y.sum() > 1 else None
    )

    dev = device()
    mean = torch.tensor(X_train.mean(0))
    scale = torch.tensor(X_train.std(0)).clamp(min=1e-6)

    model = FraudMLP(X.shape[1], mean, scale).to(dev)
    # Reweight the positive class by its scarcity so the gradient does not get
    # swamped by the 98% of legitimate transactions.
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    Xt = torch.from_numpy(X_train).to(dev)
    yt = torch.from_numpy(y_train).to(dev)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xt), device=dev)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if len(batch) < 2:  # BatchNorm needs more than one row
                continue
            optimiser.zero_grad()
            loss = criterion(model(Xt[batch]), yt[batch])
            loss.backward()
            optimiser.step()
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:>3}  loss {loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            model(torch.from_numpy(X_test).to(dev))
        ).cpu().numpy()

    result = TorchEvaluation(
        average_precision=float(average_precision_score(y_test, scores)),
        roc_auc=float(roc_auc_score(y_test, scores)),
        n=len(y_test), positives=int(y_test.sum()),
        device=str(dev), epochs=epochs,
    )
    return model, result


def export(model, path: Path = SCRIPTED_PATH) -> Path:
    """Trace to TorchScript on CPU -- the serving path has no GPU."""
    model = model.to("cpu").eval()
    scripted = torch.jit.script(model)
    scripted = torch.jit.optimize_for_inference(scripted)
    path.parent.mkdir(exist_ok=True)
    scripted.save(str(path))
    return path


def load_scripted(path: Path = SCRIPTED_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python run.py train --torch`")
    return torch.jit.load(str(path)).eval()


def score_one(scripted, vector) -> float:
    with torch.no_grad():
        x = torch.tensor([vector], dtype=torch.float32)
        return float(torch.sigmoid(scripted(x))[0])


def score_batch(scripted, vectors) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(np.asarray(vectors, dtype=np.float32))
        return torch.sigmoid(scripted(x)).numpy()
