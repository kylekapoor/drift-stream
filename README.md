# drift-stream

You deploy a model that flags fraudulent card transactions. Six months later, is
it still working? Nobody tells you when a model goes stale — it just quietly
starts being wrong.

This scores transactions live as they stream in, watches for the world changing
underneath the model, and retrains and swaps itself when it does. It also
demonstrates one failure mode **no monitoring can catch**.

`Python` · `PyTorch` · `Apache Kafka` · `Docker` · `MLflow` · `scikit-learn` · `FastAPI`

---

## The result

A model goes stale two ways. Only one is detectable.

| Regime | feature PSI | prediction PSI | avg precision | verdict |
|---|---|---|---|---|
| baseline | 0.002 | 0.001 | 0.451 | stable |
| **inputs change** | **1.929** | 0.916 | 0.309 | material |
| **rule changes** | 0.002 | 0.001 | **0.026** | *stable* |

Read the last row. Accuracy collapses to near-useless and every monitor that
doesn't use labels says "stable".

That's structural, not a tuning failure: when fraudsters change tactics but
transactions look identical, the same inputs go through the same model and give
the same outputs. Nothing to detect. Only labels reveal it — and real fraud
labels arrive weeks later with the chargebacks. **That lag is the actual
problem.**

Row two is the good news: when inputs shift, PSI screams immediately, no labels
needed.

## Latency

Tick-to-decision — producer timestamp to decision, broker included. Scoring runs
inside the consumer loop, not behind HTTP.

| Path | p50 | p99 |
|---|---|---|
| **TorchScript** | **10.9 µs** | 20.5 µs |
| sklearn `predict_proba` | 1,250 µs | 3,754 µs |
| end-to-end through Kafka | 2.12 ms | 44.5 ms |

**115× faster per row** — almost all of sklearn's 1.25 ms is input validation,
not arithmetic. The PyTorch model also scores better (0.472 vs 0.437 AP), with
imbalance handled by weighting the loss rather than resampling, which keeps
probabilities calibrated for thresholding.

## Design

```
Generator ──> Kafka ──> consumer: score inline ──> decision
                            │
                            ├─ every N events: PSI + KS + windowed accuracy
                            └─ trigger ──> retrain (worker thread) ──> MLflow ──> hot swap
```

Two retrain triggers, catching different failures:

- **input-drift** — material PSI. No labels, fires immediately.
- **performance-decay** — accuracy below 60% of baseline. Needs labels, and is
  the only trigger that sees a rule change.

PSI is hand-written, not imported: bin edges must come from the reference window
and stay frozen (re-binning compares a distribution to itself and collapses PSI
exactly when drift is worst), and empty bins must be clamped or the log returns
infinity.

## Bugs worth keeping

All produced believable output while being wrong.

- **4 partitions** — Kafka orders within a partition, not across, so pre/post-drift data interleaved and every window was a blend.
- **Retraining fed itself** — after a swap, prediction PSI still compared to the old model's scores, so it measured the swap and retrained forever.
- **Trigger fired on noise** — ~20 positives per window made accuracy swing 0.14–0.55 on identical data. Now abstains below 40 positives and needs two consecutive breaches.
- **Retraining blocked the consumer** — p99 of 3.7 *seconds* against a p50 of 3.7 ms. Now on a worker thread.
- **MLflow artifacts vanished in Docker** — defaults to `./mlruns`, not the mounted volume, so the database persisted while model files disappeared and loading failed with `No such artifact` on a valid-looking row.

## Usage

```bash
docker compose run --rm pipeline python run.py train
docker compose run --rm pipeline python run.py demo --regime covariate_shift
docker compose --profile api up          # scoring API on :8000
```

Brings up Kafka 4.x (KRaft, no ZooKeeper) and the pipeline together — no JDK, no
manual storage format, no broker left on the host. The broker advertises two
listeners (`kafka:9092` for containers, `localhost:29092` for the host);
advertising one silently breaks the other side.

Without Docker:

```bash
brew install kafka && export JAVA_HOME=$(brew --prefix openjdk)
kafka-storage format -t "$(kafka-storage random-uuid)" -c kafka/server.properties --standalone
kafka-server-start kafka/server.properties

python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run.py train
./.venv/bin/python run.py drift      # the headline table, no Kafka needed
./.venv/bin/python run.py demo --regime concept_drift --window 4000 --events 16000
./.venv/bin/python run.py bench      # latency across serving paths
./.venv/bin/python test_stream.py    # 23 checks
```

## Limits

- **Labels arrive instantly here.** Real ones lag weeks, so the only trigger that
  catches a rule change would fire long after the damage.
- Synthetic data — the generating process is known, which makes the experiment
  clean and means the model never meets real messiness.
- Retraining uses only the triggering window; production would blend history.
- Single broker, single partition, one consumer.
