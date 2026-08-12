# drift-stream

You deploy a model that flags fraudulent card transactions. Six months later, is
it still working? Nobody tells you when a model goes stale — it just quietly
starts being wrong.

This is a live pipeline that scores transactions as they stream in, watches for
the world changing underneath the model, and retrains and swaps itself when it
does. It also demonstrates one failure mode that **no amount of monitoring can
catch**.

`Python` · `PyTorch` · `Apache Kafka` · `MLflow` · `scikit-learn` · `FastAPI` · `NumPy` · `SciPy`

---

## The result

A model goes stale in two different ways, and only one is detectable.

| Regime | max feature PSI | prediction PSI | average precision | verdict |
|---|---|---|---|---|
| baseline | 0.002 | 0.001 | 0.451 | stable |
| **inputs change** | **1.929** | 0.916 | 0.309 | material |
| **rule changes** | 0.002 | 0.001 | **0.026** | *stable* |

Read the last row. Accuracy collapses from 0.451 to 0.026 — the model is
worthless — and every monitor that doesn't use the correct answers reports
"stable".

That isn't a tuning failure, it's structural. When fraudsters change tactics but
the transactions look identical, the same inputs go through the same model and
produce the same outputs. There is nothing to detect. Only labels reveal it, and
real fraud labels arrive weeks later with the chargebacks — **that lag is the
actual problem, and monitoring cannot fix it.**

The middle row is the good news: when the inputs themselves shift, PSI screams at
1.93 immediately, with no labels needed.

## Latency

Scoring runs inside the consumer loop, not behind an HTTP call. The number
reported is tick-to-decision — producer timestamp to decision, broker included.

| Path | p50 | p99 |
|---|---|---|
| **TorchScript (PyTorch, compiled)** | **10.9 µs** | 20.5 µs |
| sklearn `predict_proba` | 1,250 µs | 3,754 µs |
| end-to-end through Kafka | 2.12 ms | 44.5 ms |

**115× faster per row.** Almost all of sklearn's 1.25 ms is Python input
validation rather than arithmetic. The PyTorch model also scores slightly better
(0.472 vs 0.437 average precision), with class imbalance handled by weighting the
loss rather than resampling, which keeps the output probabilities calibrated.

## How it works

```
Generator ──> Kafka ──> consumer: score inline ──> decision
                            │
                            ├─ every N events: PSI + KS + windowed accuracy
                            └─ trigger ──> retrain (worker thread) ──> MLflow ──> hot swap
```

Two independent retrain triggers, because they catch different failures:

- **input-drift** — material PSI. No labels needed, fires immediately.
- **performance-decay** — accuracy below 60% of baseline. Needs labels, and is the
  only trigger that ever sees the second kind of staleness.

PSI is implemented directly rather than imported, because the two details that
decide whether it works are invisible in a library call: bin edges must come from
the reference window and stay frozen (re-binning compares a distribution to
itself and collapses PSI toward zero exactly when drift is worst), and empty bins
must be clamped or the logarithm returns infinity.

## Four bugs worth keeping

All four produced believable output while being wrong.

1. **Four partitions destroyed the experiment.** Kafka orders messages within a
   partition, not across them, so the before- and after-drift data interleaved and
   every analysis window was a blend. One partition now.
2. **Retraining fed itself.** After a model swap, prediction PSI still compared
   against the *old* model's scores — so it measured the swap and retrained again,
   forever.
3. **The performance trigger fired on noise.** At 2% fraud a 1,000-event window
   holds ~20 positives, and accuracy estimated from 20 points swung 0.14–0.55 on
   identical data. Now it abstains below 40 positives, needs two consecutive
   breaches, and says so when it abstains.
4. **Retraining blocked the consumer**, giving a p99 of 3.7 *seconds* against a
   p50 of 3.7 ms. Now on a worker thread while the current model keeps serving.

## Usage

No Docker. Kafka 4.x runs in KRaft mode with no ZooKeeper, scoped to this repo.

```bash
brew install kafka
export JAVA_HOME=$(brew --prefix openjdk) && export PATH="$JAVA_HOME/bin:$PATH"
kafka-storage format -t "$(kafka-storage random-uuid)" -c kafka/server.properties --standalone
kafka-server-start kafka/server.properties
```

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run.py train                    # sklearn + PyTorch
./.venv/bin/python run.py drift                    # the headline table, no Kafka
./.venv/bin/python run.py demo --regime covariate_shift
./.venv/bin/python run.py demo --regime concept_drift --window 4000 --events 16000
./.venv/bin/python run.py bench                    # latency across serving paths
./.venv/bin/python test_stream.py                  # 23 checks, no Kafka
```

MLflow uses a local SQLite file. No server, no cost.

## Limits

- **Labels arrive instantly here.** The biggest gap from reality: real fraud
  labels lag by weeks, so the only trigger that catches a rule change would fire
  long after the damage.
- Synthetic data, so the generating process is known — which is what makes the
  experiment clean and also means the model never meets real messiness.
- Retraining uses only the triggering window; production would blend history.
- Single broker, single partition, one consumer.
