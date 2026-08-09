# drift-stream

**Real-time fraud scoring on Apache Kafka**, with drift detection, automated
retraining and hot model swaps — built to demonstrate one uncomfortable result:

> No unsupervised drift monitor can detect concept drift.

Not "struggles with". Cannot. This repo measures it rather than asserting it.

`Python` · `PyTorch` · `Apache Kafka` · `MLflow` · `scikit-learn` · `FastAPI` · `NumPy` · `Pandas` · `SciPy`

---

## The result

`python run.py drift` runs three data regimes past the same trained model:

| Regime | max feature PSI | prediction PSI | average precision | verdict |
|---|---|---|---|---|
| baseline | 0.002 | 0.001 | 0.451 | stable |
| covariate shift | **1.929** | 0.916 | 0.309 | material |
| concept drift | 0.002 | 0.001 | **0.026** | *stable* |

Read the last row. Average precision falls from 0.451 to 0.026 — the model is
worthless — and **every** monitor that does not use labels reports "stable".

That is structural, not a tuning failure. Concept drift changes the mapping from
inputs to labels while leaving the input distribution untouched, so unchanged
inputs through an unchanged model produce an unchanged score distribution.
Prediction PSI is ~0.00 by construction. Only labels reveal it, and in card fraud
labels arrive weeks late with the chargebacks.

The covariate row is the encouraging half: PSI screams at 1.93, immediately, with
no labels needed.

## Serving latency

Scoring runs *inside* the consumer loop rather than behind an HTTP call, and the
reported figure is **tick-to-decision** — producer timestamp to decision, broker
included.

| Path | p50 | p99 |
|---|---|---|
| TorchScript (PyTorch, compiled) | **10.9 µs** | 20.5 µs |
| sklearn `predict_proba` | 1,250 µs | 3,754 µs |
| End-to-end through Kafka | 2.12 ms | 44.5 ms |

**TorchScript is 115× faster per row.** Almost all of sklearn's 1.25 ms is Python
input validation rather than arithmetic; a scripted module skips the dispatch
entirely. The PyTorch MLP also scores slightly better — 0.472 average precision
against 0.437 — with class imbalance handled by `pos_weight` on the loss rather
than resampling, which keeps the output probabilities calibrated.

## Architecture

```
Generator ──> Kafka topic ──> consumer: score inline ──> decision
                                  │
                                  ├─ every N events: PSI + KS + windowed AP
                                  │
                                  └─ trigger ──> retrain (worker thread)
                                                  └──> MLflow ──> hot swap
```

Two independent retrain triggers, because they catch different failures:

- **input-drift** — material PSI on any feature. No labels needed, fires immediately.
- **performance-decay** — windowed average precision below 60% of baseline. Needs
  labels, and is the only trigger that ever sees concept drift.

PSI is implemented directly rather than imported. The two details that decide
whether it works are invisible in a library call: bin edges must come from the
reference window and stay frozen (re-binning on the current window compares a
distribution to itself and collapses PSI toward zero exactly when drift is
worst), and empty bins must be clamped or the logarithm returns infinity.

## Four bugs worth keeping

Every one produced plausible-looking output while being wrong.

1. **Four partitions destroyed the experiment.** Kafka orders within a partition,
   not across them, so the baseline and post-drift publishes interleaved and every
   analysis window was a blend. Now one partition — ordering matters here more
   than throughput.
2. **Retraining fed itself.** After a hot swap, prediction PSI still compared
   against the *old* model's reference scores, so it measured the swap and
   retrained again, forever.
3. **The performance trigger fired on noise.** At 2% fraud a 1,000-event window
   holds ~20 positives, and AP estimated from 20 points swung 0.14–0.55 on
   identically distributed data. Now abstains below 40 positives, requires two
   consecutive breaches, and says so when it abstains.
4. **Retraining blocked the consumer**, giving a p99 of 3.7 *seconds* against a
   p50 of 3.7 ms. Now on a worker thread while the current model keeps serving.

## Setup

No Docker. Kafka 4.x runs in KRaft mode with no ZooKeeper, scoped to this repo.

```bash
brew install kafka
export JAVA_HOME=$(brew --prefix openjdk) && export PATH="$JAVA_HOME/bin:$PATH"
kafka-storage format -t "$(kafka-storage random-uuid)" -c kafka/server.properties --standalone
kafka-server-start kafka/server.properties
```

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run.py train                      # sklearn + PyTorch, both registered
./.venv/bin/python run.py drift                      # the headline table, no Kafka
./.venv/bin/python run.py demo --regime covariate_shift
./.venv/bin/python run.py demo --regime concept_drift --window 4000 --events 16000
./.venv/bin/python run.py bench                      # latency across serving paths
./.venv/bin/python test_stream.py                    # 23 checks, no Kafka
```

MLflow uses a local SQLite file — no server, no cost.

## Limits

- **Labels arrive instantly.** The biggest gap from reality: real fraud labels lag
  by weeks, so the only trigger that catches concept drift would fire long after
  the damage.
- Synthetic data, so the generating process is known — which is what makes the
  drift experiment clean and also means the model never meets real messiness.
- Retraining uses only the triggering window; production would blend historical data.
- Single broker, single partition, one consumer. No rebalancing or exactly-once
  semantics tested.
