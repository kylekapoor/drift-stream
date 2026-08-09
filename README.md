# drift-stream

Real-time fraud scoring on Apache Kafka, with drift detection and automated
retraining — built to demonstrate one specific, uncomfortable result:

> **No unsupervised drift monitor can detect concept drift.**

Not "struggles with". Cannot. The repo measures it rather than asserting it.

---

## The result

`python run.py drift` compares three data regimes against the same trained model:

| Regime | max feature PSI | prediction PSI | average precision | verdict |
|---|---|---|---|---|
| baseline | 0.002 | 0.001 | 0.451 | stable |
| covariate_shift | **1.929** | 0.916 | 0.309 | material |
| concept_drift | 0.002 | 0.001 | **0.026** | *stable* |

Read the last row. Average precision falls from 0.451 to 0.026 — the model has
become worthless — and **every** monitor that does not use labels reports
"stable" at PSI 0.002.

This is not a tuning problem. Concept drift changes the mapping from inputs to
labels while leaving the input distribution untouched. Unchanged inputs through
an unchanged model produce an unchanged score distribution, so prediction PSI is
~0.00 by construction. The information simply is not there.

Only labels reveal it, and in card fraud labels arrive weeks late with the
chargebacks. That lag is the actual operational problem, and no amount of input
monitoring removes it.

The covariate row is the encouraging half: PSI screams at 1.93, immediately, with
no labels required. Input monitoring is genuinely valuable — for the failure mode
it can see.

---

## Architecture

```
Generator ──> Kafka topic ──> consumer: score inline ──> decision
                                  │
                                  ├─ every N events: PSI + KS + windowed AP
                                  │
                                  └─ trigger ──> retrain (worker thread)
                                                    └──> MLflow registry ──> hot swap
```

Two independent retrain triggers, because they catch different failures:

- **input-drift** — material PSI on any feature. Needs no labels, fires immediately.
- **performance-decay** — windowed average precision below 60% of baseline.
  Needs labels, and is the only trigger that ever sees concept drift.

## Latency

Scoring happens *inside* the consumer loop rather than behind an HTTP call, and
the reported number is **tick-to-decision**: from the producer's timestamp to a
decision existing. That includes the broker. Inference time alone flatters itself
by excluding most of the budget.

| Run | throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| concept_drift, 16k events @ 400/s | 320 events/s | **2.12 ms** | 10.36 ms | 44.50 ms |
| covariate_shift, 12k events @ 250/s | 201 events/s | 2.50 ms | 146 ms | 628 ms |

The covariate run's tail is worse because it retrained twice and the second run
retrained once, at the very end. Fitting a model steals CPU from the consumer on
the same laptop. Real deployments train on separate hardware; this one is honest
about paying for it.

Single-row `predict_proba` costs ~2 ms of the p50, almost all of it sklearn's
input validation rather than tree traversal. Batching would cut it substantially
and is the obvious next optimisation.

## Three bugs worth keeping

Every one of these produced plausible-looking output while being wrong.

**1. Four partitions destroyed the experiment.** Kafka orders messages within a
partition and makes no promise across them. With four partitions the baseline and
post-drift publishes interleaved, so every analysis window was a blend of both
regimes and drift fired on windows that should have been clean. The topic now uses
one partition — total ordering matters here, throughput does not.

**2. Retraining fed itself.** After a hot swap, prediction PSI was still comparing
against reference scores computed by the *old* model, so it measured the swap
rather than the data, and retrained again. Retrain, swap, "drift", retrain, for
ever. It looked like a monitor working hard. The reference scores are now
recomputed whenever the model changes.

**3. The performance trigger fired on noise.** At 2% fraud a 1,000-event window
holds ~20 positives, and average precision estimated from 20 points swung between
0.14 and 0.55 on *identically distributed* windows — enough to repeatedly breach a
60%-of-baseline threshold. Now the check abstains below 40 positives and requires
two consecutive breaches. It says so out loud when it abstains, because silence
reads as "the model is fine" when it can equally mean "I cannot tell".

There was a fourth: retraining ran inline in the consumer loop and blocked it for
seconds, producing a p99 of 3.7 *seconds* against a p50 of 3.7 ms. It now runs on
a worker thread while the current model keeps serving.

## Setup

No Docker. Kafka 4.x runs in KRaft mode with no ZooKeeper, and the broker is
scoped entirely to this repo.

```bash
brew install kafka
export JAVA_HOME=$(brew --prefix openjdk)
export PATH="$JAVA_HOME/bin:$PATH"

kafka-storage format -t "$(kafka-storage random-uuid)" \
  -c kafka/server.properties --standalone
kafka-server-start kafka/server.properties
```

Then:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run.py train
./.venv/bin/python run.py drift                       # the headline table, no Kafka
./.venv/bin/python run.py demo --regime covariate_shift
./.venv/bin/python run.py demo --regime concept_drift --window 4000 --events 16000
./.venv/bin/python test_stream.py                     # 23 checks, no Kafka
```

MLflow uses a local SQLite file. No server, no cost. Browse it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Honest limits

- **Labels arrive instantly.** This is the biggest gap between the demo and
  reality. Real fraud labels lag by weeks, which means the performance trigger —
  the only one that catches concept drift — would fire long after the damage.
- **Synthetic data.** The generating process is known, which is what makes the
  drift experiment clean and also means the model never faces the messiness of
  real transactions.
- Retraining uses only the triggering window, so a spurious trigger fits on a
  small sample. A production system would blend recent and historical data.
- Single broker, single partition, one consumer. Nothing here is tested against
  rebalancing, partition assignment, or exactly-once semantics.
