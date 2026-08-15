# drift-stream

[![tests](https://github.com/kylekapoor/drift-stream/actions/workflows/tests.yml/badge.svg)](https://github.com/kylekapoor/drift-stream/actions/workflows/tests.yml)

A fraud model works the day you ship it. Six months later it might not, and
nothing in the system will tell you.

This pipeline scores card transactions as they arrive, watches for the data
shifting under the model, and retrains when it does. It also shows one kind of
staleness that monitoring cannot find.

`Python` · `PyTorch` · `Apache Kafka` · `PostgreSQL` · `Docker` · `MLflow` · `scikit-learn` · `FastAPI`

---

## The result

Models go stale in two ways. I could only detect one of them.

| Regime | feature PSI | prediction PSI | avg precision | verdict |
|---|---|---|---|---|
| baseline | 0.002 | 0.001 | 0.451 | stable |
| inputs change | **1.929** | 0.916 | 0.309 | material |
| rule changes | 0.002 | 0.001 | **0.026** | *stable* |

In the third row the model has stopped working. Its average precision drops from
0.451 to 0.026. Every monitor that runs without labels still reports "stable".

No amount of tuning fixes that. If fraudsters change tactics while the
transactions themselves look the same, identical inputs pass through an
identical model and produce identical outputs. There is no signal to find. Labels
expose it, and card-fraud labels show up weeks later with the chargebacks. That
lag is the expensive part.

Row two is friendlier. When the inputs move, PSI hits 1.93 right away and needs
no labels at all.

## Latency

I report tick-to-decision: producer timestamp to decision, broker included.
Scoring runs inside the consumer loop rather than behind an HTTP call.

| Path | p50 | p99 |
|---|---|---|
| TorchScript | **10.9 µs** | 20.5 µs |
| sklearn `predict_proba` | 1,250 µs | 3,754 µs |
| end-to-end through Kafka | 2.12 ms | 44.5 ms |

TorchScript scores a row 115x faster. Most of sklearn's 1.25 ms goes to input
validation rather than arithmetic. The PyTorch model also scores better, 0.472
against 0.437 average precision. I handle the 2% fraud rate by weighting the
loss instead of resampling, which leaves the output probabilities usable for
thresholding.

## Design

```
Generator ──> Kafka ──> consumer: score inline ──> decision
                            │                          │
                            │                          └─> Postgres (batched)
                            ├─ every N events: PSI + KS + windowed accuracy
                            └─ trigger ──> retrain (worker thread) ──> MLflow ──> hot swap
```

Postgres stores every decision, so afterwards I can query instead of grepping
logs: precision by model version, which features drifted before a retrain fired,
what score a disputed transaction received. Writes go out in batches of 500. One
insert per event would drop a network round trip into a loop I measure in
microseconds. Without `DATABASE_URL` the sink does nothing and the pipeline
behaves the same.

```
version   scored  fraud  mean(fraud)  mean(clean)   p50 us
      1     6000    126        0.323        0.040    162.8
```

Watch the gap between those two means. A model that has stopped working shows
them converging, and reading that needs no threshold bookkeeping.

Two retrain triggers cover the two failure modes:

- **input-drift** fires on material PSI, without labels, straight away.
- **performance-decay** fires when accuracy falls below 60% of baseline. It needs
  labels, and it is the only trigger that catches a changed rule.

I wrote PSI by hand rather than importing it, because two details decide whether
it works and neither shows up in a library call. Bin edges have to come from the
reference window and stay frozen; re-binning on the current window compares a
distribution against itself and pushes PSI toward zero at the worst possible
moment. Empty bins need clamping or the logarithm returns infinity.

## Bugs worth keeping

Each of these produced believable output while being wrong.

- **4 partitions.** Kafka orders messages inside a partition and not across
  partitions, so my pre-drift and post-drift data interleaved and every analysis
  window held a mix of both.
- **Retraining fed itself.** After a swap, prediction PSI still compared against
  the old model's scores, so it measured the swap, retrained, and repeated.
- **The trigger fired on noise.** About 20 positives per window made accuracy
  swing between 0.14 and 0.55 on identical data. It now abstains below 40
  positives and waits for two consecutive breaches.
- **Retraining blocked the consumer**, giving a p99 of 3.7 seconds against a p50
  of 3.7 ms. It runs on a worker thread now.
- **MLflow artifacts vanished in Docker.** MLflow defaults to `./mlruns`, which I
  had not mounted, so the database survived while the model files disappeared and
  loading failed with `No such artifact` on a row that looked fine.

## Usage

```bash
docker compose run --rm pipeline python run.py train
docker compose run --rm pipeline python run.py demo --regime covariate_shift
docker compose --profile api up          # scoring API on :8000
docker compose run --rm pipeline python run.py report --run <consumer-group>
```

Compose starts Kafka 4.x in KRaft mode, Postgres, and the pipeline. No JDK, no
manual storage format, no broker left running on your machine afterwards. The
broker advertises two listeners, `kafka:9092` for containers and
`localhost:29092` for the host; advertise one and the other side breaks without
saying so.

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
./.venv/bin/python test_stream.py    # 26 checks
```

## Limits

- Labels arrive with the event here. Real ones lag by weeks, so the trigger
  that catches a changed rule would fire long after the money was gone.
- The data is synthetic. I know the generating process, which keeps the
  experiment clean and also means the model never meets real mess.
- Retraining uses only the window that triggered it. Production would blend in
  history.
- One broker, one partition, one consumer.
