"""Kafka producer, scoring consumer, drift monitor and hot model swap.

Scoring happens *inside* the consumer loop, not behind an HTTP call. That is the
whole latency argument: a REST hop adds serialisation, a socket round trip and a
web framework's overhead to a prediction that takes well under a millisecond.
`api.py` exists alongside this to make the same model available over HTTP so the
two paths can be measured against each other rather than argued about.

Latency is reported as tick-to-decision -- from the instant the producer stamped
the event to the instant a decision existed -- because that is the number a
trading or fraud system is actually judged on. Inference time alone flatters
itself by excluding the broker, which is most of the budget.

The model is swapped by rebinding a single attribute. There is no restart, no
drain, and no dropped message: in-flight events finish scoring against whichever
model they started with, which for a monitoring pipeline is entirely fine.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from confluent_kafka import Consumer, KafkaException, Producer

from . import drift, model as model_mod, sink as sink_mod
from .events import FEATURES, Event, Generator

# Overridable so the same code runs against a broker on localhost or one in
# a sibling container, where the hostname is the compose service name.
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")


def producer(bootstrap: str = BOOTSTRAP) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap,
        # Latency over throughput: send immediately rather than batching for the
        # default 5 ms, which would otherwise be most of the measured budget.
        "linger.ms": 0,
        "acks": 1,
        "compression.type": "none",
    })


def consumer(bootstrap: str = BOOTSTRAP, group: str = "scorer") -> Consumer:
    return Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "fetch.wait.max.ms": 10,
    })


def publish(n: int, regime: str = "baseline", seed: int = 0, topic: str = TOPIC,
            bootstrap: str = BOOTSTRAP, rate: float = 0.0) -> int:
    """Push `n` events onto the topic. `rate` throttles to events per second."""
    p = producer(bootstrap)
    gen = Generator(seed=seed, regime=regime)
    interval = 1.0 / rate if rate > 0 else 0.0

    for i in range(n):
        event = gen.next()
        # Stamp as late as possible so queueing here is not counted as latency.
        event.produced_ns = time.time_ns()
        p.produce(topic, value=event.to_json())
        if i % 500 == 0:
            p.poll(0)
        if interval:
            time.sleep(interval)
    p.flush(30)
    return n


@dataclass
class ScoringRun:
    consumed: int = 0
    tick_to_decision_us: list = field(default_factory=list)
    inference_us: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    vectors: list = field(default_factory=list)
    drift_checks: list = field(default_factory=list)
    swaps: list = field(default_factory=list)
    model_version: int | None = None
    elapsed_s: float = 0.0
    consecutive_perf_alarms: int = 0
    persisted: int = 0

    def percentiles(self, values) -> dict:
        if not values:
            return {}
        a = np.array(values)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
            "mean": float(a.mean()),
        }

    def summary(self) -> dict:
        return {
            "consumed": self.consumed,
            "elapsed_s": round(self.elapsed_s, 2),
            "throughput_eps": round(self.consumed / self.elapsed_s, 1) if self.elapsed_s else 0.0,
            "tick_to_decision_us": self.percentiles(self.tick_to_decision_us),
            "inference_us": self.percentiles(self.inference_us),
            "model_version": self.model_version,
            "drift_checks": self.drift_checks,
            "swaps": self.swaps,
        }


class Scorer:
    """Holds the live model. Rebinding `.model` is the hot swap."""

    def __init__(self, model, version: int | None, reference_scores=None):
        self.model = model
        self.version = version
        # Owned by the scorer, not the caller, because it must be recomputed
        # whenever the model changes.
        self.reference_scores = reference_scores
        # Guards against stacking retrains: successive windows keep detecting
        # the same drift while the first fit is still running.
        self.retraining = False

    def score(self, vector) -> float:
        return model_mod.score_one(self.model, vector)

    def swap(self, model, version: int | None) -> None:
        self.model, self.version = model, version


def run_scoring(
    max_events: int,
    reference_X: np.ndarray,
    reference_scores: np.ndarray,
    window: int = 2000,
    topic: str = TOPIC,
    bootstrap: str = BOOTSTRAP,
    group: str = "scorer",
    auto_retrain: bool = True,
    reference_ap: float = 0.0,
    timeout_s: float = 60.0,
    verbose: bool = True,
    sink=None,
) -> ScoringRun:
    """Consume, score inline, monitor drift, retrain and swap when it fires."""
    live_model, version = model_mod.load()
    scorer = Scorer(live_model, version, reference_scores)
    sink = sink or sink_mod.Sink(run_id=group).connect()

    c = consumer(bootstrap, group)
    c.subscribe([topic])

    run = ScoringRun(model_version=version)
    window_vectors, window_scores, window_labels = [], [], []
    started = time.time()
    first_message_at = None

    try:
        while run.consumed < max_events and (time.time() - started) < timeout_s:
            message = c.poll(0.5)
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())

            if first_message_at is None:
                first_message_at = time.time()
            event = Event.from_json(message.value())
            vector = event.vector()

            t0 = time.perf_counter_ns()
            score = scorer.score(vector)
            t1 = time.perf_counter_ns()

            run.inference_us.append((t1 - t0) / 1000.0)
            run.tick_to_decision_us.append((time.time_ns() - event.produced_ns) / 1000.0)
            run.scores.append(score)
            run.labels.append(event.is_fraud)
            run.vectors.append(vector)
            run.consumed += 1

            window_vectors.append(vector)
            window_scores.append(score)
            window_labels.append(event.is_fraud)

            # Batched, so the database never enters the scoring hot path.
            sink.record(event.event_id, score, event.is_fraud, scorer.version,
                        (t1 - t0) / 1000.0)

            if len(window_vectors) >= window:
                check = _check_window(
                    run, scorer, reference_X, scorer.reference_scores,
                    np.array(window_vectors, dtype=float),
                    np.array(window_scores), np.array(window_labels),
                    auto_retrain, reference_ap, verbose,
                )
                run.drift_checks.append(check)
                sink.record_drift(check)
                window_vectors, window_scores, window_labels = [], [], []
    finally:
        c.close()
        sink.close()

    run.model_version = scorer.version
    run.elapsed_s = (time.time() - first_message_at) if first_message_at else 0.0
    run.persisted = sink.written
    return run


def _check_window(run, scorer, reference_X, reference_scores,
                  X, scores, labels, auto_retrain, reference_ap, verbose) -> dict:
    reports = drift.compare(reference_X, X, FEATURES)
    pred_psi = drift.prediction_drift(reference_scores, scores)
    summary = drift.summarise(reports, pred_psi)

    evaluation = model_mod.evaluate(scorer.model, X, labels)
    summary["average_precision"] = evaluation.average_precision
    summary["at_event"] = run.consumed

    # Two independent triggers, because they detect different failures. Input
    # drift fires without labels and catches covariate shift early. Performance
    # decay needs labels and is the *only* thing that sees concept drift, where
    # every input statistic reads clean while the model quietly stops working.
    raw_alarm = drift.performance_drop(
        reference_ap, evaluation.average_precision, evaluation.positives
    )
    # Debounce. One bad window is an estimate; two in a row is a trend. Input
    # drift is not debounced because PSI over a whole window is already a stable
    # statistic -- it does not need a second opinion.
    run.consecutive_perf_alarms = run.consecutive_perf_alarms + 1 if raw_alarm else 0
    performance_alarm = run.consecutive_perf_alarms >= 2

    summary["performance_alarm"] = performance_alarm
    summary["window_positives"] = evaluation.positives
    summary["reference_ap"] = reference_ap
    trigger = ("input-drift" if summary["verdict"] == "material"
               else "performance-decay" if performance_alarm else None)
    summary["trigger"] = trigger

    if verbose:
        # Say when the performance check abstained. Silence here reads as "the
        # model is fine" when it can equally mean "too few labelled positives to
        # judge", and those are opposite situations.
        if evaluation.positives < drift.MIN_POSITIVES_FOR_AP:
            perf = f"AP n/a ({evaluation.positives} pos < {drift.MIN_POSITIVES_FOR_AP})"
        else:
            perf = f"AP {evaluation.average_precision:.3f} ({evaluation.positives} pos)"
        print(f"  [{run.consumed:>6}] verdict={summary['verdict']:<9} "
              f"max PSI {summary['max_psi']:.3f}  pred PSI {pred_psi:.3f}  "
              f"{perf}  trigger={trigger or '-'}  "
              f"drifted={','.join(summary['drifted_features']) or '-'}", flush=True)

    if auto_retrain and trigger and not scorer.retraining:
        # Retrain on a worker thread, never in the consumer loop. Training takes
        # seconds; doing it inline stops consumption for that whole time and the
        # backlog shows up as a p99 tick-to-decision of 3.7 SECONDS against a
        # p50 of 3.7 ms. The stream keeps being scored by the current model
        # while the replacement is fitted, which is also how you would want a
        # real deployment to behave.
        #
        # In production the labels would lag by days. Here they arrive with the
        # event, and that is the biggest gap between this and a real system.
        scorer.retraining = True
        at_event = run.consumed
        before = evaluation.average_precision

        def _retrain():
            try:
                new_model, result, _ = model_mod.train(
                    X, labels, tag=f"retrain@{at_event}"
                )
                version = model_mod.latest_version()
                new_reference = new_model.predict_proba(reference_X)[:, 1]
                scorer.swap(new_model, version)
                # Re-baseline against the model now serving. Without this,
                # prediction PSI compares the new model's output to the old
                # model's and reports drift caused entirely by the swap --
                # retrain, swap, drift, retrain, for ever. It looked like drift
                # detection working hard; it was a feedback loop.
                scorer.reference_scores = new_reference
                run.swaps.append({
                    "at_event": at_event,
                    "new_version": version,
                    "average_precision_before": before,
                    "average_precision_after": result.average_precision,
                    "trigger": trigger,
                    "drifted_features": summary["drifted_features"],
                    "prediction_psi": pred_psi,
                })
                if verbose:
                    print(f"          retrained off-thread -> v{version}  "
                          f"AP {before:.3f} -> {result.average_precision:.3f}", flush=True)
            finally:
                scorer.retraining = False

        threading.Thread(target=_retrain, daemon=True).start()
    return summary


def ensure_topic(topic: str = TOPIC, bootstrap: str = BOOTSTRAP,
                 partitions: int = 1) -> None:
    """Create the topic if absent.

    One partition, deliberately. Kafka orders messages *within* a partition and
    makes no promise across them, so a multi-partition topic interleaves the
    baseline and post-drift publishes and every analysis window ends up a blend
    of both regimes -- which shows up as drift firing on windows that should be
    clean. Total ordering is what this experiment needs; throughput is not the
    constraint at these volumes.
    """
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    if topic in admin.list_topics(timeout=10).topics:
        return
    for _, future in admin.create_topics([NewTopic(topic, num_partitions=partitions,
                                                   replication_factor=1)]).items():
        try:
            future.result()
        except Exception:
            pass  # already exists, or a concurrent creator won


def broker_available(bootstrap: str = BOOTSTRAP) -> bool:
    from confluent_kafka.admin import AdminClient

    try:
        AdminClient({"bootstrap.servers": bootstrap}).list_topics(timeout=5)
        return True
    except Exception:
        return False
