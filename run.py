#!/usr/bin/env python3
"""drift-stream CLI:  train -> demo.

    python run.py train
    python run.py demo --events 12000
    python run.py bench
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from stream import drift, model as model_mod, pipeline
from stream.events import FEATURES, Generator, to_arrays

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results.json"

REFERENCE_N = 30000


def _reference(seed: int = 0):
    """The baseline sample everything downstream is compared against."""
    events = Generator(seed=seed, regime="baseline").batch(REFERENCE_N)
    return to_arrays(events)


def cmd_train(args):
    X, y = _reference(seed=args.seed)
    print(f"{len(X):,} baseline events, {y.sum():,} fraudulent ({y.mean():.2%})")
    model, result, run_id = model_mod.train(X, y, tag="initial")
    print(f"\nsklearn HistGradientBoosting")
    print(f"  average precision {result.average_precision:.4f}   "
          f"ROC-AUC {result.roc_auc:.4f}")
    print(f"  precision {result.precision_at_threshold:.3f}  "
          f"recall {result.recall_at_threshold:.3f}  @ threshold {result.threshold}")
    print(f"  registered as {model_mod.MODEL_NAME} v{model_mod.latest_version()}  "
          f"(run {run_id})")

    if not args.no_torch:
        from stream import torch_model
        net, torch_result = torch_model.train(X, y, epochs=args.epochs)
        path = torch_model.export(net)
        print(f"\nPyTorch MLP  (device {torch_result.device}, {torch_result.epochs} epochs)")
        print(f"  average precision {torch_result.average_precision:.4f}   "
              f"ROC-AUC {torch_result.roc_auc:.4f}")
        print(f"  TorchScript exported -> {path.name}")


def cmd_demo(args):
    if not pipeline.broker_available():
        raise SystemExit(
            "no Kafka broker on localhost:9092.\n"
            "  export JAVA_HOME=/usr/local/opt/openjdk\n"
            "  kafka-server-start kafka/server.properties"
        )
    pipeline.ensure_topic()

    X_ref, y_ref = _reference(seed=args.seed)
    model, version = model_mod.load()
    reference_scores = model.predict_proba(X_ref)[:, 1]
    reference_ap = model_mod.evaluate(model, X_ref, y_ref).average_precision
    print(f"model v{version}, reference sample {len(X_ref):,} events, "
          f"baseline AP {reference_ap:.3f}\n")

    half = args.events // 2
    # Publish on a thread WHILE consuming. Publishing everything up front and
    # then consuming measures how long events sat in the log -- the first
    # version of this reported a p50 "latency" of 103 seconds, which was
    # entirely backlog. Producing live at a sustainable rate is the only way
    # tick-to-decision means anything.
    def publish_stream():
        pipeline.publish(half, regime="baseline", seed=args.seed + 1,
                         topic=args.topic, rate=args.rate)
        pipeline.publish(half, regime=args.regime, seed=args.seed + 2,
                         topic=args.topic, rate=args.rate)

    producer_thread = threading.Thread(target=publish_stream, daemon=True)
    print(f"streaming {half:,} baseline + {half:,} {args.regime} events "
          f"at {args.rate:,.0f}/s while scoring")
    producer_thread.start()

    print(f"\nconsuming and scoring inline (window {args.window:,}):")
    run = pipeline.run_scoring(
        max_events=args.events,
        reference_X=X_ref,
        reference_scores=reference_scores,
        window=args.window,
        topic=args.topic,
        group=args.group,
        auto_retrain=not args.no_retrain,
        reference_ap=reference_ap,
        timeout_s=args.timeout,
    )

    producer_thread.join(timeout=5)
    summary = run.summary()
    t2d, inf = summary["tick_to_decision_us"], summary["inference_us"]
    print(f"\nconsumed {summary['consumed']:,} events in {summary['elapsed_s']:.1f}s "
          f"({summary['throughput_eps']:,.0f} events/s)")
    print(f"  tick-to-decision   p50 {t2d['p50'] / 1000:7.2f} ms   "
          f"p95 {t2d['p95'] / 1000:7.2f} ms   p99 {t2d['p99'] / 1000:7.2f} ms")
    print(f"  inference only     p50 {inf['p50']:7.0f} us   "
          f"p95 {inf['p95']:7.0f} us   p99 {inf['p99']:7.0f} us")

    if run.swaps:
        for s in run.swaps:
            print(f"\n  retrained at event {s['at_event']:,} -> v{s['new_version']}")
            print(f"    trigger: {s['trigger']}  "
                  f"({', '.join(s['drifted_features']) or 'no feature drift'})")
            print(f"    average precision {s['average_precision_before']:.3f} "
                  f"-> {s['average_precision_after']:.3f}")
    else:
        print("\n  no retrain triggered")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "regime": args.regime, "events": args.events, "window": args.window,
        "summary": summary,
    }, indent=2, default=float))
    print(f"\nresults -> {RESULTS}")


def cmd_drift(args):
    """Show what each drift regime does to the statistics, no Kafka involved."""
    X_ref, y_ref = _reference(seed=args.seed)
    model, _ = model_mod.load()
    ref_scores = model.predict_proba(X_ref)[:, 1]

    print(f"{'regime':<16} {'max PSI':>9} {'pred PSI':>9} {'AP':>7} {'verdict':>10}  drifted")
    for regime in ("baseline", "covariate_shift", "concept_drift"):
        events = Generator(seed=args.seed + 99, regime=regime).batch(8000)
        X, y = to_arrays(events)
        reports = drift.compare(X_ref, X, FEATURES)
        pred_psi = drift.prediction_drift(ref_scores, model.predict_proba(X)[:, 1])
        s = drift.summarise(reports, pred_psi)
        ap = model_mod.evaluate(model, X, y).average_precision
        print(f"{regime:<16} {s['max_psi']:>9.3f} {pred_psi:>9.3f} {ap:>7.3f} "
              f"{s['verdict']:>10}  {','.join(s['drifted_features']) or '-'}")

    print(
        "\nRead the concept_drift row carefully. Average precision collapses -- the "
        "model is\nworthless -- yet feature PSI AND prediction PSI both read ~0.00 and "
        "the verdict is\n'stable'. Unchanged inputs through an unchanged model give an "
        "unchanged score\ndistribution, so no unsupervised monitor can see this. Only "
        "labels can, and in card\nfraud labels arrive weeks late with the chargebacks.\n"
        "\ncovariate_shift is the opposite case: PSI screams, and it does so immediately, "
        "with\nno labels required."
    )


def cmd_bench(args):
    """Three serving paths, same model family, measured rather than argued about."""
    import time
    import requests

    model, _ = model_mod.load()
    events = Generator(seed=7).batch(args.n)
    vectors = [e.vector() for e in events]

    def measure(fn, items):
        out = []
        for item in items:
            t0 = time.perf_counter_ns()
            fn(item)
            out.append((time.perf_counter_ns() - t0) / 1000.0)
        return out

    inline = measure(lambda v: model_mod.score_one(model, v), vectors)
    print(f"sklearn inline   p50 {np.percentile(inline, 50):8.1f} us  "
          f"p99 {np.percentile(inline, 99):8.1f} us")

    try:
        from stream import torch_model
        scripted = torch_model.load_scripted()
    except Exception as exc:
        print(f"torchscript      (unavailable: {type(exc).__name__})")
    else:
        ts = measure(lambda v: torch_model.score_one(scripted, v), vectors)
        print(f"torchscript      p50 {np.percentile(ts, 50):8.1f} us  "
              f"p99 {np.percentile(ts, 99):8.1f} us   "
              f"({np.percentile(inline, 50) / np.percentile(ts, 50):.0f}x faster "
              f"than sklearn)")

    try:
        requests.get(f"{args.url}/health", timeout=2)
    except Exception:
        print(f"http    (no server at {args.url} -- "
              f"start it with: uvicorn stream.api:app --port 8000)")
        return

    http = []
    for e in events[: min(args.n, 300)]:
        payload = {f: getattr(e, f) for f in FEATURES}
        t0 = time.perf_counter_ns()
        requests.post(f"{args.url}/score", json=payload, timeout=5)
        http.append((time.perf_counter_ns() - t0) / 1000.0)
    print(f"http    p50 {np.percentile(http, 50):7.1f} us  "
          f"p99 {np.percentile(http, 99):7.1f} us")
    print(f"\nHTTP costs {np.percentile(http, 50) / np.percentile(inline, 50):.0f}x "
          f"the median of scoring in-process.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on baseline data and register")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--no-torch", action="store_true", help="skip the PyTorch model")
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("demo", help="publish, score inline, detect drift, retrain")
    d.add_argument("--events", type=int, default=12000)
    d.add_argument("--window", type=int, default=2000)
    d.add_argument("--regime", default="covariate_shift",
                   choices=["baseline", "covariate_shift", "concept_drift"])
    d.add_argument("--topic", default=pipeline.TOPIC)
    # A fresh consumer group per run, so a rerun replays from the start of the
    # topic instead of resuming at the previous run's committed offset and
    # consuming nothing.
    d.add_argument("--group", default=f"scorer-{int(time.time())}")
    d.add_argument("--rate", type=float, default=250.0,
                   help="events/s published; keep at or below what the "
                        "consumer can score or latency measures backlog")
    d.add_argument("--timeout", type=float, default=180.0)
    d.add_argument("--no-retrain", action="store_true")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("drift", help="compare drift statistics across regimes")
    r.set_defaults(func=cmd_drift)

    b = sub.add_parser("bench", help="inline vs HTTP scoring latency")
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--url", default="http://localhost:8000")
    b.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
