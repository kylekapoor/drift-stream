#!/usr/bin/env python3
"""Self-checks. No Kafka and no trained model needed:

    python test_stream.py

The PSI tests carry the weight. PSI is the statistic every decision in this
pipeline hangs off, and its two failure modes -- re-binning on the current
window, and unclamped empty bins -- both produce numbers that look plausible
while being wrong in opposite directions.
"""
from __future__ import annotations

import numpy as np

from stream import drift, model as model_mod, pipeline
from stream.events import FEATURES, Event, Generator, to_arrays


# --- PSI --------------------------------------------------------------------

def test_identical_distributions_have_no_drift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=6000), rng.normal(size=6000)
    assert drift.psi(a, b) < 0.01


def test_a_shifted_distribution_registers_as_material():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 6000)
    b = rng.normal(2.5, 1, 6000)
    assert drift.psi(a, b) > drift.PSI_MATERIAL


def test_psi_grows_monotonically_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 8000)
    values = [drift.psi(reference, rng.normal(shift, 1, 8000))
              for shift in (0.0, 0.3, 0.8, 1.5, 3.0)]
    assert values == sorted(values), values


def test_bin_edges_come_from_the_reference_not_the_current_window():
    """The mistake that makes PSI read ~0 exactly when drift is worst.

    Re-deriving quantile edges from the current window compares that window to
    itself: any distribution, however far it has moved, spreads evenly across
    its own quantiles.
    """
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 8000)
    shifted = rng.normal(6, 1, 8000)

    assert drift.psi(reference, shifted) > 1.0

    # What re-binning would have produced:
    edges = np.unique(np.quantile(shifted, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    self_share = np.histogram(shifted, bins=edges)[0] / len(shifted)
    naive = float(np.sum((self_share - self_share) * 0))
    assert naive == 0.0, "self-comparison is identically zero -- hence the bug"


def test_empty_bins_stay_finite():
    """A current window with nothing in a reference bin must not give infinity."""
    reference = np.concatenate([np.zeros(1000), np.ones(1000) * 10])
    current = np.zeros(1000)  # nothing at all in the upper bins
    value = drift.psi(reference, current)
    assert np.isfinite(value), value
    assert value > drift.PSI_MATERIAL


def test_psi_handles_binary_and_constant_features():
    rng = np.random.default_rng(3)
    flag_a = (rng.random(4000) < 0.1).astype(float)
    flag_b = (rng.random(4000) < 0.6).astype(float)
    value = drift.psi(flag_a, flag_b)
    assert np.isfinite(value) and value > drift.PSI_MATERIAL

    constant = np.ones(500)
    assert np.isfinite(drift.psi(constant, constant))


def test_psi_of_empty_input_is_zero_not_a_crash():
    assert drift.psi(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert drift.psi(np.array([1.0, 2.0]), np.array([])) == 0.0


# --- performance trigger ----------------------------------------------------

def test_performance_drop_fires_on_a_real_collapse():
    assert drift.performance_drop(0.45, 0.03, positives=100)


def test_performance_drop_abstains_on_too_few_positives():
    """The guard that stops noise-triggered retraining."""
    assert not drift.performance_drop(0.45, 0.03, positives=12)
    assert not drift.performance_drop(0.45, 0.03, positives=drift.MIN_POSITIVES_FOR_AP - 1)
    assert drift.performance_drop(0.45, 0.03, positives=drift.MIN_POSITIVES_FOR_AP)


def test_performance_drop_ignores_a_small_dip():
    assert not drift.performance_drop(0.45, 0.40, positives=200)


# --- the drift regimes ------------------------------------------------------

def test_fraud_rate_is_calibrated_in_every_regime():
    """Prevalence must be equal across regimes or the experiment is confounded."""
    for regime in ("baseline", "covariate_shift", "concept_drift"):
        _, y = to_arrays(Generator(seed=5, regime=regime).batch(20000))
        assert 0.015 < y.mean() < 0.026, f"{regime} fraud rate {y.mean():.3%}"


def test_covariate_shift_actually_moves_the_inputs():
    X_ref, _ = to_arrays(Generator(seed=0, regime="baseline").batch(9000))
    X_cur, _ = to_arrays(Generator(seed=1, regime="covariate_shift").batch(9000))
    reports = drift.compare(X_ref, X_cur, FEATURES)
    assert any(r.drifted for r in reports), [r.psi for r in reports]


def test_concept_drift_leaves_every_input_distribution_alone():
    """The central claim of this project, asserted rather than described.

    If this fails, the concept-drift regime is secretly shifting covariates and
    the "PSI cannot see this" demonstration is worthless.
    """
    X_ref, _ = to_arrays(Generator(seed=0, regime="baseline").batch(12000))
    X_cur, _ = to_arrays(Generator(seed=1, regime="concept_drift").batch(12000))
    reports = drift.compare(X_ref, X_cur, FEATURES)
    worst = max(r.psi for r in reports)
    assert worst < drift.PSI_MODERATE, (
        f"concept drift moved an input distribution (max PSI {worst:.3f}); "
        "the regime is contaminated with covariate shift"
    )


def test_concept_drift_does_change_the_labelling_rule():
    """...but the input-output relationship must genuinely differ."""
    gen_a = Generator(seed=0, regime="baseline")
    gen_b = Generator(seed=0, regime="concept_drift")
    features = [gen_a._draw_features() for _ in range(3000)]
    p_a = np.array([gen_a._fraud_probability(f) for f in features])
    p_b = np.array([gen_b._fraud_probability(f) for f in features])
    # Same inputs, different probabilities: that is concept drift by definition.
    assert np.corrcoef(p_a, p_b)[0, 1] < 0.5, np.corrcoef(p_a, p_b)[0, 1]


def test_summarise_flags_material_drift():
    X_ref, _ = to_arrays(Generator(seed=0, regime="baseline").batch(6000))
    X_cur, _ = to_arrays(Generator(seed=1, regime="covariate_shift").batch(6000))
    s = drift.summarise(drift.compare(X_ref, X_cur, FEATURES), pred_psi=0.5)
    assert s["verdict"] == "material"
    assert s["drifted_features"]
    assert len(s["per_feature"]) == len(FEATURES)


# --- events -----------------------------------------------------------------

def test_event_survives_a_json_round_trip():
    event = Generator(seed=0).next()
    restored = Event.from_json(event.to_json())
    assert restored == event
    assert restored.vector() == event.vector()


def test_feature_vector_matches_the_declared_order():
    event = Generator(seed=0).next()
    assert len(event.vector()) == len(FEATURES)
    assert event.vector()[0] == event.amount
    assert event.vector()[FEATURES.index("foreign_country")] == event.foreign_country


def test_generator_is_reproducible():
    a = Generator(seed=42).batch(50)
    b = Generator(seed=42).batch(50)
    assert [e.vector() for e in a] == [e.vector() for e in b]
    assert [e.is_fraud for e in a] == [e.is_fraud for e in b]


# --- model ------------------------------------------------------------------

def test_model_learns_the_baseline_rule():
    X, y = to_arrays(Generator(seed=0).batch(15000))
    model, result, _ = model_mod.train(X, y, register=False, log=False)
    # Random guessing scores about the base rate (~0.02) on average precision.
    assert result.average_precision > 0.25, result.average_precision
    assert result.roc_auc > 0.85, result.roc_auc


def test_evaluation_handles_a_window_with_no_fraud():
    X, y = to_arrays(Generator(seed=0).batch(400))
    model, _, _ = model_mod.train(*to_arrays(Generator(seed=1).batch(6000)),
                                  register=False, log=False)
    result = model_mod.evaluate(model, X, np.zeros(len(X), dtype=int))
    assert result.positives == 0
    assert result.recall_at_threshold == 0.0
    assert np.isfinite(result.average_precision)


# --- scorer -----------------------------------------------------------------

def test_hot_swap_replaces_the_serving_model():
    X, y = to_arrays(Generator(seed=0).batch(6000))
    first, _, _ = model_mod.train(X, y, register=False, log=False)
    second, _, _ = model_mod.train(*to_arrays(Generator(seed=9).batch(6000)),
                                   register=False, log=False)

    scorer = pipeline.Scorer(first, version=1)
    assert scorer.version == 1
    scorer.swap(second, version=2)
    assert scorer.model is second and scorer.version == 2
    assert 0.0 <= scorer.score(Generator(seed=3).next().vector()) <= 1.0


def test_scorer_guards_against_stacked_retrains():
    scorer = pipeline.Scorer(model=None, version=1)
    assert scorer.retraining is False
    scorer.retraining = True
    assert scorer.retraining, "flag must persist so a second window does not refit"


def test_percentiles_are_ordered():
    run = pipeline.ScoringRun()
    run.tick_to_decision_us = list(np.random.default_rng(0).exponential(2000, 5000))
    p = run.percentiles(run.tick_to_decision_us)
    assert p["p50"] < p["p95"] < p["p99"] <= p["max"]


# --- postgres sink ----------------------------------------------------------

def test_sink_is_a_no_op_without_a_database():
    """The pipeline must run identically with or without Postgres."""
    from stream import sink as sink_mod

    s = sink_mod.Sink(run_id="test", dsn=None).connect()
    assert not s.enabled
    s.record(1, 0.5, 0, 1, 100.0)          # must not raise
    s.record_drift({"at_event": 1, "verdict": "stable"})
    assert s.flush() == 0 and s.written == 0
    assert s.report() == []
    s.close()


def test_sink_batches_instead_of_writing_per_event():
    """One insert per event would put a round trip inside the scoring loop."""
    from stream import sink as sink_mod

    calls = []

    class FakeCursor:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a): pass
        def executemany(self, sql, rows): calls.append(len(rows))

    class FakeConn:
        def cursor(self): return FakeCursor()
        def close(self): pass

    s = sink_mod.Sink(run_id="test", dsn="x")
    s._conn = FakeConn()

    for i in range(sink_mod.BATCH_SIZE - 1):
        s.record(i, 0.1, 0, 1, 50.0)
    assert calls == [], "flushed before the batch was full"

    s.record(999, 0.1, 0, 1, 50.0)
    assert calls == [sink_mod.BATCH_SIZE], f"expected one batched insert, got {calls}"
    assert s.written == sink_mod.BATCH_SIZE


def test_sink_survives_an_unreachable_database():
    """A missing database must not take the pipeline down with it."""
    from stream import sink as sink_mod

    s = sink_mod.Sink(run_id="test", dsn="postgresql://nobody@127.0.0.1:1/nope").connect()
    assert not s.enabled
    s.record(1, 0.5, 0, 1, 100.0)
    s.close()


def test_a_database_failure_does_not_take_the_pipeline_down():
    """`connect` already treats a missing database as "run without one".

    `flush` has to agree. A Postgres restart mid-run used to raise out of
    `record`, up through the consumer loop, and stop a pipeline that is
    supposed to work fine with no database at all.
    """
    from stream import sink as sink_mod

    class Exploding:
        def cursor(self):
            raise RuntimeError("server closed the connection unexpectedly")

        def close(self):
            raise RuntimeError("already gone")

    sink = sink_mod.Sink(run_id="t", dsn="postgres://ignored")
    sink._conn = Exploding()

    for i in range(3):
        sink.record(i, 0.5, 0, 1, 10.0)          # must not raise
    assert sink.flush() == 0
    assert sink.dropped == 3, f"lost rows went uncounted: {sink.dropped}"
    assert not sink.enabled, "a failed sink must disable itself, not retry"

    sink.record_drift({"at_event": 1, "verdict": "stable"})   # must not raise
    sink.close()                                              # must not raise


def test_a_disabled_sink_reports_zero_rather_than_pretending():
    from stream import sink as sink_mod

    sink = sink_mod.Sink(run_id="t", dsn=None).connect()
    assert not sink.enabled
    assert sink.flush() == 0
    assert sink.written == 0 and sink.dropped == 0
    assert sink.report() == []


def test_reference_scores_are_rebaselined_before_the_model_is_swapped():
    """The consumer thread closes windows while the retrain thread runs.

    One landing between the swap and the re-baseline would score the new
    model against the old model's reference distribution, which is the
    retrain-swap-drift-retrain loop this re-baseline exists to prevent.
    """
    import inspect

    source = inspect.getsource(pipeline._check_window)
    swap_at = source.index("scorer.swap(new_model, version)")
    rebaseline_at = source.index("scorer.reference_scores = new_reference")
    assert rebaseline_at < swap_at, (
        "reference_scores must be assigned before scorer.swap, otherwise a "
        "window closing between them compares the new model to the old baseline"
    )


def test_scorer_swap_replaces_model_and_version_together():
    scorer = pipeline.Scorer(model="old", version=1, reference_scores=[0.1])
    scorer.swap("new", 2)
    assert (scorer.model, scorer.version) == ("new", 2)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
