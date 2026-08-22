"""Postgres sink for scored events and drift checks.

A scoring pipeline that writes its decisions to a JSON file at the end of a run
is a demo. A real one persists every decision as it happens, because the
questions that matter afterwards are queries: how did precision move by model
version, which features drifted before the retrain fired, what did we score the
transaction the customer is now disputing.

Two things make this non-trivial and both are handled here:

**Writes are batched.** Inserting one row per event would put a network round
trip inside the scoring loop -- the exact hot path this project measures in
microseconds. Rows accumulate and flush in blocks, so the database never appears
in the tick-to-decision budget.

**The sink is optional.** With no DATABASE_URL it becomes a no-op, so the
pipeline runs identically with or without Postgres and the tests need no
database.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT        NOT NULL,
    event_id      BIGINT      NOT NULL,
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version INTEGER,
    score         DOUBLE PRECISION NOT NULL,
    is_fraud      SMALLINT,
    latency_us    DOUBLE PRECISION
);

-- The queries this table exists for are "recent rows for one model version"
-- and "rows for this run", so those are the indexes. A primary key alone would
-- force a sequential scan for both.
CREATE INDEX IF NOT EXISTS predictions_version_time
    ON predictions (model_version, scored_at DESC);
CREATE INDEX IF NOT EXISTS predictions_run
    ON predictions (run_id, event_id);

CREATE TABLE IF NOT EXISTS drift_checks (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT        NOT NULL,
    checked_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    at_event       INTEGER     NOT NULL,
    verdict        TEXT        NOT NULL,
    trigger        TEXT,
    max_psi        DOUBLE PRECISION,
    prediction_psi DOUBLE PRECISION,
    avg_precision  DOUBLE PRECISION,
    drifted        JSONB
);
CREATE INDEX IF NOT EXISTS drift_checks_run ON drift_checks (run_id, at_event);
"""

# Big enough that the database is never in the scoring loop's way, small enough
# that a crash loses a trivial amount.
BATCH_SIZE = 500


def available() -> bool:
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        import psycopg  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class Sink:
    """Batched writer. A no-op when DATABASE_URL is unset."""

    run_id: str
    dsn: str | None = None
    _conn: object | None = None
    _pending: list = field(default_factory=list)
    written: int = 0
    # Rows lost to a database failure. Reported, never silently zero.
    dropped: int = 0

    def __post_init__(self):
        self.dsn = self.dsn or os.getenv("DATABASE_URL")

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def connect(self) -> "Sink":
        if not self.dsn:
            return self
        try:
            import psycopg

            self._conn = psycopg.connect(self.dsn, autocommit=True)
            with self._conn.cursor() as cur:
                cur.execute(SCHEMA)
        except Exception:
            # A missing database must not take the pipeline down with it.
            self._conn = None
        return self

    def record(self, event_id, score, is_fraud, model_version, latency_us) -> None:
        if not self._conn:
            return
        self._pending.append(
            (self.run_id, int(event_id), model_version, float(score),
             int(is_fraud), float(latency_us))
        )
        if len(self._pending) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> int:
        """Write the pending batch. Returns rows written, 0 if the write failed.

        `connect` already treats a missing database as "run without one", and
        this has to agree: a Postgres restart mid-run used to raise out of
        `record`, through the consumer loop, and kill a pipeline that is
        supposed to work fine with no database at all. The batch is dropped and
        the sink disables itself rather than retrying, because these rows are
        analysis, not the decision the pipeline just made.
        """
        if not self._conn or not self._pending:
            return 0
        rows, self._pending = self._pending, []
        try:
            with self._conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO predictions "
                    "(run_id, event_id, model_version, score, is_fraud, latency_us) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    rows,
                )
        except Exception:
            self.dropped += len(rows)
            self._conn = None
            return 0
        self.written += len(rows)
        return len(rows)

    def record_drift(self, check: dict) -> None:
        if not self._conn:
            return
        try:
            self._insert_drift(check)
        except Exception:
            self._conn = None

    def _insert_drift(self, check: dict) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO drift_checks (run_id, at_event, verdict, trigger, "
                "max_psi, prediction_psi, avg_precision, drifted) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (self.run_id, check.get("at_event", 0), check.get("verdict", ""),
                 check.get("trigger"), check.get("max_psi"),
                 check.get("prediction_psi"), check.get("average_precision"),
                 json.dumps(check.get("drifted_features", []))),
            )

    def close(self) -> None:
        self.flush()
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # --- reporting ---------------------------------------------------------

    PRECISION_BY_VERSION = """
        SELECT model_version,
               COUNT(*)                                        AS scored,
               SUM(is_fraud)                                   AS actual_fraud,
               AVG(score) FILTER (WHERE is_fraud = 1)          AS mean_score_fraud,
               AVG(score) FILTER (WHERE is_fraud = 0)          AS mean_score_clean,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_us) AS p50_latency_us
        FROM predictions
        WHERE run_id = %s
        GROUP BY model_version
        ORDER BY model_version
    """

    def report(self) -> list:
        """Separation between fraud and clean scores, per model version.

        A model that has stopped working shows the two means converging, which
        is visible here without any labels-versus-threshold bookkeeping.
        """
        if not self._conn:
            return []
        with self._conn.cursor() as cur:
            cur.execute(self.PRECISION_BY_VERSION, (self.run_id,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
