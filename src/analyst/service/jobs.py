"""Durable job queue with leases, fencing, retries and coalescing.

A research run takes minutes and costs money, so the queue has to be correct about
three things a naive "status column plus polling" design gets wrong:

**Exactly one worker owns a job at a time.** Claiming is a single
`UPDATE ... WHERE id = (SELECT ... LIMIT 1) RETURNING *` inside `BEGIN IMMEDIATE`,
so two workers cannot select the same row. Ownership is a *lease*: the claim sets
`lease_until`, and the worker renews it with heartbeats. If the worker dies, the
lease lapses and another worker takes the job over.

**A takeover does not redo finished work, or corrupt it.** Delivery is
at-least-once -- a job can be claimed again after its first worker vanished. That
is safe because the pipeline resumes from its last checkpoint, so the second worker
skips every node the first one finished. What is *not* safe is the first worker
coming back to life (a long GC pause, a stalled network call) and writing over the
second. Every claim increments `attempts`, which doubles as a fencing token: lease
renewals, completions, and run checkpoints all carry it, and all are rejected once
another worker has claimed the job. A stale worker finds out at its next write and
stops.

**Duplicate requests coalesce.** At most one job per research target and
configuration may be in flight, enforced by a partial unique index rather than by
check-then-insert. Two simultaneous requests for the same company produce one paid
run, and the second caller is handed the first caller's job.

Moving to Azure SQL keeps every one of these: the partial index becomes a filtered
unique index, the claim becomes `UPDATE TOP (1) ... WITH (UPDLOCK, READPAST) ...
OUTPUT inserted.*`, and the fenced checkpoint is the same statement.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from enum import Enum

from ..db import connect, transaction
from ..models import ResearchRun, new_id
from ..orchestrator.store import UPDATE_ASSIGNMENTS, RunStore, update_values

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    query            TEXT NOT NULL,
    dedupe_key       TEXT NOT NULL,
    idempotency_key  TEXT UNIQUE,
    status           TEXT NOT NULL,
    priority         INTEGER NOT NULL DEFAULT 100,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL,
    available_at     REAL NOT NULL,
    lease_until      REAL,
    worker_id        TEXT,
    last_error       TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    finished_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, available_at, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_run ON jobs(run_id);
CREATE INDEX IF NOT EXISTS idx_jobs_fresh ON jobs(dedupe_key, status, finished_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_inflight
    ON jobs(dedupe_key) WHERE status IN ('queued', 'running');
"""

CLAIM = """
UPDATE jobs
   SET status = 'running',
       attempts = attempts + 1,
       lease_until = ?,
       worker_id = ?,
       updated_at = ?
 WHERE id = (
       SELECT id FROM jobs
        WHERE cancel_requested = 0
          AND ((status = 'queued' AND available_at <= ?)
               OR (status = 'running' AND lease_until < ? AND attempts < max_attempts))
        ORDER BY priority, created_at
        LIMIT 1)
RETURNING *
"""

# A held lease is the fencing token: same job, same attempt, still running.
HOLDS_LEASE = "id = ? AND attempts = ? AND status = 'running'"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # deterministic failure: retrying would produce the same result
    DEAD = "dead"  # retryable failures exhausted every attempt
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.DEAD, JobStatus.CANCELLED)


class LostLease(RuntimeError):
    """This worker no longer owns the job; another worker has taken it over."""


@dataclass(frozen=True)
class Job:
    id: str
    run_id: str
    query: str
    dedupe_key: str
    idempotency_key: str | None
    status: JobStatus
    priority: int
    attempts: int
    max_attempts: int
    available_at: float
    lease_until: float | None
    worker_id: str | None
    last_error: str | None
    cancel_requested: bool
    created_at: float
    updated_at: float
    finished_at: float | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            query=row["query"],
            dedupe_key=row["dedupe_key"],
            idempotency_key=row["idempotency_key"],
            status=JobStatus(row["status"]),
            priority=row["priority"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            available_at=row["available_at"],
            lease_until=row["lease_until"],
            worker_id=row["worker_id"],
            last_error=row["last_error"],
            cancel_requested=bool(row["cancel_requested"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "available_at": self.available_at,
        }


@dataclass(frozen=True)
class Lease:
    """A worker's claim on a job. `token` is the fencing token."""

    job_id: str
    run_id: str
    token: int
    worker_id: str
    expires_at: float


@dataclass(frozen=True)
class Enqueued:
    job: Job
    created: bool
    reason: str  # "created" | "coalesced" | "idempotent"


def backoff_seconds(job_id: str, attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with deterministic jitter.

    Jitter matters because failures arrive in bursts -- a provider outage fails
    every in-flight job at once -- and identical delays would bring them all back
    at the same instant. Deriving it from the job id keeps it reproducible in tests.
    """
    raw = min(cap, base * (2 ** max(attempt - 1, 0)))
    digest = hashlib.blake2b(f"{job_id}:{attempt}".encode(), digest_size=2).digest()
    spread = int.from_bytes(digest, "big") / 65535
    return round(raw * (0.85 + 0.3 * spread), 3)


class JobQueue:
    def __init__(
        self,
        store: RunStore,
        *,
        clock: Callable[[], float] = time.time,
        max_attempts: int = 3,
        backoff_base_s: float = 30.0,
        backoff_cap_s: float = 600.0,
    ) -> None:
        # Jobs share the run store's database so enqueue and fenced checkpoints
        # can be single transactions.
        self.store = store
        self.path = store.path
        self.clock = clock
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return connect(self.path)

    # -- producing --------------------------------------------------------- #

    def enqueue(
        self,
        run: ResearchRun,
        *,
        dedupe_key: str,
        idempotency_key: str | None = None,
        priority: int = 100,
    ) -> Enqueued:
        """Create the run and its job atomically, or return the job already doing it.

        Check order matters: an idempotency key identifies one specific earlier
        request and wins outright; coalescing then catches a different request for
        the same work.
        """
        now = self.clock()
        job_id = new_id("job")
        try:
            with closing(self._connect()) as conn, transaction(conn):
                if idempotency_key:
                    row = conn.execute(
                        "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                    ).fetchone()
                    if row is not None:
                        return Enqueued(Job.from_row(row), created=False, reason="idempotent")
                row = conn.execute(
                    "SELECT * FROM jobs WHERE dedupe_key = ? AND status IN ('queued', 'running')",
                    (dedupe_key,),
                ).fetchone()
                if row is not None:
                    return Enqueued(Job.from_row(row), created=False, reason="coalesced")

                self.store.save(run, conn=conn)
                conn.execute(
                    """INSERT INTO jobs (id, run_id, query, dedupe_key, idempotency_key, status,
                                         priority, attempts, max_attempts, available_at,
                                         created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?)""",
                    (
                        job_id,
                        run.id,
                        run.query,
                        dedupe_key,
                        idempotency_key,
                        priority,
                        self.max_attempts,
                        now,
                        now,
                        now,
                    ),
                )
                created = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return Enqueued(Job.from_row(created), created=True, reason="created")
        except sqlite3.IntegrityError:
            # The transaction above already serialises enqueues on SQLite. This is
            # the backstop for backends without a database-wide write lock: the
            # schema's unique indexes reject the duplicate, the run insert rolls
            # back with it, and the caller gets whichever job won.
            winner = self._existing(idempotency_key, dedupe_key)
            if winner is None:
                raise
            reason = (
                "idempotent"
                if idempotency_key and winner.idempotency_key == idempotency_key
                else "coalesced"
            )
            return Enqueued(winner, created=False, reason=reason)

    def _existing(self, idempotency_key: str | None, dedupe_key: str) -> Job | None:
        with closing(self._connect()) as conn:
            if idempotency_key:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if row is not None:
                    return Job.from_row(row)
            row = conn.execute(
                "SELECT * FROM jobs WHERE dedupe_key = ? AND status IN ('queued', 'running')",
                (dedupe_key,),
            ).fetchone()
        return Job.from_row(row) if row else None

    # -- consuming --------------------------------------------------------- #

    def claim(self, worker_id: str, lease_seconds: float) -> Lease | None:
        """Take the next available job, or None if there is nothing to do."""
        now = self.clock()
        with closing(self._connect()) as conn, transaction(conn):
            self._reap(conn, now)
            rows = conn.execute(CLAIM, (now + lease_seconds, worker_id, now, now, now)).fetchall()
        if not rows:
            return None
        job = Job.from_row(rows[0])
        return Lease(
            job_id=job.id,
            run_id=job.run_id,
            token=job.attempts,
            worker_id=worker_id,
            expires_at=job.lease_until or now + lease_seconds,
        )

    def _reap(self, conn: sqlite3.Connection, now: float) -> None:
        """Settle expired leases that must not simply be claimed again."""
        # The worker died after a cancel was requested but before it noticed.
        conn.execute(
            """UPDATE jobs SET status = 'cancelled', lease_until = NULL,
                              finished_at = ?, updated_at = ?
                WHERE status = 'running' AND lease_until < ? AND cancel_requested = 1""",
            (now, now, now),
        )
        # The worker died on the final attempt; there is no retry left to give it.
        conn.execute(
            """UPDATE jobs SET status = 'dead', lease_until = NULL,
                              finished_at = ?, updated_at = ?,
                              last_error = COALESCE(last_error || ' | ', '')
                                  || 'lease expired on the final attempt; worker presumed dead'
                WHERE status = 'running' AND lease_until < ? AND attempts >= max_attempts""",
            (now, now, now),
        )

    def heartbeat(self, lease: Lease, lease_seconds: float) -> bool:
        """Extend the lease. False means it was lost -- stop working on this job."""
        now = self.clock()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"UPDATE jobs SET lease_until = ?, updated_at = ? WHERE {HOLDS_LEASE}",
                (now + lease_seconds, now, lease.job_id, lease.token),
            )
            return cursor.rowcount == 1

    def stop_reason(self, lease: Lease) -> str | None:
        """Why the holder of `lease` should stop, or None to carry on.

        Polled by the pipeline between nodes. Reading the row directly, rather than
        relying on a flag the heartbeat thread sets, keeps cancellation
        deterministic: the decision never depends on heartbeat timing.
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT attempts, status, cancel_requested FROM jobs WHERE id = ?",
                (lease.job_id,),
            ).fetchone()
        if row is None or row["attempts"] != lease.token or row["status"] != "running":
            return "lease lost"
        if row["cancel_requested"]:
            return "cancelled"
        return None

    def complete(self, lease: Lease) -> bool:
        return self._settle(lease, JobStatus.SUCCEEDED)

    def mark_cancelled(self, lease: Lease) -> bool:
        return self._settle(lease, JobStatus.CANCELLED)

    def _settle(self, lease: Lease, status: JobStatus) -> bool:
        now = self.clock()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"""UPDATE jobs SET status = ?, lease_until = NULL,
                                    finished_at = ?, updated_at = ?
                     WHERE {HOLDS_LEASE}""",
                (status.value, now, now, lease.job_id, lease.token),
            )
            return cursor.rowcount == 1

    def fail(self, lease: Lease, error: str, *, retryable: bool) -> Job | None:
        """Record a failure. Retryable failures are re-queued with backoff until
        the job runs out of attempts, then dead-lettered. Returns None if the lease
        had already been lost."""
        now = self.clock()
        with closing(self._connect()) as conn, transaction(conn):
            row = conn.execute(
                f"SELECT * FROM jobs WHERE {HOLDS_LEASE}", (lease.job_id, lease.token)
            ).fetchone()
            if row is None:
                return None
            job = Job.from_row(row)
            message = error[:2000]
            if job.cancel_requested:
                # Failing and being cancelled converge: the caller wanted it stopped.
                status, available_at = JobStatus.CANCELLED, None
            elif retryable and job.attempts < job.max_attempts:
                delay = backoff_seconds(
                    job.id, job.attempts, self.backoff_base_s, self.backoff_cap_s
                )
                status, available_at = JobStatus.QUEUED, now + delay
            else:
                status = JobStatus.DEAD if retryable else JobStatus.FAILED
                available_at = None

            if status is JobStatus.QUEUED:
                conn.execute(
                    """UPDATE jobs SET status = 'queued', available_at = ?, lease_until = NULL,
                                      worker_id = NULL, last_error = ?, updated_at = ?
                        WHERE id = ?""",
                    (available_at, message, now, job.id),
                )
            else:
                conn.execute(
                    """UPDATE jobs SET status = ?, lease_until = NULL, last_error = ?,
                                      finished_at = ?, updated_at = ?
                        WHERE id = ?""",
                    (status.value, message, now, now, job.id),
                )
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job.id,)).fetchone()
        return Job.from_row(updated)

    def request_cancel(self, job_id: str) -> Job | None:
        """Cancel a queued job immediately; ask a running one to stop between nodes."""
        now = self.clock()
        with closing(self._connect()) as conn, transaction(conn):
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            job = Job.from_row(row)
            if job.status is JobStatus.QUEUED:
                conn.execute(
                    """UPDATE jobs SET status = 'cancelled', cancel_requested = 1,
                                      finished_at = ?, updated_at = ?
                        WHERE id = ?""",
                    (now, now, job_id),
                )
            elif job.status is JobStatus.RUNNING:
                conn.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (now, job_id),
                )
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(updated)

    # -- reading ----------------------------------------------------------- #

    def get(self, job_id: str) -> Job | None:
        return self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))

    def by_idempotency_key(self, key: str) -> Job | None:
        return self._one("SELECT * FROM jobs WHERE idempotency_key = ?", (key,))

    def for_run(self, run_id: str) -> Job | None:
        return self._one(
            "SELECT * FROM jobs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1", (run_id,)
        )

    def fresh_success(self, dedupe_key: str, max_age_s: float) -> Job | None:
        """The newest completed job for this work, if it is young enough to reuse."""
        return self._one(
            """SELECT * FROM jobs
                WHERE dedupe_key = ? AND status = 'succeeded' AND finished_at >= ?
                ORDER BY finished_at DESC LIMIT 1""",
            (dedupe_key, self.clock() - max_age_s),
        )

    def list(self, limit: int = 50, *, status: JobStatus | None = None) -> list[Job]:
        sql, params = "SELECT * FROM jobs", []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status.value)
        sql += " ORDER BY created_at DESC LIMIT ?"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, [*params, limit]).fetchall()
        return [Job.from_row(r) for r in rows]

    def depth(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        counts = {s.value: 0 for s in JobStatus}
        counts.update({r["status"]: r["n"] for r in rows})
        return counts

    def _one(self, sql: str, params: tuple[object, ...]) -> Job | None:
        with closing(self._connect()) as conn:
            row = conn.execute(sql, params).fetchone()
        return Job.from_row(row) if row else None


class FencedRunStore(RunStore):
    """A RunStore whose writes land only while the writer still holds its lease.

    The fence and the write are one statement, so there is no gap between "do I
    still own this job?" and "write the checkpoint" for another worker to claim the
    job in. A worker that lost its lease gets `LostLease` instead of a write.
    """

    def __init__(self, inner: RunStore, lease: Lease) -> None:
        # Deliberately skips RunStore.__init__: the schema already exists, and
        # this object only redirects writes.
        self.path = inner.path
        self.lease = lease

    def save(self, run: ResearchRun, *, conn: sqlite3.Connection | None = None) -> None:
        run.touch()
        with closing(self._connect()) as own:
            cursor = own.execute(
                f"""UPDATE runs SET {UPDATE_ASSIGNMENTS}
                     WHERE id = ?
                       AND EXISTS (SELECT 1 FROM jobs WHERE {HOLDS_LEASE})""",
                (*update_values(run), run.id, self.lease.job_id, self.lease.token),
            )
            written = cursor.rowcount == 1
        if not written:
            raise LostLease(
                f"job {self.lease.job_id} attempt {self.lease.token}: lease lost, "
                f"checkpoint for {run.id} refused"
            )
