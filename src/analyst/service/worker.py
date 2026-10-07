"""Background worker: claim a job, run the pipeline, record the outcome.

One job at a time per process. Throughput scales by running more worker processes
(`docker compose up --scale worker=3`, or more replicas in the cloud) rather than
threads inside one, because each replica is then independently restartable and the
queue already guarantees they never share a job.

Every outcome maps to a definite job state:

    pipeline finishes          -> succeeded
    pipeline raises            -> re-queued with backoff, or dead after the last attempt
    model refuses / ambiguous  -> failed (retrying would produce the same answer)
    cancel requested           -> cancelled, at the next node boundary
    lease lost                 -> nothing: another worker owns the job now

`run_once` is also the unit an event-driven deployment wants: a scale-to-zero job
runner starts a container, the container processes one job, and exits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

from ..cache import FetchCache
from ..config import settings
from ..llm.client import LLMClient, LLMRefusal, build_client
from ..models import ResearchRun, RunStatus
from ..orchestrator.pipeline import Pipeline, PipelineDeps, PipelineStopped
from ..tools.fetch import Fetcher
from .events import EventLog
from .jobs import FencedRunStore, JobQueue, JobStatus, Lease, LostLease

log = logging.getLogger(__name__)

DepsFactory = Callable[[ResearchRun], tuple[LLMClient, Fetcher]]
Note = Callable[[str, str], None]


def default_deps(run: ResearchRun) -> tuple[LLMClient, Fetcher]:
    config = settings()
    llm = build_client(run.config.model, config.api_key, stub=run.config.stub)
    return llm, Fetcher(FetchCache(), config)


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(frozen=True)
class Outcome:
    job_id: str
    run_id: str
    result: str  # succeeded | retrying | dead | failed | cancelled | lost_lease
    detail: str = ""


class Heartbeat:
    """Renews a lease on a dedicated thread for as long as the job runs.

    This has to be a thread rather than an asyncio task. The pipeline's model calls
    are synchronous and hold the event loop for seconds at a time; a heartbeat
    scheduled on that loop would starve, the lease would lapse mid-run, and a second
    worker would pick up a job that is still being worked on.
    """

    def __init__(self, queue: JobQueue, lease: Lease, *, interval_s: float, lease_s: float) -> None:
        self._queue = queue
        self._lease = lease
        self._interval_s = interval_s
        self._lease_s = lease_s
        self._stop = threading.Event()
        self.lost = False
        self.beats = 0
        self._thread = threading.Thread(
            target=self._loop, name=f"heartbeat-{lease.job_id}", daemon=True
        )

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval_s + 5)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                held = self._queue.heartbeat(self._lease, self._lease_s)
            except Exception:  # noqa: BLE001 - one failed renewal is survivable
                # The lease outlasts several missed beats by design, so a transient
                # database error is logged and retried rather than treated as loss.
                log.exception("heartbeat for %s failed; retrying", self._lease.job_id)
                continue
            if not held:
                self.lost = True
                return
            self.beats += 1


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        events: EventLog,
        *,
        deps_factory: DepsFactory = default_deps,
        worker_id: str | None = None,
        lease_s: float = 120.0,
        heartbeat_s: float = 20.0,
        poll_s: float = 2.0,
    ) -> None:
        if heartbeat_s * 3 > lease_s:
            raise ValueError(
                f"heartbeat every {heartbeat_s}s cannot keep a {lease_s}s lease safely; "
                f"the lease should survive at least three missed beats"
            )
        self.queue = queue
        self.events = events
        self.deps_factory = deps_factory
        self.worker_id = worker_id or default_worker_id()
        self.lease_s = lease_s
        self.heartbeat_s = heartbeat_s
        self.poll_s = poll_s

    # -- loop -------------------------------------------------------------- #

    def run_forever(
        self,
        stop: threading.Event,
        *,
        max_jobs: int | None = None,
        on_outcome: Callable[[Outcome], None] | None = None,
    ) -> int:
        """Process jobs until `stop` is set. A job in progress always finishes;
        stopping only prevents the next claim."""
        processed = 0
        while not stop.is_set():
            outcome = self.run_once()
            if outcome is None:
                stop.wait(self.poll_s)
                continue
            processed += 1
            if on_outcome is not None:
                on_outcome(outcome)
            if max_jobs is not None and processed >= max_jobs:
                break
        return processed

    def run_once(self) -> Outcome | None:
        """Claim and fully process one job. None if the queue was empty."""
        lease = self.queue.claim(self.worker_id, self.lease_s)
        if lease is None:
            return None

        def note(status: str, detail: str = "") -> None:
            self.events.append(lease.run_id, status, detail=detail, source="worker")

        note("claimed", f"{self.worker_id} took attempt {lease.token}")
        run = self.queue.store.load(lease.run_id)
        if run is None:
            self.queue.fail(lease, "run record missing", retryable=False)
            note("failed", "run record missing")
            return Outcome(lease.job_id, lease.run_id, "failed", "run record missing")

        with Heartbeat(self.queue, lease, interval_s=self.heartbeat_s, lease_s=self.lease_s):
            return self._process(lease, run, note)

    # -- one job ----------------------------------------------------------- #

    def _process(self, lease: Lease, run: ResearchRun, note: Note) -> Outcome:
        fenced = FencedRunStore(self.queue.store, lease)
        try:
            llm, fetcher = self.deps_factory(run)
            deps = PipelineDeps(
                llm=llm,
                fetcher=fetcher,
                store=fenced,
                progress=self.events.recorder(run.id),
                should_stop=lambda: self.queue.stop_reason(lease),
            )
            finished = asyncio.run(Pipeline(deps).run(run))
        except PipelineStopped as stop:
            if stop.reason == "cancelled":
                return self._cancel(lease, run, fenced, note)
            return self._lost(lease, note, stop.reason)
        except LostLease as exc:
            return self._lost(lease, note, str(exc))
        except LLMRefusal as exc:
            return self._failed(lease, note, exc, retryable=False)
        except Exception as exc:  # noqa: BLE001 - any other failure is a retry candidate
            log.exception("job %s attempt %s failed", lease.job_id, lease.token)
            return self._failed(lease, note, exc, retryable=True)

        if finished.status is RunStatus.AMBIGUOUS:
            # Resolution is deterministic for a given SEC index; retrying cannot help.
            message = finished.error or "could not resolve the research target"
            self.queue.fail(lease, message, retryable=False)
            note("failed", message)
            return Outcome(lease.job_id, lease.run_id, "failed", message)

        if not self.queue.complete(lease):
            return self._lost(lease, note, "finished, but the lease was lost before completion")
        note(
            "done",
            f"{sum(1 for c in finished.claims if c.survived)} claims published, "
            f"${finished.ledger.total_usd:.4f}",
        )
        return Outcome(lease.job_id, lease.run_id, "succeeded")

    def _cancel(
        self, lease: Lease, run: ResearchRun, fenced: FencedRunStore, note: Note
    ) -> Outcome:
        run.status = RunStatus.CANCELLED
        try:
            fenced.save(run)
        except LostLease as exc:
            return self._lost(lease, note, str(exc))
        self.queue.mark_cancelled(lease)
        note("cancelled", "stopped at a node boundary at the caller's request")
        return Outcome(lease.job_id, lease.run_id, "cancelled")

    def _lost(self, lease: Lease, note: Note, detail: str) -> Outcome:
        # Deliberately writes nothing to the job or the run: they belong to
        # whichever worker holds the lease now.
        note("lease lost", f"{self.worker_id} stopped; {detail}")
        return Outcome(lease.job_id, lease.run_id, "lost_lease", detail)

    def _failed(self, lease: Lease, note: Note, exc: BaseException, *, retryable: bool) -> Outcome:
        message = f"{type(exc).__name__}: {exc}"
        job = self.queue.fail(lease, message, retryable=retryable)
        if job is None:
            return self._lost(lease, note, "lease lost while recording a failure")
        if job.status is JobStatus.QUEUED:
            delay = max(0.0, job.available_at - self.queue.clock())
            note(
                "retry scheduled",
                f"attempt {job.attempts} of {job.max_attempts} failed ({message[:160]}); "
                f"retrying in {delay:.0f}s",
            )
            return Outcome(lease.job_id, lease.run_id, "retrying", message)
        note(job.status.value, message[:300])
        return Outcome(lease.job_id, lease.run_id, job.status.value, message)
