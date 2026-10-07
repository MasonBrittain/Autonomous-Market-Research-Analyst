"""Job queue and worker tests.

The properties under test are the ones that cost money when they break:

* a job is claimed by exactly one worker, even under real contention;
* a crashed worker's job is taken over -- and the takeover resumes from the
  checkpoint instead of redoing finished work;
* a worker that lost its lease cannot overwrite the worker that replaced it;
* duplicate requests coalesce into one paid run;
* failures retry with backoff, then dead-letter; refusals do not retry.

Most tests drive time with a fake clock, so lease expiry is instant and exact.
The heartbeat test is the deliberate exception: it has to prove a real thread keeps
a lease alive while the main thread is blocked, so it uses real time.
"""

from __future__ import annotations

import threading
import time

import pytest

from analyst.llm.client import LLMRefusal
from analyst.llm.stub import StubLLM
from analyst.models import NodeStatus, ResearchRun, RunConfig, RunStatus
from analyst.orchestrator.store import RunStore
from analyst.service.events import EventLog
from analyst.service.jobs import (
    FencedRunStore,
    JobQueue,
    JobStatus,
    LostLease,
    backoff_seconds,
)
from analyst.service.worker import Heartbeat, Worker


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store() -> RunStore:
    return RunStore()


@pytest.fixture
def queue(store: RunStore, clock: FakeClock) -> JobQueue:
    return JobQueue(store, clock=clock, max_attempts=3, backoff_base_s=30, backoff_cap_s=600)


def _run(query: str = "Apple") -> ResearchRun:
    return ResearchRun(query=query, config=RunConfig(stub=True))


def _enqueue(queue: JobQueue, key: str = "ticker:AAPL@cfg", query: str = "Apple"):
    return queue.enqueue(_run(query), dedupe_key=key)


def _worker(queue: JobQueue, fetcher, *, llm=None, worker_id: str = "w1") -> Worker:
    return Worker(
        queue,
        EventLog(queue.path),
        deps_factory=lambda run: (llm or StubLLM(), fetcher),
        worker_id=worker_id,
        lease_s=60,
        heartbeat_s=10,
    )


# --------------------------------------------------------------------------- #
# Enqueue: atomicity, coalescing, idempotency
# --------------------------------------------------------------------------- #


def test_enqueue_creates_the_run_and_the_job_together(queue: JobQueue, store: RunStore):
    result = _enqueue(queue)
    assert result.created and result.reason == "created"
    assert result.job.status is JobStatus.QUEUED
    assert store.load(result.job.run_id) is not None, "job points at a run that does not exist"


def test_duplicate_in_flight_request_coalesces(queue: JobQueue, store: RunStore):
    first = _enqueue(queue, query="Apple")
    second = _enqueue(queue, query="AAPL")

    assert not second.created
    assert second.reason == "coalesced"
    assert second.job.id == first.job.id
    # The losing request's run was never written.
    assert len(store.list_runs(50)) == 1


def test_coalescing_ends_once_the_job_is_finished(queue: JobQueue):
    first = _enqueue(queue)
    lease = queue.claim("w1", 60)
    assert lease is not None
    queue.complete(lease)

    second = _enqueue(queue)
    assert second.created, "a finished job must not absorb new requests"
    assert second.job.id != first.job.id


def test_concurrent_duplicate_requests_create_exactly_one_job(queue: JobQueue, store: RunStore):
    """Eight simultaneous requests for the same company, on eight connections,
    must produce one job and one run."""
    barrier = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def request() -> None:
        barrier.wait()
        outcome = _enqueue(queue)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=request) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(r.created for r in results) == 1
    assert len({r.job.id for r in results}) == 1
    assert len(store.list_runs(50)) == 1
    assert queue.depth()["queued"] == 1


def test_idempotency_key_replays_the_original_job(queue: JobQueue):
    first = queue.enqueue(_run(), dedupe_key="ticker:AAPL@a", idempotency_key="req-1")
    replay = queue.enqueue(_run(), dedupe_key="ticker:AAPL@a", idempotency_key="req-1")
    assert replay.reason == "idempotent"
    assert replay.job.id == first.job.id
    assert queue.by_idempotency_key("req-1") is not None


def test_idempotency_key_wins_over_coalescing_after_completion(queue: JobQueue):
    """A replayed request gets its own original job back even once it has finished
    and a newer identical job exists."""
    first = queue.enqueue(_run(), dedupe_key="k", idempotency_key="req-1")
    lease = queue.claim("w1", 60)
    assert lease is not None
    queue.complete(lease)
    queue.enqueue(_run(), dedupe_key="k")

    replay = queue.enqueue(_run(), dedupe_key="k", idempotency_key="req-1")
    assert replay.job.id == first.job.id


# --------------------------------------------------------------------------- #
# Claiming
# --------------------------------------------------------------------------- #


def test_claim_returns_none_on_an_empty_queue(queue: JobQueue):
    assert queue.claim("w1", 60) is None


def test_claim_marks_the_job_running_and_issues_a_fencing_token(queue: JobQueue):
    _enqueue(queue)
    lease = queue.claim("w1", 60)
    assert lease is not None
    assert lease.token == 1
    job = queue.get(lease.job_id)
    assert job is not None and job.status is JobStatus.RUNNING and job.worker_id == "w1"


def test_a_held_lease_is_not_claimable(queue: JobQueue, clock: FakeClock):
    _enqueue(queue)
    assert queue.claim("w1", 60) is not None
    clock.advance(59)
    assert queue.claim("w2", 60) is None


def test_concurrent_workers_never_claim_the_same_job(store: RunStore):
    """Eight threads with their own connections race over forty jobs. Every job
    must be claimed exactly once -- no duplicates, none missed."""
    queue = JobQueue(store)
    for i in range(40):
        queue.enqueue(_run(f"Company {i}"), dedupe_key=f"ticker:T{i}@cfg")

    claimed: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def drain(worker: str) -> None:
        barrier.wait()
        while (lease := queue.claim(worker, 300)) is not None:
            with lock:
                claimed.append(lease.job_id)

    threads = [threading.Thread(target=drain, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == 40
    assert len(set(claimed)) == 40, "a job was handed to two workers"


def test_higher_priority_jobs_are_claimed_first(queue: JobQueue, clock: FakeClock):
    queue.enqueue(_run("Low"), dedupe_key="low", priority=200)
    clock.advance(1)
    urgent = queue.enqueue(_run("High"), dedupe_key="high", priority=10)
    lease = queue.claim("w1", 60)
    assert lease is not None and lease.job_id == urgent.job.id


# --------------------------------------------------------------------------- #
# Leases and fencing
# --------------------------------------------------------------------------- #


def test_an_expired_lease_is_taken_over_with_a_new_token(queue: JobQueue, clock: FakeClock):
    _enqueue(queue)
    first = queue.claim("w1", 60)
    clock.advance(61)
    second = queue.claim("w2", 60)

    assert first is not None and second is not None
    assert second.job_id == first.job_id
    assert second.token == first.token + 1


def test_a_stale_worker_cannot_complete_or_renew(queue: JobQueue, clock: FakeClock):
    _enqueue(queue)
    stale = queue.claim("w1", 60)
    clock.advance(61)
    current = queue.claim("w2", 60)
    assert stale is not None and current is not None

    assert not queue.heartbeat(stale, 60), "a stale worker renewed someone else's lease"
    assert not queue.complete(stale), "a stale worker completed someone else's job"
    assert queue.fail(stale, "boom", retryable=True) is None
    assert queue.stop_reason(stale) == "lease lost"

    job = queue.get(current.job_id)
    assert job is not None and job.status is JobStatus.RUNNING and job.worker_id == "w2"
    assert queue.complete(current)


def test_a_stale_worker_cannot_overwrite_the_checkpoint(
    queue: JobQueue, store: RunStore, clock: FakeClock
):
    """The split-brain case: the first worker comes back after its lease lapsed and
    tries to write its (now outdated) view of the run over the second worker's."""
    result = _enqueue(queue)
    stale = queue.claim("w1", 60)
    clock.advance(61)
    current = queue.claim("w2", 60)
    assert stale is not None and current is not None

    newer = store.load(result.job.run_id)
    assert newer is not None
    newer.open_questions = ["written by the current owner"]
    FencedRunStore(store, current).save(newer)

    outdated = store.load(result.job.run_id)
    assert outdated is not None
    outdated.open_questions = ["written by a worker that lost its lease"]
    with pytest.raises(LostLease):
        FencedRunStore(store, stale).save(outdated)

    persisted = store.load(result.job.run_id)
    assert persisted is not None
    assert persisted.open_questions == ["written by the current owner"]


def test_heartbeat_extends_the_lease(queue: JobQueue, clock: FakeClock):
    _enqueue(queue)
    lease = queue.claim("w1", 60)
    assert lease is not None
    clock.advance(50)
    assert queue.heartbeat(lease, 60)
    clock.advance(50)  # 100s after the claim, but only 50s after the renewal
    assert queue.claim("w2", 60) is None


def test_heartbeat_thread_keeps_a_lease_alive_while_the_main_thread_blocks(store: RunStore):
    """Real time, on purpose. The pipeline's model calls block the main thread for
    seconds; the heartbeat must keep renewing anyway, or a second worker would start
    a job that is still in progress. Lease 1.0s, renewed every 0.2s, while the main
    thread blocks for 2.5s -- well past the original expiry."""
    queue = JobQueue(store)
    queue.enqueue(_run(), dedupe_key="k")
    lease = queue.claim("w1", 1.0)
    assert lease is not None

    stolen = []
    with Heartbeat(queue, lease, interval_s=0.2, lease_s=1.0) as heartbeat:
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            time.sleep(0.5)  # blocking, like a synchronous model call
            if (rival := queue.claim("w2", 1.0)) is not None:
                stolen.append(rival)

    assert not stolen, "the lease lapsed while its owner was still working"
    assert heartbeat.beats >= 5
    assert not heartbeat.lost


def test_without_a_heartbeat_the_same_lease_does_lapse(store: RunStore):
    """Control for the test above: the setup must be able to fail."""
    queue = JobQueue(store)
    queue.enqueue(_run(), dedupe_key="k")
    assert queue.claim("w1", 0.3) is not None
    time.sleep(0.6)
    assert queue.claim("w2", 0.3) is not None


def test_worker_refuses_a_lease_too_short_for_its_heartbeat(queue: JobQueue, fake_fetcher):
    with pytest.raises(ValueError, match="missed beats"):
        Worker(queue, EventLog(queue.path), lease_s=30, heartbeat_s=20)


# --------------------------------------------------------------------------- #
# Retries, dead letters, cancellation
# --------------------------------------------------------------------------- #


def test_retryable_failure_requeues_with_backoff(queue: JobQueue, clock: FakeClock):
    _enqueue(queue)
    lease = queue.claim("w1", 60)
    assert lease is not None
    job = queue.fail(lease, "RateLimitError: slow down", retryable=True)

    assert job is not None and job.status is JobStatus.QUEUED
    delay = job.available_at - clock.now
    assert 30 * 0.85 <= delay <= 30 * 1.15
    assert job.last_error and "RateLimitError" in job.last_error

    assert queue.claim("w2", 60) is None, "backoff was not honoured"
    clock.advance(delay + 0.01)
    retry = queue.claim("w2", 60)
    assert retry is not None and retry.token == 2


def test_backoff_grows_and_is_capped():
    delays = [backoff_seconds("job_x", attempt, 30, 600) for attempt in range(1, 8)]
    assert delays[1] > delays[0]
    assert delays[2] > delays[1]
    assert max(delays) <= 600 * 1.15


def test_backoff_jitter_spreads_jobs_that_failed_together():
    delays = {backoff_seconds(f"job_{i}", 1, 30, 600) for i in range(20)}
    assert len(delays) > 10, "jobs failed at the same moment would retry in lockstep"


def test_exhausted_retries_dead_letter_the_job(queue: JobQueue, clock: FakeClock):
    _enqueue(queue)
    for attempt in range(1, 4):
        lease = queue.claim("w1", 60)
        assert lease is not None and lease.token == attempt
        job = queue.fail(lease, f"failure {attempt}", retryable=True)
        assert job is not None
        clock.advance(10_000)

    assert job.status is JobStatus.DEAD
    assert job.last_error == "failure 3"
    assert queue.claim("w1", 60) is None


def test_non_retryable_failure_fails_immediately(queue: JobQueue):
    _enqueue(queue)
    lease = queue.claim("w1", 60)
    assert lease is not None
    job = queue.fail(lease, "refused", retryable=False)
    assert job is not None and job.status is JobStatus.FAILED and job.attempts == 1


def test_a_worker_dying_on_its_last_attempt_is_dead_lettered(queue: JobQueue, clock: FakeClock):
    _enqueue(queue)
    for _ in range(3):
        assert queue.claim("w1", 60) is not None
        clock.advance(61)  # the worker vanished without reporting anything

    assert queue.claim("w1", 60) is None
    job = queue.list(1)[0]
    assert job.status is JobStatus.DEAD
    assert job.last_error and "worker presumed dead" in job.last_error


def test_cancelling_a_queued_job_is_immediate(queue: JobQueue):
    result = _enqueue(queue)
    cancelled = queue.request_cancel(result.job.id)
    assert cancelled is not None and cancelled.status is JobStatus.CANCELLED
    assert queue.claim("w1", 60) is None


def test_cancelling_a_running_job_asks_it_to_stop(queue: JobQueue):
    result = _enqueue(queue)
    lease = queue.claim("w1", 60)
    assert lease is not None
    job = queue.request_cancel(result.job.id)
    assert job is not None and job.status is JobStatus.RUNNING and job.cancel_requested
    assert queue.stop_reason(lease) == "cancelled"


def test_a_cancelled_job_whose_worker_died_is_settled(queue: JobQueue, clock: FakeClock):
    result = _enqueue(queue)
    assert queue.claim("w1", 60) is not None
    queue.request_cancel(result.job.id)
    clock.advance(61)
    assert queue.claim("w2", 60) is None
    job = queue.get(result.job.id)
    assert job is not None and job.status is JobStatus.CANCELLED


def test_fresh_success_respects_its_window(queue: JobQueue, clock: FakeClock):
    _enqueue(queue, key="k")
    lease = queue.claim("w1", 60)
    assert lease is not None
    queue.complete(lease)

    assert queue.fresh_success("k", 3600) is not None
    clock.advance(3601)
    assert queue.fresh_success("k", 3600) is None
    assert queue.fresh_success("other", 3600) is None


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


def test_worker_runs_a_job_to_completion(queue: JobQueue, store: RunStore, fake_fetcher):
    result = _enqueue(queue)
    outcome = _worker(queue, fake_fetcher).run_once()

    assert outcome is not None and outcome.result == "succeeded"
    job = queue.get(result.job.id)
    assert job is not None and job.status is JobStatus.SUCCEEDED
    run = store.load(result.job.run_id)
    assert run is not None and run.status is RunStatus.DONE and run.report is not None

    statuses = [e.status for e in EventLog(queue.path).since(run.id)]
    assert statuses[0] == "claimed"
    assert statuses[-1] == "done"


def test_worker_returns_none_when_idle(queue: JobQueue, fake_fetcher):
    assert _worker(queue, fake_fetcher).run_once() is None


def test_takeover_after_a_crash_resumes_instead_of_restarting(
    queue: JobQueue, store: RunStore, clock: FakeClock, fake_fetcher
):
    """The point of pairing at-least-once delivery with a checkpointed pipeline.

    Worker A gets as far as finishing Scout, then dies without a word. Once its
    lease lapses, worker B takes the job and must pick up at the Librarian: no
    document fetched twice, no model call repeated, and the finished brief must
    still contain the risk section Scout produced in the other process.
    """
    import asyncio

    from analyst.orchestrator.pipeline import Pipeline, PipelineDeps, PipelineStopped

    result = _enqueue(queue)
    lease_a = queue.claim("worker-a", 60)
    assert lease_a is not None
    run = store.load(result.job.run_id)
    assert run is not None

    def dies_after_scout() -> str | None:
        return "process killed" if run.node("scout").status is NodeStatus.DONE else None

    deps = PipelineDeps(
        llm=StubLLM(),
        fetcher=fake_fetcher,
        store=FencedRunStore(store, lease_a),
        should_stop=dies_after_scout,
    )
    with pytest.raises(PipelineStopped):
        asyncio.run(Pipeline(deps).run(run))
    fetches_by_a = len(fake_fetcher.calls or [])
    assert fetches_by_a > 0

    clock.advance(61)
    llm_b = StubLLM()
    outcome = _worker(queue, fake_fetcher, llm=llm_b, worker_id="worker-b").run_once()

    assert outcome is not None and outcome.result == "succeeded"
    finished = store.load(result.job.run_id)
    assert finished is not None and finished.report is not None
    job = queue.get(result.job.id)
    assert job is not None and job.attempts == 2

    # Resumed, not restarted: only the cached SEC index is re-read for peer lookup.
    refetched = [u for u in (fake_fetcher.calls or [])[fetches_by_a:] if "company_tickers" not in u]
    assert refetched == [], f"worker B re-fetched work worker A finished: {refetched[:3]}"
    assert not any(node.startswith("scout") for node, _ in llm_b.calls), "Scout ran twice"
    assert finished.stated_risks, "the takeover lost the risk section produced by worker A"
    assert finished.node("scout").attempts == 1


def test_worker_schedules_a_retry_when_a_node_raises(
    queue: JobQueue, store: RunStore, clock: FakeClock, fake_fetcher
):
    class ScribeOutage(StubLLM):
        def structured(self, **kwargs):  # type: ignore[override]
            if kwargs["node"] == "scribe.summary":
                raise RuntimeError("provider unavailable")
            return super().structured(**kwargs)

    result = _enqueue(queue)
    outcome = _worker(queue, fake_fetcher, llm=ScribeOutage()).run_once()

    assert outcome is not None and outcome.result == "retrying"
    job = queue.get(result.job.id)
    assert job is not None and job.status is JobStatus.QUEUED
    run = store.load(result.job.run_id)
    assert run is not None
    assert run.node("challenge").status is NodeStatus.DONE, "progress before the failure was lost"
    assert run.node("compose").status is NodeStatus.FAILED

    events = [e.status for e in EventLog(queue.path).since(run.id)]
    assert "retry scheduled" in events

    clock.advance(1000)
    retry = _worker(queue, fake_fetcher, worker_id="w2").run_once()
    assert retry is not None and retry.result == "succeeded"
    done = store.load(result.job.run_id)
    assert done is not None and done.status is RunStatus.DONE
    assert done.error is None, "a run that succeeded on retry still reports the old error"


def test_a_model_refusal_is_not_retried(queue: JobQueue, fake_fetcher):
    class Refuses(StubLLM):
        def structured(self, **kwargs):  # type: ignore[override]
            raise LLMRefusal("policy", "declined")

    result = _enqueue(queue)
    outcome = _worker(queue, fake_fetcher, llm=Refuses()).run_once()
    assert outcome is not None and outcome.result == "failed"
    job = queue.get(result.job.id)
    assert job is not None and job.status is JobStatus.FAILED


def test_worker_honours_a_cancel_between_nodes(queue: JobQueue, store: RunStore, fake_fetcher):
    result = _enqueue(queue)

    class CancelDuringScout(StubLLM):
        def structured(self, **kwargs):  # type: ignore[override]
            if kwargs["node"] == "scout.plan":
                queue.request_cancel(result.job.id)
            return super().structured(**kwargs)

    outcome = _worker(queue, fake_fetcher, llm=CancelDuringScout()).run_once()

    assert outcome is not None and outcome.result == "cancelled"
    job = queue.get(result.job.id)
    assert job is not None and job.status is JobStatus.CANCELLED
    run = store.load(result.job.run_id)
    assert run is not None and run.status is RunStatus.CANCELLED
    assert run.node("scout").status is NodeStatus.DONE, "the node in flight should finish"
    assert run.node("curate").status is NodeStatus.PENDING, "no node should start after a cancel"


def test_worker_abandons_a_job_it_lost_without_writing(
    queue: JobQueue, store: RunStore, clock: FakeClock, fake_fetcher
):
    """Worker A stalls past its lease mid-run; B takes over. When A resumes it must
    stop at its next checkpoint and leave both the job and the run to B."""
    result = _enqueue(queue)

    class StallsDuringScout(StubLLM):
        def structured(self, **kwargs):  # type: ignore[override]
            if kwargs["node"] == "scout.plan":
                clock.advance(61)
                assert queue.claim("worker-b", 60) is not None
            return super().structured(**kwargs)

    outcome = _worker(queue, fake_fetcher, llm=StallsDuringScout(), worker_id="worker-a").run_once()

    assert outcome is not None and outcome.result == "lost_lease"
    job = queue.get(result.job.id)
    assert job is not None and job.status is JobStatus.RUNNING and job.worker_id == "worker-b"
    run = store.load(result.job.run_id)
    assert run is not None
    assert run.node("scout").status is not NodeStatus.DONE, "the stale worker's checkpoint landed"


def test_run_forever_stops_when_asked(queue: JobQueue, fake_fetcher):
    for i in range(3):
        _enqueue(queue, key=f"k{i}", query=f"Apple {i}")
    stop = threading.Event()
    seen = []
    worker = _worker(queue, fake_fetcher)
    processed = worker.run_forever(stop, max_jobs=2, on_outcome=seen.append)
    assert processed == 2 and len(seen) == 2
    assert queue.depth()["queued"] == 1
