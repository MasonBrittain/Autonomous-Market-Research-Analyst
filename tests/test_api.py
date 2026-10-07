"""HTTP API tests.

The service runs in-process against fixtures: FakeFetcher behind the resolver and
the worker, StubLLM in place of the model. Workers are driven explicitly with
`run_once`, so each test controls exactly when work happens.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from analyst.config import Settings
from analyst.llm.stub import StubLLM
from analyst.models import Entity, RunConfig
from analyst.orchestrator.store import RunStore
from analyst.service.api import create_app, public_status
from analyst.service.events import EventLog
from analyst.service.jobs import JobQueue
from analyst.service.resolve import EntityResolver, config_fingerprint, dedupe_key, entity_key
from analyst.service.worker import Worker


@pytest.fixture
def service(isolated_settings: Settings, fake_fetcher):
    """Build a client plus the worker that serves it, sharing one database."""

    def build(
        config: Settings | None = None, *, llm: StubLLM | None = None, **app_kwargs
    ) -> tuple[TestClient, Worker, JobQueue]:
        store = RunStore()
        queue = JobQueue(store)
        events = EventLog(store.path)
        app = create_app(
            config=config or isolated_settings,
            store=store,
            queue=queue,
            events=events,
            resolver=EntityResolver(lambda: fake_fetcher),
            poll_s=0.01,
            **app_kwargs,
        )
        worker = Worker(
            queue,
            events,
            deps_factory=lambda run: (llm or StubLLM(), fake_fetcher),
            lease_s=60,
            heartbeat_s=10,
        )
        return TestClient(app), worker, queue

    return build


def sse(text: str) -> list[tuple[int | None, str, dict]]:
    out = []
    for block in text.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith(":") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key] = value.strip()
        if "event" in fields:
            event_id = int(fields["id"]) if "id" in fields else None
            out.append((event_id, fields["event"], json.loads(fields["data"])))
    return out


def stream(client: TestClient, run_id: str, **headers: str) -> list[tuple[int | None, str, dict]]:
    with client.stream("GET", f"/research/{run_id}/events", headers=headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return sse("".join(response.iter_text()))


# --------------------------------------------------------------------------- #
# POST /research: the four answers
# --------------------------------------------------------------------------- #


def test_new_research_is_queued(service):
    client, _, _ = service()
    response = client.post("/research", json={"query": "Apple"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["mode"] == "stub"
    assert body["entity"]["ticker"] == "AAPL"
    assert response.headers["location"] == f"/research/{body['run_id']}"
    assert body["links"]["events"].endswith("/events")


def test_same_company_by_another_name_joins_the_run_in_flight(service):
    """Resolution happens before queueing, so 'AAPL' and 'Apple' are one paid run."""
    client, _, queue = service()
    first = client.post("/research", json={"query": "Apple"}).json()
    second = client.post("/research", json={"query": "AAPL"})

    assert second.status_code == 202
    assert second.json()["coalesced"] is True
    assert second.json()["run_id"] == first["run_id"]
    assert queue.depth()["queued"] == 1


def test_ambiguous_query_is_rejected_before_anything_is_queued(service):
    client, _, queue = service()
    response = client.post("/research", json={"query": "Delta"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ambiguous"
    assert "candidates" in body
    assert sum(queue.depth().values()) == 0, "an ambiguous query must not cost a run"


def test_recent_identical_brief_is_reused(service):
    client, worker, _ = service()
    first = client.post("/research", json={"query": "Apple"}).json()
    worker.run_once()

    again = client.post("/research", json={"query": "AAPL"})
    assert again.status_code == 200
    assert again.json()["cached"] is True
    assert again.json()["run_id"] == first["run_id"]


def test_force_bypasses_the_reuse(service):
    client, worker, _ = service()
    first = client.post("/research", json={"query": "Apple"}).json()
    worker.run_once()

    forced = client.post("/research", json={"query": "Apple", "force": True})
    assert forced.status_code == 202
    assert forced.json()["run_id"] != first["run_id"]


def test_reuse_respects_the_freshness_window(service, isolated_settings):
    client, worker, _ = service(dataclasses.replace(isolated_settings, report_ttl_s=0))
    first = client.post("/research", json={"query": "Apple"}).json()
    worker.run_once()

    again = client.post("/research", json={"query": "Apple"})
    assert again.status_code == 202
    assert again.json()["run_id"] != first["run_id"]


def test_a_different_window_is_different_work(service):
    client, worker, _ = service()
    client.post("/research", json={"query": "Apple", "lookback_days": 120})
    worker.run_once()
    other = client.post("/research", json={"query": "Apple", "lookback_days": 30})
    assert other.status_code == 202, "a 120-day brief must not answer a 30-day request"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 201},
        {"query": "Apple", "lookback_days": 3},
        {"query": "Apple", "lookback_days": 400},
        {"query": "Apple", "surprise": True},
        {},
    ],
)
def test_invalid_requests_are_rejected(service, payload):
    client, _, queue = service()
    assert client.post("/research", json=payload).status_code == 422
    assert sum(queue.depth().values()) == 0


def test_resolution_outage_is_a_503_not_a_500(isolated_settings):
    class Down(EntityResolver):
        async def __call__(self, query: str) -> Entity:
            raise ConnectionError("sec.gov unreachable")

    client = TestClient(create_app(config=isolated_settings, resolver=Down(lambda: None)))  # type: ignore[arg-type]
    response = client.post("/research", json={"query": "Apple"})
    assert response.status_code == 503
    assert response.json()["error"] == "resolution_unavailable"


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def test_idempotency_key_replays_the_original_answer(service):
    client, _, queue = service()
    headers = {"Idempotency-Key": "order-7"}
    first = client.post("/research", json={"query": "Apple"}, headers=headers).json()
    replay = client.post("/research", json={"query": "Apple"}, headers=headers)

    assert replay.status_code == 202
    assert replay.json()["run_id"] == first["run_id"]
    assert replay.json()["idempotent_replay"] is True
    assert sum(queue.depth().values()) == 1


def test_reusing_an_idempotency_key_for_a_different_request_conflicts(service):
    client, _, _ = service()
    client.post("/research", json={"query": "Apple"}, headers={"Idempotency-Key": "k1"})
    clash = client.post("/research", json={"query": "Microsoft"}, headers={"Idempotency-Key": "k1"})
    assert clash.status_code == 409
    assert clash.json()["error"] == "idempotency_key_reused"


# --------------------------------------------------------------------------- #
# Reading a run
# --------------------------------------------------------------------------- #


def test_full_lifecycle(service):
    client, worker, _ = service()
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]

    queued = client.get(f"/research/{run_id}").json()
    assert queued["status"] == "queued"
    assert queued["metrics"] is None
    assert [n["status"] for n in queued["nodes"]] == ["pending"] * 6

    outcome = worker.run_once()
    assert outcome is not None and outcome.result == "succeeded"

    done = client.get(f"/research/{run_id}").json()
    assert done["status"] == "done"
    assert [n["status"] for n in done["nodes"]] == ["done"] * 6
    assert [n["label"] for n in done["nodes"]] == [
        "Resolve",
        "Scout",
        "Librarian",
        "Analyst",
        "Adversary",
        "Scribe",
    ]
    assert done["metrics"]["claims_published"] > 0
    assert done["metrics"]["cost_usd"] > 0
    assert done["headline"]
    assert done["job"]["attempts"] == 1


def test_unknown_run_is_404(service):
    client, _, _ = service()
    response = client.get("/research/run_missing")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_report_is_409_until_finished(service):
    client, _, _ = service()
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]
    response = client.get(f"/research/{run_id}/report")
    assert response.status_code == 409
    assert response.json()["error"] == "not_ready"
    assert response.json()["status"] == "queued"


def test_report_formats(service):
    client, worker, _ = service()
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]
    worker.run_once()

    html = client.get(f"/research/{run_id}/report")
    assert html.status_code == 200
    assert html.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in html.text

    md = client.get(f"/research/{run_id}/report", params={"format": "md"})
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.text.startswith("# ")

    brief = client.get(f"/research/{run_id}/report", params={"format": "json"}).json()
    assert set(brief["swot"]) == {"Strengths", "Weaknesses", "Opportunities", "Threats"}
    assert brief["citations"], "a structured brief without sources is useless to a frontend"
    assert brief["provenance"]["run_id"] == run_id
    assert brief["headline"]

    assert client.get(f"/research/{run_id}/report", params={"format": "pdf"}).status_code == 422


def test_list_runs_and_filter_by_ticker(service):
    client, worker, _ = service()
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]
    worker.run_once()

    runs = client.get("/research").json()["runs"]
    assert [r["run_id"] for r in runs] == [run_id]
    assert runs[0]["status"] == "done"
    assert client.get("/research", params={"ticker": "aapl"}).json()["runs"]
    assert client.get("/research", params={"ticker": "MSFT"}).json()["runs"] == []


# --------------------------------------------------------------------------- #
# Event stream
# --------------------------------------------------------------------------- #


def test_event_stream_replays_progress_and_ends(service):
    client, worker, _ = service()
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]
    worker.run_once()

    events = stream(client, run_id)
    kinds = [kind for _, kind, _ in events]
    assert kinds[-1] == "end"
    assert events[-1][2]["status"] == "done"

    progress = [data for _, kind, data in events if kind == "progress"]
    assert progress[0]["status"] == "claimed"
    labels = [p["label"] for p in progress if p["node"] and p["status"] == "done"]
    assert labels == ["Resolve", "Scout", "Librarian", "Analyst", "Adversary", "Scribe"]

    ids = [event_id for event_id, kind, _ in events if kind == "progress"]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_event_stream_resumes_after_last_event_id(service):
    """A reconnecting browser sends Last-Event-ID and must get only what it missed."""
    client, worker, _ = service()
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]
    worker.run_once()

    full = [e for e in stream(client, run_id) if e[1] == "progress"]
    midpoint = full[len(full) // 2][0]
    assert midpoint is not None
    resumed = [
        e for e in stream(client, run_id, **{"Last-Event-ID": str(midpoint)}) if e[1] == "progress"
    ]

    assert resumed == [e for e in full if e[0] > midpoint]


def test_event_stream_does_not_end_while_a_retry_is_pending(service):
    """A failed attempt with retries left is not a final answer; the stream must
    stay open rather than tell the client the run failed."""

    class ScribeOutage(StubLLM):
        def structured(self, **kwargs):  # type: ignore[override]
            if kwargs["node"] == "scribe.summary":
                raise RuntimeError("provider unavailable")
            return super().structured(**kwargs)

    client, worker, _ = service(llm=ScribeOutage(), stream_max_s=0.3)
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]
    outcome = worker.run_once()
    assert outcome is not None and outcome.result == "retrying"

    status = client.get(f"/research/{run_id}").json()
    assert status["status"] == "retrying"
    assert "provider unavailable" in status["error"]

    kinds = [kind for _, kind, _ in stream(client, run_id)]
    assert "end" not in kinds
    assert kinds[-1] == "timeout"


def test_event_stream_for_unknown_run_is_404(service):
    client, _, _ = service()
    assert client.get("/research/run_missing/events").status_code == 404


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_cancelling_a_queued_run(service):
    client, worker, _ = service()
    run_id = client.post("/research", json={"query": "Apple"}).json()["run_id"]

    response = client.delete(f"/research/{run_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert worker.run_once() is None, "a cancelled job was still processed"
    assert client.get(f"/research/{run_id}").json()["status"] == "cancelled"

    again = client.delete(f"/research/{run_id}")
    assert again.status_code == 409
    assert again.json()["error"] == "already_finished"


def test_cancelling_unknown_run_is_404(service):
    client, _, _ = service()
    assert client.delete("/research/run_missing").status_code == 404


def test_cancelling_frees_the_slot_for_a_new_request(service):
    client, _, _ = service()
    first = client.post("/research", json={"query": "Apple"}).json()["run_id"]
    client.delete(f"/research/{first}")
    second = client.post("/research", json={"query": "Apple"})
    assert second.status_code == 202
    assert second.json()["run_id"] != first


# --------------------------------------------------------------------------- #
# Auth, CORS, health
# --------------------------------------------------------------------------- #


def test_writes_require_the_key_when_one_is_set(service, isolated_settings):
    client, _, queue = service(dataclasses.replace(isolated_settings, service_api_key="s3cret"))

    assert client.post("/research", json={"query": "Apple"}).status_code == 401
    wrong = client.post(
        "/research", json={"query": "Apple"}, headers={"Authorization": "Bearer nope"}
    )
    assert wrong.status_code == 401
    assert sum(queue.depth().values()) == 0

    ok = client.post(
        "/research", json={"query": "Apple"}, headers={"Authorization": "Bearer s3cret"}
    )
    assert ok.status_code == 202
    run_id = ok.json()["run_id"]

    # Reads cost nothing and stay open.
    assert client.get(f"/research/{run_id}").status_code == 200
    assert client.get("/research").status_code == 200
    # Cancelling is a write.
    assert client.delete(f"/research/{run_id}").status_code == 401


def test_key_must_be_a_bearer_token(service, isolated_settings):
    client, _, _ = service(dataclasses.replace(isolated_settings, service_api_key="s3cret"))
    bare = client.post("/research", json={"query": "Apple"}, headers={"Authorization": "s3cret"})
    assert bare.status_code == 401


def test_cors_allows_only_configured_origins(service, isolated_settings):
    client, _, _ = service(
        dataclasses.replace(isolated_settings, cors_origins=("https://app.example.com",))
    )
    preflight = {"Access-Control-Request-Method": "POST"}
    allowed = client.options(
        "/research", headers={"Origin": "https://app.example.com", **preflight}
    )
    assert allowed.headers.get("access-control-allow-origin") == "https://app.example.com"
    denied = client.options("/research", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in denied.headers


def test_health_and_readiness(service):
    client, _, _ = service()
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    body = ready.json()
    assert body["ready"] is True
    assert body["mode"] == "stub"
    assert "queued" in body["queue"]


def test_not_ready_without_a_valid_sec_contact(service, isolated_settings):
    client, _, _ = service(
        dataclasses.replace(isolated_settings, sec_user_agent="analyst/0.1 (no email)")
    )
    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert any("SEC_USER_AGENT" in p for p in ready.json()["problems"])


def test_demo_page_and_openapi_are_served(service):
    client, _, _ = service()
    page = client.get("/")
    assert page.status_code == 200
    assert "Market Research Analyst" in page.text
    # Dynamic text must never be written as HTML.
    assert "innerHTML" not in page.text
    schema = client.get("/openapi.json").json()
    assert "/research" in schema["paths"]
    assert "/research/{run_id}/events" in schema["paths"]


# --------------------------------------------------------------------------- #
# What counts as the same work
# --------------------------------------------------------------------------- #


def test_dedupe_key_separates_stub_and_live_runs():
    """A placeholder brief must never be served in answer to a request for a real one."""
    entity = Entity(query="Apple", name="Apple Inc.", ticker="AAPL", cik="320193")
    assert dedupe_key(entity, RunConfig(stub=True)) != dedupe_key(entity, RunConfig(stub=False))


def test_dedupe_key_changes_when_the_prompts_change(monkeypatch):
    from analyst.service import resolve

    config = RunConfig()
    before = config_fingerprint(config)
    monkeypatch.setattr(resolve, "prompt_fingerprint", lambda version: "edited")
    assert config_fingerprint(config) != before, "a brief from old prompts would be reused"


def test_entity_key_normalises_names_and_tickers():
    assert entity_key(Entity(query="aapl", name="Apple Inc.", ticker="aapl")) == "ticker:AAPL"
    industry = Entity(query="Airlines industry", name="Airlines Industry", is_industry=True)
    assert entity_key(industry).startswith("industry:")


def test_public_status_reports_retrying_not_failed():
    from analyst.service.jobs import Job, JobStatus

    def job(status: JobStatus, attempts: int) -> Job:
        return Job(
            id="j",
            run_id="r",
            query="q",
            dedupe_key="k",
            idempotency_key=None,
            status=status,
            priority=100,
            attempts=attempts,
            max_attempts=3,
            available_at=0,
            lease_until=None,
            worker_id=None,
            last_error=None,
            cancel_requested=False,
            created_at=0,
            updated_at=0,
            finished_at=None,
        )

    assert public_status(None, job(JobStatus.QUEUED, 0)) == "queued"
    assert public_status(None, job(JobStatus.QUEUED, 1)) == "retrying"
    assert public_status(None, job(JobStatus.RUNNING, 1)) == "running"
    assert public_status(None, job(JobStatus.SUCCEEDED, 1)) == "done"
    assert public_status(None, job(JobStatus.DEAD, 3)) == "failed"
    assert public_status(None, job(JobStatus.CANCELLED, 0)) == "cancelled"
