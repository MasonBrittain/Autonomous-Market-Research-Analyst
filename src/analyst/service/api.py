"""HTTP API.

    POST   /research                  start a brief, or reuse one     202 / 200 / 422
    GET    /research                  recent runs
    GET    /research/{run_id}         status, per-agent progress, metrics
    GET    /research/{run_id}/report  ?format=html|md|json          409 until done
    GET    /research/{run_id}/events  live progress, Server-Sent Events
    DELETE /research/{run_id}         cancel
    GET    /healthz  /readyz          liveness, readiness
    GET    /                          demo page

The API never runs research itself. It resolves the query, decides whether the work
already exists, and queues it; workers do the rest. That keeps request latency in
milliseconds for work that takes minutes, and lets the API and workers scale and
restart independently.

`POST /research` answers in one of four ways, and the distinction is the point:

* **422** -- the query is ambiguous. Candidates come back in the response, before
  any money is spent.
* **200** -- an identical brief (same company, same configuration, same prompts)
  finished recently. It is returned instead of paying for it twice.
* **202, coalesced** -- the same brief is already being produced. The caller is
  attached to that run rather than starting a second one.
* **202, created** -- new work was queued.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from ..cache import FetchCache
from ..config import Settings, settings
from ..models import Entity, ResearchRun, RunConfig, RunStatus
from ..orchestrator.pipeline import NODES
from ..orchestrator.store import RunStore
from ..scribe.render import build_context
from ..tools.fetch import Fetcher
from .events import NODE_LABELS, EventLog
from .jobs import Job, JobQueue, JobStatus
from .resolve import EntityResolver, dedupe_key

STATIC_DIR = Path(__file__).parent / "static"

# Public statuses. Derived from the job when there is one -- a run whose last
# attempt failed is "retrying", not "failed", while the job still has attempts left.
TERMINAL = frozenset({"done", "failed", "cancelled", "ambiguous"})


class ApiError(Exception):
    """An error rendered as `{"error": code, "message": ..., **extra}`.

    The HTTP code is `http_status`, not `status`, because several errors carry the
    run's status as an extra field and the two must not collide.
    """

    def __init__(self, http_status: int, error: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.body = {"error": error, "message": message, **extra}


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1, max_length=200, description="Company name, ticker, or industry"
    )
    lookback_days: int = Field(default=120, ge=7, le=365, description="Evidence window")
    force: bool = Field(
        default=False, description="Produce a new brief even if a fresh identical one exists"
    )

    @field_validator("query")
    @classmethod
    def _normalise(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


def public_status(run: ResearchRun | None, job: Job | None) -> str:
    if job is not None:
        if job.status is JobStatus.QUEUED:
            return "retrying" if job.attempts > 0 else "queued"
        if job.status is JobStatus.RUNNING:
            return "running"
        if job.status is JobStatus.SUCCEEDED:
            return "done"
        if job.status is JobStatus.CANCELLED:
            return "cancelled"
        if run is not None and run.status is RunStatus.AMBIGUOUS:
            return "ambiguous"
        return "failed"
    if run is None:
        return "unknown"
    return {
        RunStatus.DONE: "done",
        RunStatus.FAILED: "failed",
        RunStatus.AMBIGUOUS: "ambiguous",
        RunStatus.CANCELLED: "cancelled",
        RunStatus.PENDING: "queued",
    }.get(run.status, "running")


def _links(run_id: str) -> dict[str, str]:
    base = f"/research/{run_id}"
    return {
        "self": base,
        "events": f"{base}/events",
        "report_html": f"{base}/report?format=html",
        "report_md": f"{base}/report?format=md",
        "report_json": f"{base}/report?format=json",
    }


def _entity_view(entity: Entity | None) -> dict[str, Any] | None:
    if entity is None:
        return None
    return {
        "name": entity.name,
        "ticker": entity.ticker,
        "cik": entity.cik,
        "industry": entity.industry,
        "is_industry": entity.is_industry,
    }


def _metrics(run: ResearchRun) -> dict[str, Any]:
    published = sum(1 for c in run.claims if c.survived)
    coverage = run.latest_coverage
    return {
        "documents": len(run.evidence),
        "documents_used": len(run.usable_evidence()),
        "facts": len(run.facts),
        "claims_published": published,
        "claims_rejected": len(run.claims) - published,
        "rejection_rate": run.rejection_rate,
        "stated_risks": len(run.stated_risks),
        "coverage": coverage.score if coverage else 0.0,
        "llm_calls": run.ledger.total_calls,
        "cache_read_tokens": run.ledger.cache_read_tokens,
        "cost_usd": run.ledger.total_usd,
    }


def _nodes(run: ResearchRun) -> list[dict[str, Any]]:
    out = []
    for name in NODES:
        state = run.nodes.get(name)
        out.append(
            {
                "node": name,
                "label": NODE_LABELS.get(name, name),
                "status": state.status.value if state else "pending",
                "attempts": state.attempts if state else 0,
                "duration_ms": state.duration_ms if state else 0,
                "error": state.error if state else None,
            }
        )
    return out


def _sse(data: dict[str, Any], *, event: str, event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


def create_app(
    *,
    config: Settings | None = None,
    store: RunStore | None = None,
    queue: JobQueue | None = None,
    events: EventLog | None = None,
    resolver: EntityResolver | None = None,
    poll_s: float = 0.5,
    keepalive_s: float = 15.0,
    stream_max_s: float = 30 * 60,
) -> FastAPI:
    """Build the app. Every collaborator is injectable, which is how the tests run
    the whole service against fixtures with no network."""
    cfg = config or settings()
    run_store = store or RunStore(cfg.runs_db)
    job_queue = queue or JobQueue(run_store, max_attempts=cfg.job_max_attempts)
    event_log = events or EventLog(run_store.path)
    resolve = resolver or EntityResolver(lambda: Fetcher(FetchCache(cfg.cache_dir), cfg))

    app = FastAPI(
        title="Autonomous Market Research Analyst",
        version="0.2.0",
        summary="Evidence-gathering research briefs that red-team their own claims.",
    )
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
            expose_headers=["Location"],
        )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.body)

    def require_key(authorization: str | None = Header(default=None)) -> None:
        """Endpoints that start paid work need the service key when one is set.

        Reads stay open: they cost nothing, and a brief is meant to be shared.
        Compared in constant time so the key cannot be recovered byte by byte
        from response timing.
        """
        expected = cfg.service_api_key
        if not expected:
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        # Check the scheme explicitly: `removeprefix("Bearer ")` would silently
        # accept a bare key, because a missing prefix is simply not removed.
        valid = scheme == "Bearer" and secrets.compare_digest(
            supplied.strip().encode(), expected.encode()
        )
        if not valid:
            raise ApiError(
                401, "unauthorized", "a valid bearer token is required to start or cancel work"
            )

    def load_run(run_id: str) -> ResearchRun:
        run = run_store.load(run_id)
        if run is None:
            raise ApiError(404, "not_found", f"no run {run_id!r}")
        return run

    def current_status(run_id: str) -> str:
        return public_status(run_store.load(run_id), job_queue.for_run(run_id))

    # -- writes ------------------------------------------------------------ #

    @app.post("/research", status_code=202, dependencies=[Depends(require_key)])
    async def create_research(
        body: ResearchRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        # 1. A replayed request returns its original answer, without re-resolving.
        if idempotency_key:
            prior = await run_in_threadpool(job_queue.by_idempotency_key, idempotency_key)
            if prior is not None:
                if prior.query.casefold() != body.query.casefold():
                    raise ApiError(
                        409,
                        "idempotency_key_reused",
                        "this Idempotency-Key was already used for a different request",
                    )
                response.headers["Location"] = f"/research/{prior.run_id}"
                return {
                    "run_id": prior.run_id,
                    "job_id": prior.id,
                    "status": current_status(prior.run_id),
                    "idempotent_replay": True,
                    "links": _links(prior.run_id),
                }

        # 2. Resolve before queueing anything.
        try:
            entity = await resolve(body.query)
        except Exception as exc:  # noqa: BLE001 - surfaced as a 503, not a 500
            raise ApiError(
                503, "resolution_unavailable", f"could not load the SEC company index: {exc}"
            ) from exc
        if entity.needs_clarification:
            raise ApiError(
                422,
                "ambiguous",
                f"{body.query!r} does not identify one company with confidence. "
                f"Retry with a ticker or the full legal name.",
                query=body.query,
                candidates=[c.model_dump() for c in entity.candidates[:8]],
            )

        run_config = RunConfig(
            model=cfg.model, lookback_days=body.lookback_days, stub=not cfg.has_api_key
        )
        key = dedupe_key(entity, run_config)

        # 3. Reuse a recent identical brief rather than paying for it twice.
        if not body.force:
            fresh = await run_in_threadpool(job_queue.fresh_success, key, cfg.report_ttl_s)
            if fresh is not None and fresh.finished_at is not None:
                response.status_code = 200
                response.headers["Location"] = f"/research/{fresh.run_id}"
                return {
                    "run_id": fresh.run_id,
                    "job_id": fresh.id,
                    "status": "done",
                    "cached": True,
                    "age_seconds": round(time.time() - fresh.finished_at, 1),
                    "entity": _entity_view(entity),
                    "links": _links(fresh.run_id),
                }

        # 4. Queue it, or attach to the identical run already in flight.
        run = ResearchRun(query=body.query, config=run_config, entity=entity)
        result = await run_in_threadpool(
            lambda: job_queue.enqueue(run, dedupe_key=key, idempotency_key=idempotency_key)
        )
        run_id = result.job.run_id
        response.headers["Location"] = f"/research/{run_id}"
        return {
            "run_id": run_id,
            "job_id": result.job.id,
            "status": current_status(run_id),
            "coalesced": result.reason == "coalesced",
            "idempotent_replay": result.reason == "idempotent",
            "mode": "stub" if run_config.stub else "live",
            "entity": _entity_view(entity),
            "links": _links(run_id),
        }

    @app.delete("/research/{run_id}", dependencies=[Depends(require_key)])
    def cancel(run_id: str) -> dict[str, Any]:
        job = job_queue.for_run(run_id)
        if job is None:
            raise ApiError(404, "not_found", f"no job for run {run_id!r}")
        if job.status.terminal:
            raise ApiError(
                409,
                "already_finished",
                f"run {run_id!r} already finished as {job.status.value}",
                status=current_status(run_id),
            )
        updated = job_queue.request_cancel(job.id)
        assert updated is not None
        return {
            "run_id": run_id,
            # A queued job is cancelled outright; a running one stops at its next
            # node boundary, so the caller is told which of the two happened.
            "status": "cancelled" if updated.status is JobStatus.CANCELLED else "cancelling",
            "links": _links(run_id),
        }

    # -- reads ------------------------------------------------------------- #

    @app.get("/research")
    def list_research(
        ticker: str | None = None, limit: int = Query(default=20, ge=1, le=100)
    ) -> dict[str, Any]:
        runs = []
        for summary in run_store.list_runs(limit, ticker=ticker):
            runs.append(
                {
                    "run_id": summary.id,
                    "query": summary.query,
                    "entity": summary.entity_name,
                    "ticker": summary.ticker,
                    "status": current_status(summary.id),
                    "updated_at": summary.updated_at,
                    "cost_usd": summary.cost_usd,
                    "links": _links(summary.id),
                }
            )
        return {"runs": runs}

    @app.get("/research/{run_id}")
    def get_research(run_id: str) -> dict[str, Any]:
        run = load_run(run_id)
        job = job_queue.for_run(run_id)
        status = public_status(run, job)
        return {
            "run_id": run.id,
            "query": run.query,
            "status": status,
            "mode": "stub" if run.config.stub else "live",
            "entity": _entity_view(run.entity),
            "job": job.as_dict() if job else None,
            "nodes": _nodes(run),
            "metrics": _metrics(run) if status == "done" else None,
            "headline": run.report.headline if run.report else None,
            "error": run.error if status in ("failed", "ambiguous", "retrying") else None,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
            "links": _links(run.id),
        }

    @app.get("/research/{run_id}/report", response_model=None)
    def get_report(
        run_id: str,
        fmt: Literal["html", "md", "json"] = Query(default="html", alias="format"),
    ) -> Response:
        run = load_run(run_id)
        if run.report is None:
            raise ApiError(
                409,
                "not_ready",
                "the brief is not finished yet; follow the events stream or poll the run",
                status=current_status(run_id),
            )
        if fmt == "md":
            return PlainTextResponse(run.report.markdown, media_type="text/markdown; charset=utf-8")
        if fmt == "json":
            # The structured brief a frontend renders from: claims with their
            # citations, the risk register, peers, provenance.
            context = build_context(
                run, summary=run.report.executive_summary, headline=run.report.headline
            )
            return JSONResponse(context)
        return HTMLResponse(run.report.html)

    @app.get("/research/{run_id}/events", response_model=None)
    async def stream_events(
        run_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        await run_in_threadpool(load_run, run_id)
        try:
            # A reconnecting browser sends the last id it saw; resume after it.
            start = int(last_event_id or 0)
        except ValueError:
            start = 0

        async def stream(cursor: int) -> AsyncIterator[str]:
            started = last_sent = time.monotonic()
            yield "retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    return
                batch = await run_in_threadpool(event_log.since, run_id, cursor)
                for event in batch:
                    cursor = event.id
                    yield _sse(event.as_dict(), event="progress", event_id=event.id)
                    last_sent = time.monotonic()
                if batch:
                    continue
                # End only on an empty poll, so the final events always go out first.
                status = await run_in_threadpool(current_status, run_id)
                if status in TERMINAL:
                    yield _sse({"status": status, "links": _links(run_id)}, event="end")
                    return
                if time.monotonic() - started > stream_max_s:
                    yield _sse({"status": status}, event="timeout")
                    return
                if time.monotonic() - last_sent > keepalive_s:
                    yield ": keepalive\n\n"
                    last_sent = time.monotonic()
                await asyncio.sleep(poll_s)

        return StreamingResponse(
            stream(start),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=None)
    def readyz() -> JSONResponse:
        problems = []
        depth: dict[str, int] = {}
        try:
            depth = job_queue.depth()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"database unavailable: {exc}")
        sec_problem = cfg.sec_user_agent_problem()
        if sec_problem:
            problems.append(sec_problem.splitlines()[0])
        return JSONResponse(
            status_code=503 if problems else 200,
            content={
                "ready": not problems,
                "problems": problems,
                "mode": "live" if cfg.has_api_key else "stub",
                "auth_required": bool(cfg.service_api_key),
                "queue": depth,
            },
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    return app
