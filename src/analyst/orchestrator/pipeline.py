"""The run loop.

Hand-rolled rather than built on a graph framework, for two reasons: the DAG is
six nodes long and linear, and owning the loop is what makes crash-resume,
per-node cost accounting, and typed error handling straightforward instead of
framework-mediated.

Resume works because every node's output lives on the `ResearchRun` and the whole
run is checkpointed after each node. Restarting skips completed nodes, so a run
killed during analysis does not re-fetch a single document or re-pay for a single
extraction. The fact pack is rebuilt deterministically from stored facts -- pinned
to `run.created_at` rather than wall-clock time, so recency ranking is identical
on resume.

The rule that makes this hold: **a node may not hand data to a later node through
`self`.** A resumed run starts in a fresh Pipeline, possibly in a different
process, so anything kept on the instance is gone. Derived state that is cheap to
rebuild (the fact pack, the SEC index) is rebuilt on demand; everything else goes
on the run. `tests/test_pipeline.py` resumes from each node boundary and asserts
the report matches a clean run, which is what caught the two places this was
broken.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..adversary.agent import Adversary
from ..analyst.agent import Analyst
from ..cache import FetchCache
from ..config import settings
from ..librarian.agent import Librarian
from ..llm.client import LLMClient, build_client
from ..models import (
    Entity,
    NodeStatus,
    Report,
    ResearchRun,
    RunConfig,
    RunStatus,
    Snapshot,
)
from ..scout.agent import Scout
from ..scribe.render import Scribe
from ..tools import entity as entity_tools
from ..tools import financials
from ..tools.fetch import Fetcher
from .store import RunStore

NODES = ("resolve", "scout", "curate", "analyze", "challenge", "compose")

ProgressFn = Callable[[str, str, str], None]
StopFn = Callable[[], str | None]


def _noop(node: str, status: str, detail: str) -> None:
    return None


def _never() -> str | None:
    return None


class PipelineStopped(RuntimeError):
    """Raised between nodes when the caller asks the run to stop.

    Raised *before* a node starts, so nothing is half-written: the run on disk is
    exactly the last checkpoint. What happens next -- marking it cancelled, or
    walking away because another worker now owns it -- is the caller's decision.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class AmbiguousTarget(RuntimeError):
    """Raised when the query cannot be resolved to one company with confidence."""

    def __init__(self, entity: Entity) -> None:
        options = ", ".join(
            f"{c.name}{f' ({c.ticker})' if c.ticker else ''}" for c in entity.candidates[:5]
        )
        super().__init__(
            f"could not resolve {entity.query!r} with confidence "
            f"(best guess {entity.name!r} at {entity.confidence:.2f})."
            + (f" Candidates: {options}" if options else " No candidates found.")
        )
        self.entity = entity


@dataclass
class PipelineDeps:
    llm: LLMClient
    fetcher: Fetcher
    store: RunStore | None = None
    progress: ProgressFn = _noop
    # Polled before each node. Returning a reason stops the run cleanly; this is
    # how the worker implements cancellation and backs off after losing a lease.
    should_stop: StopFn = _never
    stats: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    def __init__(self, deps: PipelineDeps) -> None:
        self.deps = deps
        # Caches only. Both are rebuilt on demand, so a resumed run in a fresh
        # instance gets the same values -- see the module docstring.
        self._index: entity_tools.CompanyIndex | None = None
        self._pack: str = ""

    # -- public ------------------------------------------------------------ #

    async def run(self, run: ResearchRun, *, resume: bool = True) -> ResearchRun:
        for node in NODES:
            state = run.node(node)
            if resume and state.status is NodeStatus.DONE:
                self.deps.progress(node, "skipped", "already complete")
                continue

            reason = self.deps.should_stop()
            if reason:
                raise PipelineStopped(reason)

            state.status = NodeStatus.RUNNING
            state.attempts += 1
            state.started_at = datetime.now(UTC)
            self.deps.progress(node, "start", "")
            try:
                detail = await getattr(self, f"_node_{node}")(run)
            except AmbiguousTarget as exc:
                state.status = NodeStatus.FAILED
                state.error = str(exc)
                state.finished_at = datetime.now(UTC)
                run.status = RunStatus.AMBIGUOUS
                run.error = str(exc)
                self._checkpoint(run)
                self.deps.progress(node, "ambiguous", str(exc))
                return run
            except Exception as exc:  # noqa: BLE001 - recorded, checkpointed, re-raised
                state.status = NodeStatus.FAILED
                state.error = f"{type(exc).__name__}: {exc}"
                state.finished_at = datetime.now(UTC)
                run.status = RunStatus.FAILED
                run.error = state.error
                self._checkpoint(run)
                self.deps.progress(node, "failed", state.error)
                raise

            state.status = NodeStatus.DONE
            state.finished_at = datetime.now(UTC)
            self._checkpoint(run)
            self.deps.progress(node, "done", detail or "")

        run.status = RunStatus.DONE
        # A run that succeeds on a retry should not keep reporting the error
        # from the attempt that failed.
        run.error = None
        self._checkpoint(run)
        return run

    # -- nodes ------------------------------------------------------------- #

    async def _company_index(self) -> entity_tools.CompanyIndex:
        """The SEC ticker index, loaded on first use.

        Needed by resolve and again by scout for peer lookup. Loading it lazily
        rather than only in resolve is what lets a run resumed after resolve still
        map peer CIKs to tickers. The underlying fetch is disk-cached, so the
        reload costs a file read, not a request.
        """
        if self._index is None:
            self._index = await entity_tools.load_index(self.deps.fetcher)
        return self._index

    async def _node_resolve(self, run: ResearchRun) -> str:
        run.status = RunStatus.RESOLVING
        entity = entity_tools.resolve_from_index(run.query, await self._company_index())
        run.entity = entity
        if entity.needs_clarification:
            # Researching a guess produces a confident brief about the wrong
            # company, which is worse than no brief.
            raise AmbiguousTarget(entity)
        return (
            f"{entity.name}"
            f"{f' ({entity.ticker})' if entity.ticker else ''} "
            f"conf={entity.confidence:.2f}"
        )

    async def _node_scout(self, run: ResearchRun) -> str:
        run.status = RunStatus.SCOUTING
        assert run.entity is not None
        scout = Scout(self.deps.llm, self.deps.fetcher, run.config)
        result = await scout.gather(run.entity)

        run.evidence = result.evidence
        run.coverage = result.assessments
        run.risk_factors = result.risk_chunks
        for usage in scout.usage_sink:
            run.ledger.add(usage)

        if result.snapshot_info:
            run.snapshot = financials.snapshot_from_info(result.snapshot_info)
            run.entity.industry = (
                run.entity.industry or str(run.snapshot.extras.get("industry") or "") or None
            )
            run.entity.exchange = str(run.snapshot.extras.get("exchange") or "") or None
        else:
            run.snapshot = run.snapshot or Snapshot()

        run.peers = await self._resolve_peers(result.peer_ciks)
        if run.peers:
            run.entity.peers = [p.ticker for p in run.peers]

        self.deps.stats["scout"] = {
            "tool_calls": result.tool_calls,
            "rounds": result.rounds,
            "evidence": len(result.evidence),
            "risk_factors": len(run.risk_factors),
            "coverage": run.latest_coverage.score if run.latest_coverage else 0.0,
        }
        coverage = run.latest_coverage
        return (
            f"{len(result.evidence)} docs, {result.rounds} round(s), "
            f"coverage {int((coverage.score if coverage else 0) * 100)}%, "
            f"{result.tool_calls} tool calls"
        )

    async def _node_curate(self, run: ResearchRun) -> str:
        run.status = RunStatus.CURATING
        assert run.entity is not None
        librarian = Librarian(self.deps.llm, run.config)
        evidence, facts = librarian.curate(run.entity, run.evidence)
        run.evidence = evidence
        run.facts = facts
        for usage in librarian.usage_sink:
            run.ledger.add(usage)

        kept, pack = librarian.build_pack(facts, evidence, now=run.created_at)
        run.facts = kept
        self._pack = pack
        self.deps.stats["librarian"] = librarian.stats.as_dict()
        return (
            f"{len(kept)} facts from {len(run.usable_evidence())} usable docs "
            f"({librarian.stats.rejected_duplicate} dup, "
            f"{librarian.stats.rejected_stub} stub, "
            f"{librarian.stats.rejected_irrelevant} off-topic), "
            f"quote integrity {librarian.stats.quote_integrity:.0%}"
        )

    async def _node_analyze(self, run: ResearchRun) -> str:
        run.status = RunStatus.ANALYZING
        assert run.entity is not None
        self._ensure_pack(run)

        analyst = Analyst(self.deps.llm, run.config)
        analyst.load_pack(self._pack, run.facts)

        claims = analyst.swot(run.entity)
        claims += analyst.catalysts(run.entity)
        claims += analyst.competitive(run.entity, run.snapshot, run.peers)
        run.claims = claims

        if run.risk_factors:
            run.stated_risks = analyst.stated_risks(run.entity, run.risk_factors)

        gaps = [d.value for d in (run.latest_coverage.gaps if run.latest_coverage else [])]
        run.open_questions = analyst.open_questions(run.entity, gaps, claims)

        for usage in analyst.usage_sink:
            run.ledger.add(usage)
        self.deps.stats["analyst"] = analyst.stats.as_dict()
        return (
            f"{len(claims)} claims, {len(run.stated_risks)} risks assessed, "
            f"{analyst.stats.claims_dropped_no_citation} dropped uncited, "
            f"{analyst.stats.fact_ids_hallucinated} invalid citations"
        )

    async def _node_challenge(self, run: ResearchRun) -> str:
        run.status = RunStatus.CHALLENGING
        self._ensure_pack(run)
        adversary = Adversary(self.deps.llm, run.config)
        adversary.load_pack(self._pack, run.facts)
        run.claims = adversary.challenge(run.claims)
        for usage in adversary.usage_sink:
            run.ledger.add(usage)
        self.deps.stats["adversary"] = adversary.stats.as_dict()
        return (
            f"{adversary.stats.accepted} accepted, {adversary.stats.revised} revised, "
            f"{adversary.stats.rejected} rejected "
            f"({adversary.stats.rejection_rate:.0%}), "
            f"{adversary.stats.contradictions_found} contradictions"
        )

    async def _node_compose(self, run: ResearchRun) -> str:
        run.status = RunStatus.COMPOSING
        scribe = Scribe(self.deps.llm, run)
        summary, headline = scribe.write_summary()
        for usage in scribe.usage_sink:
            run.ledger.add(usage)
        markdown, html = scribe.render(summary=summary, headline=headline)
        run.report = Report(
            markdown=markdown, html=html, executive_summary=summary, headline=headline
        )
        published = sum(1 for c in run.claims if c.survived)
        return f"{published} claims published, ${run.ledger.total_usd:.4f} total"

    # -- helpers ----------------------------------------------------------- #

    def _ensure_pack(self, run: ResearchRun) -> None:
        """Rebuild the fact pack after a resume that skipped `curate`."""
        if self._pack:
            return
        librarian = Librarian(self.deps.llm, run.config)
        _, self._pack = librarian.build_pack(run.facts, run.evidence, now=run.created_at)

    async def _resolve_peers(self, peer_ciks: list[str]) -> list:
        if not peer_ciks:
            return []
        index = await self._company_index()
        tickers = [t for t in (financials.ticker_for_cik(cik, index) for cik in peer_ciks) if t]
        if not tickers:
            return []
        return await financials.get_peer_metrics(tickers)

    def _checkpoint(self, run: ResearchRun) -> None:
        if self.deps.store is not None:
            self.deps.store.save(run)


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


async def research(
    query: str,
    *,
    config: RunConfig | None = None,
    progress: ProgressFn = _noop,
    store: RunStore | None = None,
    offline: bool = False,
    run_id: str | None = None,
) -> tuple[ResearchRun, dict[str, Any]]:
    cfg = config or RunConfig()
    cfg_settings = settings()
    cfg_settings.ensure_dirs()

    llm = build_client(cfg.model, cfg_settings.api_key, stub=cfg.stub)
    fetcher = Fetcher(FetchCache(), cfg_settings, offline=offline)
    run_store = store if store is not None else RunStore()

    run: ResearchRun | None = run_store.load(run_id) if run_id else None
    if run is None:
        run = ResearchRun(query=query, config=cfg)
        run_store.save(run)

    deps = PipelineDeps(llm=llm, fetcher=fetcher, store=run_store, progress=progress)
    pipeline = Pipeline(deps)
    finished = await pipeline.run(run)
    deps.stats["fetch"] = fetcher.stats()
    return finished, deps.stats


def research_sync(query: str, **kwargs: Any) -> tuple[ResearchRun, dict[str, Any]]:
    return asyncio.run(research(query, **kwargs))
