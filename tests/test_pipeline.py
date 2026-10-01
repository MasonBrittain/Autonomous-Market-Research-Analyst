"""End-to-end pipeline tests against fixtures.

The whole six-node pipeline runs here with no network and no API key: FakeFetcher
serves canned RSS/EDGAR/article documents and StubLLM stands in for the model.
That makes the orchestration, resume, and cost-accounting paths testable in
milliseconds, which is the point of having built both stand-ins.
"""

from __future__ import annotations

import pytest

from analyst.llm.stub import StubLLM
from analyst.models import NodeStatus, ResearchRun, RunConfig, RunStatus
from analyst.orchestrator.pipeline import NODES, AmbiguousTarget, Pipeline, PipelineDeps
from analyst.orchestrator.store import RunStore


def _deps(fake_fetcher, store=None, progress=None):
    events: list[tuple[str, str, str]] = []

    def record(node: str, status: str, detail: str) -> None:
        events.append((node, status, detail))
        if progress is not None:
            progress(node, status, detail)

    deps = PipelineDeps(llm=StubLLM(), fetcher=fake_fetcher, store=store, progress=record)
    return deps, events


async def test_full_pipeline_produces_a_cited_brief(fake_fetcher):
    run = ResearchRun(query="Apple", config=RunConfig(stub=True, max_scout_rounds=3))
    deps, events = _deps(fake_fetcher)

    finished = await Pipeline(deps).run(run)

    assert finished.status is RunStatus.DONE
    assert all(finished.node(n).status is NodeStatus.DONE for n in NODES)

    assert finished.entity is not None and finished.entity.ticker == "AAPL"
    assert finished.evidence, "scout gathered nothing"
    assert finished.facts, "librarian extracted no facts"
    assert finished.claims, "analyst produced no claims"
    assert finished.stated_risks, "no risk cross-reference from the 10-K"
    assert finished.report is not None
    assert "## SWOT" in finished.report.markdown
    assert "## Sources" in finished.report.markdown

    # Every node reported progress exactly once.
    done_nodes = [n for n, status, _ in events if status == "done"]
    assert done_nodes == list(NODES)


async def test_every_published_claim_cites_a_real_fact(fake_fetcher):
    """The invariant the whole design rests on."""
    run = ResearchRun(query="Apple", config=RunConfig(stub=True))
    deps, _ = _deps(fake_fetcher)
    finished = await Pipeline(deps).run(run)

    fact_ids = {f.id for f in finished.facts}
    evidence_ids = {e.id for e in finished.evidence}

    published = [c for c in finished.claims if c.survived]
    assert published, "nothing survived to check"
    for claim in published:
        assert claim.fact_ids, f"claim with no citation: {claim.statement!r}"
        assert set(claim.fact_ids) <= fact_ids, "claim cites a fact that does not exist"
        assert set(claim.evidence_ids) <= evidence_ids


async def test_extracted_quotes_are_genuine_substrings(fake_fetcher):
    """The Librarian's quote gate must leave nothing ungrounded behind."""
    from analyst.librarian.agent import quote_is_grounded

    run = ResearchRun(query="Apple", config=RunConfig(stub=True))
    deps, _ = _deps(fake_fetcher)
    finished = await Pipeline(deps).run(run)

    assert finished.facts
    for fact in finished.facts:
        evidence = finished.evidence_by_id(fact.evidence_id)
        assert evidence is not None
        assert quote_is_grounded(fact.verbatim_quote, evidence.clean_text), (
            f"ungrounded quote survived: {fact.verbatim_quote[:60]!r}"
        )


async def test_duplicate_syndicated_article_is_detected(fake_fetcher):
    """Fixture a5 duplicates a1 verbatim; it must not be counted twice."""
    run = ResearchRun(query="Apple", config=RunConfig(stub=True))
    deps, _ = _deps(fake_fetcher)
    finished = await Pipeline(deps).run(run)

    duplicates = [e for e in finished.evidence if e.duplicate_of is not None]
    assert duplicates, "no duplicate detected among syndicated fixtures"
    assert finished.dedup_rate > 0
    # Duplicates are marked, not deleted, so corroboration can still be counted.
    assert all(e.cluster_id for e in duplicates)


async def test_adversary_rejects_at_least_one_claim(fake_fetcher):
    """A review pass that never rejects anything is not reviewing."""
    run = ResearchRun(query="Apple", config=RunConfig(stub=True))
    deps, _ = _deps(fake_fetcher)
    finished = await Pipeline(deps).run(run)

    judged = [c for c in finished.claims if c.verdict is not None]
    assert judged, "no claim was judged"
    assert finished.rejection_rate > 0
    rejected = [c for c in finished.claims if not c.survived]
    assert rejected
    assert all(c.verdict and c.verdict.reasoning for c in rejected)


async def test_cost_is_accounted_per_node(fake_fetcher):
    run = ResearchRun(query="Apple", config=RunConfig(stub=True))
    deps, _ = _deps(fake_fetcher)
    finished = await Pipeline(deps).run(run)

    assert finished.ledger.total_calls > 0
    assert finished.ledger.total_usd > 0
    by_node = finished.ledger.by_node()
    for expected in ("scout.plan", "librarian.triage", "adversary.judge", "scribe.summary"):
        assert expected in by_node, f"no cost recorded for {expected}"


async def test_resume_skips_completed_nodes_and_refetches_nothing(fake_fetcher):
    """The crash-resume guarantee: no duplicate fetches, no double charges."""
    store = RunStore()
    run = ResearchRun(query="Apple", config=RunConfig(stub=True))
    deps, _ = _deps(fake_fetcher, store=store)
    finished = await Pipeline(deps).run(run)

    fetches_before = len(fake_fetcher.calls or [])
    cost_before = finished.ledger.total_usd
    calls_before = finished.ledger.total_calls

    reloaded = store.load(finished.id)
    assert reloaded is not None

    deps2, events2 = _deps(fake_fetcher, store=store)
    resumed = await Pipeline(deps2).run(reloaded)

    assert resumed.status is RunStatus.DONE
    assert len(fake_fetcher.calls or []) == fetches_before, "resume re-fetched documents"
    assert resumed.ledger.total_usd == cost_before, "resume re-paid for completed work"
    assert resumed.ledger.total_calls == calls_before
    assert all(status == "skipped" for _, status, _ in events2 if status != "start")


async def test_interrupted_run_resumes_from_the_failed_node(fake_fetcher):
    """Kill the run mid-analysis, reload, finish -- without redoing scout work."""
    store = RunStore()
    run = ResearchRun(query="Apple", config=RunConfig(stub=True))
    deps, _ = _deps(fake_fetcher, store=store)
    pipeline = Pipeline(deps)

    # Simulate a crash during `analyze`.
    boom = RuntimeError("simulated crash")

    async def explode(_run):
        raise boom

    original = pipeline._node_analyze
    pipeline._node_analyze = explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await pipeline.run(run)

    crashed = store.load(run.id)
    assert crashed is not None
    assert crashed.status is RunStatus.FAILED
    assert crashed.node("scout").status is NodeStatus.DONE
    assert crashed.node("curate").status is NodeStatus.DONE
    assert crashed.node("analyze").status is NodeStatus.FAILED
    assert crashed.facts, "facts were not checkpointed before the crash"

    fetches_after_crash = len(fake_fetcher.calls or [])

    deps2, _ = _deps(fake_fetcher, store=store)
    recovered = await Pipeline(deps2).run(crashed)

    assert recovered.status is RunStatus.DONE
    assert recovered.report is not None
    assert len(fake_fetcher.calls or []) == fetches_after_crash, "recovery re-fetched"
    pipeline._node_analyze = original  # type: ignore[method-assign]


async def test_ambiguous_query_stops_instead_of_guessing(fake_fetcher):
    """A confident brief about the wrong company is the worst possible output."""
    run = ResearchRun(query="Delta", config=RunConfig(stub=True))
    deps, events = _deps(fake_fetcher)

    finished = await Pipeline(deps).run(run)

    assert finished.status is RunStatus.AMBIGUOUS
    assert finished.error and "resolve" in finished.error.lower()
    assert finished.report is None
    assert ("resolve", "ambiguous", finished.error) in events
    # Nothing downstream ran.
    assert finished.node("scout").status is NodeStatus.PENDING


async def test_scout_respects_its_round_ceiling(fake_fetcher):
    run = ResearchRun(query="Apple", config=RunConfig(stub=True, max_scout_rounds=1))
    deps, _ = _deps(fake_fetcher)
    finished = await Pipeline(deps).run(run)

    assert len(finished.coverage) == 1
    assert finished.coverage[0].round_number == 1


async def test_ambiguous_target_exception_lists_candidates(fake_fetcher):
    from analyst.tools import entity as entity_tools

    index = await entity_tools.load_index(fake_fetcher)
    entity = entity_tools.resolve_from_index("Delta", index)
    exc = AmbiguousTarget(entity)
    assert "Delta" in str(exc)
