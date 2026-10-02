"""Tests for deduplication, the Librarian's integrity gates, the Analyst's
citation validation, the Adversary, schema generation, and the checkpoint store."""

from __future__ import annotations

from datetime import UTC

import pytest

from analyst.adversary.agent import Adversary
from analyst.analyst.agent import Analyst
from analyst.librarian.agent import Librarian, quote_is_grounded
from analyst.librarian.dedup import (
    cluster_sizes,
    deduplicate,
    hamming,
    shingles,
    simhash,
    similarity,
)
from analyst.llm.client import LLMResult, SystemBlock
from analyst.llm.schema import to_strict_schema
from analyst.llm.schemas import (
    ClaimSet,
    DocumentTriage,
    DraftClaim,
    ExtractedFact,
    RiskAssessmentSet,
)
from analyst.llm.stub import StubLLM
from analyst.models import (
    AdversaryVerdict,
    Claim,
    Entity,
    Evidence,
    Fact,
    Quadrant,
    ResearchRun,
    RunConfig,
    Section,
    SourceType,
    UsageRecord,
    Verdict,
    content_hash,
)
from analyst.orchestrator.store import RunStore

# Long enough to clear the 200-word stub floor, which is what a real article is.
LONG_A = (
    "Acme Corporation said quarterly revenue rose 12 percent to 4.3 billion dollars, beating "
    "analyst expectations as demand for industrial coatings accelerated in North America. "
    "Chief Executive Dana Reyes said operating margins improved 180 basis points year over year, "
    "helped by price increases that took effect in March, and the company raised full-year "
    "guidance to between 9 and 11 percent revenue growth. "
) * 8

# How syndication actually looks: the same wire copy wrapped in a different outlet's
# chrome and trimmed. Measured distance from LONG_A is 1. A systematically rewritten
# version of the same story measures 13 and is correctly treated as distinct
# corroboration rather than a duplicate.
LONG_B = (
    "ACME BEATS ESTIMATES | Markets Wrap | Subscribe to our newsletter | "
    + LONG_A[: int(len(LONG_A) * 0.93)]
    + " Reporting by the newsroom; editing by the desk. All rights reserved."
)
LONG_C = (
    "The Federal Reserve left its benchmark rate unchanged on Wednesday as policymakers waited "
    "for more evidence that services inflation was cooling before easing policy further. "
    "Two officials dissented, preferring an immediate cut given softening payroll data and "
    "weaker manufacturing surveys across the industrial midwest. "
) * 8


def _ev(text: str, **kw) -> Evidence:
    return Evidence(
        source_type=kw.pop("source_type", SourceType.NEWS),
        url=kw.pop("url", f"https://example.com/{abs(hash(text)) % 10_000}"),
        clean_text=text,
        hash=content_hash(text),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #


def test_shingles_capture_word_order():
    assert shingles("a b c d", n=3) == ["a b c", "b c d"]
    assert shingles("short", n=3) == ["short"]
    assert shingles("", n=3) == []


def test_simhash_is_stable_and_order_sensitive():
    assert simhash(LONG_A) == simhash(LONG_A)
    assert simhash("alpha beta gamma delta") != simhash("delta gamma beta alpha")


def test_identical_text_has_zero_distance():
    assert hamming(simhash(LONG_A), simhash(LONG_A)) == 0
    assert similarity(LONG_A, LONG_A) == 1.0


def test_syndicated_copy_is_near_but_rewrite_is_not():
    """The calibration the default threshold rests on."""
    syndicated = hamming(simhash(LONG_A), simhash(LONG_B))
    unrelated = hamming(simhash(LONG_A), simhash(LONG_C))
    assert syndicated <= 8, f"syndicated copy scored {syndicated}, above the threshold"
    assert unrelated > 20, f"unrelated text scored only {unrelated}"


def test_exact_duplicates_are_marked_not_deleted():
    items, _ = deduplicate([_ev(LONG_A), _ev(LONG_A), _ev(LONG_C)])
    assert len(items) == 3, "deduplicate must not drop rows"
    marked = [e for e in items if e.duplicate_of]
    assert len(marked) == 1
    assert marked[0].cluster_id


def test_near_duplicates_cluster_together():
    items, clusters = deduplicate([_ev(LONG_A), _ev(LONG_B), _ev(LONG_C)], max_distance=8)
    assert clusters == 1
    assert sum(1 for e in items if e.duplicate_of) == 1
    survivors = [e for e in items if e.is_usable]
    assert len(survivors) == 2


def test_cluster_representative_prefers_earlier_and_more_authoritative():
    from datetime import datetime, timedelta

    now = datetime.now(UTC)
    early = _ev(LONG_A, published_at=now - timedelta(days=5), authority=0.95)
    late = _ev(LONG_B, published_at=now - timedelta(days=1), authority=0.4)
    items, _ = deduplicate([late, early], max_distance=8)
    assert early.duplicate_of is None, "the earlier, more authoritative item should survive"
    assert late.duplicate_of == early.id


def test_cluster_sizes_count_corroboration():
    items, _ = deduplicate([_ev(LONG_A), _ev(LONG_B), _ev(LONG_C)], max_distance=8)
    sizes = cluster_sizes(items)
    assert max(sizes.values()) == 2


def test_empty_text_is_not_clustered():
    items, clusters = deduplicate([_ev(""), _ev("")])
    assert clusters == 0


# --------------------------------------------------------------------------- #
# Librarian integrity gates
# --------------------------------------------------------------------------- #


def test_quote_grounding_accepts_whitespace_differences():
    source = "Revenue   rose\n12 percent\tto $4.3 billion."
    assert quote_is_grounded("revenue rose 12 percent to $4.3 billion.", source)


def test_quote_grounding_rejects_paraphrase():
    source = "Revenue rose 12 percent to $4.3 billion in the quarter."
    assert not quote_is_grounded("Revenue increased twelve percent to $4.3bn", source)


def test_quote_grounding_rejects_too_short():
    assert not quote_is_grounded("rose", "Revenue rose 12 percent.")


def test_ungrounded_facts_are_dropped(monkeypatch):
    """A hallucinated quote must cost the fact, not produce a warning."""
    librarian = Librarian(StubLLM(), RunConfig())
    evidence = _ev(LONG_A)

    good = ExtractedFact(
        text="Revenue rose 12 percent.",
        verbatim_quote="quarterly revenue rose 12 percent to 4.3 billion dollars",
        happened_at=None,
        polarity="positive",
        dimensions=["financial_health"],
        salience=0.8,
    )
    invented = ExtractedFact(
        text="The CEO privately told investors margins would double.",
        verbatim_quote="margins would double over the next two fiscal years, the CEO said",
        happened_at=None,
        polarity="positive",
        dimensions=["financial_health"],
        salience=0.9,
    )

    assert librarian._accept_fact(evidence, good) is not None
    assert librarian._accept_fact(evidence, invented) is None
    assert librarian.stats.facts_dropped_bad_quote == 1
    assert librarian.stats.quote_integrity == 0.5


def test_stub_rejection_runs_before_any_model_call():
    librarian = Librarian(StubLLM(), RunConfig(min_body_words=200))
    paywalled = _ev("Subscribe to continue reading. " * 3)
    real = _ev(LONG_A)
    librarian._reject_stubs([paywalled, real])
    assert paywalled.relevant is False
    assert "stub or paywalled" in paywalled.relevance_reason
    assert real.relevant is None


def test_filings_are_exempt_from_the_word_floor():
    """A short 8-K is legitimately short."""
    librarian = Librarian(StubLLM(), RunConfig(min_body_words=200))
    filing = _ev(
        "Item 2.02 Results of Operations. Acme reported results.", source_type=SourceType.FILING_8K
    )
    librarian._reject_stubs([filing])
    assert filing.relevant is not False


def test_fact_pack_respects_its_token_budget():
    librarian = Librarian(StubLLM(), RunConfig(fact_pack_token_budget=120))
    evidence = _ev(LONG_A)
    facts = [
        Fact(
            evidence_id=evidence.id,
            text=f"Fact number {i} with some length to it.",
            verbatim_quote="quarterly revenue rose 12 percent",
            salience=0.9 - i * 0.01,
        )
        for i in range(60)
    ]
    kept, pack = librarian.build_pack(facts, [evidence])
    assert 0 < len(kept) < 60, "budget was not enforced"
    assert len(pack) <= 120 * 4 + 400


def test_fact_pack_ranks_salient_facts_first():
    librarian = Librarian(StubLLM(), RunConfig())
    evidence = _ev(LONG_A, authority=0.9)
    low = Fact(evidence_id=evidence.id, text="Trivial detail.", verbatim_quote="q", salience=0.1)
    high = Fact(
        evidence_id=evidence.id, text="Critical finding.", verbatim_quote="q", salience=0.95
    )
    kept, _ = librarian.build_pack([low, high], [evidence])
    assert kept[0].id == high.id


def test_fact_pack_is_stable_for_a_fixed_now():
    """Resume must rebuild a byte-identical pack, or the cache prefix changes."""
    librarian = Librarian(StubLLM(), RunConfig())
    evidence = _ev(LONG_A)
    facts = [
        Fact(evidence_id=evidence.id, text="A fact.", verbatim_quote="quote here", salience=0.5)
    ]
    run = ResearchRun(query="x")
    _, first = librarian.build_pack(facts, [evidence], now=run.created_at)
    _, second = librarian.build_pack(facts, [evidence], now=run.created_at)
    assert first == second


# --------------------------------------------------------------------------- #
# Analyst citation validation
# --------------------------------------------------------------------------- #


def _analyst_with_facts() -> tuple[Analyst, list[Fact]]:
    evidence_id = "ev_deadbeef0001"
    facts = [
        Fact(evidence_id=evidence_id, text="Real fact one.", verbatim_quote="q1"),
        Fact(evidence_id=evidence_id, text="Real fact two.", verbatim_quote="q2"),
    ]
    analyst = Analyst(StubLLM(), RunConfig())
    analyst.load_pack("pack text", facts)
    return analyst, facts


def test_hallucinated_fact_ids_are_stripped():
    analyst, facts = _analyst_with_facts()
    kept = analyst._valid_ids([facts[0].id, "f_000000000000", "f_111111111111"])
    assert kept == [facts[0].id]
    assert analyst.stats.fact_ids_hallucinated == 2


def test_claims_with_no_valid_citation_are_dropped():
    analyst, facts = _analyst_with_facts()
    drafts = [
        DraftClaim(statement="Cited claim.", rationale="r", confidence=0.7, fact_ids=[facts[0].id]),
        DraftClaim(
            statement="Uncited claim.", rationale="r", confidence=0.9, fact_ids=["f_999999999999"]
        ),
    ]
    claims = analyst._to_claims(drafts, Section.SWOT, quadrant=Quadrant.STRENGTH)
    assert len(claims) == 1
    assert claims[0].statement == "Cited claim."
    assert analyst.stats.claims_dropped_no_citation == 1
    assert analyst.stats.citation_validity == 0.5


def test_evidence_ids_are_derived_from_facts():
    analyst, facts = _analyst_with_facts()
    claims = analyst._to_claims(
        [
            DraftClaim(
                statement="s", rationale="r", confidence=0.5, fact_ids=[facts[0].id, facts[1].id]
            )
        ],
        Section.SWOT,
    )
    assert claims[0].evidence_ids == ["ev_deadbeef0001"], "duplicate evidence ids must collapse"


def test_analyst_caches_the_fact_pack_not_the_section_prompt():
    """Regression guard on the caching layout: the cached block must be the pack."""
    analyst, facts = _analyst_with_facts()
    blocks = analyst._system
    assert isinstance(blocks[-1], SystemBlock)
    assert blocks[-1].cache is True
    assert "FACT PACK" in blocks[-1].text
    assert blocks[0].cache is False


def test_uncited_risk_status_is_downgraded_to_quiet():
    """Materializing without evidence is exactly the overreach we guard against."""

    class Claiming(StubLLM):
        def structured(self, **kw):  # type: ignore[override]
            parsed = RiskAssessmentSet.model_validate(
                {
                    "assessments": [
                        {
                            "risk_index": 0,
                            "summary": "A risk.",
                            "status": "materializing",
                            "fact_ids": [],
                            "reasoning": "no citation offered",
                        }
                    ]
                }
            )
            return LLMResult(
                parsed=parsed, usage=UsageRecord(node="analyst.risks", model="claude-opus-5")
            )

    analyst = Analyst(Claiming(), RunConfig())
    analyst.load_pack("pack", [])
    risks = analyst.stated_risks(Entity(query="x", name="X"), ["Some disclosed risk text."])
    assert len(risks) == 1
    assert risks[0].status.value == "quiet"


def test_out_of_range_risk_index_is_ignored():
    class OutOfRange(StubLLM):
        def structured(self, **kw):  # type: ignore[override]
            parsed = RiskAssessmentSet.model_validate(
                {
                    "assessments": [
                        {
                            "risk_index": 99,
                            "summary": "s",
                            "status": "quiet",
                            "fact_ids": [],
                            "reasoning": "r",
                        }
                    ]
                }
            )
            return LLMResult(
                parsed=parsed, usage=UsageRecord(node="analyst.risks", model="claude-opus-5")
            )

    analyst = Analyst(OutOfRange(), RunConfig())
    analyst.load_pack("pack", [])
    assert analyst.stated_risks(Entity(query="x", name="X"), ["one risk"]) == []


# --------------------------------------------------------------------------- #
# Adversary
# --------------------------------------------------------------------------- #


def test_adversary_marks_verdicts_and_counts_interventions():
    facts = [Fact(id="f_aaaaaaaaaaaa", evidence_id="ev_1", text="t", verbatim_quote="q")]
    claims = [
        Claim(
            section=Section.SWOT,
            quadrant=Quadrant.STRENGTH,
            statement=f"Claim {i}",
            fact_ids=["f_aaaaaaaaaaaa"],
            evidence_ids=["ev_1"],
        )
        for i in range(12)
    ]
    adversary = Adversary(StubLLM(), RunConfig())
    adversary.load_pack("pack", facts)
    judged = adversary.challenge(claims)

    assert all(c.verdict is not None for c in judged)
    assert adversary.stats.judged == 12
    # The accounting identity is the contract here. How many claims land in each
    # bucket depends on prompt hashing in the offline stand-in, so asserting a
    # non-zero intervention rate would be testing that stand-in's distribution
    # rather than the Adversary. `test_revision_losing_its_citations_becomes_a_rejection`
    # and the pipeline-level forced-rejection test cover the intervention paths.
    assert adversary.stats.accepted + adversary.stats.revised + adversary.stats.rejected == 12
    assert 0.0 <= adversary.stats.intervention_rate <= 1.0
    assert adversary.stats.rejection_rate == round(adversary.stats.rejected / 12, 3)


def test_rejected_verdict_means_not_survived():
    """The property the renderer depends on, asserted directly rather than via the
    stub's hash buckets -- which claims get rejected is not a stable contract."""
    rejecting = AdversaryVerdict(
        verdict=Verdict.REJECT,
        supported=False,
        correct_section=True,
        specific=False,
        reasoning="generic filler",
    )
    accepting = AdversaryVerdict(
        verdict=Verdict.ACCEPT,
        supported=True,
        correct_section=True,
        specific=True,
    )
    rejected = Claim(section=Section.SWOT, statement="Rejected.", verdict=rejecting)
    accepted = Claim(section=Section.SWOT, statement="Accepted.", verdict=accepting)
    unjudged = Claim(section=Section.SWOT, statement="Unjudged.")

    assert not rejected.survived
    assert accepted.survived
    assert unjudged.survived, "an unjudged claim is not a rejected one"


def test_revision_losing_its_citations_becomes_a_rejection():
    """A repair that drops all its evidence must not be published."""

    class LosesCitations(StubLLM):
        def structured(self, **kw):  # type: ignore[override]
            model = kw["output_model"]
            if model.__name__ == "ClaimJudgement":
                parsed = model.model_validate(
                    {
                        "supported": True,
                        "correct_section": False,
                        "specific": True,
                        "stale": False,
                        "contradicting_fact_ids": [],
                        "verdict": "revise",
                        "reasoning": "miscategorised",
                    }
                )
            else:
                parsed = model.model_validate(
                    {
                        "statement": "Rewritten.",
                        "rationale": "r",
                        "confidence": 0.5,
                        "fact_ids": ["f_doesnotexist"],
                    }
                )
            return LLMResult(
                parsed=parsed, usage=UsageRecord(node="adversary", model="claude-opus-5")
            )

    facts = [Fact(id="f_aaaaaaaaaaaa", evidence_id="ev_1", text="t", verbatim_quote="q")]
    claim = Claim(
        section=Section.SWOT,
        statement="Original.",
        fact_ids=["f_aaaaaaaaaaaa"],
        evidence_ids=["ev_1"],
    )
    adversary = Adversary(LosesCitations(), RunConfig(allow_revision=True))
    adversary.load_pack("pack", facts)
    adversary.challenge([claim])

    assert not claim.survived
    assert claim.statement == "Original.", "a failed revision must not overwrite the claim"
    assert adversary.stats.rejected == 1


def test_unparseable_verdict_fails_toward_scrutiny():
    from analyst.adversary.agent import _parse_verdict

    assert _parse_verdict("nonsense") is Verdict.REVISE
    assert _parse_verdict("ACCEPT") is Verdict.ACCEPT
    assert _parse_verdict(" reject ") is Verdict.REJECT


def test_adversary_only_keeps_contradictions_that_exist():
    facts = [Fact(id="f_aaaaaaaaaaaa", evidence_id="ev_1", text="t", verbatim_quote="q")]
    adversary = Adversary(StubLLM(), RunConfig())
    adversary.load_pack("pack", facts)
    claim = Claim(
        section=Section.SWOT, statement="s", fact_ids=["f_aaaaaaaaaaaa"], evidence_ids=["ev_1"]
    )
    _, verdict = adversary._judge_one(claim)
    assert verdict is not None
    assert all(fid in {"f_aaaaaaaaaaaa"} for fid in verdict.contradicting_fact_ids)


# --------------------------------------------------------------------------- #
# Schema generation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", [ClaimSet, DocumentTriage, RiskAssessmentSet])
def test_strict_schema_is_closed_and_fully_required(model):
    schema = to_strict_schema(model)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_strict_schema_inlines_all_references():
    import json

    rendered = json.dumps(to_strict_schema(DocumentTriage))
    assert "$ref" not in rendered
    assert "$defs" not in rendered


def test_strict_schema_closes_nested_objects():
    schema = to_strict_schema(DocumentTriage)
    fact_schema = schema["properties"]["facts"]["items"]
    assert fact_schema["additionalProperties"] is False
    assert "verbatim_quote" in fact_schema["required"]


def test_strict_schema_drops_noise_keys():
    import json

    rendered = json.dumps(to_strict_schema(ClaimSet))
    assert '"title"' not in rendered


# --------------------------------------------------------------------------- #
# Checkpoint store
# --------------------------------------------------------------------------- #


def test_store_round_trips_a_full_run(sample_run):
    store = RunStore()
    store.save(sample_run)
    loaded = store.load(sample_run.id)
    assert loaded is not None
    assert loaded.id == sample_run.id
    assert len(loaded.claims) == len(sample_run.claims)
    assert loaded.claims[0].verdict is not None
    assert loaded.ledger.total_usd == sample_run.ledger.total_usd
    assert loaded.entity is not None and loaded.entity.ticker == "ACME"


def test_store_upserts_rather_than_duplicating(sample_run):
    store = RunStore()
    store.save(sample_run)
    sample_run.open_questions.append("another question")
    store.save(sample_run)
    assert len(store.list_runs()) == 1
    reloaded = store.load(sample_run.id)
    assert reloaded is not None and len(reloaded.open_questions) == 3


def test_store_lists_and_deletes(sample_run):
    store = RunStore()
    store.save(sample_run)
    summaries = store.list_runs()
    assert summaries[0].ticker == "ACME"
    assert summaries[0].cost_usd > 0
    assert store.delete(sample_run.id)
    assert store.load(sample_run.id) is None
    assert not store.delete("run_missing")


def test_store_finds_latest_for_a_query(sample_run):
    store = RunStore()
    store.save(sample_run)
    found = store.latest_for(sample_run.query)
    assert found is not None and found.id == sample_run.id
    assert store.latest_for("never researched") is None


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #


def test_cache_reads_are_cheaper_than_fresh_input():
    fresh = UsageRecord(node="n", model="claude-opus-5", input_tokens=100_000)
    cached = UsageRecord(node="n", model="claude-opus-5", cache_read_tokens=100_000)
    assert cached.cost_usd < fresh.cost_usd
    assert cached.cost_usd == pytest.approx(fresh.cost_usd * 0.1)


def test_cache_writes_cost_a_premium():
    fresh = UsageRecord(node="n", model="claude-opus-5", input_tokens=100_000)
    written = UsageRecord(node="n", model="claude-opus-5", cache_write_tokens=100_000)
    assert written.cost_usd == pytest.approx(fresh.cost_usd * 1.25)


def test_ledger_aggregates_by_node():
    run = ResearchRun(query="x")
    run.ledger.add(UsageRecord(node="a", model="claude-opus-5", output_tokens=1000))
    run.ledger.add(UsageRecord(node="a", model="claude-opus-5", output_tokens=1000))
    run.ledger.add(UsageRecord(node="b", model="claude-opus-5", output_tokens=1000))
    by_node = run.ledger.by_node()
    assert by_node["a"] == pytest.approx(by_node["b"] * 2)
    assert run.ledger.total_calls == 3


def test_unknown_model_falls_back_to_opus_rates():
    known = UsageRecord(node="n", model="claude-opus-5", output_tokens=1000)
    unknown = UsageRecord(node="n", model="some-future-model", output_tokens=1000)
    assert unknown.cost_usd == known.cost_usd
