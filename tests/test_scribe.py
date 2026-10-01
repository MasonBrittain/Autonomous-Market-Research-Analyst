"""Scribe tests.

These are the Phase 0 acceptance tests: a fully-faked run must render a complete,
correctly-footnoted brief with zero API calls and zero network. They also pin the
two invariants the design depends on -- rejected claims never reach the page, and
rendering is deterministic.
"""

from __future__ import annotations

import re

from analyst.llm.stub import StubLLM
from analyst.models import Quadrant, Section
from analyst.scribe.render import Scribe, build_context


def test_renders_markdown_and_html_with_no_api_calls(sample_run):
    scribe = Scribe(StubLLM(), sample_run)
    markdown, html = scribe.render(summary="A summary.", headline="A headline")

    assert "# Acme Corporation (ACME)" in markdown
    assert "## SWOT" in markdown
    assert "## Sources" in markdown
    assert "Not investment advice" in markdown
    assert html.startswith("<!doctype html>")
    assert "Acme Corporation" in html


def test_rejected_claims_never_reach_the_report(sample_run):
    rejected = next(
        c for c in sample_run.claims if c.verdict and c.verdict.verdict.value == "reject"
    )
    markdown, html = Scribe(StubLLM(), sample_run).render(summary="s", headline="h")

    assert rejected.statement not in markdown
    assert rejected.statement not in html
    # ...and the surviving ones do.
    survivor = next(c for c in sample_run.claims if c.survived)
    assert survivor.statement in markdown


def test_every_citation_number_resolves_to_a_source(sample_run):
    """A footnote marker with no matching source entry is a broken report."""
    markdown, _ = Scribe(StubLLM(), sample_run).render(summary="s", headline="h")

    body, sources = markdown.split("## Sources", 1)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    defined = {int(n) for n in re.findall(r"^(\d+)\. ", sources, re.MULTILINE)}

    assert cited, "expected at least one citation in the body"
    assert cited <= defined, f"dangling citations: {sorted(cited - defined)}"


def test_citations_are_numbered_in_order_of_first_appearance(sample_run):
    context = build_context(sample_run, summary="s", headline="h")
    numbers = [c["number"] for c in context["citations"]]
    assert numbers == list(range(1, len(numbers) + 1))


def test_rendering_is_deterministic(sample_run):
    """Same evidence, same report -- the property that makes eval diffs readable."""
    first, _ = Scribe(StubLLM(), sample_run).render(summary="fixed", headline="fixed")
    second, _ = Scribe(StubLLM(), sample_run).render(summary="fixed", headline="fixed")

    # The generated-at timestamp is the only permitted difference.
    strip = lambda text: re.sub(r"Generated \d{4}-\d\d-\d\d \d\d:\d\d UTC", "", text)  # noqa: E731
    assert strip(first) == strip(second)


def test_contradicting_evidence_is_surfaced_not_hidden(sample_run):
    """Surfacing disagreement is the point of the contradiction scan."""
    markdown, html = Scribe(StubLLM(), sample_run).render(summary="s", headline="h")
    assert "Contradicting evidence" in markdown
    assert "Contradicting evidence" in html


def test_stated_risk_section_reports_status(sample_run):
    markdown, _ = Scribe(StubLLM(), sample_run).render(summary="s", headline="h")
    assert "Stated risks vs. observed reality" in markdown
    assert "Materializing" in markdown
    assert "Contradicted" in markdown


def test_provenance_carries_reproducibility_fields(sample_run):
    context = build_context(sample_run)
    prov = context["provenance"]
    for key in (
        "run_id",
        "model",
        "prompt_version",
        "prompt_fingerprint",
        "documents_gathered",
        "claims_published",
        "claims_rejected",
        "rejection_rate",
        "cost_usd",
        "cache_read_tokens",
    ):
        assert key in prov, f"missing provenance field: {key}"
    assert prov["claims_rejected"] == 1
    assert prov["cost_usd"] > 0


def test_summary_falls_back_when_nothing_survives(sample_run):
    for claim in sample_run.claims:
        assert claim.verdict is not None
        claim.verdict.verdict = claim.verdict.verdict.__class__("reject")

    summary, headline = Scribe(StubLLM(), sample_run).write_summary()
    assert "insufficient" in (summary + headline).lower()


def test_swot_quadrants_only_contain_their_own_claims(sample_run):
    context = build_context(sample_run)
    strengths = context["swot"]["Strengths"]
    threats = context["swot"]["Threats"]
    strength_statements = {c["statement"] for c in strengths}
    for claim in sample_run.claims_in(Section.SWOT):
        if claim.quadrant is Quadrant.THREAT:
            assert claim.statement not in strength_statements
    assert any("inquiry" in c["statement"] for c in threats)


def test_markdown_list_items_stay_on_separate_lines(sample_run):
    """Regression: an inline citation loop plus trim_blocks ran list items together,
    producing `...[3]- next item` on one line."""
    markdown, _ = Scribe(StubLLM(), sample_run).render(summary="s", headline="h")

    for line in markdown.split("\n"):
        assert line.count("- ") <= 1 or not line.startswith("- "), (
            f"two list items collapsed onto one line: {line[:120]!r}"
        )
    # A heading must never be glued to the end of a list item.
    assert "]## " not in markdown
    assert "]- " not in markdown


def test_risks_are_ordered_materializing_first(sample_run):
    from analyst.models import RiskStatus, StatedRisk

    sample_run.stated_risks.append(
        StatedRisk(risk_text="t", summary="A quiet risk.", status=RiskStatus.QUIET)
    )
    context = build_context(sample_run)
    statuses = [r["status_key"] for r in context["risks"]]
    order = {"materializing": 0, "contradicted": 1, "quiet": 2}
    assert statuses == sorted(statuses, key=lambda s: order[s])
