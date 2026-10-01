"""Eval harness.

Two kinds of metric live here and the distinction matters:

* **Structural metrics** are computed from a stored run with no model involved --
  citation validity, quote integrity, dedup rate, coverage, cost, latency. They are
  cheap, exact, and run in CI on every commit.
* **Semantic metrics** need a judge: does the cited evidence actually support the
  claim, and did the brief cover the points a competent analyst would hit. Those
  need an API key and a calibrated judge, so they are defined here but gated.

The reason for the split is honesty about what is measured. A scoreboard that mixes
an exact number with a model's opinion, without saying which is which, is worse
than no scoreboard.

Run: python -m evals.harness score <run_id>
     python -m evals.harness report
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from analyst.librarian.agent import quote_is_grounded
from analyst.models import ResearchRun, RiskStatus, Section
from analyst.orchestrator.store import RunStore

GOLDEN_DIR = Path(__file__).parent / "golden"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class GoldenTarget:
    """One hand-labelled research target.

    `expected_points` is the tedious, high-leverage part: the 5-8 findings a
    competent analyst would hit. Writing these by hand is what converts "the output
    looks good" into a number.
    """

    query: str
    expected_entity: str
    expected_ticker: str | None = None
    should_be_ambiguous: bool = False
    should_be_industry: bool = False
    expected_points: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def load_all(cls) -> list[GoldenTarget]:
        targets: list[GoldenTarget] = []
        for path in sorted(GOLDEN_DIR.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            targets.append(cls(**payload))
        return targets


@dataclass
class StructuralScore:
    """Everything measurable without a model."""

    run_id: str
    query: str
    entity: str | None
    status: str = "unknown"

    # Integrity: these should be 1.0, and a regression is a bug, not a quality dip.
    citation_validity: float = 0.0
    quote_integrity: float = 0.0
    dangling_citations: int = 0
    uncited_published_claims: int = 0

    # Behaviour: these are tuning targets, not pass/fail.
    coverage_score: float = 0.0
    scout_rounds: int = 0
    dedup_rate: float = 0.0
    rejection_rate: float = 0.0
    documents_gathered: int = 0
    documents_used: int = 0
    facts: int = 0
    claims_published: int = 0
    risks_materializing: int = 0
    open_questions: int = 0

    # Economics.
    llm_calls: int = 0
    cost_usd: float = 0.0
    cache_read_tokens: int = 0
    wall_ms: int = 0

    @property
    def applicable(self) -> bool:
        """Integrity is only meaningful for a run that finished and produced claims.

        An aborted run (ambiguous target, fetch failure) has nothing to check, and
        scoring it zero would make the scoreboard look like a regression when the
        pipeline behaved correctly by stopping.
        """
        return self.status == "done" and self.claims_published > 0

    @property
    def integrity_ok(self) -> bool:
        if not self.applicable:
            return True
        return (
            self.citation_validity == 1.0
            and self.quote_integrity == 1.0
            and self.dangling_citations == 0
            and self.uncited_published_claims == 0
        )


def score_structural(run: ResearchRun) -> StructuralScore:
    published = [c for c in run.claims if c.survived]
    fact_ids = {f.id for f in run.facts}

    uncited = sum(1 for c in published if not c.fact_ids)
    dangling = sum(1 for c in published for fid in c.fact_ids if fid not in fact_ids)

    grounded = 0
    checked = 0
    for fact in run.facts:
        evidence = run.evidence_by_id(fact.evidence_id)
        if evidence is None or not evidence.clean_text:
            continue
        checked += 1
        if quote_is_grounded(fact.verbatim_quote, evidence.clean_text):
            grounded += 1

    coverage = run.latest_coverage
    return StructuralScore(
        run_id=run.id,
        query=run.query,
        entity=run.entity.name if run.entity else None,
        status=run.status.value,
        citation_validity=round(1 - (uncited / len(published)), 3) if published else 1.0,
        quote_integrity=round(grounded / checked, 3) if checked else 1.0,
        dangling_citations=dangling,
        uncited_published_claims=uncited,
        coverage_score=coverage.score if coverage else 0.0,
        scout_rounds=len(run.coverage),
        dedup_rate=run.dedup_rate,
        rejection_rate=run.rejection_rate,
        documents_gathered=len(run.evidence),
        documents_used=len(run.usable_evidence()),
        facts=len(run.facts),
        claims_published=len(published),
        risks_materializing=sum(
            1 for r in run.stated_risks if r.status is RiskStatus.MATERIALIZING
        ),
        open_questions=len(run.open_questions),
        llm_calls=run.ledger.total_calls,
        cost_usd=run.ledger.total_usd,
        cache_read_tokens=run.ledger.cache_read_tokens,
        wall_ms=sum(state.duration_ms for state in run.nodes.values()),
    )


def check_entity_resolution(run: ResearchRun, target: GoldenTarget) -> dict[str, object]:
    """Did resolution land on the right company, or correctly refuse to guess?"""
    entity = run.entity
    if target.should_be_ambiguous:
        return {
            "expected": "ambiguous",
            "passed": entity is not None and entity.needs_clarification,
            "actual": entity.name if entity else None,
        }
    if target.should_be_industry:
        return {
            "expected": "industry",
            "passed": entity is not None and entity.is_industry,
            "actual": entity.name if entity else None,
        }
    passed = entity is not None and (
        target.expected_ticker is None or entity.ticker == target.expected_ticker
    )
    return {
        "expected": target.expected_ticker or target.expected_entity,
        "passed": passed,
        "actual": entity.ticker if entity else None,
    }


def section_shape(run: ResearchRun) -> dict[str, int]:
    """Claims per section. A brief with 20 strengths and no threats is suspicious."""
    return {
        "swot": len(run.claims_in(Section.SWOT)),
        "catalysts": len(run.claims_in(Section.CATALYSTS)),
        "competitive": len(run.claims_in(Section.COMPETITIVE)),
        "stated_risks": len(run.stated_risks),
    }


# --------------------------------------------------------------------------- #
# Semantic metrics (require a calibrated judge -- not yet implemented)
# --------------------------------------------------------------------------- #


SEMANTIC_METRICS = {
    "citation_precision": (
        "Does the cited evidence actually establish the claim? Needs an LLM judge, "
        "calibrated against ~50 hand-labelled claims so its agreement rate with a "
        "human is a published number rather than an assumption."
    ),
    "golden_coverage": (
        "What share of each target's hand-written expected_points did the brief hit? "
        "Needs semantic matching between claims and expected points."
    ),
    "specificity": (
        "Share of published claims that name a number, date or party. Partially "
        "automatable with a regex, but the judgement call is semantic."
    ),
}


def semantic_status() -> dict[str, str]:
    """Reported explicitly so the scoreboard never implies more than it measures."""
    return {name: "not implemented" for name in SEMANTIC_METRICS}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_score(score: StructuralScore) -> None:
    print(json.dumps(asdict(score), indent=2))
    print(f"\nintegrity_ok: {score.integrity_ok}")
    if not score.integrity_ok:
        print("  -> an integrity metric below 1.0 is a bug, not a quality regression")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    command = argv[1]
    store = RunStore()

    if command == "score":
        if len(argv) < 3:
            print("usage: python -m evals.harness score <run_id>")
            return 1
        run = store.load(argv[2])
        if run is None:
            print(f"no run {argv[2]}")
            return 1
        _print_score(score_structural(run))
        print("\nsection shape:", json.dumps(section_shape(run)))
        return 0

    if command == "report":
        rows = [score_structural(r) for s in store.list_runs(50) if (r := store.load(s.id))]
        if not rows:
            print("no runs stored yet")
            return 0
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "structural.json"
        out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
        header = (
            f"{'run':<22}{'entity':<20}{'status':<11}"
            f"{'cite':>6}{'quote':>7}{'cov':>6}{'rej':>6}{'cost':>9}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            cite = f"{r.citation_validity:>6.2f}" if r.applicable else f"{'  n/a':>6}"
            quote = f"{r.quote_integrity:>7.2f}" if r.applicable else f"{'   n/a':>7}"
            print(
                f"{r.run_id:<22}{(r.entity or r.query)[:18]:<20}{r.status:<11}"
                f"{cite}{quote}{r.coverage_score:>6.2f}"
                f"{r.rejection_rate:>6.2f}{r.cost_usd:>9.4f}"
            )
        scored = [r for r in rows if r.applicable]
        failing = [r.run_id for r in scored if not r.integrity_ok]
        print(f"\n{len(scored)} of {len(rows)} runs scored (the rest did not complete)")
        if failing:
            print(f"\nintegrity failures: {', '.join(failing)}")
        print(f"\nwritten to {out}")
        print("\nsemantic metrics:", json.dumps(semantic_status(), indent=2))
        return 1 if failing else 0

    if command == "golden":
        for target in GoldenTarget.load_all():
            kind = (
                "ambiguous"
                if target.should_be_ambiguous
                else "industry"
                if target.should_be_industry
                else target.expected_ticker or "company"
            )
            print(f"  {target.query:<28} {kind:<12} {len(target.expected_points)} expected points")
        return 0

    print(f"unknown command: {command}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
