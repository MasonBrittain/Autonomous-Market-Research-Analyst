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

Run: python -m evals.harness score <run_id>        # structural metrics, one run
     python -m evals.harness report                 # structural scoreboard
     python -m evals.harness judge <run_id>         # judged metrics (--live for the real model)
     python -m evals.harness label <run_id>         # export a human-labelling worksheet
     python -m evals.harness calibrate <run_id>     # judge vs. human labels
     python -m evals.harness golden                 # list the golden set
     python -m evals.harness semantic-status        # what is blocking judged metrics
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from analyst.config import settings
from analyst.librarian.agent import quote_is_grounded
from analyst.llm.client import build_client
from analyst.models import ResearchRun, RiskStatus, Section
from analyst.orchestrator.store import RunStore

from .calibration import MIN_CALIBRATION_SAMPLE

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
# Semantic metrics (judged; gated on calibration)
# --------------------------------------------------------------------------- #


SEMANTIC_METRICS = {
    "citation_precision": (
        "Do a claim's cited facts establish it? Judged per claim. Note this measures "
        "residual error after the production Adversary has already filtered on nearly "
        "the same question, so it is not a measure of the raw analysis."
    ),
    "golden_coverage": (
        "Share of a target's hand-written expected_points the brief actually reached. "
        "Requires expected_points to be filled in for that target."
    ),
    "specificity": (
        "Share of claims that are not generic filler. Measured twice -- a mechanical "
        "detector (figures, dates, named parties) and the judge -- so the two can be "
        "compared where they disagree."
    ),
}


def semantic_status(store: RunStore | None = None) -> dict[str, object]:
    """What the semantic layer can currently say, and what is blocking it.

    Judged numbers are only quotable once the judge has been calibrated against
    human labels for that run, so this reports labelling progress rather than a
    bare "implemented".
    """
    from . import labels as labels_mod

    worksheets = (
        sorted(labels_mod.LABELS_DIR.glob("*.json")) if labels_mod.LABELS_DIR.exists() else []
    )
    labelled_total = 0
    per_run: dict[str, str] = {}
    for path in worksheets:
        label_set = labels_mod.load_labels(path)
        labelled_total += len(label_set.supported)
        per_run[path.stem] = (
            f"{len(label_set.supported)} supported, {len(label_set.specific)} specific"
        )

    targets_with_points = [t.query for t in GoldenTarget.load_all() if t.expected_points]
    return {
        "metrics": sorted(SEMANTIC_METRICS),
        "human_labels": per_run or "none",
        "labelled_claims_total": labelled_total,
        "calibration_sample_needed": MIN_CALIBRATION_SAMPLE,
        "calibration_ready": labelled_total >= MIN_CALIBRATION_SAMPLE,
        "golden_targets_with_expected_points": targets_with_points or "none",
        "blocking": _semantic_blockers(labelled_total, targets_with_points),
    }


def _semantic_blockers(labelled_total: int, targets_with_points: list[str]) -> list[str]:
    blockers: list[str] = []
    if labelled_total < MIN_CALIBRATION_SAMPLE:
        blockers.append(
            f"citation precision and specificity are not quotable until the judge is "
            f"calibrated: {labelled_total} of {MIN_CALIBRATION_SAMPLE} claims hand-labelled "
            f"(python -m evals.harness label <run_id>)"
        )
    if not targets_with_points:
        blockers.append(
            "golden coverage cannot be computed: no golden target has expected_points "
            "filled in (evals/golden/*.json)"
        )
    return blockers


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _int_flag(argv: list[str], flag: str) -> int | None:
    if flag not in argv:
        return None
    position = argv.index(flag) + 1
    if position >= len(argv):
        return None
    try:
        return int(argv[position])
    except ValueError:
        return None


def _judge_client(live: bool):  # noqa: ANN202 - returns an LLMClient implementation
    """The real client only when explicitly asked for; grading costs money."""
    if not live:
        from .stub import JudgeStub

        return JudgeStub()
    config = settings()
    if not config.has_api_key:
        raise SystemExit("--live needs ANTHROPIC_API_KEY in .env")
    return build_client(config.model, config.api_key, stub=False)


def _print_score(score: StructuralScore) -> None:
    print(json.dumps(asdict(score), indent=2))
    print(f"\nintegrity_ok: {score.integrity_ok}")
    if not score.integrity_ok:
        print("  -> an integrity metric below 1.0 is a bug, not a quality regression")


def main(argv: list[str]) -> int:
    # Imported here rather than at module scope: `semantic` imports from this
    # module for GoldenTarget, and a top-level import would be circular.
    from . import labels as labels_mod
    from .calibration import gate
    from .semantic import score_semantic, write_result

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
        rows = [
            score_structural(loaded)
            for summary in store.list_runs(50)
            if (loaded := store.load(summary.id))
        ]
        if not rows:
            print("no runs stored yet")
            return 0
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "structural.json"
        out.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
        header = (
            f"{'run':<22}{'entity':<20}{'status':<11}"
            f"{'cite':>6}{'quote':>7}{'cov':>6}{'rej':>6}{'cost':>9}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            cite = f"{row.citation_validity:>6.2f}" if row.applicable else f"{'  n/a':>6}"
            quote = f"{row.quote_integrity:>7.2f}" if row.applicable else f"{'   n/a':>7}"
            print(
                f"{row.run_id:<22}{(row.entity or row.query)[:18]:<20}{row.status:<11}"
                f"{cite}{quote}{row.coverage_score:>6.2f}"
                f"{row.rejection_rate:>6.2f}{row.cost_usd:>9.4f}"
            )
        scored = [row for row in rows if row.applicable]
        failing = [row.run_id for row in scored if not row.integrity_ok]
        print(f"\n{len(scored)} of {len(rows)} runs scored (the rest did not complete)")
        if failing:
            print(f"\nintegrity failures: {', '.join(failing)}")
        print(f"\nwritten to {out}")
        print("\nsemantic metrics:", json.dumps(semantic_status(store), indent=2))
        return 1 if failing else 0

    if command == "judge":
        if len(argv) < 3:
            print("usage: python -m evals.harness judge <run_id> [--limit N] [--live]")
            return 1
        run = store.load(argv[2])
        if run is None:
            print(f"no run {argv[2]}")
            return 1

        limit = _int_flag(argv, "--limit")
        live = "--live" in argv
        llm = _judge_client(live)
        if not live:
            print("[offline stand-in -- pass --live to grade with the real model]\n")

        score = score_semantic(run, llm, limit=limit)
        out = write_result(score, RESULTS_DIR)
        print("\n".join(score.summary_lines()))
        if score.citation_failures:
            print(f"\ncitation failures ({len(score.citation_failures)}):")
            for cid, why in list(score.citation_failures.items())[:10]:
                claim = next((c for c in run.claims if c.id == cid), None)
                print(f"  {cid}  {(claim.statement[:84] if claim else '')!r}")
                print(f"      {why[:110]}")
        if score.coverage_misses:
            print(f"\nmissed reference points ({len(score.coverage_misses)}):")
            for point, why in score.coverage_misses[:10]:
                print(f"  - {point[:90]}")
                print(f"      {why[:110]}")
        if score.specificity_disagreements:
            print(
                f"\ndetector/judge disagreed on {len(score.specificity_disagreements)} claims "
                f"-- worth reading, one of them is wrong"
            )
        print(f"\nwritten to {out}")
        return 0 if score.quotable else 0

    if command == "label":
        if len(argv) < 3:
            print("usage: python -m evals.harness label <run_id> [--overwrite]")
            return 1
        run = store.load(argv[2])
        if run is None:
            print(f"no run {argv[2]}")
            return 1
        try:
            path = labels_mod.export_worksheet(run, overwrite="--overwrite" in argv)
        except FileExistsError as exc:
            print(f"{exc}")
            return 1
        published = sum(1 for c in run.claims if c.survived)
        print(f"worksheet for {published} published claims -> {path}")
        print("\n".join(f"  {line}" for line in labels_mod.INSTRUCTIONS))
        return 0

    if command == "calibrate":
        if len(argv) < 3:
            print("usage: python -m evals.harness calibrate <run_id> [--live]")
            return 1
        run = store.load(argv[2])
        if run is None:
            print(f"no run {argv[2]}")
            return 1
        worksheet = labels_mod.worksheet_path(run.id)
        if not worksheet.exists():
            print(f"no labels for {run.id}. Create a worksheet first:")
            print(f"    python -m evals.harness label {run.id}")
            return 1
        print(f"labels: {labels_mod.label_progress(run.id)}\n")

        live = "--live" in argv
        if not live:
            print("[offline stand-in -- pass --live to calibrate the real judge]\n")
        score = score_semantic(run, _judge_client(live))
        if not score.calibrations:
            print("worksheet exists but holds no filled labels yet")
            return 1
        for cal in score.calibrations:
            print(f"--- {cal.question} ---")
            print(json.dumps(cal.matrix.as_dict(), indent=2))
            print(f"trust: {cal.trust.value}")
            print(f"{cal.caveat()}\n")
            if cal.disagreements:
                print(f"disagreements ({len(cal.disagreements)}):")
                for d in cal.disagreements[:10]:
                    print(f"  judge={d['judge']} human={d['human']}  {str(d['context'])[:84]!r}")
                print()
        ok, blockers = gate(score.calibrations)
        print(f"judged metrics quotable: {ok}")
        for blocker in blockers:
            print(f"  - {blocker}")
        return 0 if ok else 1

    if command == "semantic-status":
        print(json.dumps(semantic_status(store), indent=2))
        return 0

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
