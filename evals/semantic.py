"""Semantic scoring: run the judge, then gate its numbers on calibration.

`SemanticScore` is deliberately awkward to misuse. Every judged proportion is
carried as a Wilson interval rather than a bare float, and `quotable` is false
until the judge has been checked against human labels. `summary_lines()` prints
the caveat alongside the number in the same breath, so a figure cannot travel
without the thing that qualifies it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from analyst.llm.client import LLMClient
from analyst.models import ResearchRun

from . import labels as labels_mod
from .calibration import Calibration, Interval, Trust, calibrate, gate, wilson
from .harness import GoldenTarget
from .judge import CitationResult, CoverageResult, Judge, SpecificityResult, total_cost


@dataclass
class SemanticScore:
    run_id: str
    query: str

    citation_precision: Interval = field(default_factory=lambda: wilson(0, 0))
    golden_coverage: Interval = field(default_factory=lambda: wilson(0, 0))
    specificity_detector: Interval = field(default_factory=lambda: wilson(0, 0))
    specificity_judge: Interval = field(default_factory=lambda: wilson(0, 0))

    calibrations: list[Calibration] = field(default_factory=list)
    coverage_misses: list[tuple[str, str]] = field(default_factory=list)
    citation_failures: dict[str, str] = field(default_factory=dict)
    specificity_disagreements: list[str] = field(default_factory=list)
    judge_cost_usd: float = 0.0

    @property
    def trust(self) -> Trust:
        """The weakest link: one uncalibrated question taints the set."""
        if not self.calibrations:
            return Trust.UNCALIBRATED
        order = [Trust.UNCALIBRATED, Trust.INSUFFICIENT, Trust.UNRELIABLE, Trust.TRUSTED]
        return min((c.trust for c in self.calibrations), key=order.index)

    @property
    def quotable(self) -> bool:
        ok, _ = gate(self.calibrations)
        return ok and bool(self.calibrations)

    def summary_lines(self) -> list[str]:
        """Numbers and their caveats together, never apart."""
        lines = [
            f"citation precision     {self.citation_precision}",
            f"golden coverage        {self.golden_coverage}",
            f"specificity (detector) {self.specificity_detector}",
            f"specificity (judge)    {self.specificity_judge}",
            f"judge cost             ${self.judge_cost_usd:.4f}",
            "",
            f"trust: {self.trust.value}  quotable: {self.quotable}",
        ]
        if not self.quotable:
            lines.append("")
            lines.append("NOT QUOTABLE as a result. Blocking reasons:")
            _, blockers = gate(self.calibrations)
            lines.extend(f"  - {b}" for b in blockers or ["no human labels for this run"])
        else:
            lines.extend(f"  {c.caveat()}" for c in self.calibrations)
        return lines

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "trust": self.trust.value,
            "quotable": self.quotable,
            "citation_precision": {
                "point": self.citation_precision.point,
                "low": self.citation_precision.low,
                "high": self.citation_precision.high,
                "n": self.citation_precision.n,
            },
            "golden_coverage": {
                "point": self.golden_coverage.point,
                "low": self.golden_coverage.low,
                "high": self.golden_coverage.high,
                "n": self.golden_coverage.n,
            },
            "specificity_detector": self.specificity_detector.point,
            "specificity_judge": self.specificity_judge.point,
            "specificity_disagreements": self.specificity_disagreements,
            "coverage_misses": [{"point": p, "why": w} for p, w in self.coverage_misses],
            "citation_failures": self.citation_failures,
            "calibrations": [c.as_dict() for c in self.calibrations],
            "judge_cost_usd": self.judge_cost_usd,
        }


def _golden_for(query: str) -> GoldenTarget | None:
    wanted = query.strip().lower()
    for target in GoldenTarget.load_all():
        if target.query.strip().lower() == wanted:
            return target
    return None


def score_semantic(
    run: ResearchRun,
    llm: LLMClient,
    *,
    limit: int | None = None,
    include_specificity: bool = True,
) -> SemanticScore:
    """Grade one run, then calibrate the grades against any human labels."""
    judge = Judge(llm)
    score = SemanticScore(run_id=run.id, query=run.query)

    citations: CitationResult = judge.grade_citations(run, limit=limit)
    score.citation_precision = wilson(citations.established, citations.total)
    score.citation_failures = {
        cid: (grade.unsupported_part or grade.reasoning)[:200]
        for cid, grade in citations.failures().items()
    }

    target = _golden_for(run.query)
    coverage = CoverageResult(query=run.query)
    if target and target.expected_points:
        coverage = judge.grade_coverage(run, target.expected_points)
        score.golden_coverage = wilson(coverage.covered, coverage.total)
        score.coverage_misses = coverage.misses()

    spec = SpecificityResult()
    if include_specificity:
        spec = judge.grade_specificity(run, limit=limit)
        score.specificity_detector = wilson(sum(spec.detector.values()), len(spec.detector))
        score.specificity_judge = wilson(sum(spec.judge.values()), len(spec.judge))
        score.specificity_disagreements = spec.disagreements

    score.judge_cost_usd = total_cost(citations, coverage, spec)

    # Calibration: compare the judge to whatever humans have labelled.
    worksheet = labels_mod.worksheet_path(run.id)
    if worksheet.exists():
        human = labels_mod.load_labels(worksheet)
        context = labels_mod.claim_context(run)
        if human.supported:
            score.calibrations.append(
                calibrate("citation precision", citations.verdicts, human.supported, context)
            )
        if human.specific and spec.judge:
            score.calibrations.append(calibrate("specificity", spec.judge, human.specific, context))
    return score


def write_result(score: SemanticScore, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"semantic_{score.run_id}.json"
    out.write_text(json.dumps(score.as_dict(), indent=2), encoding="utf-8")
    return out
