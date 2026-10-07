"""LLM-as-judge for the semantic eval metrics.

Two questions the structural harness cannot answer:

* **Citation precision** -- do a claim's cited facts actually establish it?
* **Golden coverage** -- did the brief reach the findings a competent analyst would?

One honesty problem is worth stating plainly, because it is easy to hide. The
production Adversary already filters claims on very nearly the citation-precision
question. Re-asking the same model the same thing at eval time therefore measures
**residual** error after that filter, not error in the raw analysis -- and because
judge and Adversary share a model, their mistakes correlate. The judge prompt is
deliberately framed differently (an outside auditor grading a finished brief, not
a reviewer improving a teammate's work) to decorrelate them a little, but the only
real defence is `calibration.py`: until the judge has been checked against human
labels, its numbers are not results. The gate enforces that rather than trusting
anyone to remember it.

The judge runs off the same `LLMClient` protocol as the pipeline, so it works
against the offline stand-in for tests and against the real client in anger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from analyst.llm.client import LLMClient, SystemBlock
from analyst.models import Claim, ResearchRun, UsageRecord

from . import specificity

# Judge prompts are eval machinery, so they live beside the evals rather than in
# the product package that ships in the wheel.
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, version: str = "v1") -> str:
    path = PROMPTS_DIR / version / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"no judge prompt {name!r} in {PROMPTS_DIR / version}")
    return path.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Judge output schemas
# --------------------------------------------------------------------------- #


class CitationGrade(BaseModel):
    """Whether one claim's cited evidence establishes it."""

    established: bool = Field(description="Do the cited facts alone establish the claim?")
    unsupported_part: str = Field(
        description="Empty when established; otherwise the part of the claim the evidence misses"
    )
    reasoning: str


class CoveragePoint(BaseModel):
    point_index: int = Field(description="Index of the reference point being graded")
    covered: bool
    matching_claim_id: str = Field(description="Best-matching claim id, or empty string")
    reasoning: str


class CoverageGrade(BaseModel):
    points: list[CoveragePoint]


class SpecificityGrade(BaseModel):
    """The judge's half of the specificity question."""

    generic: bool = Field(
        description="True if this sentence would be equally true of a random competitor"
    )
    reasoning: str


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class CitationResult:
    run_id: str
    grades: dict[str, CitationGrade] = field(default_factory=dict)
    usage: list[UsageRecord] = field(default_factory=list)

    @property
    def verdicts(self) -> dict[str, bool]:
        """Shape `calibration.calibrate` expects."""
        return {cid: g.established for cid, g in self.grades.items()}

    @property
    def established(self) -> int:
        return sum(1 for g in self.grades.values() if g.established)

    @property
    def total(self) -> int:
        return len(self.grades)

    def failures(self) -> dict[str, CitationGrade]:
        return {cid: g for cid, g in self.grades.items() if not g.established}


@dataclass
class CoverageResult:
    query: str
    expected_points: list[str] = field(default_factory=list)
    graded: list[CoveragePoint] = field(default_factory=list)
    usage: list[UsageRecord] = field(default_factory=list)

    @property
    def covered(self) -> int:
        return sum(1 for p in self.graded if p.covered)

    @property
    def total(self) -> int:
        return len(self.graded)

    @property
    def verdicts(self) -> dict[str, bool]:
        return {f"point_{p.point_index}": p.covered for p in self.graded}

    def misses(self) -> list[tuple[str, str]]:
        """(reference point, why it was missed) for each uncovered point."""
        out: list[tuple[str, str]] = []
        for p in self.graded:
            if p.covered:
                continue
            text = (
                self.expected_points[p.point_index]
                if 0 <= p.point_index < len(self.expected_points)
                else f"<point {p.point_index}>"
            )
            out.append((text, p.reasoning))
        return out


@dataclass
class SpecificityResult:
    """Detector and judge side by side, plus where they disagree."""

    detector: dict[str, bool] = field(default_factory=dict)
    judge: dict[str, bool] = field(default_factory=dict)
    details: dict[str, dict[str, object]] = field(default_factory=dict)
    usage: list[UsageRecord] = field(default_factory=list)

    @property
    def disagreements(self) -> list[str]:
        shared = set(self.detector) & set(self.judge)
        return sorted(cid for cid in shared if self.detector[cid] != self.judge[cid])

    @property
    def detector_rate(self) -> float:
        return round(sum(self.detector.values()) / len(self.detector), 3) if self.detector else 0.0

    @property
    def judge_rate(self) -> float:
        return round(sum(self.judge.values()) / len(self.judge), 3) if self.judge else 0.0


# --------------------------------------------------------------------------- #
# The judge
# --------------------------------------------------------------------------- #


class Judge:
    def __init__(self, llm: LLMClient, *, prompt_version: str = "v1", effort: str = "high") -> None:
        self.llm = llm
        self.prompt_version = prompt_version
        self.effort = effort

    # -- citation precision ------------------------------------------------ #

    def grade_citations(self, run: ResearchRun, *, limit: int | None = None) -> CitationResult:
        """Grade each published claim against only the facts it cited."""
        result = CitationResult(run_id=run.id)
        published = [c for c in run.claims if c.survived]
        if limit is not None:
            published = published[:limit]

        system = [SystemBlock(load_prompt("judge_citation", self.prompt_version), cache=True)]
        for claim in published:
            cited = self._render_cited(run, claim)
            user = (
                f"Section: {claim.section.value}"
                f"{f' / {claim.quadrant.value}' if claim.quadrant else ''}\n\n"
                f"CLAIM:\n{claim.statement}\n\n"
                f"CITED FACTS (the only evidence you may use):\n{cited}"
            )
            try:
                out = self.llm.structured(
                    node="judge.citation",
                    system=system,
                    user=user,
                    output_model=CitationGrade,
                    effort=self.effort,
                    max_tokens=1500,
                )
            except Exception:  # noqa: BLE001 - an ungraded claim is excluded, not fatal
                continue
            result.usage.append(out.usage)
            result.grades[claim.id] = out.parsed
        return result

    def _render_cited(self, run: ResearchRun, claim: Claim) -> str:
        lines: list[str] = []
        for fid in claim.fact_ids:
            fact = run.fact_by_id(fid)
            if fact is None:
                continue
            evidence = run.evidence_by_id(fact.evidence_id)
            when = fact.happened_at.isoformat() if fact.happened_at else "undated"
            where = evidence.publisher if evidence else "unknown"
            lines.append(
                f"[{fact.id}] ({where}; {when}) {fact.text}\n"
                f'    quote: "{fact.verbatim_quote[:300]}"'
            )
        return "\n".join(lines) if lines else "(no cited facts)"

    # -- golden coverage --------------------------------------------------- #

    def grade_coverage(self, run: ResearchRun, expected_points: list[str]) -> CoverageResult:
        """Score the brief against a target's hand-written reference findings."""
        result = CoverageResult(query=run.query, expected_points=list(expected_points))
        if not expected_points:
            return result

        published = [c for c in run.claims if c.survived]
        if not published:
            # No claims at all: every point is missed, and that needs no model call.
            result.graded = [
                CoveragePoint(
                    point_index=i,
                    covered=False,
                    matching_claim_id="",
                    reasoning="the brief published no claims",
                )
                for i in range(len(expected_points))
            ]
            return result

        claim_list = "\n".join(
            f"[{c.id}] ({c.section.value}"
            f"{f'/{c.quadrant.value}' if c.quadrant else ''}) {c.statement}"
            for c in published
        )
        points = "\n".join(f"[{i}] {p}" for i, p in enumerate(expected_points))
        system = [SystemBlock(load_prompt("judge_coverage", self.prompt_version), cache=True)]
        user = (
            f"Subject: {run.entity.name if run.entity else run.query}\n\n"
            f"REFERENCE POINTS (grade every one, keep the index):\n{points}\n\n"
            f"CLAIMS THE BRIEF MADE:\n{claim_list}"
        )
        try:
            out = self.llm.structured(
                node="judge.coverage",
                system=system,
                user=user,
                output_model=CoverageGrade,
                effort=self.effort,
                max_tokens=6000,
            )
        except Exception:  # noqa: BLE001
            return result
        result.usage.append(out.usage)

        # Keep only in-range indices, and fill in any point the judge skipped as
        # uncovered -- a silently dropped point would otherwise inflate recall.
        graded: dict[int, CoveragePoint] = {}
        for point in out.parsed.points:
            if 0 <= point.point_index < len(expected_points):
                graded.setdefault(point.point_index, point)
        valid_claim_ids = {c.id for c in published}
        for index in range(len(expected_points)):
            existing = graded.get(index)
            if existing is None:
                graded[index] = CoveragePoint(
                    point_index=index,
                    covered=False,
                    matching_claim_id="",
                    reasoning="the judge returned no grade for this point",
                )
            elif existing.matching_claim_id and existing.matching_claim_id not in valid_claim_ids:
                # Cited a claim that does not exist; the match cannot be trusted.
                graded[index] = CoveragePoint(
                    point_index=index,
                    covered=False,
                    matching_claim_id="",
                    reasoning=(
                        f"judge matched a nonexistent claim id ({existing.matching_claim_id})"
                    ),
                )
        result.graded = [graded[i] for i in sorted(graded)]
        return result

    # -- specificity ------------------------------------------------------- #

    def grade_specificity(self, run: ResearchRun, *, limit: int | None = None) -> SpecificityResult:
        """Run the mechanical detector and the judge over the same claims."""
        result = SpecificityResult()
        published = [c for c in run.claims if c.survived]
        if limit is not None:
            published = published[:limit]
        subject = run.entity.name if run.entity else run.query

        system = [
            SystemBlock(
                "You are auditing a research brief for generic filler. For the single "
                "sentence given, answer one question: would this be equally true of a "
                "randomly chosen competitor in the same industry? If yes, it is generic "
                "and carries no information. A sentence naming a figure, a date, or a "
                "specific third party is not generic. Judge the sentence as written.",
                cache=True,
            )
        ]

        for claim in published:
            verdict = specificity.assess(claim.statement, subject)
            result.detector[claim.id] = verdict.specific
            result.details[claim.id] = {
                "statement": claim.statement,
                "detector": verdict.as_dict(),
            }
            try:
                out = self.llm.structured(
                    node="judge.specificity",
                    system=system,
                    user=f"Subject: {subject}\n\nSENTENCE:\n{claim.statement}",
                    output_model=SpecificityGrade,
                    effort="low",
                    max_tokens=800,
                )
            except Exception:  # noqa: BLE001
                continue
            result.usage.append(out.usage)
            result.judge[claim.id] = not out.parsed.generic
            result.details[claim.id]["judge"] = {
                "specific": not out.parsed.generic,
                "reasoning": out.parsed.reasoning,
            }
        return result


def total_cost(*results: CitationResult | CoverageResult | SpecificityResult) -> float:
    """What the grading itself cost. Eval spend is real spend."""
    return round(sum(u.cost_usd for r in results for u in r.usage), 6)
