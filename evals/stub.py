"""Offline stand-in for the judge.

Extends the pipeline's stand-in with generators for the judge schemas, so the
whole eval path -- grading, calibration arithmetic, worksheet export, reporting --
runs and is tested with no API key.

It lives in `evals/` rather than in the product package because the judge is eval
machinery, and the shipped library should not carry generators for it.

As with the pipeline stand-in, outputs are hash-seeded and derived from the actual
prompt, so coverage grades reference claim ids that really exist and the specificity
grade tracks the mechanical detector closely enough that the disagreement path is
exercised rather than hypothetical. And as with the pipeline stand-in: never assert
on *which* items it marks which way -- that is hashing, not a contract.
"""

from __future__ import annotations

import hashlib
import re

from analyst.llm.stub import StubLLM

from .judge import CitationGrade, CoverageGrade, CoveragePoint, SpecificityGrade
from .specificity import assess

_CLAIM_ID = re.compile(r"\bc_[0-9a-f]{12}\b")
_POINT_INDEX = re.compile(r"^\s*\[(\d+)\]", re.MULTILINE)
_SENTENCE_BLOCK = re.compile(r"SENTENCE:\s*\n(.+)", re.DOTALL)


def _seed(text: str) -> int:
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest(), 16)


class JudgeStub(StubLLM):
    """`StubLLM` plus the three judge schemas."""

    def _make_citation_grade(self, node: str, system: str, user: str) -> CitationGrade:
        # Fail roughly one claim in five so the failure path and a non-trivial
        # confusion matrix both get exercised.
        established = _seed(user) % 5 != 0
        return CitationGrade(
            established=established,
            unsupported_part="" if established else "the comparative part of the claim",
            reasoning=(
                "Stub grade: the cited facts carry the stated figure."
                if established
                else "Stub grade: the cited facts do not reach the comparison asserted."
            ),
        )

    def _make_coverage_grade(self, node: str, system: str, user: str) -> CoverageGrade:
        indices = sorted({int(i) for i in _POINT_INDEX.findall(user)})
        claim_ids = _CLAIM_ID.findall(user)
        points: list[CoveragePoint] = []
        for position, index in enumerate(indices):
            # Cover about two thirds, deterministically per point.
            covered = (_seed(f"{user[:200]}:{index}") + position) % 3 != 0
            points.append(
                CoveragePoint(
                    point_index=index,
                    covered=covered,
                    matching_claim_id=(
                        claim_ids[position % len(claim_ids)] if covered and claim_ids else ""
                    ),
                    reasoning=(
                        "Stub grade: a claim states the same finding."
                        if covered
                        else "Stub grade: the brief touches the topic without making the finding."
                    ),
                )
            )
        return CoverageGrade(points=points)

    def _make_specificity_grade(self, node: str, system: str, user: str) -> SpecificityGrade:
        """Track the mechanical detector, then diverge on a deterministic slice.

        Agreeing most of the time is realistic; the engineered disagreements are
        what make `SpecificityResult.disagreements` testable.
        """
        match = _SENTENCE_BLOCK.search(user)
        sentence = match.group(1).strip() if match else user
        detector_says_specific = assess(sentence).specific
        if _seed(sentence) % 7 == 0:
            detector_says_specific = not detector_says_specific
        return SpecificityGrade(
            generic=not detector_says_specific,
            reasoning="Stub grade: judged against the sentence as written.",
        )
