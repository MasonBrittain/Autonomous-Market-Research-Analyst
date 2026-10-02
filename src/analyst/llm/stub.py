"""A no-network implementation of the `LLMClient` protocol.

This is not a mock for unit tests alone -- it is how the pipeline is developed.
With `--stub` the orchestrator runs end to end, the Scribe renders a real report,
and the checkpoint/resume and citation-integrity paths are all exercised without
an API key or a cent of spend.

Two properties make it useful rather than decorative:

* Outputs are derived from the actual prompt, so claims cite fact IDs that really
  exist in the pack and quotes are genuine substrings of the source text. Dangling
  citations would otherwise hide real bugs in the renderer and the Adversary.
* It is deterministic (hash-seeded), so two stub runs over the same evidence
  produce identical reports -- which is exactly the invariant the Scribe tests
  assert.
"""

from __future__ import annotations

import hashlib
import re
from typing import TypeVar

from pydantic import BaseModel

from ..models import UsageRecord
from .client import LLMResult, SystemBlock
from .schemas import (
    ClaimJudgement,
    ClaimSet,
    CoverageVerdict,
    DocumentTriage,
    DraftClaim,
    ExecutiveSummary,
    ExtractedFact,
    FactExtraction,
    OpenQuestionSet,
    RelevanceVerdict,
    RevisedClaim,
    RiskAssessment,
    RiskAssessmentSet,
    ScoutPlan,
)

T = TypeVar("T", bound=BaseModel)

_FACT_ID = re.compile(r"\bf_[0-9a-f]{12}\b")
_RISK_INDEX = re.compile(r"^\s*\[(\d+)\]", re.MULTILINE)
_SENTENCE = re.compile(r"[^.!?\n]{60,300}[.!?]")


def _seed(text: str) -> int:
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest(), 16)


class StubLLM:
    """Deterministic offline stand-in for `AnthropicLLM`."""

    def __init__(self, model: str = "claude-opus-5") -> None:
        self.model = model
        self.calls: list[tuple[str, str]] = []
        self._rounds: dict[str, int] = {}

    def structured(
        self,
        *,
        node: str,
        system: list[SystemBlock] | str,
        user: str,
        output_model: type[T],
        effort: str = "high",
        max_tokens: int = 8000,
    ) -> LLMResult[T]:
        self.calls.append((node, output_model.__name__))
        self._rounds[node] = self._rounds.get(node, 0) + 1

        system_text = system if isinstance(system, str) else "\n".join(b.text for b in system)
        payload = self._dispatch(output_model, node, system_text, user)

        # Plausible token accounting so cost reporting and the ledger are exercised.
        approx_in = (len(system_text) + len(user)) // 4
        usage = UsageRecord(
            node=node,
            model=self.model,
            input_tokens=approx_in,
            output_tokens=max(len(payload.model_dump_json()) // 4, 24),
            cache_read_tokens=0,
            latency_ms=1,
            stop_reason="end_turn",
        )
        return LLMResult(parsed=payload, usage=usage, raw_text=payload.model_dump_json())

    # -- dispatch ---------------------------------------------------------- #

    def _dispatch(self, model: type[T], node: str, system: str, user: str) -> T:
        name = model.__name__
        handler = getattr(self, f"_make_{_snake(name)}", None)
        if handler is None:
            raise NotImplementedError(
                f"StubLLM has no generator for {name}; add _make_{_snake(name)}"
            )
        return handler(node, system, user)  # type: ignore[no-any-return]

    # -- generators -------------------------------------------------------- #

    def _make_scout_plan(self, node: str, system: str, user: str) -> ScoutPlan:
        subject = _subject(user)
        return ScoutPlan(
            queries=[
                f"{subject} earnings results",
                f"{subject} competition market share",
                f"{subject} lawsuit regulatory investigation",
                f"{subject} executive leadership change",
                f"{subject} product launch strategy",
            ],
            dimensions_targeted=[
                "financial_health",
                "competitive_position",
                "regulatory_legal",
                "leadership",
                "product_tech",
            ],
            rationale="Stub plan: one query per uncovered dimension.",
        )

    def _make_coverage_verdict(self, node: str, system: str, user: str) -> CoverageVerdict:
        # Saturate on the second assessment so the loop terminates promptly but
        # the multi-round path still gets exercised.
        round_number = self._rounds.get(node, 1)
        saturated = round_number >= 2
        covered = [
            "financial_health",
            "recent_events",
            "competitive_position",
            "product_tech",
        ]
        gaps = [] if saturated else ["regulatory_legal", "leadership"]
        return CoverageVerdict(
            covered=covered if saturated else covered[:2],
            gaps=gaps,
            next_queries=[] if saturated else [f"{_subject(user)} regulatory inquiry"],
            saturated=saturated,
            reasoning="Stub assessment.",
        )

    def _make_document_triage(self, node: str, system: str, user: str) -> DocumentTriage:
        quotes = _quotes(user, limit=2)
        facts = [
            ExtractedFact(
                text=f"Reported: {q[:120].strip()}",
                verbatim_quote=q.strip(),
                happened_at=None,
                polarity="neutral" if i % 2 else "positive",
                dimensions=["recent_events"],
                salience=0.6,
            )
            for i, q in enumerate(quotes)
        ]
        return DocumentTriage(
            relevant=bool(facts),
            reason="Stub triage: document mentions the subject."
            if facts
            else "No extractable body.",
            dimensions=["recent_events", "financial_health"],
            facts=facts,
        )

    def _make_relevance_verdict(self, node: str, system: str, user: str) -> RelevanceVerdict:
        return RelevanceVerdict(
            relevant=True, reason="Stub relevance.", dimensions=["recent_events"]
        )

    def _make_fact_extraction(self, node: str, system: str, user: str) -> FactExtraction:
        return FactExtraction(facts=self._make_document_triage(node, system, user).facts)

    def _make_claim_set(self, node: str, system: str, user: str) -> ClaimSet:
        fact_ids = _FACT_ID.findall(system) or _FACT_ID.findall(user)
        if not fact_ids:
            return ClaimSet(
                claims=[],
                insufficient_evidence=True,
                note="Stub: no facts in pack.",
            )
        section = _section_hint(user)
        claims = [
            DraftClaim(
                statement=f"[stub:{section}] Claim {i + 1} derived from {len(fact_ids)} available facts.",
                rationale="Stub rationale referencing the cited facts.",
                confidence=0.55 + (i * 0.1),
                fact_ids=fact_ids[i * 2 : i * 2 + 2] or fact_ids[:1],
            )
            for i in range(min(3, max(1, len(fact_ids) // 2)))
        ]
        return ClaimSet(claims=claims, insufficient_evidence=False, note="")

    def _make_risk_assessment_set(self, node: str, system: str, user: str) -> RiskAssessmentSet:
        indices = [int(i) for i in _RISK_INDEX.findall(user)][:8]
        fact_ids = _FACT_ID.findall(system) or _FACT_ID.findall(user)
        statuses = ("materializing", "quiet", "contradicted")
        return RiskAssessmentSet(
            assessments=[
                RiskAssessment(
                    risk_index=idx,
                    summary=f"Stub summary of disclosed risk {idx}.",
                    status=statuses[(_seed(str(idx)) + idx) % 3],
                    fact_ids=fact_ids[:1],
                    reasoning="Stub cross-reference.",
                )
                for idx in indices
            ]
        )

    def _make_open_question_set(self, node: str, system: str, user: str) -> OpenQuestionSet:
        return OpenQuestionSet(
            questions=[
                "Stub: no evidence found on supply chain concentration.",
                "Stub: pricing power outside the core segment is unquantified.",
            ]
        )

    def _make_claim_judgement(self, node: str, system: str, user: str) -> ClaimJudgement:
        # Reject roughly one claim in six, deterministically, so the rejection path
        # and the rejection-rate metric are both exercised.
        bucket = _seed(user) % 6
        if bucket == 0:
            return ClaimJudgement(
                supported=False,
                correct_section=True,
                specific=False,
                stale=False,
                contradicting_fact_ids=[],
                verdict="reject",
                reasoning="Stub rejection: generic and not established by the cited facts.",
            )
        if bucket == 1:
            return ClaimJudgement(
                supported=True,
                correct_section=False,
                specific=True,
                stale=False,
                contradicting_fact_ids=[],
                verdict="revise",
                reasoning="Stub revision: forward-looking claim filed as a present strength.",
            )
        return ClaimJudgement(
            supported=True,
            correct_section=True,
            specific=True,
            stale=False,
            contradicting_fact_ids=[],
            verdict="accept",
            reasoning="Stub acceptance.",
        )

    def _make_revised_claim(self, node: str, system: str, user: str) -> RevisedClaim:
        fact_ids = _FACT_ID.findall(user) or _FACT_ID.findall(system)
        return RevisedClaim(
            statement="[stub:revised] Narrowed claim tied directly to the cited evidence.",
            rationale="Stub revision rationale.",
            confidence=0.6,
            fact_ids=fact_ids[:2],
        )

    def _make_executive_summary(self, node: str, system: str, user: str) -> ExecutiveSummary:
        return ExecutiveSummary(
            summary=(
                "Stub executive summary. Generated offline from validated claims only, "
                "so this text introduces no statement absent from the claim set. Run "
                "with an API key to produce real analysis."
            ),
            headline=f"{_subject(user)}: stub brief generated without model access",
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _subject(text: str) -> str:
    match = re.search(r"(?:Subject|Company|Target|Entity)\s*:\s*(.+)", text)
    if match:
        return match.group(1).strip().split("\n")[0][:60]
    return "the subject"


def _section_hint(text: str) -> str:
    """Read the quadrant off the final instruction line.

    Scanning the whole prompt would always match "strength", because the SWOT
    instructions define all four quadrants before asking for one.
    """
    marker = "Write only this quadrant:"
    tail = text.split(marker, 1)[1] if marker in text else text[-400:]
    for key in ("strength", "weakness", "opportunit", "threat", "competitive", "catalyst"):
        if key in tail.lower():
            return key.rstrip("i") + "y" if key == "opportunit" else key
    return "claim"


def _quotes(text: str, limit: int = 2) -> list[str]:
    """Real substrings of the prompt, so the quote-integrity check has teeth."""
    found = [m.group(0).strip() for m in _SENTENCE.finditer(text)]
    return found[:limit]
