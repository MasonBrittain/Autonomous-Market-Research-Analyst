"""Adversary: attack every claim before it reaches the page.

This is the quality gate the project is built around. An Analyst asked to produce
a SWOT will always produce one, and absent a check roughly a third of what it
produces is either uncited, miscategorised, or filler that would fit any company
in the index. The Adversary judges each claim on four independent axes and the
rejection rate is reported as a headline metric -- a pass that never rejects
anything is a pass that is not doing its job.

It also does something no single-pass design can: it scans the whole fact pack for
evidence that *contradicts* the claim. News genuinely disagrees -- the same layoff
is cost discipline in one outlet and distress in another -- and surfacing that
disagreement is more useful than silently picking a side.

Caching: the judge prompt and the fact pack are identical for every claim, so the
cached prefix is reused across all judgements. With twenty claims that is nineteen
cache reads instead of nineteen full-price sends of the pack.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..llm.client import LLMClient, SystemBlock
from ..llm.schemas import ClaimJudgement, RevisedClaim
from ..models import (
    AdversaryVerdict,
    Claim,
    Fact,
    RunConfig,
    Section,
    Verdict,
)
from ..prompts import load_prompt

SECTION_RULES: dict[Section, str] = {
    Section.SWOT: (
        "SWOT categorisation: Strengths/Weaknesses are internal and present. "
        "Opportunities/Threats are external and forward-looking."
    ),
    Section.CATALYSTS: "A catalyst must be a discrete, dated event, not a condition.",
    Section.COMPETITIVE: "A competitive claim must name the peer or metric it compares against.",
    Section.STATED_RISKS: "A risk assessment must be grounded in cited evidence.",
}


@dataclass
class AdversaryStats:
    judged: int = 0
    accepted: int = 0
    revised: int = 0
    rejected: int = 0
    failed_support: int = 0
    failed_section: int = 0
    failed_specific: int = 0
    stale: int = 0
    contradictions_found: int = 0
    revision_calls: int = 0

    @property
    def rejection_rate(self) -> float:
        return round(self.rejected / self.judged, 3) if self.judged else 0.0

    @property
    def intervention_rate(self) -> float:
        """Share of claims the pass changed or removed -- the value it added."""
        if not self.judged:
            return 0.0
        return round((self.rejected + self.revised) / self.judged, 3)

    def as_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "rejection_rate": self.rejection_rate,
            "intervention_rate": self.intervention_rate,
        }


def _parse_verdict(value: str) -> Verdict:
    try:
        return Verdict(str(value).strip().lower())
    except ValueError:
        # An unparseable verdict is treated as needing revision rather than as an
        # acceptance -- fail toward scrutiny, not toward publication.
        return Verdict.REVISE


class Adversary:
    def __init__(self, llm: LLMClient, config: RunConfig) -> None:
        self.llm = llm
        self.config = config
        self.stats = AdversaryStats()
        self.usage_sink: list = []
        self._pack = ""
        self._facts: dict[str, Fact] = {}

    def load_pack(self, pack: str, facts: list[Fact]) -> None:
        self._pack = pack
        self._facts = {f.id: f for f in facts}

    @property
    def _system(self) -> list[SystemBlock]:
        return [
            SystemBlock(load_prompt("adversary_judge", self.config.prompt_version), cache=False),
            SystemBlock(f"=== FACT PACK ===\n{self._pack}\n=== END FACT PACK ===", cache=True),
        ]

    @property
    def _effort(self) -> str:
        return self.config.effort.get("adversary", "high")

    # -- entry point ------------------------------------------------------- #

    def challenge(self, claims: list[Claim]) -> list[Claim]:
        """Judge every claim, then repair the ones worth repairing."""
        if not claims:
            return claims

        max_workers = min(8, len(claims))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            judged = list(pool.map(self._judge_one, claims))

        for claim, verdict in judged:
            claim.verdict = verdict
            self.stats.judged += 1
            if verdict is None:
                continue
            if not verdict.supported:
                self.stats.failed_support += 1
            if not verdict.correct_section:
                self.stats.failed_section += 1
            if not verdict.specific:
                self.stats.failed_specific += 1
            if verdict.stale:
                self.stats.stale += 1
            self.stats.contradictions_found += len(verdict.contradicting_fact_ids)
            if verdict.verdict is Verdict.ACCEPT:
                self.stats.accepted += 1
            elif verdict.verdict is Verdict.REJECT:
                self.stats.rejected += 1
            else:
                self.stats.revised += 1

        if self.config.allow_revision:
            self._revise_flagged(claims)
        return claims

    # -- judging ----------------------------------------------------------- #

    def _judge_one(self, claim: Claim) -> tuple[Claim, AdversaryVerdict | None]:
        cited = self._render_cited(claim)
        where = f"{claim.section.value}{f' / {claim.quadrant.value}' if claim.quadrant else ''}"
        user = (
            f"Section: {where}\n"
            f"{SECTION_RULES.get(claim.section, '')}\n\n"
            f"CLAIM UNDER REVIEW:\n{claim.statement}\n\n"
            f"STATED RATIONALE:\n{claim.rationale or '(none given)'}\n\n"
            f"CITED FACTS (the only evidence that can support this claim):\n{cited}\n\n"
            f"Judge the claim. Scan the full fact pack above for anything that "
            f"contradicts it and list those fact IDs."
        )
        try:
            out = self.llm.structured(
                node="adversary.judge",
                system=self._system,
                user=user,
                output_model=ClaimJudgement,
                effort=self._effort,
                max_tokens=2500,
            )
        except Exception:  # noqa: BLE001 - an unjudged claim stays unjudged, run continues
            return claim, None
        self.usage_sink.append(out.usage)
        judgement: ClaimJudgement = out.parsed

        contradicting = [
            fid.strip().strip("[]")
            for fid in judgement.contradicting_fact_ids
            if fid.strip().strip("[]") in self._facts
        ]
        return claim, AdversaryVerdict(
            verdict=_parse_verdict(judgement.verdict),
            supported=bool(judgement.supported),
            correct_section=bool(judgement.correct_section),
            specific=bool(judgement.specific),
            stale=bool(judgement.stale),
            contradicting_fact_ids=contradicting,
            reasoning=judgement.reasoning.strip(),
        )

    def _render_cited(self, claim: Claim) -> str:
        lines: list[str] = []
        for fid in claim.fact_ids:
            fact = self._facts.get(fid)
            if fact is None:
                continue
            lines.append(f'[{fact.id}] {fact.text}\n    quote: "{fact.verbatim_quote[:300]}"')
        return "\n".join(lines) if lines else "(no cited facts -- this alone fails support)"

    # -- revision ---------------------------------------------------------- #

    def _revise_flagged(self, claims: list[Claim]) -> None:
        """Give each flagged claim exactly one repair attempt.

        One attempt, not a loop: if a narrowed restatement still fails, the claim
        was not supportable and belongs out of the brief.
        """
        targets = [
            c for c in claims if c.verdict is not None and c.verdict.verdict is Verdict.REVISE
        ]
        if not targets:
            return

        instructions = load_prompt("adversary_revise", self.config.prompt_version)

        def revise_one(claim: Claim) -> tuple[Claim, RevisedClaim | None]:
            assert claim.verdict is not None
            user = (
                f"{instructions}\n\n"
                f"Section: {claim.section.value}"
                f"{f' / {claim.quadrant.value}' if claim.quadrant else ''}\n\n"
                f"ORIGINAL CLAIM:\n{claim.statement}\n\n"
                f"YOUR CRITIQUE:\n{claim.verdict.reasoning}\n"
                f"(supported={claim.verdict.supported}, "
                f"correct_section={claim.verdict.correct_section}, "
                f"specific={claim.verdict.specific})\n\n"
                f"CITED FACTS:\n{self._render_cited(claim)}"
            )
            try:
                out = self.llm.structured(
                    node="adversary.revise",
                    system=self._system,
                    user=user,
                    output_model=RevisedClaim,
                    effort=self._effort,
                    max_tokens=2000,
                )
            except Exception:  # noqa: BLE001
                return claim, None
            self.usage_sink.append(out.usage)
            return claim, out.parsed

        with ThreadPoolExecutor(max_workers=min(6, len(targets))) as pool:
            results = list(pool.map(revise_one, targets))

        for claim, revised in results:
            self.stats.revision_calls += 1
            if revised is None or not revised.statement.strip():
                continue
            kept = [
                fid.strip().strip("[]")
                for fid in revised.fact_ids
                if fid.strip().strip("[]") in self._facts
            ]
            if not kept:
                # The revision lost its evidence; treat it as a rejection.
                assert claim.verdict is not None
                claim.verdict.verdict = Verdict.REJECT
                claim.verdict.reasoning += " | revision produced no valid citations"
                self.stats.revised -= 1
                self.stats.rejected += 1
                continue
            claim.statement = revised.statement.strip()
            claim.rationale = revised.rationale.strip()
            claim.confidence = max(0.0, min(1.0, float(revised.confidence or 0.5)))
            claim.fact_ids = kept
            claim.evidence_ids = list(
                dict.fromkeys(self._facts[fid].evidence_id for fid in kept if fid in self._facts)
            )
