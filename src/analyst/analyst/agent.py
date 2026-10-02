"""Analyst: build the structured view, one focused call per section.

Caching layout is the thing to understand here. Caching is a prefix match, so the
stable content goes first and the volatile content last:

    system = [ shared analyst role , fact pack  <-- cache breakpoint ]
    user   = per-section instruction + section inputs

Because the role text and the fact pack are byte-identical across all eight
section calls, every call after the first reads the pack from cache instead of
paying for it again. If the section instruction lived in the system prompt
instead, each section would start a fresh prefix and the pack would be billed
eight times. `ledger.cache_read_tokens` is the check that this is actually
working.

Citation integrity is enforced here, not trusted: any `fact_id` the model returns
that is not in the pack is dropped, and a claim left with no valid citation is
discarded entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.client import LLMClient, SystemBlock
from ..llm.schemas import ClaimSet, DraftClaim, OpenQuestionSet, RiskAssessmentSet
from ..models import (
    Claim,
    Entity,
    Fact,
    PeerMetric,
    Quadrant,
    RiskStatus,
    RunConfig,
    Section,
    Snapshot,
    StatedRisk,
)
from ..prompts import load_prompt
from ..tools.financials import format_money, format_pct

QUADRANT_BRIEFS: dict[Quadrant, str] = {
    Quadrant.STRENGTH: "STRENGTHS -- internal attributes the company has NOW.",
    Quadrant.WEAKNESS: "WEAKNESSES -- internal deficiencies the company has NOW.",
    Quadrant.OPPORTUNITY: "OPPORTUNITIES -- external, forward-looking conditions it could exploit but has not yet.",
    Quadrant.THREAT: "THREATS -- external, forward-looking conditions that could harm it.",
}


@dataclass
class AnalystStats:
    section_calls: int = 0
    claims_drafted: int = 0
    claims_dropped_no_citation: int = 0
    fact_ids_hallucinated: int = 0
    insufficient_sections: list[str] | None = None

    def __post_init__(self) -> None:
        if self.insufficient_sections is None:
            self.insufficient_sections = []

    @property
    def citation_validity(self) -> float:
        total = self.claims_drafted + self.claims_dropped_no_citation
        return round(self.claims_drafted / total, 3) if total else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "section_calls": self.section_calls,
            "claims_drafted": self.claims_drafted,
            "claims_dropped_no_citation": self.claims_dropped_no_citation,
            "fact_ids_hallucinated": self.fact_ids_hallucinated,
            "citation_validity": self.citation_validity,
            "insufficient_sections": self.insufficient_sections,
        }


class Analyst:
    def __init__(self, llm: LLMClient, config: RunConfig) -> None:
        self.llm = llm
        self.config = config
        self.stats = AnalystStats()
        self.usage_sink: list = []
        self._pack: str = ""
        self._valid_fact_ids: set[str] = set()
        self._facts_by_id: dict[str, Fact] = {}

    # -- setup ------------------------------------------------------------- #

    def load_pack(self, pack: str, facts: list[Fact]) -> None:
        self._pack = pack
        self._facts_by_id = {f.id: f for f in facts}
        self._valid_fact_ids = set(self._facts_by_id)

    @property
    def _system(self) -> list[SystemBlock]:
        role = load_prompt("analyst_role", self.config.prompt_version)
        # Stable prefix: identical bytes on every section call, cached once.
        return [
            SystemBlock(role, cache=False),
            SystemBlock(f"=== FACT PACK ===\n{self._pack}\n=== END FACT PACK ===", cache=True),
        ]

    @property
    def _effort(self) -> str:
        return self.config.effort.get("analyst", "high")

    # -- sections ---------------------------------------------------------- #

    def swot(self, entity: Entity) -> list[Claim]:
        claims: list[Claim] = []
        instructions = load_prompt("analyst_swot", self.config.prompt_version)
        for quadrant in Quadrant:
            user = (
                f"{instructions}\n\n"
                f"Subject: {entity.name}\n"
                f"Window: last {self.config.lookback_days} days\n\n"
                f"Write only this quadrant:\n{QUADRANT_BRIEFS[quadrant]}"
            )
            result = self._claim_set("analyst.swot", user)
            if result is None:
                continue
            section_label = f"swot.{quadrant.value}"
            if result.insufficient_evidence:
                assert self.stats.insufficient_sections is not None
                self.stats.insufficient_sections.append(section_label)
            claims.extend(self._to_claims(result.claims, Section.SWOT, quadrant=quadrant))
        return claims

    def catalysts(self, entity: Entity) -> list[Claim]:
        user = (
            f"{load_prompt('analyst_catalysts', self.config.prompt_version)}\n\n"
            f"Subject: {entity.name}\n"
            f"Window: last {self.config.lookback_days} days"
        )
        result = self._claim_set("analyst.catalysts", user)
        return self._to_claims(result.claims, Section.CATALYSTS) if result else []

    def competitive(
        self, entity: Entity, snapshot: Snapshot | None, peers: list[PeerMetric]
    ) -> list[Claim]:
        user = (
            f"{load_prompt('analyst_competitive', self.config.prompt_version)}\n\n"
            f"Subject: {entity.name} ({entity.ticker or 'unlisted'})\n"
            f"SEC industry classification: {entity.industry or 'unknown'} (SIC {entity.sic or 'n/a'})\n\n"
            f"{_snapshot_table(entity, snapshot)}\n\n"
            f"{_peer_table(peers)}"
        )
        result = self._claim_set("analyst.competitive", user)
        if result and result.insufficient_evidence:
            assert self.stats.insufficient_sections is not None
            self.stats.insufficient_sections.append("competitive")
        return self._to_claims(result.claims, Section.COMPETITIVE) if result else []

    def stated_risks(self, entity: Entity, risk_chunks: list[str]) -> list[StatedRisk]:
        """Cross-reference the company's own disclosed risks against the news.

        This is the section with no equivalent in comparable tools: Item 1A is
        management's ranked list of what could go wrong, and scoring each entry
        against recent evidence turns boilerplate into a live risk register.
        """
        if not risk_chunks:
            return []
        listing = "\n\n".join(f"[{i}] {chunk[:1200]}" for i, chunk in enumerate(risk_chunks[:20]))
        user = (
            f"{load_prompt('analyst_risks', self.config.prompt_version)}\n\n"
            f"Subject: {entity.name}\n\n"
            f"=== DISCLOSED RISK FACTORS (10-K Item 1A) ===\n{listing}"
        )
        out = self.llm.structured(
            node="analyst.risks",
            system=self._system,
            user=user,
            output_model=RiskAssessmentSet,
            effort=self._effort,
            max_tokens=8000,
        )
        self.usage_sink.append(out.usage)
        self.stats.section_calls += 1

        risks: list[StatedRisk] = []
        for assessment in out.parsed.assessments:
            if not 0 <= assessment.risk_index < len(risk_chunks):
                continue
            fact_ids = self._valid_ids(assessment.fact_ids)
            try:
                status = RiskStatus(assessment.status.strip().lower())
            except ValueError:
                status = RiskStatus.QUIET
            # A non-quiet status without citations is downgraded: the prompt
            # requires evidence for "materializing" and "contradicted", and an
            # uncited claim of either is exactly the overreach we are guarding.
            if status is not RiskStatus.QUIET and not fact_ids:
                status = RiskStatus.QUIET
            risks.append(
                StatedRisk(
                    risk_text=risk_chunks[assessment.risk_index][:2000],
                    summary=assessment.summary.strip(),
                    status=status,
                    fact_ids=fact_ids,
                    reasoning=assessment.reasoning,
                )
            )
        return risks

    def open_questions(self, entity: Entity, gaps: list[str], claims: list[Claim]) -> list[str]:
        made = "\n".join(f"- {c.statement}" for c in claims[:40]) or "(none)"
        user = (
            f"{load_prompt('analyst_open_questions', self.config.prompt_version)}\n\n"
            f"Subject: {entity.name}\n"
            f"Coverage dimensions still lacking evidence: {', '.join(gaps) or 'none reported'}\n\n"
            f"Claims already made:\n{made}"
        )
        out = self.llm.structured(
            node="analyst.open_questions",
            system=self._system,
            user=user,
            output_model=OpenQuestionSet,
            effort=self._effort,
            max_tokens=2000,
        )
        self.usage_sink.append(out.usage)
        self.stats.section_calls += 1
        return [q.strip() for q in out.parsed.questions if q.strip()][:8]

    # -- internals --------------------------------------------------------- #

    def _claim_set(self, node: str, user: str) -> ClaimSet | None:
        try:
            out = self.llm.structured(
                node=node,
                system=self._system,
                user=user,
                output_model=ClaimSet,
                effort=self._effort,
                max_tokens=6000,
            )
        except Exception:  # noqa: BLE001 - a failed section degrades the brief, not the run
            return None
        self.usage_sink.append(out.usage)
        self.stats.section_calls += 1
        return out.parsed

    def _valid_ids(self, fact_ids: list[str]) -> list[str]:
        """Keep only citations that point at facts actually in the pack."""
        kept: list[str] = []
        for fid in fact_ids:
            clean = fid.strip().strip("[]")
            if clean in self._valid_fact_ids:
                if clean not in kept:
                    kept.append(clean)
            else:
                self.stats.fact_ids_hallucinated += 1
        return kept

    def _to_claims(
        self, drafts: list[DraftClaim], section: Section, *, quadrant: Quadrant | None = None
    ) -> list[Claim]:
        claims: list[Claim] = []
        for draft in drafts:
            if not draft.statement.strip():
                continue
            fact_ids = self._valid_ids(draft.fact_ids)
            if not fact_ids:
                # Uncitable claim. The whole design rests on traceability, so this
                # is dropped rather than published with an empty footnote.
                self.stats.claims_dropped_no_citation += 1
                continue
            evidence_ids: list[str] = []
            for fid in fact_ids:
                fact = self._facts_by_id.get(fid)
                if fact and fact.evidence_id not in evidence_ids:
                    evidence_ids.append(fact.evidence_id)
            self.stats.claims_drafted += 1
            claims.append(
                Claim(
                    section=section,
                    quadrant=quadrant,
                    statement=draft.statement.strip(),
                    rationale=draft.rationale.strip(),
                    confidence=max(0.0, min(1.0, float(draft.confidence or 0.5))),
                    evidence_ids=evidence_ids,
                    fact_ids=fact_ids,
                )
            )
        return claims


# --------------------------------------------------------------------------- #
# Deterministic inputs rendered for the model
# --------------------------------------------------------------------------- #


def _snapshot_table(entity: Entity, snapshot: Snapshot | None) -> str:
    if snapshot is None:
        return "=== SUBJECT METRICS ===\n(unavailable)"
    rows = [
        ("Market cap", format_money(snapshot.market_cap)),
        ("Revenue (TTM)", format_money(snapshot.revenue_ttm)),
        ("Gross margin", format_pct(snapshot.gross_margin)),
        ("Operating margin", format_pct(snapshot.operating_margin)),
        ("Net margin", format_pct(snapshot.net_margin)),
        ("P/E (trailing)", f"{snapshot.pe_ratio:.1f}" if snapshot.pe_ratio else "n/a"),
        ("Debt/equity", f"{snapshot.debt_to_equity:.1f}" if snapshot.debt_to_equity else "n/a"),
        ("Free cash flow", format_money(snapshot.free_cash_flow)),
        ("Employees", f"{snapshot.employees:,}" if snapshot.employees else "n/a"),
    ]
    body = "\n".join(f"  {label}: {value}" for label, value in rows)
    return f"=== SUBJECT METRICS ({entity.ticker or entity.name}) ===\n{body}"


def _peer_table(peers: list[PeerMetric]) -> str:
    if not peers:
        return (
            "=== PEER METRICS ===\n(no peer data available -- say so if it limits the comparison)"
        )
    header = f"  {'ticker':<8}{'mkt cap':>12}{'revenue':>12}{'gross':>9}{'op margin':>11}{'P/E':>8}"
    lines = [header]
    for p in peers:
        lines.append(
            f"  {p.ticker:<8}{format_money(p.market_cap):>12}{format_money(p.revenue_ttm):>12}"
            f"{format_pct(p.gross_margin):>9}{format_pct(p.operating_margin):>11}"
            f"{(f'{p.pe_ratio:.1f}' if p.pe_ratio else 'n/a'):>8}"
        )
    return "=== PEER METRICS (from SEC industry classification) ===\n" + "\n".join(lines)
