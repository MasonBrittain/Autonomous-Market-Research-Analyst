"""Scout: the only stage with real autonomy.

Scout runs a bounded loop -- plan, gather, assess coverage, decide whether to go
again -- rather than executing a fixed fetch list. That loop is what makes the
project autonomous in a meaningful sense, and it is also the stage most likely to
misbehave, so three guards are non-negotiable:

* Scout sees document *metadata only* (`Evidence.digest()`), never bodies. A
  retrieval decision needs a title, a date and a gist; feeding it full articles
  would multiply the loop's cost by two orders of magnitude for no gain.
* Hard ceilings on rounds and tool calls, so a non-converging loop ends.
* Saturation is an explicit, prompted-for answer. Scout is told that "we have
  enough" is a correct response, because an agent that never stops is worse than
  one that stops early.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.client import LLMClient, SystemBlock
from ..llm.schemas import CoverageVerdict, ScoutPlan
from ..models import (
    CoverageAssessment,
    Dimension,
    Entity,
    Evidence,
    RunConfig,
    SourceType,
)
from ..prompts import load_prompt
from ..tools import edgar, financials, news
from ..tools.fetch import Fetcher


@dataclass
class ScoutResult:
    evidence: list[Evidence] = field(default_factory=list)
    assessments: list[CoverageAssessment] = field(default_factory=list)
    risk_chunks: list[str] = field(default_factory=list)
    snapshot_info: dict = field(default_factory=dict)
    peer_ciks: list[str] = field(default_factory=list)
    tool_calls: int = 0

    @property
    def rounds(self) -> int:
        return len(self.assessments)


def _parse_dimensions(names: list[str]) -> list[Dimension]:
    """Coerce model-supplied dimension names, ignoring anything unrecognised.

    The model occasionally invents a dimension. Silently dropping it is right:
    the coverage checklist is ours, not the model's, and a hallucinated dimension
    must not be able to mark a real one covered.
    """
    valid = {d.value: d for d in Dimension}
    out: list[Dimension] = []
    for name in names:
        key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        dim = valid.get(key)
        if dim is not None and dim not in out:
            out.append(dim)
    return out


class Scout:
    def __init__(self, llm: LLMClient, fetcher: Fetcher, config: RunConfig) -> None:
        self.llm = llm
        self.fetcher = fetcher
        self.config = config
        self.usage_sink: list = []

    # -- main loop --------------------------------------------------------- #

    async def gather(self, entity: Entity) -> ScoutResult:
        result = ScoutResult()
        seen_urls: set[str] = set()

        # Structured sources are unconditional: filings and financials are the
        # backbone of the brief and need no model judgement to decide on.
        await self._gather_structured(entity, result, seen_urls)

        plan = self._plan(entity)
        queries = plan.queries

        for round_number in range(1, self.config.max_scout_rounds + 1):
            if result.tool_calls >= self.config.max_scout_tool_calls:
                break
            if len(result.evidence) >= self.config.max_evidence:
                break

            batch = await self._search(entity, queries, seen_urls, result)
            result.evidence.extend(batch)

            assessment = self._assess(entity, result, round_number)
            result.assessments.append(assessment)

            if assessment.saturated or not assessment.next_queries:
                break
            # Nothing new arrived: another round with similar queries will not help.
            if not batch and round_number > 1:
                assessment.saturated = True
                assessment.reasoning += " (forced: last round returned no new documents)"
                break
            queries = assessment.next_queries

        return result

    # -- structured sources ------------------------------------------------ #

    async def _gather_structured(
        self, entity: Entity, result: ScoutResult, seen_urls: set[str]
    ) -> None:
        if entity.cik:
            filings, risk_chunks = await edgar.fetch_filings(self.fetcher, entity.cik, entity.name)
            result.tool_calls += 1
            result.risk_chunks = risk_chunks
            for ev in filings:
                if ev.url not in seen_urls or ev.source_type is SourceType.FILING_10K:
                    seen_urls.add(ev.url)
                    result.evidence.append(ev)

            sic, sic_desc = await edgar.company_sic(self.fetcher, entity.cik)
            entity.sic = sic or entity.sic
            entity.industry = entity.industry or sic_desc or None
            if sic:
                result.peer_ciks = await edgar.peers_by_sic(self.fetcher, sic, entity.cik)
                result.tool_calls += 1

        if entity.ticker:
            _, info = await financials.get_snapshot(entity.ticker)
            result.snapshot_info = info
            result.tool_calls += 1

    # -- model calls ------------------------------------------------------- #

    def _plan(self, entity: Entity) -> ScoutPlan:
        system = [SystemBlock(load_prompt("scout_plan", self.config.prompt_version), cache=True)]
        user = (
            f"Subject: {entity.name}\n"
            f"Ticker: {entity.ticker or 'n/a'}\n"
            f"Industry: {entity.industry or 'unknown'}\n"
            f"Also known as: {', '.join(entity.aliases) or 'n/a'}\n"
            f"Window: last {self.config.lookback_days} days\n\n"
            f"Coverage dimensions to evidence:\n"
            + "\n".join(f"- {d.value}" for d in Dimension.all())
        )
        out = self.llm.structured(
            node="scout.plan",
            system=system,
            user=user,
            output_model=ScoutPlan,
            effort=self.config.effort.get("scout", "medium"),
            max_tokens=2000,
        )
        self.usage_sink.append(out.usage)
        return out.parsed

    def _assess(self, entity: Entity, result: ScoutResult, round_number: int) -> CoverageAssessment:
        # Metadata only. This is the line that keeps the loop affordable.
        digests = [ev.digest() for ev in result.evidence]
        inventory = "\n".join(
            f"- [{d['type']}] {d['date'] or 'undated'} | {d['publisher'] or 'unknown'} | "
            f"{d['title']} | {d['gist']}"
            for d in digests
        )
        system = [
            SystemBlock(load_prompt("scout_coverage", self.config.prompt_version), cache=True)
        ]
        user = (
            f"Subject: {entity.name}\n"
            f"Round: {round_number} of {self.config.max_scout_rounds}\n"
            f"Documents gathered: {len(digests)}\n\n"
            f"Coverage dimensions:\n"
            + "\n".join(f"- {d.value}" for d in Dimension.all())
            + f"\n\nGathered so far (metadata only):\n{inventory or '(nothing yet)'}"
        )
        out = self.llm.structured(
            node="scout.assess",
            system=system,
            user=user,
            output_model=CoverageVerdict,
            effort=self.config.effort.get("scout", "medium"),
            max_tokens=2000,
        )
        self.usage_sink.append(out.usage)
        verdict: CoverageVerdict = out.parsed

        return CoverageAssessment(
            round_number=round_number,
            covered=_parse_dimensions(verdict.covered),
            gaps=_parse_dimensions(verdict.gaps),
            next_queries=[q for q in verdict.next_queries if q.strip()][:6],
            saturated=verdict.saturated,
            reasoning=verdict.reasoning,
        )

    # -- retrieval --------------------------------------------------------- #

    async def _search(
        self,
        entity: Entity,
        queries: list[str],
        seen_urls: set[str],
        result: ScoutResult,
    ) -> list[Evidence]:
        remaining = self.config.max_evidence - len(result.evidence)
        if remaining <= 0:
            return []
        batch = await news.search_news(
            self.fetcher,
            entity.search_terms(),
            queries=queries,
            ticker=entity.ticker,
            lookback_days=self.config.lookback_days,
            limit=min(remaining, 30),
            min_body_words=self.config.min_body_words,
        )
        result.tool_calls += 1
        fresh: list[Evidence] = []
        for ev in batch:
            if ev.url in seen_urls:
                continue
            seen_urls.add(ev.url)
            fresh.append(ev)
        return fresh
