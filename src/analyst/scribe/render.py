"""Scribe: assemble the brief.

Almost nothing here is a model call. The sections are rendered from claims that
already survived the Adversary, with footnotes generated from the evidence those
claims cite. The model writes only the executive summary, and it writes it from
the validated claim list -- so the single place a hallucination could reach the
page is constrained to summarising statements that are already cited.

A consequence worth keeping: two runs over the same evidence produce byte-identical
reports apart from the summary. That invariant is what the Scribe tests assert, and
it is what makes eval diffs readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..llm.client import LLMClient, SystemBlock
from ..llm.schemas import ExecutiveSummary
from ..models import (
    Claim,
    Evidence,
    Quadrant,
    ResearchRun,
    RiskStatus,
    Section,
    StatedRisk,
)
from ..prompts import load_prompt, prompt_fingerprint
from ..tools.financials import format_money, format_pct

TEMPLATE_DIR = Path(__file__).parent / "templates"

QUADRANT_TITLES = {
    Quadrant.STRENGTH: "Strengths",
    Quadrant.WEAKNESS: "Weaknesses",
    Quadrant.OPPORTUNITY: "Opportunities",
    Quadrant.THREAT: "Threats",
}

RISK_LABELS = {
    RiskStatus.MATERIALIZING: "Materializing",
    RiskStatus.QUIET: "Quiet",
    RiskStatus.CONTRADICTED: "Contradicted",
}


@dataclass
class Citation:
    number: int
    evidence: Evidence
    quotes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        date = (
            self.evidence.published_at.date().isoformat()
            if self.evidence.published_at
            else "undated"
        )
        return f"{self.evidence.publisher or 'unknown source'}, {date}"


class CitationIndex:
    """Assigns footnote numbers in order of first appearance."""

    def __init__(self, run: ResearchRun) -> None:
        self._run = run
        self._by_evidence: dict[str, Citation] = {}
        self._order: list[Citation] = []

    def cite(self, claim: Claim) -> list[int]:
        numbers: list[int] = []
        for eid in claim.evidence_ids:
            evidence = self._run.evidence_by_id(eid)
            if evidence is None:
                continue
            citation = self._by_evidence.get(eid)
            if citation is None:
                citation = Citation(number=len(self._order) + 1, evidence=evidence)
                self._by_evidence[eid] = citation
                self._order.append(citation)
            for fid in claim.fact_ids:
                fact = self._run.fact_by_id(fid)
                if fact and fact.evidence_id == eid and fact.verbatim_quote not in citation.quotes:
                    citation.quotes.append(fact.verbatim_quote)
            if citation.number not in numbers:
                numbers.append(citation.number)
        return sorted(numbers)

    def cite_fact_ids(self, fact_ids: list[str]) -> list[int]:
        pseudo = Claim(
            section=Section.STATED_RISKS,
            statement="",
            fact_ids=fact_ids,
            evidence_ids=list(
                dict.fromkeys(
                    f.evidence_id
                    for f in (self._run.fact_by_id(fid) for fid in fact_ids)
                    if f is not None
                )
            ),
        )
        return self.cite(pseudo)

    @property
    def citations(self) -> list[Citation]:
        return self._order


def _claim_view(claim: Claim, index: CitationIndex, run: ResearchRun) -> dict[str, object]:
    contradictions: list[str] = []
    if claim.verdict and claim.verdict.contradicting_fact_ids:
        for fid in claim.verdict.contradicting_fact_ids:
            fact = run.fact_by_id(fid)
            if fact:
                contradictions.append(fact.text)
    return {
        "statement": claim.statement,
        "rationale": claim.rationale,
        "confidence": claim.confidence,
        "confidence_label": _confidence_label(claim.confidence),
        "citations": index.cite(claim),
        "revised": claim.verdict is not None and claim.verdict.verdict.value == "revise",
        "contradictions": contradictions,
    }


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "moderate"
    return "low"


# Materializing risks are the point of the section, so they lead; contradicted ones
# are the next most interesting; quiet boilerplate goes last.
_RISK_ORDER = {RiskStatus.MATERIALIZING: 0, RiskStatus.CONTRADICTED: 1, RiskStatus.QUIET: 2}


def _risk_sort_key(risk: StatedRisk) -> tuple[int, str]:
    return (_RISK_ORDER.get(risk.status, 3), risk.summary)


def _snapshot_rows(run: ResearchRun) -> list[tuple[str, str]]:
    snap = run.snapshot
    if snap is None:
        return []
    rows = [
        ("Market cap", format_money(snap.market_cap)),
        ("Revenue (TTM)", format_money(snap.revenue_ttm)),
        ("Gross margin", format_pct(snap.gross_margin)),
        ("Operating margin", format_pct(snap.operating_margin)),
        ("Net margin", format_pct(snap.net_margin)),
        ("P/E (trailing)", f"{snap.pe_ratio:.1f}" if snap.pe_ratio else "n/a"),
        ("Debt/equity", f"{snap.debt_to_equity:.1f}" if snap.debt_to_equity else "n/a"),
        ("Free cash flow", format_money(snap.free_cash_flow)),
        ("Employees", f"{snap.employees:,}" if snap.employees else "n/a"),
    ]
    return [(label, value) for label, value in rows if value != "n/a"]


def build_context(run: ResearchRun, *, summary: str = "", headline: str = "") -> dict[str, object]:
    index = CitationIndex(run)
    entity = run.entity

    swot = {QUADRANT_TITLES[q]: [_claim_view(c, index, run) for c in run.swot(q)] for q in Quadrant}

    catalysts = sorted(
        run.claims_in(Section.CATALYSTS),
        key=lambda c: (
            min(
                (
                    f.happened_at
                    for f in (run.fact_by_id(i) for i in c.fact_ids)
                    if f and f.happened_at
                ),
                default=None,
            )
            or datetime.min.date()
        ),
        reverse=True,
    )

    risks = [
        {
            "summary": r.summary,
            "status": RISK_LABELS[r.status],
            "status_key": r.status.value,
            "reasoning": r.reasoning,
            "citations": index.cite_fact_ids(r.fact_ids),
            "excerpt": " ".join(r.risk_text.split())[:260],
        }
        for r in sorted(run.stated_risks, key=_risk_sort_key)
    ]

    coverage = run.latest_coverage
    return {
        "entity_name": entity.name if entity else run.query,
        "ticker": entity.ticker if entity else None,
        "industry": entity.industry if entity else None,
        "is_industry": entity.is_industry if entity else False,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "window_days": run.config.lookback_days,
        "headline": headline,
        "summary": summary,
        "snapshot_rows": _snapshot_rows(run),
        "swot": swot,
        "catalysts": [_claim_view(c, index, run) for c in catalysts],
        "competitive": [_claim_view(c, index, run) for c in run.claims_in(Section.COMPETITIVE)],
        "risks": risks,
        "risks_materializing": sum(
            1 for r in run.stated_risks if r.status is RiskStatus.MATERIALIZING
        ),
        "open_questions": run.open_questions,
        "peers": [
            {
                "ticker": p.ticker,
                "name": p.name,
                "market_cap": format_money(p.market_cap),
                "gross_margin": format_pct(p.gross_margin),
                "operating_margin": format_pct(p.operating_margin),
                "pe": f"{p.pe_ratio:.1f}" if p.pe_ratio else "n/a",
            }
            for p in run.peers
        ],
        "citations": [
            {
                "number": c.number,
                "label": c.label,
                "title": c.evidence.title,
                "url": c.evidence.url,
                "quotes": c.quotes[:3],
            }
            for c in index.citations
        ],
        # Provenance: everything needed to reproduce or distrust this run.
        "provenance": {
            "run_id": run.id,
            "model": run.config.model,
            "prompt_version": run.config.prompt_version,
            "prompt_fingerprint": prompt_fingerprint(run.config.prompt_version),
            "documents_gathered": len(run.evidence),
            "documents_used": len(run.usable_evidence()),
            "facts": len(run.facts),
            "claims_published": sum(1 for c in run.claims if c.survived),
            "claims_rejected": sum(1 for c in run.claims if not c.survived),
            "rejection_rate": run.rejection_rate,
            "dedup_rate": run.dedup_rate,
            "coverage_score": coverage.score if coverage else 0.0,
            "scout_rounds": len(run.coverage),
            "llm_calls": run.ledger.total_calls,
            "cost_usd": run.ledger.total_usd,
            "cache_read_tokens": run.ledger.cache_read_tokens,
            "cost_by_node": run.ledger.by_node(),
            "stub": run.config.stub,
        },
    }


def _cites_md(numbers: list[int]) -> str:
    return "".join(f" [{n}]" for n in numbers)


def _cites_html(numbers: list[int]) -> str:
    return "".join(f'<sup><a href="#src{n}">[{n}]</a></sup>' for n in numbers)


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    # Rendering citations through a filter rather than an inline `{% for %}` keeps
    # trim_blocks from eating the newline that ends each list item.
    env.filters["cites"] = _cites_md
    env.filters["cite_links"] = _cites_html
    return env


class Scribe:
    def __init__(self, llm: LLMClient, run: ResearchRun) -> None:
        self.llm = llm
        self.run = run
        self.usage_sink: list = []

    def write_summary(self) -> tuple[str, str]:
        """The only free text the model contributes to the report."""
        surviving = [c for c in self.run.claims if c.survived]
        if not surviving:
            return (
                "No claim in this brief survived adversarial review. The evidence "
                "gathered was insufficient to support analysis.",
                f"{self.run.entity.name if self.run.entity else self.run.query}: insufficient evidence",
            )

        claim_list = "\n".join(
            f"- [{c.section.value}{f'/{c.quadrant.value}' if c.quadrant else ''}] {c.statement}"
            for c in surviving
        )
        snapshot = (
            "\n".join(f"  {k}: {v}" for k, v in _snapshot_rows(self.run)) or "  (unavailable)"
        )
        entity = self.run.entity
        user = (
            f"Subject: {entity.name if entity else self.run.query}\n"
            f"Ticker: {entity.ticker if entity and entity.ticker else 'n/a'}\n"
            f"Window: last {self.run.config.lookback_days} days\n\n"
            f"METRICS:\n{snapshot}\n\n"
            f"VALIDATED CLAIMS (the only material you may use):\n{claim_list}"
        )
        out = self.llm.structured(
            node="scribe.summary",
            system=[
                SystemBlock(
                    load_prompt("scribe_summary", self.run.config.prompt_version), cache=True
                )
            ],
            user=user,
            output_model=ExecutiveSummary,
            effort=self.run.config.effort.get("scribe", "medium"),
            max_tokens=1500,
        )
        self.usage_sink.append(out.usage)
        parsed: ExecutiveSummary = out.parsed
        return parsed.summary.strip(), parsed.headline.strip()

    def render(self, summary: str = "", headline: str = "") -> tuple[str, str]:
        context = build_context(self.run, summary=summary, headline=headline)
        env = _environment()
        markdown = env.get_template("report.md.j2").render(**context)
        html = env.get_template("report.html.j2").render(**context)
        return markdown, html
