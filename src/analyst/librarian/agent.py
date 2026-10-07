"""Librarian: turn a pile of fetched documents into a ranked, citable fact pack.

Deterministic work happens first and the model is used only where judgement is
actually required. The ordering matters for cost: stub rejection and deduplication
run before any model call, so we never pay to extract facts from the eighth copy
of a wire story or from a paywall interstitial.

The quote-integrity gate is the most important thing in this file. A fact whose
`verbatim_quote` is not a literal substring of its source document is dropped --
not warned about, dropped. Everything downstream (the Adversary's support check,
the report's footnotes) assumes a quote can be found in the source, so a fact that
breaks that assumption is worse than a missing fact.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime

from ..llm.client import LLMClient, SystemBlock
from ..llm.schemas import DocumentTriage, ExtractedFact
from ..models import (
    Dimension,
    Entity,
    Evidence,
    Fact,
    Polarity,
    RunConfig,
    SourceType,
    content_hash,
)
from ..prompts import load_prompt
from ..tools.textutil import looks_like_stub, make_gist, word_count
from .dedup import cluster_sizes, deduplicate

_WS = re.compile(r"\s+")


@dataclass
class LibrarianStats:
    documents_in: int = 0
    rejected_stub: int = 0
    rejected_duplicate: int = 0
    rejected_irrelevant: int = 0
    facts_extracted: int = 0
    facts_dropped_bad_quote: int = 0
    triage_calls: int = 0
    clusters: int = 0
    pack_facts: int = 0
    pack_tokens: int = 0

    @property
    def quote_integrity(self) -> float:
        """Share of extracted facts whose quote was genuinely in the source."""
        total = self.facts_extracted + self.facts_dropped_bad_quote
        return round(self.facts_extracted / total, 3) if total else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "quote_integrity": self.quote_integrity,
        }


def _normalize_for_match(text: str) -> str:
    """Collapse whitespace for substring matching.

    Whitespace differences are an artefact of HTML extraction, not a sign the model
    invented a quote, so they are forgiven. Nothing else is.
    """
    return _WS.sub(" ", text).strip().lower()


def quote_is_grounded(quote: str, source: str, min_chars: int = 25) -> bool:
    if not quote or len(quote.strip()) < min_chars:
        return False
    return _normalize_for_match(quote) in _normalize_for_match(source)


def _parse_dimensions(names: list[str]) -> list[Dimension]:
    valid = {d.value: d for d in Dimension}
    out: list[Dimension] = []
    for name in names:
        dim = valid.get(str(name).strip().lower().replace(" ", "_").replace("-", "_"))
        if dim and dim not in out:
            out.append(dim)
    return out


def _parse_polarity(value: str) -> Polarity:
    try:
        return Polarity(str(value).strip().lower())
    except ValueError:
        return Polarity.NEUTRAL


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


class Librarian:
    def __init__(self, llm: LLMClient, config: RunConfig) -> None:
        self.llm = llm
        self.config = config
        self.stats = LibrarianStats()
        self.usage_sink: list = []

    # -- entry point ------------------------------------------------------- #

    def curate(self, entity: Entity, evidence: list[Evidence]) -> tuple[list[Evidence], list[Fact]]:
        self.stats.documents_in = len(evidence)

        self._reject_stubs(evidence)
        _, clusters = deduplicate(
            evidence,
            max_distance=self.config.near_duplicate_threshold,
        )
        self.stats.clusters = clusters
        self.stats.rejected_duplicate = sum(1 for e in evidence if e.duplicate_of is not None)

        candidates = [e for e in evidence if e.is_usable and e.clean_text.strip()]
        facts = self._triage(entity, candidates)
        return evidence, facts

    # -- deterministic passes ---------------------------------------------- #

    def _reject_stubs(self, evidence: list[Evidence]) -> None:
        """Drop paywall pages and teasers before paying to read them.

        Filings are exempt: a short 8-K is legitimately short, and the minimum
        word count is calibrated for news articles.
        """
        for ev in evidence:
            if ev.source_type in (SourceType.FILING_10K, SourceType.FILING_8K):
                continue
            text = ev.clean_text
            if not text.strip() or looks_like_stub(text, self.config.min_body_words):
                ev.relevant = False
                ev.relevance_reason = (
                    f"stub or paywalled ({word_count(text)} words < {self.config.min_body_words})"
                )
                self.stats.rejected_stub += 1
            elif not ev.hash:
                ev.hash = content_hash(text)
            if not ev.gist:
                ev.gist = make_gist(text)

    # -- model pass -------------------------------------------------------- #

    def _triage(self, entity: Entity, candidates: list[Evidence]) -> list[Fact]:
        """One call per document: relevance and extraction together.

        Splitting these into two calls would double the cost for no quality gain,
        since both need the same document in context.
        """
        if not candidates:
            return []

        system = [
            SystemBlock(load_prompt("librarian_triage", self.config.prompt_version), cache=True)
        ]
        effort = self.config.effort.get("librarian", "low")

        def run_one(ev: Evidence) -> tuple[Evidence, DocumentTriage | None]:
            # Long filings are truncated for the triage call only; the full text
            # is retained on the Evidence for quote verification and the report.
            body = ev.clean_text[:24_000]
            user = (
                f"Subject: {entity.name}"
                f"{f' ({entity.ticker})' if entity.ticker else ''}\n"
                f"Also known as: {', '.join(entity.aliases) or 'n/a'}\n"
                f"Source type: {ev.source_type.value}\n"
                f"Publisher: {ev.publisher or 'unknown'}\n"
                f"Published: {ev.published_at.date().isoformat() if ev.published_at else 'undated'}\n"
                f"Title: {ev.title}\n\n"
                f"--- DOCUMENT TEXT ---\n{body}\n--- END DOCUMENT TEXT ---"
            )
            try:
                out = self.llm.structured(
                    node="librarian.triage",
                    system=system,
                    user=user,
                    output_model=DocumentTriage,
                    effort=effort,
                    max_tokens=4000,
                )
            except Exception as exc:  # noqa: BLE001 - one bad document must not kill the run
                ev.relevance_reason = f"triage failed: {type(exc).__name__}: {exc}"
                return ev, None
            self.usage_sink.append(out.usage)
            return ev, out.parsed

        max_workers = min(8, max(1, len(candidates)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(run_one, candidates))

        facts: list[Fact] = []
        for ev, triage in results:
            self.stats.triage_calls += 1
            if triage is None:
                continue
            ev.relevant = bool(triage.relevant)
            ev.relevance_reason = triage.reason
            ev.dimensions = _parse_dimensions(triage.dimensions)
            if not ev.relevant:
                # Counted here, where the model rejects it -- not from `relevant is
                # False` afterwards, which the stub filter also sets and which
                # double-counted every paywalled page as off-topic.
                self.stats.rejected_irrelevant += 1
                continue
            for raw in triage.facts:
                fact = self._accept_fact(ev, raw)
                if fact is not None:
                    facts.append(fact)
        return facts

    def _accept_fact(self, ev: Evidence, raw: ExtractedFact) -> Fact | None:
        if not raw.text.strip():
            return None
        if not quote_is_grounded(raw.verbatim_quote, ev.clean_text):
            # The model paraphrased instead of quoting, or invented the quote.
            # Either way the fact is not verifiable, so it does not exist.
            self.stats.facts_dropped_bad_quote += 1
            return None
        self.stats.facts_extracted += 1
        return Fact(
            evidence_id=ev.id,
            text=raw.text.strip(),
            verbatim_quote=raw.verbatim_quote.strip(),
            happened_at=_parse_date(raw.happened_at)
            or (ev.published_at.date() if ev.published_at else None),
            polarity=_parse_polarity(raw.polarity),
            dimensions=_parse_dimensions(raw.dimensions) or ev.dimensions,
            salience=max(0.0, min(1.0, float(raw.salience or 0.5))),
        )

    # -- fact pack --------------------------------------------------------- #

    def build_pack(
        self, facts: list[Fact], evidence: list[Evidence], *, now: datetime | None = None
    ) -> tuple[list[Fact], str]:
        """Rank facts and render the pack that every Analyst call shares.

        The pack is the cached prefix for the whole analysis stage, so it must be
        byte-stable across calls -- which is why nothing time-varying is rendered
        into it.
        """
        now = now or datetime.now(UTC)
        by_id = {e.id: e for e in evidence}
        corroboration = cluster_sizes([e for e in evidence if e.is_usable])

        def score(fact: Fact) -> float:
            ev = by_id.get(fact.evidence_id)
            authority = ev.authority if ev else 0.4
            age_days = 0.0
            if ev and ev.published_at:
                age_days = max((now - ev.published_at).total_seconds() / 86400.0, 0.0)
            # Linear decay over the window, floored so older filings still count.
            recency = max(0.25, 1.0 - (age_days / max(self.config.lookback_days, 1)))
            corrob = min(corroboration.get(fact.evidence_id, 1) / 4.0, 1.0)
            return (fact.salience * 0.45) + (authority * 0.3) + (recency * 0.15) + (corrob * 0.1)

        ranked = sorted(facts, key=score, reverse=True)[: self.config.max_facts_in_pack]

        lines: list[str] = []
        budget_chars = self.config.fact_pack_token_budget * 4
        used = 0
        kept: list[Fact] = []
        for fact in ranked:
            ev = by_id.get(fact.evidence_id)
            source = (
                f"{ev.publisher or 'unknown'}, {ev.published_at.date().isoformat() if ev and ev.published_at else 'undated'}"
                if ev
                else "unknown"
            )
            line = (
                f"[{fact.id}] ({source}; {fact.polarity.value}) {fact.text}\n"
                f'    quote: "{fact.verbatim_quote[:320]}"'
            )
            if used + len(line) > budget_chars:
                break
            lines.append(line)
            used += len(line)
            kept.append(fact)

        self.stats.pack_facts = len(kept)
        pack = "\n".join(lines) if lines else "(no facts available)"
        self.stats.pack_tokens = self._measure(pack)
        return kept, pack

    def _measure(self, pack: str) -> int:
        """Measure the pack with the API's own counter when available.

        One extra call per run buys an exact number instead of a chars/4 guess,
        which matters because the pack is the cached prefix and its size drives
        most of the run's cost.
        """
        counter = getattr(self.llm, "count_tokens", None)
        if callable(counter):
            try:
                return int(counter("", pack))
            except Exception:  # noqa: BLE001 - fall back to the estimate
                pass
        return len(pack) // 4
