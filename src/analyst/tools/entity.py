"""Resolve a free-text query to a concrete research target.

This runs before anything else because every later relevance judgement depends
on it. "Apple" must become Apple Inc. / AAPL / CIK 0000320193 before we start
collecting articles, or the Librarian spends its budget rejecting stories about
fruit. Ambiguity is surfaced rather than guessed at: a low-confidence resolution
returns candidates and the run stops to ask.

The SEC's company_tickers.json is the authority -- it is free, official, and maps
ticker <-> CIK <-> legal name in one file.
"""

from __future__ import annotations

import difflib
import json
import re

from ..models import Entity, EntityCandidate
from .fetch import Fetcher

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Legal-form suffixes and filler that carry no identifying signal.
_NOISE = {
    "inc",
    "inc.",
    "incorporated",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "ltd",
    "ltd.",
    "limited",
    "plc",
    "llc",
    "lp",
    "holdings",
    "holding",
    "group",
    "the",
    "sa",
    "ag",
    "nv",
    "se",
    "class",
    "a",
    "b",
    "c",
    "common",
    "stock",
}

# Words that mean the user asked about a sector, not a single issuer.
_INDUSTRY_MARKERS = {
    "industry",
    "sector",
    "market",
    "companies",
    "space",
    "vertical",
    "airlines",
    "banks",
    "banking",
    "semiconductors",
    "semiconductor",
    "retail",
    "retailers",
    "biotech",
    "biotechnology",
    "pharma",
    "pharmaceuticals",
    "utilities",
    "insurers",
    "insurance",
    "automakers",
    "automotive",
    "shipbuilding",
    "shipyards",
    "defense",
    "aerospace",
    "mining",
    "energy",
    "renewables",
    "solar",
    "telecom",
    "telecommunications",
    "healthcare",
    "logistics",
    "freight",
    "railroads",
    "homebuilders",
    "restaurants",
    "grocers",
    "streaming",
    "gaming",
    "cybersecurity",
    "fintech",
    "cloud",
    "datacenters",
}

_PUNCT = re.compile(r"[^a-z0-9& ]+")
_WS = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop legal-form noise words."""
    s = _PUNCT.sub(" ", name.lower())
    s = _WS.sub(" ", s).strip()
    kept = [w for w in s.split() if w not in _NOISE]
    return " ".join(kept) if kept else s


def looks_like_industry(query: str) -> bool:
    words = set(normalize_name(query).split())
    return bool(words & _INDUSTRY_MARKERS)


def derive_aliases(legal_name: str) -> list[str]:
    """Short forms a journalist would actually use: 'Apple Inc.' -> 'Apple'."""
    aliases: list[str] = []
    short = normalize_name(legal_name)
    if short and short != legal_name.lower():
        aliases.append(short.title() if short.islower() else short)
    first = short.split(" ")[0] if short else ""
    if len(first) > 3 and first != short:
        aliases.append(first.title())
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        if a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out


class CompanyIndex:
    """In-memory index over the SEC ticker file."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.by_ticker: dict[str, dict[str, str]] = {}
        self.by_norm: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            ticker = (row.get("ticker") or "").upper()
            if ticker:
                self.by_ticker.setdefault(ticker, row)
            norm = normalize_name(row.get("title") or "")
            if norm:
                self.by_norm.setdefault(norm, []).append(row)

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def from_payload(cls, payload: dict | list) -> CompanyIndex:
        """SEC ships this as an object keyed by row index, not an array."""
        raw = payload.values() if isinstance(payload, dict) else payload
        rows: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cik = item.get("cik_str") or item.get("cik")
            rows.append(
                {
                    "cik": str(cik).zfill(10) if cik is not None else "",
                    "ticker": str(item.get("ticker", "")).upper(),
                    "title": str(item.get("title", "")),
                }
            )
        return cls(rows)

    def score(self, query: str, limit: int = 6) -> list[tuple[dict[str, str], float]]:
        """Rank companies against a query. Exact ticker beats everything."""
        q_norm = normalize_name(query)
        q_upper = query.strip().upper()
        scored: dict[str, tuple[dict[str, str], float]] = {}

        def consider(row: dict[str, str], score: float) -> None:
            key = row.get("cik") or row.get("ticker") or row.get("title", "")
            prior = scored.get(key)
            if prior is None or score > prior[1]:
                scored[key] = (row, score)

        # A bare ticker is an unambiguous request.
        if q_upper in self.by_ticker and len(q_upper) <= 5:
            consider(self.by_ticker[q_upper], 1.0)

        for row in self.by_norm.get(q_norm, []):
            consider(row, 0.97)

        if q_norm:
            close = difflib.get_close_matches(q_norm, self.by_norm.keys(), n=limit, cutoff=0.72)
            for name in close:
                ratio = difflib.SequenceMatcher(None, q_norm, name).ratio()
                for row in self.by_norm[name]:
                    consider(row, round(min(ratio, 0.96), 4))

            # Word-boundary prefix matches, which edit-distance alone misses:
            # "delta" vs "delta air lines" scores only 0.53 on SequenceMatcher
            # because of the length difference, so without this pass a query that
            # is genuinely ambiguous between several longer names would surface no
            # candidates at all -- and the run could not tell the user what it was
            # torn between. Score decays with how much of the name is unexplained.
            prefix = f"{q_norm} "
            for name, rows in self.by_norm.items():
                if not name.startswith(prefix):
                    continue
                coverage = len(q_norm) / len(name)
                score = round(0.55 + (0.3 * coverage), 4)
                for row in rows:
                    consider(row, score)

        ranked = sorted(scored.values(), key=lambda pair: -pair[1])
        return ranked[:limit]


async def load_index(fetcher: Fetcher) -> CompanyIndex:
    result = await fetcher.get(SEC_TICKERS_URL, check_robots=False)
    if not result.ok:
        raise RuntimeError(f"could not load SEC ticker index: {result.error or result.status}")
    return CompanyIndex.from_payload(json.loads(result.body))


def resolve_from_index(query: str, index: CompanyIndex) -> Entity:
    """Pure resolution logic, separated from I/O so it is unit-testable offline."""
    if looks_like_industry(query):
        return Entity(
            query=query,
            name=query.strip().title(),
            is_industry=True,
            confidence=0.8,
            aliases=[query.strip()],
        )

    ranked = index.score(query)
    if not ranked:
        return Entity(query=query, name=query.strip(), confidence=0.0)

    candidates = [
        EntityCandidate(
            name=row.get("title", ""),
            ticker=row.get("ticker") or None,
            cik=row.get("cik") or None,
            score=score,
        )
        for row, score in ranked
    ]

    top_row, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    # Confidence is about separation, not absolute similarity: two companies
    # scoring 0.95 and 0.94 is the ambiguous case we must not resolve silently.
    margin = top_score - runner_up
    confidence = top_score if margin >= 0.08 else round(top_score * 0.55, 4)

    name = top_row.get("title", query)
    return Entity(
        query=query,
        name=name,
        ticker=top_row.get("ticker") or None,
        cik=top_row.get("cik") or None,
        aliases=derive_aliases(name),
        confidence=round(confidence, 4),
        candidates=candidates,
    )


async def resolve_entity(query: str, fetcher: Fetcher) -> Entity:
    """Resolve `query` to an `Entity`, consulting the SEC index."""
    index = await load_index(fetcher)
    return resolve_from_index(query, index)
