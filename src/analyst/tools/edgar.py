"""SEC EDGAR adapter.

This is the most valuable source in the project and the least used by comparable
tools. A 10-K Item 1A is a company's own ranked list of threats, written under
legal liability; Item 7 (MD&A) is management explaining its own numbers. Both are
free, structured, and far better evidence than a news aggregator.

Item extraction from filing HTML is heuristic by necessity -- filers are not
consistent -- so every extractor here degrades to an empty result rather than
raising, and the pipeline treats missing sections as a coverage gap for Scout.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from ..models import Evidence, SourceType, content_hash
from .fetch import Fetcher
from .textutil import html_to_text, make_gist, strip_xbrl_noise, word_count

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"
FILING_INDEX_URL = "https://www.sec.gov/cgi-bin/browse-edgar"


def _accession_nodash(accession: str) -> str:
    return accession.replace("-", "")


async def fetch_submissions(fetcher: Fetcher, cik: str) -> dict:
    """Company submission history. data.sec.gov is a documented JSON API."""
    url = SUBMISSIONS_URL.format(cik=cik.zfill(10))
    result = await fetcher.get(url, check_robots=False)
    if not result.ok:
        return {}
    try:
        return json.loads(result.body)
    except json.JSONDecodeError:
        return {}


def recent_filings(
    submissions: dict, forms: tuple[str, ...] = ("10-K", "8-K", "10-Q")
) -> list[dict[str, str]]:
    """Flatten the column-oriented `filings.recent` block into row dicts."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    accession = recent.get("accessionNumber") or []
    out: list[dict[str, str]] = []
    for i in range(len(accession)):
        form = (recent.get("form") or [""] * len(accession))[i]
        if forms and form not in forms:
            continue
        out.append(
            {
                "form": form,
                "accession": accession[i],
                "filing_date": (recent.get("filingDate") or [""] * len(accession))[i],
                "report_date": (recent.get("reportDate") or [""] * len(accession))[i],
                "document": (recent.get("primaryDocument") or [""] * len(accession))[i],
                "description": (recent.get("primaryDocDescription") or [""] * len(accession))[i],
            }
        )
    return out


def document_url(cik: str, accession: str, document: str) -> str:
    return ARCHIVE_URL.format(
        cik_int=int(cik), accession=_accession_nodash(accession), document=document
    )


# --------------------------------------------------------------------------- #
# Item extraction
# --------------------------------------------------------------------------- #

_ITEM_PATTERNS: dict[str, str] = {
    "1a": r"item\s*1a[\.\:\s\-—]*\s*risk\s*factors",
    "1b": r"item\s*1b[\.\:\s\-—]*\s*unresolved",
    "2": r"item\s*2[\.\:\s\-—]*\s*propert",
    "7": r"item\s*7[\.\:\s\-—]*\s*management",
    "7a": r"item\s*7a[\.\:\s\-—]*\s*quantitative",
    "8": r"item\s*8[\.\:\s\-—]*\s*financial\s*statements",
}


def extract_item(text: str, start_key: str, end_keys: tuple[str, ...], min_chars: int = 800) -> str:
    """Return the body of one 10-K item.

    Every filing mentions "Item 1A. Risk Factors" at least twice -- once in the
    table of contents and once as the real heading -- so we take the longest
    plausible span between a start marker and a following end marker rather than
    the first match. `min_chars` then discards table-of-contents spans, which are
    a line long; it is not a quality bar on the section itself.
    """
    if not text:
        return ""
    start_re = re.compile(_ITEM_PATTERNS[start_key], re.IGNORECASE)
    end_res = [
        re.compile(_ITEM_PATTERNS[k], re.IGNORECASE) for k in end_keys if k in _ITEM_PATTERNS
    ]

    best = ""
    for start_match in start_re.finditer(text):
        begin = start_match.end()
        candidate_ends = [
            m.start() for r in end_res for m in r.finditer(text, begin) if m.start() > begin
        ]
        stop = min(candidate_ends) if candidate_ends else len(text)
        span = text[begin:stop].strip()
        if len(span) > len(best):
            best = span
    return best if len(best) >= min_chars else ""


_RISK_SPLIT = re.compile(r"\n{1,}")


def split_risk_factors(item_1a: str, max_risks: int = 25, min_chars: int = 350) -> list[str]:
    """Break Item 1A into individual risks.

    Filers format risk headings inconsistently, so instead of guessing at heading
    style we accumulate paragraphs into chunks of at least `min_chars`. The model
    summarises each chunk later; over-splitting would cost more than it gains.
    """
    if not item_1a:
        return []
    paragraphs = [p.strip() for p in _RISK_SPLIT.split(item_1a) if len(p.strip()) > 40]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""
        if len(chunks) >= max_risks:
            break
    if buffer and len(chunks) < max_risks:
        chunks.append(buffer)
    return chunks


# --------------------------------------------------------------------------- #
# Public adapter surface
# --------------------------------------------------------------------------- #


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


async def fetch_filings(
    fetcher: Fetcher,
    cik: str,
    company_name: str = "",
    *,
    want_10k: bool = True,
    max_8k: int = 6,
) -> tuple[list[Evidence], list[str]]:
    """Fetch the latest 10-K sections and recent 8-Ks.

    Returns (evidence, risk_factor_chunks). Risk chunks are returned separately
    because they drive the stated-risk cross-reference rather than the fact pack.
    """
    submissions = await fetch_submissions(fetcher, cik)
    if not submissions:
        return [], []

    name = submissions.get("name") or company_name
    filings = recent_filings(submissions, forms=("10-K", "8-K"))

    evidence: list[Evidence] = []
    risk_chunks: list[str] = []

    if want_10k:
        tenk = next((f for f in filings if f["form"] == "10-K" and f["document"]), None)
        if tenk:
            url = document_url(cik, tenk["accession"], tenk["document"])
            result = await fetcher.get(url)
            if result.ok:
                text = strip_xbrl_noise(html_to_text(result.body))
                item_1a = extract_item(text, "1a", ("1b", "2"))
                item_7 = extract_item(text, "7", ("7a", "8"))
                filed = _parse_date(tenk["filing_date"])
                if item_1a:
                    risk_chunks = split_risk_factors(item_1a)
                    evidence.append(
                        Evidence(
                            source_type=SourceType.FILING_10K,
                            url=url,
                            title=f"{name} 10-K Item 1A Risk Factors ({tenk['filing_date']})",
                            publisher="SEC EDGAR",
                            published_at=filed,
                            clean_text=item_1a,
                            gist=make_gist(item_1a),
                            authority=1.0,
                            hash=content_hash(item_1a),
                        )
                    )
                if item_7:
                    evidence.append(
                        Evidence(
                            source_type=SourceType.FILING_10K,
                            url=url,
                            title=f"{name} 10-K Item 7 MD&A ({tenk['filing_date']})",
                            publisher="SEC EDGAR",
                            published_at=filed,
                            clean_text=item_7,
                            gist=make_gist(item_7),
                            authority=1.0,
                            hash=content_hash(item_7),
                        )
                    )

    eights = [f for f in filings if f["form"] == "8-K" and f["document"]][:max_8k]
    urls = [document_url(cik, f["accession"], f["document"]) for f in eights]
    if urls:
        results = await fetcher.get_many(urls)
        for filing, result in zip(eights, results, strict=True):
            if not result.ok:
                continue
            # 8-K primary documents are mostly inline XBRL; strip it and skip the
            # filing entirely if no prose survives.
            body = strip_xbrl_noise(html_to_text(result.body))
            if word_count(body) < 60:
                continue
            evidence.append(
                Evidence(
                    source_type=SourceType.FILING_8K,
                    url=result.url,
                    title=f"{name} 8-K {filing['filing_date']}: {filing['description'] or 'current report'}",
                    publisher="SEC EDGAR",
                    published_at=_parse_date(filing["filing_date"]),
                    clean_text=body,
                    gist=make_gist(body),
                    authority=1.0,
                    hash=content_hash(body),
                )
            )

    return evidence, risk_chunks


async def company_sic(fetcher: Fetcher, cik: str) -> tuple[str, str]:
    submissions = await fetch_submissions(fetcher, cik)
    return str(submissions.get("sic") or ""), str(submissions.get("sicDescription") or "")


_TICKER_IN_ATOM = re.compile(r"CIK=(\d+)", re.IGNORECASE)


async def peers_by_sic(
    fetcher: Fetcher, sic: str, exclude_cik: str = "", limit: int = 8
) -> list[str]:
    """Discover same-industry filers via EDGAR's browse-by-SIC listing.

    yfinance has no reliable peer list, so the peer set is derived from the SEC's
    own industry classification. If EDGAR's robots.txt blocks the CGI endpoint
    this returns an empty list and the competitive section degrades gracefully.
    """
    if not sic:
        return []
    url = (
        f"{FILING_INDEX_URL}?action=getcompany&SIC={sic}&type=10-K"
        f"&dateb=&owner=include&count=40&output=atom"
    )
    result = await fetcher.get(url)
    if not result.ok:
        return []
    ciks: list[str] = []
    for match in _TICKER_IN_ATOM.finditer(result.body):
        cik_found = match.group(1).zfill(10)
        if cik_found == exclude_cik.zfill(10) or cik_found in ciks:
            continue
        ciks.append(cik_found)
        if len(ciks) >= limit * 2:
            break
    return ciks[: limit * 2]
