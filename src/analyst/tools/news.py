"""News retrieval over public RSS.

RSS rather than scraping: it is the publisher-sanctioned interface, it carries
clean titles, dates and source attribution, and it keeps us out of terms-of-service
territory. The cost is recall -- RSS gives headlines and summaries, and roughly a
third of linked bodies will be paywalled. That is an accepted trade (see
`docs/decisions.md`); EDGAR carries the analytical weight.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from urllib.parse import quote_plus, urlparse

from ..models import Evidence, SourceType, content_hash
from .fetch import Fetcher
from .textutil import extract_article, looks_like_stub, make_gist, word_count

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
YAHOO_FINANCE_RSS = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
)

# Rough authority weights. Used to rank the fact pack and to pick the canonical
# member of a duplicate cluster -- not to decide truth.
AUTHORITY: dict[str, float] = {
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "wsj.com": 0.9,
    "ft.com": 0.9,
    "bloomberg.com": 0.9,
    "cnbc.com": 0.8,
    "barrons.com": 0.8,
    "forbes.com": 0.65,
    "marketwatch.com": 0.7,
    "businesswire.com": 0.75,
    "prnewswire.com": 0.6,
    "globenewswire.com": 0.6,
    "seekingalpha.com": 0.5,
    "fool.com": 0.4,
    "benzinga.com": 0.4,
    "investing.com": 0.5,
    "yahoo.com": 0.55,
    "techcrunch.com": 0.7,
    "theinformation.com": 0.8,
}


def authority_for(url: str, publisher: str = "") -> float:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for domain, score in AUTHORITY.items():
        if host.endswith(domain):
            return score
    if publisher:
        p = publisher.lower()
        for domain, score in AUTHORITY.items():
            if domain.split(".")[0] in p:
                return score
    return 0.45


def build_queries(
    terms: list[str], extra: list[str] | None = None, lookback_days: int = 120
) -> list[str]:
    """Build Google News RSS URLs.

    `when:` keeps the window tight so the Librarian is not spending extraction
    budget rejecting three-year-old articles.
    """
    window = f"when:{max(lookback_days, 1)}d"
    queries: list[str] = []
    for term in terms[:3]:
        queries.append(f'"{term}" {window}')
    for e in (extra or [])[:8]:
        queries.append(f"{e} {window}")
    return [GOOGLE_NEWS_RSS.format(query=quote_plus(q)) for q in queries]


def _parse_feed(body: str) -> list[dict[str, str]]:
    import feedparser

    parsed = feedparser.parse(body)
    out: list[dict[str, str]] = []
    for entry in parsed.entries:
        published: datetime | None = None
        struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if struct:
            try:
                published = datetime(
                    struct[0], struct[1], struct[2], struct[3], struct[4], struct[5], tzinfo=UTC
                )
            except (TypeError, ValueError):
                published = None
        source = ""
        src = getattr(entry, "source", None)
        if src is not None:
            source = getattr(src, "title", "") or ""
        out.append(
            {
                "url": getattr(entry, "link", "") or "",
                "title": getattr(entry, "title", "") or "",
                "summary": getattr(entry, "summary", "") or "",
                "publisher": source,
                "published": published.isoformat() if published else "",
            }
        )
    return out


async def search_news(
    fetcher: Fetcher,
    terms: list[str],
    *,
    queries: list[str] | None = None,
    ticker: str | None = None,
    lookback_days: int = 120,
    limit: int = 40,
    fetch_bodies: bool = True,
    min_body_words: int = 200,
) -> list[Evidence]:
    """Search news and return Evidence, bodies filled in where retrievable."""
    feed_urls = build_queries(terms, queries, lookback_days)
    if ticker:
        feed_urls.append(YAHOO_FINANCE_RSS.format(ticker=ticker))

    feeds = await fetcher.get_many(feed_urls, check_robots=False)

    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    seen_urls: set[str] = set()
    entries: list[dict[str, str]] = []
    for feed in feeds:
        if not feed.ok:
            continue
        for entry in _parse_feed(feed.body):
            url = entry["url"]
            if not url or url in seen_urls:
                continue
            if entry["published"]:
                try:
                    if datetime.fromisoformat(entry["published"]) < cutoff:
                        continue
                except ValueError:
                    pass
            seen_urls.add(url)
            entries.append(entry)

    entries.sort(key=lambda e: e["published"], reverse=True)
    entries = entries[:limit]

    evidence: list[Evidence] = []
    for entry in entries:
        published = None
        if entry["published"]:
            try:
                published = datetime.fromisoformat(entry["published"])
            except ValueError:
                published = None
        summary_text = extract_article(entry["summary"]) if entry["summary"] else ""
        ev = Evidence(
            source_type=SourceType.NEWS,
            url=entry["url"],
            title=entry["title"],
            publisher=entry["publisher"] or urlparse(entry["url"]).netloc,
            published_at=published,
            clean_text=summary_text,
            gist=make_gist(summary_text or entry["title"]),
            authority=authority_for(entry["url"], entry["publisher"]),
        )
        evidence.append(ev)

    if fetch_bodies and evidence:
        await _hydrate_bodies(fetcher, evidence, min_body_words)

    for ev in evidence:
        ev.hash = content_hash(ev.clean_text or ev.title)
    return evidence


async def _hydrate_bodies(fetcher: Fetcher, evidence: list[Evidence], min_body_words: int) -> None:
    """Fetch and extract article bodies, keeping the RSS summary on failure.

    Many Google News links are redirect shims and many publishers paywall; both
    are expected. When the body is unusable we keep the summary, which is still
    a dated, attributable sentence or two.
    """
    results = await fetcher.get_many([e.url for e in evidence])
    for ev, result in zip(evidence, results, strict=True):
        if not result.ok:
            continue
        body = extract_article(result.body, ev.url)
        if not body or looks_like_stub(body, min_body_words):
            continue
        if word_count(body) > word_count(ev.clean_text):
            ev.clean_text = body
            ev.gist = make_gist(body)


async def search_news_sync_safe(*args: object, **kwargs: object) -> list[Evidence]:
    """Convenience wrapper so sync callers do not have to manage a loop."""
    return await search_news(*args, **kwargs)  # type: ignore[arg-type]


def run_search(**kwargs: object) -> list[Evidence]:
    fetcher = kwargs.pop("fetcher")
    return asyncio.run(search_news(fetcher, **kwargs))  # type: ignore[arg-type]
