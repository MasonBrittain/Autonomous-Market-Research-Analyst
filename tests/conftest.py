"""Test fixtures.

The important one is `FakeFetcher`, which pattern-matches URLs and serves canned
documents. It duck-types the real `Fetcher` surface (`get`, `get_many`, `stats`)
so the adapters, Scout's loop, and the full pipeline all run against it unchanged
-- no network, no API key, no mocking of our own internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from analyst.models import (
    Evidence,
    Fact,
    Polarity,
    ResearchRun,
    RunConfig,
    SourceType,
    content_hash,
)
from analyst.tools.fetch import FetchResult

TICKER_INDEX = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1061219, "ticker": "APLE", "title": "Apple Hospitality REIT, Inc."},
    "2": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    "3": {"cik_str": 1551152, "ticker": "ABBV", "title": "AbbVie Inc."},
    "4": {"cik_str": 97745, "ticker": "TMO", "title": "Thermo Fisher Scientific Inc."},
}

RISK_BODY = (
    "Our business depends on a small number of contract manufacturers concentrated in a "
    "single geographic region, and disruption at any one facility could materially reduce "
    "our ability to meet demand. We have experienced component shortages in prior periods "
    "and expect volatility to continue. " * 4
)
MDA_BODY = (
    "Revenue for fiscal 2026 increased 11% to $4.3 billion, driven by volume growth in the "
    "industrial segment and price realization of approximately 300 basis points. Gross margin "
    "expanded 180 basis points to 46.2%. Operating expenses grew more slowly than revenue. "
    "Free cash flow of $610 million funded $240 million of buybacks and reduced net leverage "
    "to 2.1 times trailing earnings. We expect full-year revenue growth of 9% to 11%. " * 8
)

TENK_HTML = f"""<html><body>
<p>TABLE OF CONTENTS</p>
<p>Item 1A. Risk Factors .... 12</p>
<p>Item 1B. Unresolved Staff Comments .... 30</p>
<p>Item 7. Management's Discussion and Analysis .... 44</p>
<p>Item 7A. Quantitative and Qualitative Disclosures .... 70</p>
<h2>Item 1A. Risk Factors</h2>
<p>{RISK_BODY}</p>
<p>Competition from larger and better capitalized rivals may compress our pricing and
reduce our share of the industrial coatings market over time. {"Competitive pressure persists. " * 20}</p>
<h2>Item 1B. Unresolved Staff Comments</h2><p>None.</p>
<h2>Item 7. Management's Discussion and Analysis</h2>
<p>{MDA_BODY}</p>
<h2>Item 7A. Quantitative and Qualitative Disclosures</h2><p>Interest rate risk.</p>
</body></html>"""

ARTICLE_BODY = (
    "Acme Corporation said on Tuesday that quarterly revenue rose 12 percent to $4.3 billion, "
    "beating analyst expectations, as demand for its industrial coatings business accelerated "
    "in North America. Chief Executive Dana Reyes said operating margins improved 180 basis "
    "points year over year, helped by price increases that took effect in March. The company "
    "raised its full-year guidance and now expects revenue growth of 9 to 11 percent. Shares "
    "rose 6 percent in after-hours trading. Analysts at three banks lifted their price targets "
    "following the report, citing better than expected free cash flow and a reduction in net "
    "leverage to 2.1 times trailing earnings. The company also disclosed that a federal "
    "regulator had opened an inquiry into its supplier certification process, which it said it "
    "was cooperating with fully. "
)


def _article_html(headline: str, extra: str = "") -> str:
    return (
        f"<html><head><title>{headline}</title></head><body><nav>Menu Home Markets</nav>"
        f"<article><h1>{headline}</h1><p>{ARTICLE_BODY}</p><p>{extra}</p>"
        f"<p>{ARTICLE_BODY}</p></article><footer>Subscribe to our newsletter</footer></body></html>"
    )


def _rss(entries: list[tuple[str, str, str, int]]) -> str:
    items = []
    for title, link, publisher, days_ago in entries:
        pub = (datetime.now(UTC) - timedelta(days=days_ago)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        items.append(
            f"<item><title>{title}</title><link>{link}</link>"
            f"<pubDate>{pub}</pubDate>"
            f"<source url='http://{publisher}'>{publisher}</source>"
            f"<description>{ARTICLE_BODY[:200]}</description></item>"
        )
    return (
        "<?xml version='1.0'?><rss version='2.0'><channel><title>News</title>"
        + "".join(items)
        + "</channel></rss>"
    )


FEED_ENTRIES = [
    ("Acme beats estimates and raises guidance", "https://reuters.com/a1", "reuters.com", 3),
    ("Acme margins expand on March price increases", "https://cnbc.com/a2", "cnbc.com", 8),
    (
        "Regulator opens inquiry into Acme supplier certification",
        "https://wsj.com/a3",
        "wsj.com",
        14,
    ),
    ("Acme names new chief operating officer", "https://apnews.com/a4", "apnews.com", 21),
    ("Acme beats estimates and raises guidance", "https://fool.com/a5", "fool.com", 4),
]

SUBMISSIONS = {
    "name": "Acme Corporation",
    "sic": "2851",
    "sicDescription": "Paints, Varnishes, Lacquers, Enamels & Allied Products",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-26-000010", "0000320193-26-000008"],
            "form": ["10-K", "8-K"],
            "filingDate": ["2026-02-12", "2026-08-04"],
            "reportDate": ["2026-01-31", "2026-08-04"],
            "primaryDocument": ["acme-10k.htm", "acme-8k.htm"],
            "primaryDocDescription": ["10-K", "Results of Operations"],
        }
    },
}

PEER_ATOM = (
    "<?xml version='1.0'?><feed><entry><link href='/cgi-bin/browse-edgar?action=getcompany&CIK=0000789019'/>"
    "</entry><entry><link href='/cgi-bin/browse-edgar?action=getcompany&CIK=0001551152'/></entry>"
    "<entry><link href='/cgi-bin/browse-edgar?action=getcompany&CIK=0000097745'/></entry></feed>"
)

YF_INFO = {
    "shortName": "Acme Corporation",
    "longName": "Acme Corporation",
    "marketCap": 52_400_000_000,
    "totalRevenue": 4_300_000_000,
    "grossMargins": 0.462,
    "operatingMargins": 0.183,
    "profitMargins": 0.121,
    "trailingPE": 24.6,
    "debtToEquity": 88.4,
    "freeCashflow": 610_000_000,
    "fullTimeEmployees": 14_200,
    "currentPrice": 212.4,
    "fiftyTwoWeekLow": 168.0,
    "fiftyTwoWeekHigh": 229.5,
    "sector": "Basic Materials",
    "industry": "Specialty Chemicals",
    "exchange": "NYQ",
    "longBusinessSummary": "Acme Corporation manufactures industrial coatings and sealants.",
}


@dataclass
class FakeFetcher:
    """Serves fixtures by URL pattern. Duck-types the real Fetcher."""

    offline: bool = False
    calls: list[str] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    async def get(
        self, url: str, *, headers=None, use_cache=True, check_robots=True
    ) -> FetchResult:  # noqa: ANN001, ARG002
        assert self.calls is not None
        self.calls.append(url)
        body = self._body_for(url)
        if body is None:
            return FetchResult(url=url, status=404, body="", error="fixture miss")
        return FetchResult(url=url, status=200, body=body)

    async def get_many(
        self, urls: list[str], *, headers=None, check_robots=True
    ) -> list[FetchResult]:  # noqa: ANN001, ARG002
        return [await self.get(u) for u in urls]

    def stats(self) -> dict[str, object]:
        assert self.calls is not None
        return {"cache": {"hits": 0, "misses": len(self.calls)}, "hosts_seen": 1}

    def _body_for(self, url: str) -> str | None:
        if "company_tickers.json" in url:
            return json.dumps(TICKER_INDEX)
        if "data.sec.gov/submissions" in url:
            return json.dumps(SUBMISSIONS)
        if "browse-edgar" in url:
            return PEER_ATOM
        if "/Archives/" in url:
            return TENK_HTML
        if "news.google.com/rss" in url or "feeds.finance.yahoo.com" in url:
            return _rss(FEED_ENTRIES)
        for title, link, _publisher, _days in FEED_ENTRIES:
            if url == link:
                # a5 duplicates a1 verbatim, which exercises deduplication.
                return _article_html(title, extra="" if link.endswith("a5") else url)
        return None


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point caches, runs db and output at a temp dir; clear the settings cache.

    Absolute paths in these env vars win over the project root, because
    `Path("C:/repo") / "C:/tmp/x"` resolves to the absolute right-hand side.
    """
    from analyst import config, prompts

    monkeypatch.setenv("ANALYST_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ANALYST_RUNS_DB", str(tmp_path / "runs.sqlite3"))
    monkeypatch.setenv("ANALYST_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config.settings.cache_clear()
    prompts.load_prompt.cache_clear()
    yield config.settings()
    config.settings.cache_clear()
    prompts.load_prompt.cache_clear()


@pytest.fixture
def fake_fetcher() -> FakeFetcher:
    return FakeFetcher()


@pytest.fixture(autouse=True)
def no_yfinance(monkeypatch):
    """yfinance is the one dependency that would reach the network in tests."""
    from analyst.tools import financials

    monkeypatch.setattr(financials, "_info", lambda ticker: dict(YF_INFO))


@pytest.fixture
def sample_run() -> ResearchRun:
    """A fully-populated run with no model or network involvement.

    This is the Phase 0 artifact: it proves the state model and the renderer work
    before a single API call exists.
    """
    from analyst.models import (
        AdversaryVerdict,
        Claim,
        Entity,
        Quadrant,
        RiskStatus,
        Section,
        Snapshot,
        StatedRisk,
        UsageRecord,
        Verdict,
    )

    run = ResearchRun(query="Acme Corporation", config=RunConfig(stub=True))
    run.entity = Entity(
        query="Acme Corporation",
        name="Acme Corporation",
        ticker="ACME",
        cik="0000320193",
        aliases=["Acme"],
        industry="Specialty Chemicals",
        confidence=0.97,
    )
    now = datetime.now(UTC)
    ev1 = Evidence(
        source_type=SourceType.NEWS,
        url="https://reuters.com/a1",
        title="Acme beats estimates and raises guidance",
        publisher="reuters.com",
        published_at=now - timedelta(days=3),
        clean_text=ARTICLE_BODY,
        hash=content_hash(ARTICLE_BODY),
        authority=0.95,
        relevant=True,
    )
    ev2 = Evidence(
        source_type=SourceType.FILING_10K,
        url="https://sec.gov/Archives/acme-10k.htm",
        title="Acme 10-K Item 1A Risk Factors (2026-02-12)",
        publisher="SEC EDGAR",
        published_at=now - timedelta(days=200),
        clean_text=RISK_BODY,
        hash=content_hash(RISK_BODY),
        authority=1.0,
        relevant=True,
    )
    run.evidence = [ev1, ev2]

    f1 = Fact(
        evidence_id=ev1.id,
        text="Quarterly revenue rose 12% to $4.3 billion, beating expectations.",
        verbatim_quote="quarterly revenue rose 12 percent to $4.3 billion",
        happened_at=(now - timedelta(days=3)).date(),
        polarity=Polarity.POSITIVE,
        salience=0.9,
    )
    f2 = Fact(
        evidence_id=ev1.id,
        text="A federal regulator opened an inquiry into supplier certification.",
        verbatim_quote="a federal regulator had opened an inquiry into its supplier certification process",
        happened_at=(now - timedelta(days=14)).date(),
        polarity=Polarity.NEGATIVE,
        salience=0.8,
    )
    f3 = Fact(
        evidence_id=ev2.id,
        text="Contract manufacturing is concentrated in a single geographic region.",
        verbatim_quote="concentrated in a single geographic region",
        polarity=Polarity.NEGATIVE,
        salience=0.7,
    )
    run.facts = [f1, f2, f3]

    accept = AdversaryVerdict(
        verdict=Verdict.ACCEPT,
        supported=True,
        correct_section=True,
        specific=True,
        reasoning="Directly supported by the cited figure.",
    )
    run.claims = [
        Claim(
            section=Section.SWOT,
            quadrant=Quadrant.STRENGTH,
            statement="Revenue growth of 12% with 180bps of margin expansion indicates pricing power.",
            rationale="Price increases flowed through without volume loss.",
            confidence=0.8,
            evidence_ids=[ev1.id],
            fact_ids=[f1.id],
            verdict=accept,
        ),
        Claim(
            section=Section.SWOT,
            quadrant=Quadrant.WEAKNESS,
            statement="Contract manufacturing concentrated in one region is a single point of failure.",
            confidence=0.65,
            evidence_ids=[ev2.id],
            fact_ids=[f3.id],
            verdict=accept,
        ),
        Claim(
            section=Section.SWOT,
            quadrant=Quadrant.THREAT,
            statement="The federal inquiry into supplier certification could restrict sourcing.",
            confidence=0.55,
            evidence_ids=[ev1.id],
            fact_ids=[f2.id],
            verdict=AdversaryVerdict(
                verdict=Verdict.ACCEPT,
                supported=True,
                correct_section=True,
                specific=True,
                contradicting_fact_ids=[f1.id],
                reasoning="Supported, but note the same quarter showed record results.",
            ),
        ),
        Claim(
            section=Section.SWOT,
            quadrant=Quadrant.STRENGTH,
            statement="Acme has a strong brand and experienced management.",
            confidence=0.4,
            evidence_ids=[ev1.id],
            fact_ids=[f1.id],
            verdict=AdversaryVerdict(
                verdict=Verdict.REJECT,
                supported=False,
                correct_section=True,
                specific=False,
                reasoning="Generic filler; the cited fact says nothing about brand.",
            ),
        ),
        Claim(
            section=Section.CATALYSTS,
            statement="Full-year guidance raised to 9-11% revenue growth.",
            confidence=0.85,
            evidence_ids=[ev1.id],
            fact_ids=[f1.id],
            verdict=accept,
        ),
    ]
    run.stated_risks = [
        StatedRisk(
            risk_text=RISK_BODY,
            summary="Manufacturing is concentrated with few contract partners in one region.",
            status=RiskStatus.MATERIALIZING,
            fact_ids=[f3.id],
            reasoning="The 10-K discloses it and no mitigation has been reported.",
        ),
        StatedRisk(
            risk_text="Competition may compress pricing.",
            summary="Larger rivals may compress pricing.",
            status=RiskStatus.CONTRADICTED,
            fact_ids=[f1.id],
            reasoning="Price increases held, suggesting pricing power rather than compression.",
        ),
    ]
    run.snapshot = Snapshot(
        market_cap=52_400_000_000,
        revenue_ttm=4_300_000_000,
        gross_margin=0.462,
        operating_margin=0.183,
        net_margin=0.121,
        pe_ratio=24.6,
        employees=14_200,
    )
    run.open_questions = [
        "No evidence quantifies customer concentration, though the 10-K flags it.",
        "The regulatory inquiry's scope is undisclosed.",
    ]
    run.ledger.add(
        UsageRecord(
            node="analyst.swot",
            model="claude-opus-5",
            input_tokens=12_000,
            output_tokens=900,
            cache_read_tokens=38_000,
        )
    )
    return run
