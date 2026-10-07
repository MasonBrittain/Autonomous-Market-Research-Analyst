"""Adapter and utility tests. All offline."""

from __future__ import annotations

import json

import pytest

from analyst.cache import FetchCache
from analyst.tools import edgar, financials, news
from analyst.tools.entity import (
    CompanyIndex,
    derive_aliases,
    looks_like_industry,
    normalize_name,
    resolve_from_index,
)
from analyst.tools.textutil import html_to_text, looks_like_stub, make_gist
from tests.conftest import SUBMISSIONS, TENK_HTML, TICKER_INDEX, YF_INFO

# --------------------------------------------------------------------------- #
# Entity resolution
# --------------------------------------------------------------------------- #


@pytest.fixture
def index() -> CompanyIndex:
    return CompanyIndex.from_payload(TICKER_INDEX)


def test_normalize_name_strips_legal_forms():
    assert normalize_name("Apple Inc.") == "apple"
    assert normalize_name("Microsoft Corporation") == "microsoft"
    assert normalize_name("The Kroger Co.") == "kroger"
    assert normalize_name("Acme Holdings Group Ltd") == "acme"


def test_normalize_name_never_returns_empty_for_real_input():
    """A name made entirely of noise words must not normalize to nothing."""
    assert normalize_name("The Company Inc") != ""


def test_bare_ticker_resolves_with_full_confidence(index):
    entity = resolve_from_index("AAPL", index)
    assert entity.ticker == "AAPL"
    assert entity.cik == "0000320193"
    assert entity.confidence == 1.0
    assert not entity.needs_clarification


def test_close_name_match_resolves(index):
    entity = resolve_from_index("Microsoft", index)
    assert entity.ticker == "MSFT"
    assert entity.confidence > 0.9


def test_prefix_collision_prefers_whole_name_match(index):
    """'Apple' must resolve to Apple Inc., not Apple Hospitality REIT."""
    entity = resolve_from_index("Apple", index)
    assert entity.ticker == "AAPL"
    # ...but the other candidate is still offered.
    assert any(c.ticker == "APLE" for c in entity.candidates)


def test_unknown_company_requires_clarification(index):
    entity = resolve_from_index("Totally Fictional Widgets", index)
    assert entity.confidence < 0.6
    assert entity.needs_clarification


def test_industry_query_is_detected_not_resolved(index):
    entity = resolve_from_index("semiconductor industry", index)
    assert entity.is_industry
    assert not entity.needs_clarification
    assert entity.ticker is None


def test_industry_markers():
    assert looks_like_industry("airlines industry")
    assert looks_like_industry("cybersecurity sector")
    assert not looks_like_industry("Apple Inc")


def test_derive_aliases_produces_short_forms():
    aliases = derive_aliases("Thermo Fisher Scientific Inc.")
    assert any("Thermo" in a for a in aliases)


def test_search_terms_deduplicates(index):
    entity = resolve_from_index("AAPL", index)
    terms = entity.search_terms()
    assert len(terms) == len({t.lower() for t in terms})


# --------------------------------------------------------------------------- #
# EDGAR
# --------------------------------------------------------------------------- #


def test_recent_filings_flattens_column_arrays():
    rows = edgar.recent_filings(SUBMISSIONS)
    assert len(rows) == 2
    assert rows[0]["form"] == "10-K"
    assert rows[0]["document"] == "acme-10k.htm"


def test_recent_filings_filters_by_form():
    rows = edgar.recent_filings(SUBMISSIONS, forms=("8-K",))
    assert [r["form"] for r in rows] == ["8-K"]


def test_document_url_strips_cik_zeros_and_accession_dashes():
    url = edgar.document_url("0000320193", "0000320193-26-000010", "acme-10k.htm")
    assert "/data/320193/" in url
    assert "000032019326000010" in url


def test_extract_item_skips_table_of_contents():
    """Every 10-K names Item 1A twice; the TOC entry must not win."""
    text = html_to_text(TENK_HTML)
    item = edgar.extract_item(text, "1a", ("1b", "2"))
    assert len(item) > 1500
    assert "contract manufacturers" in item
    # The MD&A must not bleed in.
    assert "Revenue for fiscal 2026" not in item


def test_extract_item_returns_empty_when_too_short():
    assert edgar.extract_item("Item 1A. Risk Factors\nnone\nItem 1B.", "1a", ("1b",)) == ""


def test_extract_mda():
    text = html_to_text(TENK_HTML)
    item = edgar.extract_item(text, "7", ("7a", "8"))
    assert "Revenue for fiscal 2026" in item


def test_split_risk_factors_produces_chunks():
    text = html_to_text(TENK_HTML)
    item = edgar.extract_item(text, "1a", ("1b", "2"))
    chunks = edgar.split_risk_factors(item)
    assert chunks
    assert all(len(c) > 100 for c in chunks)


async def test_fetch_filings_returns_evidence_and_risks(fake_fetcher):
    evidence, risks = await edgar.fetch_filings(fake_fetcher, "0000320193", "Acme Corporation")
    assert risks
    kinds = {e.source_type.value for e in evidence}
    assert "filing_10k" in kinds
    assert all(e.authority == 1.0 for e in evidence if e.publisher == "SEC EDGAR")


async def test_peers_by_sic_extracts_ciks(fake_fetcher):
    ciks = await edgar.peers_by_sic(fake_fetcher, "2851", exclude_cik="0000320193")
    assert "0000789019" in ciks
    assert "0000320193" not in ciks


async def test_fetch_filings_handles_missing_submissions(fake_fetcher):
    """An unknown CIK degrades to empty, it does not raise."""
    evidence, risks = await edgar.fetch_filings(fake_fetcher, "9999999999")
    assert evidence == [] or isinstance(evidence, list)
    assert isinstance(risks, list)


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #


def test_build_queries_applies_the_time_window():
    urls = news.build_queries(["Acme Corp"], ["Acme lawsuit"], lookback_days=30)
    assert all("when%3A30d" in u for u in urls)
    assert len(urls) == 2


def test_authority_scores_wire_services_above_blogs():
    assert news.authority_for("https://reuters.com/x") > news.authority_for("https://fool.com/x")
    assert news.authority_for("https://unknown-site.example/x") == 0.45


async def test_search_news_parses_feed_and_hydrates_bodies(fake_fetcher):
    evidence = await news.search_news(
        fake_fetcher, ["Acme Corporation"], ticker="ACME", lookback_days=120, limit=10
    )
    assert evidence
    assert all(e.url for e in evidence)
    assert all(e.hash for e in evidence)
    # Bodies were hydrated beyond the RSS summary.
    assert any(len(e.clean_text.split()) > 150 for e in evidence)
    assert any(e.published_at is not None for e in evidence)


async def test_search_news_respects_lookback_cutoff(fake_fetcher):
    evidence = await news.search_news(fake_fetcher, ["Acme Corporation"], lookback_days=5, limit=10)
    # Fixtures are 3, 4, 8, 14 and 21 days old; only the first two qualify.
    assert len(evidence) <= 2


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #


def test_html_to_text_preserves_block_structure():
    text = html_to_text("<p>One</p><p>Two</p><script>bad()</script>")
    assert "One" in text and "Two" in text
    assert "bad()" not in text


def test_looks_like_stub_catches_paywalls():
    assert looks_like_stub("Subscribe to continue reading this article.", min_words=10)
    assert looks_like_stub("short", min_words=50)
    assert not looks_like_stub("word " * 300, min_words=50)


def test_make_gist_prefers_sentence_boundaries():
    gist = make_gist("First sentence here. Second sentence follows and is long enough.", limit=30)
    assert gist.endswith(".") or gist.endswith("...")


# --------------------------------------------------------------------------- #
# Financials
# --------------------------------------------------------------------------- #


def test_snapshot_from_info_coerces_and_filters():
    snap = financials.snapshot_from_info(YF_INFO)
    assert snap.market_cap == 52_400_000_000
    assert snap.gross_margin == pytest.approx(0.462)
    assert snap.extras["industry"] == "Specialty Chemicals"


def test_snapshot_rejects_nan_and_missing():
    snap = financials.snapshot_from_info({"marketCap": float("nan"), "trailingPE": None})
    assert snap.market_cap is None
    assert snap.pe_ratio is None


def test_snapshot_from_empty_info_is_safe():
    snap = financials.snapshot_from_info({})
    assert snap.market_cap is None
    assert snap.extras == {}


def test_format_helpers():
    assert financials.format_money(52_400_000_000) == "$52.40B"
    assert financials.format_money(None) == "n/a"
    assert financials.format_pct(0.4623) == "46.2%"
    assert financials.format_pct(None) == "n/a"


def test_ticker_for_cik_uses_the_index(index):
    assert financials.ticker_for_cik("0000789019", index) == "MSFT"
    assert financials.ticker_for_cik("0000000001", index) is None


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_cache_round_trips_and_tracks_hit_rate(tmp_path):
    cache = FetchCache(root=tmp_path / "c")
    assert cache.get("https://example.com/a") is None
    cache.put("https://example.com/a", "<html>body</html>")
    payload = cache.get("https://example.com/a")
    assert payload is not None and payload["body"] == "<html>body</html>"
    assert cache.hits == 1 and cache.misses == 1
    assert cache.hit_rate == 0.5


def test_cache_shards_by_hash_prefix(tmp_path):
    cache = FetchCache(root=tmp_path / "c")
    cache.put("https://example.com/a", "x")
    files = list((tmp_path / "c").rglob("*.json"))
    assert len(files) == 1
    assert len(files[0].parent.name) == 2


def test_cache_survives_corrupt_entry(tmp_path):
    cache = FetchCache(root=tmp_path / "c")
    cache.put("https://example.com/a", "x")
    path = next((tmp_path / "c").rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("https://example.com/a") is None


def test_cache_ttl_expires(tmp_path):
    cache = FetchCache(root=tmp_path / "c", ttl_seconds=0)
    cache.put("https://example.com/a", "x")
    assert cache.get("https://example.com/a") is None


def test_company_index_handles_both_payload_shapes():
    as_dict = CompanyIndex.from_payload(TICKER_INDEX)
    as_list = CompanyIndex.from_payload(list(TICKER_INDEX.values()))
    assert len(as_dict) == len(as_list) == len(TICKER_INDEX)
    assert json.loads(json.dumps(as_dict.rows))[0]["ticker"] == "AAPL"


def test_extract_item_finds_a_heading_split_by_inline_markup():
    """Regression: Microsoft's FY2026 10-K styles the heading letter by letter, so
    flattened HTML reads "RIS K FACTORS". The strict pattern matched only the table
    of contents and the whole stated-risks section silently disappeared."""
    body = (
        "Our operations and financial results are subject to various risks and "
        "uncertainties that could adversely affect our business. "
    ) * 20
    html = (
        "<p>Item 1A.</p><p>Risk Factors</p><p>14</p>"
        "<p>Item 1B.</p><p>Unresolved Staff Comments</p><p>29</p>"
        f"<p>ITEM 1A. RIS<span style='letter-spacing:1px'>K</span> FACTORS</p><p>{body}</p>"
        "<p>ITEM 1B. UNRESOLVED STAFF COMMENTS</p><p>None.</p>"
    )
    text = html_to_text(html)
    assert "RIS K FACTORS" in text, "fixture no longer reproduces the split heading"

    item = edgar.extract_item(text, "1a", ("1b", "2"))
    assert "subject to various risks" in item
    assert "Unresolved" not in item
    assert edgar.split_risk_factors(item)


def test_loose_heading_patterns_stay_anchored_to_the_item_number():
    """Tolerating split letters must not let one item's heading match another."""
    text = "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES\n" + ("Rates. " * 300)
    assert edgar.extract_item(text, "7", ("8",)) == "", "Item 7A was mistaken for Item 7"
