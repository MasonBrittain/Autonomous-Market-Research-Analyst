"""Text extraction helpers shared by the adapters.

Two different jobs live here because they need different tools: news pages need
boilerplate stripping (nav, cookie banners, related-articles rails), which
trafilatura does well; SEC filings need plain tag-stripping that preserves
document order, because the "boilerplate" in a 10-K is the content.
"""

from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")
_BLOCK_END = re.compile(r"</(p|div|tr|li|h[1-6]|table|section)>", re.IGNORECASE)


def html_to_text(raw: str) -> str:
    """Flatten HTML to text while keeping block structure as newlines."""
    if not raw:
        return ""
    text = _SCRIPT.sub(" ", raw)
    text = _BLOCK_END.sub("\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def extract_article(raw_html: str, url: str = "") -> str:
    """Pull the article body out of a news page, discarding chrome."""
    if not raw_html:
        return ""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            raw_html,
            url=url or None,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted and len(extracted.split()) >= 40:
            return extracted.strip()
    except Exception:
        # trafilatura is best-effort; a parse failure should degrade, not crash.
        pass
    return html_to_text(raw_html)


def word_count(text: str) -> int:
    return len(text.split())


def make_gist(text: str, limit: int = 240) -> str:
    """First meaningful sentences, used as the metadata-only view for Scout."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary > limit * 0.5:
        return cut[: boundary + 1]
    return cut.rsplit(" ", 1)[0] + "..."


def looks_like_stub(text: str, min_words: int = 200) -> bool:
    """Detect paywall interstitials and teaser pages.

    These are worse than useless: a paywall page is on-topic enough to pass a
    relevance check while containing no facts, so it burns extraction budget and
    pollutes the fact pack.
    """
    if word_count(text) < min_words:
        return True
    lowered = text.lower()
    markers = (
        "subscribe to continue",
        "subscribe now to read",
        "already a subscriber",
        "sign in to read",
        "this article is for subscribers",
        "enable javascript",
        "you have reached your article limit",
        "create a free account to read",
    )
    head = lowered[:1200]
    return any(m in head for m in markers)


# Inline XBRL is embedded in every modern SEC filing's primary document. Flattened
# to text it becomes long runs of identifiers and dates -- "aapl-20260730 false
# 0000320193 us-gaap:CommonStockMember ..." -- which read as on-topic to a relevance
# check while containing no facts. Stripping it is what keeps 8-K noise out of the
# fact pack.
_XBRL_TOKEN = re.compile(r"[:_]|\d")


def _xbrl_density(line: str) -> float:
    tokens = line.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if _XBRL_TOKEN.search(t))
    return hits / len(tokens)


def strip_xbrl_noise(text: str, *, min_tokens: int = 6, density: float = 0.45) -> str:
    """Drop lines dominated by XBRL identifiers, keeping prose."""
    kept: list[str] = []
    for line in text.split("\n"):
        tokens = line.split()
        if len(tokens) >= min_tokens and _xbrl_density(line) >= density:
            continue
        kept.append(line)
    return _BLANKS.sub("\n\n", "\n".join(kept)).strip()


def prose_word_count(text: str) -> int:
    """Words outside XBRL-looking lines -- the real content length of a filing."""
    return word_count(strip_xbrl_noise(text))
