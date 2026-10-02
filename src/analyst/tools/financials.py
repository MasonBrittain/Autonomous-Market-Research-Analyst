"""Financial snapshot and peer metrics via yfinance.

The Snapshot section of the report is built entirely from these numbers with no
model involved. That is deliberate: an LLM adds nothing to "revenue was X" except
the possibility of getting it wrong.

yfinance scrapes a public endpoint and is therefore the flakiest dependency in
the project, so every accessor here is defensive and partial results are normal.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date
from typing import Any

from ..models import PeerMetric, Snapshot


def _num(value: Any) -> float | None:
    """Coerce to a finite float, rejecting NaN/inf that yfinance returns freely."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _ratio(numerator: Any, denominator: Any) -> float | None:
    n, d = _num(numerator), _num(denominator)
    if n is None or d in (None, 0):
        return None
    return round(n / d, 4)  # type: ignore[operator]


def _info(ticker: str) -> dict[str, Any]:
    try:
        import yfinance

        data = yfinance.Ticker(ticker).info
        return data if isinstance(data, dict) else {}
    except Exception:
        # Network failure, rate limit, delisted ticker, or an upstream layout
        # change -- all of which should degrade the snapshot, not fail the run.
        return {}


def snapshot_from_info(info: dict[str, Any]) -> Snapshot:
    revenue = _num(info.get("totalRevenue"))
    return Snapshot(
        as_of=date.today(),
        market_cap=_num(info.get("marketCap")),
        revenue_ttm=revenue,
        gross_margin=_num(info.get("grossMargins")),
        operating_margin=_num(info.get("operatingMargins")),
        net_margin=_num(info.get("profitMargins")),
        pe_ratio=_num(info.get("trailingPE")),
        debt_to_equity=_num(info.get("debtToEquity")),
        free_cash_flow=_num(info.get("freeCashflow")),
        employees=int(_num(info.get("fullTimeEmployees")) or 0) or None,
        price=_num(info.get("currentPrice")) or _num(info.get("regularMarketPrice")),
        price_change_52w=_ratio(
            (_num(info.get("currentPrice")) or 0) - (_num(info.get("fiftyTwoWeekLow")) or 0),
            _num(info.get("fiftyTwoWeekLow")),
        ),
        extras={
            k: v
            for k, v in {
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "country": info.get("country"),
                "summary": (info.get("longBusinessSummary") or "")[:800] or None,
                "fifty_two_week_high": _num(info.get("fiftyTwoWeekHigh")),
                "fifty_two_week_low": _num(info.get("fiftyTwoWeekLow")),
                "beta": _num(info.get("beta")),
                "dividend_yield": _num(info.get("dividendYield")),
            }.items()
            if v is not None
        },
    )


async def get_snapshot(ticker: str) -> tuple[Snapshot, dict[str, Any]]:
    """Fetch the snapshot off-thread; yfinance is synchronous and blocking."""
    info = await asyncio.to_thread(_info, ticker)
    return snapshot_from_info(info), info


def peer_metric_from_info(ticker: str, info: dict[str, Any]) -> PeerMetric:
    return PeerMetric(
        ticker=ticker,
        name=str(info.get("shortName") or info.get("longName") or ticker),
        market_cap=_num(info.get("marketCap")),
        revenue_ttm=_num(info.get("totalRevenue")),
        gross_margin=_num(info.get("grossMargins")),
        operating_margin=_num(info.get("operatingMargins")),
        pe_ratio=_num(info.get("trailingPE")),
    )


async def get_peer_metrics(tickers: list[str], limit: int = 6) -> list[PeerMetric]:
    """Enrich a candidate peer list, keeping the largest by market cap.

    Peers arrive from EDGAR's SIC classification, which is broad -- a SIC code can
    hold both a $3T company and a shell. Ranking by market cap and truncating
    keeps the comparison meaningful.
    """
    if not tickers:
        return []
    infos = await asyncio.gather(*(asyncio.to_thread(_info, t) for t in tickers[: limit * 3]))
    metrics = [
        peer_metric_from_info(t, info)
        for t, info in zip(tickers[: limit * 3], infos, strict=True)
        if info
    ]
    metrics = [m for m in metrics if m.market_cap]
    metrics.sort(key=lambda m: -(m.market_cap or 0))
    return metrics[:limit]


def ticker_for_cik(cik: str, index: Any) -> str | None:
    """Map a CIK back to a ticker using the already-loaded SEC index."""
    target = cik.zfill(10)
    for row in getattr(index, "rows", []):
        if row.get("cik") == target and row.get("ticker"):
            return row["ticker"]
    return None


def format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    abs_v = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs_v >= threshold:
            return f"${value / threshold:,.2f}{suffix}"
    return f"${value:,.0f}"


def format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
