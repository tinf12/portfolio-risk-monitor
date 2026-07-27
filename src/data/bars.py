"""Daily bar fetching.

Requests always end at a completed session. The Alpaca free tier rejects
queries touching the last 15 minutes of SIP data, so asking for today's bar
during market hours fails outright (CLAUDE.md, "Data constraints").
"""

from __future__ import annotations

import datetime as dt
import logging

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.data.alpaca_client import get_data_client, with_retry
from src.portfolio.spec import SYMBOLS

logger = logging.getLogger(__name__)

PriceRow = tuple[str, str, float]


def fetch_daily_closes(
    start: str | dt.date,
    end: str | dt.date,
    symbols: tuple[str, ...] = SYMBOLS,
) -> list[PriceRow]:
    """Fetch daily closing prices, inclusive of both endpoints.

    Returns rows of (trade_date, symbol, close) sorted by date then symbol,
    ready for `upsert_prices`. Sorting is for deterministic output and stable
    diffs, not correctness.

    Uses the SIP feed for full-market consolidated prices. IEX is roughly 2% of
    volume and its closes are not representative.
    """
    start_date = dt.date.fromisoformat(start) if isinstance(start, str) else start
    end_date = dt.date.fromisoformat(end) if isinstance(end, str) else end

    if start_date > end_date:
        raise ValueError(f"start {start_date} is after end {end_date}")

    client = get_data_client()
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.UTC),
        end=dt.datetime.combine(end_date, dt.time.max, tzinfo=dt.UTC),
        feed=DataFeed.SIP,
        adjustment="all",
    )

    bar_set = with_retry(
        lambda: client.get_stock_bars(request),
        description=f"fetch bars {start_date}..{end_date}",
    )

    rows: list[PriceRow] = []
    for symbol, bars in bar_set.data.items():
        for bar in bars:
            rows.append((bar.timestamp.date().isoformat(), symbol, float(bar.close)))

    rows.sort(key=lambda r: (r[0], r[1]))
    logger.info(
        "Fetched %d bars for %d symbols, %s..%s",
        len(rows),
        len(symbols),
        start_date,
        end_date,
    )
    return rows


def missing_symbols(rows: list[PriceRow], trade_date: str) -> set[str]:
    """Return spec symbols with no price row on `trade_date`.

    A non-empty result on a trading day means the panel is ragged and the day
    must not be treated as complete.
    """
    present = {symbol for date, symbol, _ in rows if date == trade_date}
    return set(SYMBOLS) - present
