"""Historical price backfill, 2019 onward.

Populates `prices` only. It does not write positions or P&L: the account did
not exist historically, and inventing position rows would put fiction in a
table the dashboard treats as observed fact. Synthetic historical positions
belong to Phase 3 replay, which keeps them clearly separate.

Run deliberately and commit as its own commit, apart from any daily heartbeat
commit (CLAUDE.md, "Committing data/risk.db").
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from src.data.bars import fetch_daily_closes
from src.data.calendar import most_recent_completed_session, trading_days_between
from src.db.connection import get_connection
from src.db.upserts import upsert_prices
from src.portfolio.spec import BACKFILL_START

logger = logging.getLogger(__name__)

# Alpaca caps the rows per response, so long ranges are chunked. A year per
# request keeps each response well inside the limit for 11 symbols.
CHUNK_DAYS = 365


def backfill_prices(start: str = BACKFILL_START, end: str | None = None) -> int:
    """Fetch and store daily closes over the range. Returns rows written.

    Idempotent: re-running overwrites the same primary keys rather than
    duplicating.
    """
    end_date = end or most_recent_completed_session()
    logger.info("Backfilling prices %s..%s", start, end_date)

    cursor = dt.date.fromisoformat(start)
    final = dt.date.fromisoformat(end_date)
    written = 0

    with get_connection() as conn:
        while cursor <= final:
            chunk_end = min(cursor + dt.timedelta(days=CHUNK_DAYS), final)
            rows = fetch_daily_closes(start=cursor, end=chunk_end)
            written += upsert_prices(conn, rows)
            logger.info("  %s..%s -> %d rows", cursor, chunk_end, len(rows))
            cursor = chunk_end + dt.timedelta(days=1)

    logger.info("Backfill complete: %d rows written.", written)
    return written


def report_coverage(start: str = BACKFILL_START, end: str | None = None) -> None:
    """Log any trading day whose panel is not complete for all 11 tickers.

    A ragged day would silently distort return series, so surface it here
    rather than discovering it during Phase 3 validation.
    """
    from src.portfolio.spec import SYMBOLS

    end_date = end or most_recent_completed_session()
    expected = len(SYMBOLS)

    with get_connection() as conn:
        counts = {
            row["trade_date"]: row["n"]
            for row in conn.execute(
                """
                SELECT trade_date, COUNT(*) AS n
                FROM prices
                WHERE trade_date BETWEEN ? AND ?
                GROUP BY trade_date
                """,
                (start, end_date),
            )
        }

    incomplete = [
        (day, counts.get(day, 0))
        for day in trading_days_between(start, end_date)
        if counts.get(day, 0) != expected
    ]

    if not incomplete:
        logger.info("Coverage complete: all sessions have %d symbols.", expected)
        return

    logger.warning("%d incomplete sessions:", len(incomplete))
    for day, n in incomplete[:20]:
        logger.warning("  %s: %d/%d symbols", day, n, expected)
    if len(incomplete) > 20:
        logger.warning("  ... and %d more", len(incomplete) - 20)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Backfill historical prices.")
    parser.add_argument("--start", default=BACKFILL_START)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report coverage gaps without fetching.",
    )
    args = parser.parse_args()

    if not args.check_only:
        backfill_prices(args.start, args.end)
    report_coverage(args.start, args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
