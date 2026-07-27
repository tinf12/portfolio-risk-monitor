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
from src.db.upserts import PriceWriteResult, insert_prices
from src.portfolio.spec import BACKFILL_START

logger = logging.getLogger(__name__)

# Alpaca caps the rows per response, so long ranges are chunked. A year per
# request keeps each response well inside the limit for 11 symbols.
CHUNK_DAYS = 365


def backfill_prices(
    start: str = BACKFILL_START,
    end: str | None = None,
    *,
    allow_restate: bool = False,
) -> PriceWriteResult:
    """Fetch and store daily closes over the range.

    Safe to re-run: `prices` is write-once, so an existing close is never
    changed. Re-running fills gaps and reports any date where the vendor now
    reports a different close than the one stored.

    `allow_restate=True` applies those differences. Use it only deliberately —
    it rewrites history and invalidates the reproducibility of every risk
    figure already computed from the affected dates.
    """
    end_date = end or most_recent_completed_session()
    logger.info("Backfilling prices %s..%s", start, end_date)

    cursor = dt.date.fromisoformat(start)
    final = dt.date.fromisoformat(end_date)
    total = PriceWriteResult()

    with get_connection() as conn:
        while cursor <= final:
            chunk_end = min(cursor + dt.timedelta(days=CHUNK_DAYS), final)
            rows = fetch_daily_closes(start=cursor, end=chunk_end)
            chunk = insert_prices(conn, rows, allow_restate=allow_restate)

            total.inserted += chunk.inserted
            total.unchanged += chunk.unchanged
            total.restatements.extend(chunk.restatements)
            total.applied_restatements += chunk.applied_restatements

            logger.info(
                "  %s..%s -> %d new, %d unchanged, %d restated",
                cursor,
                chunk_end,
                chunk.inserted,
                chunk.unchanged,
                len(chunk.restatements),
            )
            cursor = chunk_end + dt.timedelta(days=1)

    _log_restatements(total, allow_restate=allow_restate)
    logger.info(
        "Backfill complete: %d inserted, %d unchanged.",
        total.inserted,
        total.unchanged,
    )
    return total


def _log_restatements(result: PriceWriteResult, *, allow_restate: bool) -> None:
    """Report vendor drift against stored history.

    Expected over time: adjusted closes are restated downward by every
    subsequent distribution. Surfaced rather than ignored so the divergence
    between the database and the vendor is never invisible.
    """
    if not result.has_restatements:
        return

    verb = "APPLIED" if allow_restate else "REJECTED"
    logger.warning(
        "%s %d restatement(s): the vendor now reports different closes for "
        "dates already stored.",
        verb,
        len(result.restatements),
    )
    for trade_date, symbol, stored, incoming in result.restatements[:10]:
        drift = (incoming / stored - 1) * 100 if stored else float("nan")
        logger.warning(
            "  %s %-5s stored=%.4f incoming=%.4f (%+.2f%%)",
            trade_date,
            symbol,
            stored,
            incoming,
            drift,
        )
    if len(result.restatements) > 10:
        logger.warning("  ... and %d more", len(result.restatements) - 10)

    if not allow_restate:
        logger.warning(
            "Stored values kept. Re-run with --allow-restate to overwrite, "
            "which invalidates reproducibility of figures already computed "
            "from these dates."
        )


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
    parser.add_argument(
        "--allow-restate",
        action="store_true",
        help=(
            "Overwrite stored closes when the vendor now reports different "
            "values. Rewrites history; invalidates reproducibility of figures "
            "already computed from the affected dates."
        ),
    )
    args = parser.parse_args()

    if not args.check_only:
        backfill_prices(args.start, args.end, allow_restate=args.allow_restate)
    report_coverage(args.start, args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
