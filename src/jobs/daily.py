"""The daily run.

Order of operations, and why:

1. Resolve the target session — the last *completed* one, never today's.
2. Fetch and store closes for all 11 tickers.
3. Snapshot account and positions, store them, compute P&L.
4. Rebalance if the target session is the month's first.
5. Record a heartbeat row either way.

Step 5 runs even on failure. GitHub does not notify on a failed scheduled
workflow, so an absent or failed `runs` row is the only failure signal
(CLAUDE.md, "GitHub Actions").

Risk metrics are deliberately not computed here yet — that is Phase 2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from src.data.bars import fetch_daily_closes, missing_symbols
from src.data.calendar import (
    is_first_trading_day_of_month,
    most_recent_completed_session,
)
from src.db.connection import get_connection
from src.db.upserts import (
    get_previous_total_value,
    insert_prices,
    record_run,
    upsert_portfolio_pnl,
    upsert_positions,
)
from src.portfolio.orders import submit_orders
from src.portfolio.positions import fetch_account_snapshot
from src.portfolio.rebalance import compute_orders, target_quantities

logger = logging.getLogger(__name__)


def run_daily(
    trade_date: str | None = None,
    *,
    dry_run: bool = False,
    skip_orders: bool = False,
) -> str:
    """Execute one daily cycle. Returns the session date processed.

    `trade_date` is injectable for replaying a specific session; it defaults to
    the most recent completed one.
    """
    session = trade_date or most_recent_completed_session()
    logger.info("Daily run for session %s", session)

    with get_connection() as conn:
        try:
            price_rows = fetch_daily_closes(start=session, end=session)
            gaps = missing_symbols(price_rows, session)
            if gaps:
                raise RuntimeError(
                    f"Missing closes for {sorted(gaps)} on {session}. "
                    "Refusing to store an incomplete session."
                )
            written = insert_prices(conn, price_rows)
            if written.has_restatements:
                # The vendor changed a close this job already stored. That is
                # never expected for the session just fetched, so treat it as a
                # data integrity failure rather than writing on top of it.
                sample = ", ".join(
                    f"{d} {s} {old:.4f}->{new:.4f}"
                    for d, s, old, new in written.restatements[:5]
                )
                raise RuntimeError(
                    f"{len(written.restatements)} restatement(s) on {session}: "
                    f"{sample}. Stored values kept; investigate before rerunning."
                )

            snapshot = fetch_account_snapshot()
            upsert_positions(conn, snapshot.position_rows(session))

            previous_value = get_previous_total_value(conn, session)
            if previous_value is not None and previous_value > 0:
                daily_pnl = snapshot.total_value - previous_value
                daily_return = daily_pnl / previous_value
            else:
                daily_pnl = None
                daily_return = None

            upsert_portfolio_pnl(
                conn,
                trade_date=session,
                total_value=snapshot.total_value,
                cash=snapshot.cash,
                daily_pnl=daily_pnl,
                daily_return=daily_return,
            )

            if is_first_trading_day_of_month(session) and not skip_orders:
                logger.info("%s is the month's first session; rebalancing.", session)
                prices = {symbol: close for _, symbol, close in price_rows}
                targets = target_quantities(snapshot.total_value, prices)
                current = {sym: qty for sym, qty, _ in snapshot.positions}
                orders = compute_orders(current, targets)
                submit_orders(orders, dry_run=dry_run)
            else:
                logger.info("No rebalance for %s.", session)

            record_run(conn, "success", f"session {session}")
            logger.info("Daily run complete for %s", session)

        except Exception as exc:
            # Roll back the partial session, then record the failure in its own
            # transaction so the heartbeat survives.
            conn.rollback()
            logger.exception("Daily run failed for %s", session)
            record_run(conn, "failure", f"session {session}: {exc}")
            raise

    return session


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the daily risk snapshot.")
    parser.add_argument("--date", help="Session to process (YYYY-MM-DD).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log orders without submitting them.",
    )
    parser.add_argument(
        "--skip-orders",
        action="store_true",
        help="Snapshot data only; never rebalance.",
    )
    args = parser.parse_args()

    try:
        run_daily(args.date, dry_run=args.dry_run, skip_orders=args.skip_orders)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
