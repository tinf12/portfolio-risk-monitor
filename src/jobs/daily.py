"""The daily run.

Order of operations, and why:

1. Resolve the target session — the last *completed* one, never today's.
2. Fetch and store closes for all 11 tickers.
3. Snapshot account and positions, store them, compute P&L.
4. Compute and store risk estimates from the stored return series.
5. Rebalance if the target session is the month's first.
6. Record a heartbeat row either way.

Step 6 runs even on failure. GitHub does not notify on a failed scheduled
workflow, so an absent or failed `runs` row is the only failure signal
(CLAUDE.md, "GitHub Actions").

Step 4 follows step 3 because VaR is scaled by the total_value written there,
and estimated from the return series that step just extended. It precedes the
rebalance so the estimate describes the portfolio as it stood at the close,
not the one the next morning's orders will create.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sqlite3
import sys

from src.data.bars import fetch_daily_closes, missing_symbols
from src.data.calendar import (
    is_first_trading_day_of_month,
    most_recent_completed_session,
    next_trading_day,
)
from src.db.connection import get_connection
from src.db.upserts import (
    get_previous_total_value,
    get_return_series,
    insert_prices,
    record_run,
    upsert_portfolio_pnl,
    upsert_positions,
    upsert_risk_estimates,
)
from src.portfolio.orders import submit_orders
from src.portfolio.positions import fetch_account_snapshot
from src.portfolio.rebalance import compute_orders, target_quantities
from src.risk.expected_shortfall import expected_shortfall
from src.risk.var import historical_var, parametric_var

logger = logging.getLogger(__name__)

# Windows to estimate over. Both are written for every session that has enough
# history, and coexist in risk_estimates because lookback_days is part of its
# key -- so the two can be compared rather than one replacing the other.
#
# The choice of windows is a methodology decision, not plumbing: 250 is roughly
# a trading year and is the conventional default; 30 exists so the live series
# produces a figure about six weeks in rather than months.
#
# 30 is a floor, not a preference. Below 25 the nearest-rank tail collapses:
# ceil(0.05 * 20) and ceil(0.01 * 20) are both 1, so the 95% and 99% estimates
# would read the same single observation and report identical numbers while
# looking like two independent measurements. At 30 they are 2 and 1 -- still
# thin, and the 99% figure is exactly its worst day, which is a caveat for the
# README rather than a reason to withhold the number.
LOOKBACK_WINDOWS = (30, 250)

CONFIDENCE_LEVELS = (0.95, 0.99)


def _write_risk_estimates(
    conn: sqlite3.Connection,
    session: str,
    total_value: float,
) -> int:
    """Estimate and store VaR/ES for `session`, returning the row count.

    Reads the return series ending at `session`, so a figure only ever uses
    information available at that close. `applies_to_date` is the next NYSE
    session -- the day the estimate predicts. Computing it as session + 1 day
    would point at weekends and holidays, and rows pointing at a non-trading
    day can never breach, which biases the Kupiec test toward acceptance
    (CLAUDE.md, "Temporal convention").

    Windows with too little history are skipped with a log line rather than
    estimated from whatever happens to be there: a VaR from 12 observations is
    not a weaker number, it is a different claim.

    Expected shortfall is stored only for the historical rows. Parametric ES
    has a closed form under the normal assumption, but choosing and writing it
    is a methodology decision reserved to the author (CLAUDE.md, "Author
    boundary"), so those rows carry NULL rather than a figure this job
    invented. The column is nullable for exactly this reason.
    """
    applies_to = next_trading_day(session)
    rows: list[tuple[str, str, str, float, float, float | None, int]] = []

    for window in LOOKBACK_WINDOWS:
        returns = get_return_series(conn, session, window)
        if len(returns) < window:
            logger.info(
                "Skipping %d-day estimates for %s: %d of %d returns available.",
                window,
                session,
                len(returns),
                window,
            )
            continue

        for confidence in CONFIDENCE_LEVELS:
            rows.append((
                session,
                applies_to,
                "historical",
                confidence,
                historical_var(returns, confidence, total_value),
                expected_shortfall(returns, confidence, total_value),
                window,
            ))
            rows.append((
                session,
                applies_to,
                "parametric",
                confidence,
                parametric_var(returns, confidence, total_value),
                None,
                window,
            ))

    if not rows:
        logger.info("No risk estimates for %s; insufficient history.", session)
        return 0

    result = upsert_risk_estimates(conn, rows)
    if result.has_changes:
        # Same inputs must give the same output. A moved value without a code
        # change means determinism broke somewhere upstream.
        for as_of, method, conf, window, stored, incoming in result.changed:
            logger.warning(
                "Risk estimate changed on re-run: %s %s %.2f %dd, %.4f -> %.4f",
                as_of, method, conf, window, stored, incoming,
            )

    logger.info(
        "Wrote %d risk estimate(s) for %s (applies to %s).",
        len(rows), session, applies_to,
    )
    return len(rows)


def run_daily(
    trade_date: str | None = None,
    *,
    dry_run: bool = False,
    skip_orders: bool = False,
) -> str:
    """Execute one daily cycle. Returns the session date processed.

    `trade_date` is injectable for replaying a specific session; it defaults to
    the most recent completed one.

    Two safeguards on what may be written:

    - `dry_run` performs every read and computation but commits nothing. A
      verification run must not leave rows behind.
    - Account state is a *live* read with no historical equivalent. When
      `trade_date` is not the most recent completed session, the snapshot
      describes today, not that date, so positions and P&L are not stored for
      it. Writing them would fabricate history — a $100k equity recorded
      against a date the account did not hold it. Prices are unaffected:
      a historical close is a fact about that date.
    """
    session = trade_date or most_recent_completed_session()
    current_session = most_recent_completed_session()
    is_current = session == current_session

    logger.info("Daily run for session %s%s", session, " (dry run)" if dry_run else "")
    if not is_current:
        logger.warning(
            "Session %s is not the most recent completed session (%s). The "
            "account snapshot is a live read, so positions and P&L will NOT "
            "be stored for %s.",
            session,
            current_session,
            session,
        )

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

            if is_current:
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

                # After the P&L write, so today's return is in the window, and
                # scaled by the total_value just stored for this session.
                _write_risk_estimates(conn, session, snapshot.total_value)
            else:
                logger.warning(
                    "Skipped positions and P&L for %s; account state is live.",
                    session,
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

            if dry_run:
                # Discard everything. A verification run must be observable
                # only through its logs, never through stored rows.
                conn.rollback()
                logger.info("Dry run complete for %s; nothing written.", session)
                return session

            record_run(conn, "success", f"session {session}")
            logger.info("Daily run complete for %s", session)

        except Exception as exc:
            # Roll back the partial session, then record the failure in its own
            # transaction so the heartbeat survives.
            conn.rollback()
            logger.exception("Daily run failed for %s", session)
            if not dry_run:
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
