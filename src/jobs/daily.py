"""The daily run.

Order of operations, and why:

1. Resolve the target session — the last *completed* one, never today's.
2. Fetch and store closes for all 11 tickers.
3. Snapshot account and positions, store them, compute P&L.
4. Compute and store risk estimates and portfolio metrics from stored rows.
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
    get_symbol_return_series,
    get_total_value_series,
    get_var_amount,
    insert_prices,
    record_run,
    upsert_portfolio_metrics,
    upsert_portfolio_pnl,
    upsert_positions,
    upsert_risk_contributions,
    upsert_risk_estimates,
)
from src.portfolio.orders import submit_orders
from src.portfolio.positions import AccountSnapshot, fetch_account_snapshot
from src.portfolio.rebalance import compute_orders, target_quantities
from src.risk.contribution import historical_contribution
from src.risk.expected_shortfall import expected_shortfall
from src.risk.metrics import current_drawdown, rolling_volatility
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

# Named to match portfolio_metrics.vol_20d. Changing one without the other puts
# a window in the column that its name denies.
VOL_WINDOW = 20

# Cash is carried as a position with a constant zero return. It is part of the
# portfolio, it has no price risk, and holding it is exactly why the portfolio's
# VaR is lower than a fully invested one's. Modelling it explicitly keeps the
# weights summing to 1.0 without normalising the sleeves, which would decompose
# a fully-invested portfolio that is not the one being held.
CASH_SYMBOL = "CASH"

# Two different checks, because two different things can be wrong.
#
# Contributions decompose the portfolio return implied by *today's* weights
# applied to each symbol's historical returns. Against that series the
# decomposition is exact by construction, so any gap beyond float noise is a
# genuine defect -- a dropped symbol, a weight that does not match its return
# series, a misaligned window. Fatal.
ADDITIVITY_TOLERANCE = 1e-6

# The stored var_amount is a different series: account equity, which reflects
# the weights as they actually were on each past day, plus cash drag, dividends
# and execution costs. Equal-weight sleeves drift between monthly rebalances, so
# the two disagree by more than rounding -- measured at 0.5% at 95% confidence
# and 4% at 99%, where the estimate rests on a single day with no averaging to
# wash the drift out.
#
# That gap is a real property of the portfolio and is logged rather than scaled
# away: rescaling would tie the table out at the cost of making every figure in
# it slightly fictional. It only becomes evidence of a fault when it is far
# larger than drift can explain.
DRIFT_ALERT_THRESHOLD = 0.25


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


def _write_risk_contributions(
    conn: sqlite3.Connection,
    session: str,
    snapshot: AccountSnapshot,
) -> int:
    """Decompose each stored historical VaR figure across the positions held.

    Weights come from the account snapshot, with cash as a zero-return position
    so they sum to 1.0 (see CASH_SYMBOL). Per-symbol returns come from stored
    closes, so this decomposes the price-only relationship between positions and
    the portfolio.

    Only windows and confidence levels that already have a stored historical
    estimate are decomposed: a contribution row exists to explain a specific
    var_amount, so writing one with nothing to join to would be meaningless.

    Raises:
        RuntimeError: If the contributions miss the stored var_amount by more
            than RESIDUAL_TOLERANCE. Small residuals are cash drag and
            dividends; a large one means the decomposition is not describing
            the portfolio the estimate was computed from.
    """
    if not snapshot.positions:
        logger.info("No positions on %s; nothing to decompose.", session)
        return 0

    weights = {
        symbol: market_value / snapshot.total_value
        for symbol, _qty, market_value in snapshot.positions
    }
    weights[CASH_SYMBOL] = snapshot.cash / snapshot.total_value

    rows: list[tuple[str, str, float, float | None, float, str, float, int]] = []

    for window in LOOKBACK_WINDOWS:
        returns_by_symbol = get_symbol_return_series(conn, session, window)
        if not returns_by_symbol:
            logger.info(
                "No %d-day contributions for %s: incomplete price history.",
                window, session,
            )
            continue

        # Cash earns nothing and moves nothing, on every day in the window.
        returns_by_symbol[CASH_SYMBOL] = [0.0] * window

        if set(returns_by_symbol) != set(weights):
            logger.warning(
                "Skipping %d-day contributions for %s: priced symbols %s do not "
                "match held positions %s.",
                window, session, sorted(returns_by_symbol), sorted(weights),
            )
            continue

        # The portfolio return series these contributions actually decompose:
        # today's weights applied to each symbol's history.
        implied = [
            sum(returns_by_symbol[s][t] * weights[s] for s in weights)
            for t in range(window)
        ]

        for confidence in CONFIDENCE_LEVELS:
            var_amount = get_var_amount(
                conn, session, "historical", confidence, window
            )
            if var_amount is None:
                continue

            contributions = historical_contribution(
                returns_by_symbol, weights, confidence, snapshot.total_value
            )
            total = sum(contributions.values())

            # Check 1: internal additivity. Exact by construction, so any gap
            # here is a defect rather than a modelling difference.
            implied_var = historical_var(implied, confidence, snapshot.total_value)
            if abs(total - implied_var) > ADDITIVITY_TOLERANCE * max(
                abs(implied_var), 1.0
            ):
                raise RuntimeError(
                    f"Contributions for {session} at {confidence} over {window}d "
                    f"sum to {total:.6f}, but the portfolio VaR implied by the "
                    f"same weights and returns is {implied_var:.6f}. The "
                    f"decomposition is not describing its own inputs."
                )

            # Check 2: distance from the stored figure. Expected to be non-zero
            # (weight drift, cash drag, dividends) and recorded so the size of
            # that effect is visible rather than assumed.
            drift = total - var_amount
            drift_pct = drift / var_amount if var_amount else 0.0
            if abs(drift_pct) > DRIFT_ALERT_THRESHOLD:
                raise RuntimeError(
                    f"Contributions for {session} at {confidence} over {window}d "
                    f"sum to {total:.2f} against a stored var_amount of "
                    f"{var_amount:.2f} ({drift_pct:+.1%}). Weight drift and cash "
                    f"drag do not explain a gap this size."
                )

            logger.info(
                "Contributions for %s at %.2f over %dd: sum %.2f, "
                "stored var_amount %.2f, drift %+.2f (%+.2f%%).",
                session, confidence, window, total, var_amount,
                drift, drift_pct * 100,
            )

            rows.extend(
                (
                    session,
                    symbol,
                    weights[symbol],
                    None,  # marginal_var: see upsert_risk_contributions
                    amount,
                    "historical",
                    confidence,
                    window,
                )
                for symbol, amount in sorted(contributions.items())
            )

    if rows:
        upsert_risk_contributions(conn, rows)
        logger.info("Wrote %d contribution row(s) for %s.", len(rows), session)

    return len(rows)


def _write_portfolio_metrics(conn: sqlite3.Connection, session: str) -> None:
    """Compute and store volatility, drawdown, and peak for `session`.

    Unlike the risk estimates, these describe `session` itself rather than
    predicting the next day, which is why portfolio_metrics is keyed on
    as_of_date alone and carries no applies_to_date.

    The two inputs are different series, and confusing them is the way this
    goes wrong: volatility is the dispersion of *returns*, drawdown is a
    position within the history of *levels*. Drawdown reads the full stored
    history because its peak is all-time; a trailing peak would understate the
    decline.

    vol_20d is stored as None until 20 returns exist. Drawdown is meaningful
    from the first row, so it is always written.
    """
    returns = get_return_series(conn, session, VOL_WINDOW)
    values = get_total_value_series(conn, session)

    vol = rolling_volatility(returns, window=VOL_WINDOW)
    if vol is None:
        logger.info(
            "No %d-day volatility for %s: %d of %d returns available.",
            VOL_WINDOW, session, len(returns), VOL_WINDOW,
        )

    drawdown, peak = current_drawdown(values)
    upsert_portfolio_metrics(conn, session, vol, drawdown, peak)

    logger.info(
        "Metrics for %s: vol_20d=%s drawdown=%.4f peak=%.2f",
        session, "None" if vol is None else f"{vol:.4f}", drawdown, peak,
    )


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
                _write_risk_contributions(conn, session, snapshot)
                _write_portfolio_metrics(conn, session)
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
