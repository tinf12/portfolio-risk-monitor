"""Writes.

Re-running a day never duplicates rows. Two different policies apply, and the
difference is deliberate:

- `prices` is **write-once**. A stored close is never silently changed. See
  `insert_prices`.
- `positions` and `portfolio_pnl` are upserts. They are snapshots of live
  account state, so re-running a session should refresh them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

PriceRow = tuple[str, str, float]

# Vendor closes are floats round-tripped through SQLite. Treat a difference
# below this as representation noise rather than a genuine restatement.
RESTATEMENT_TOLERANCE = 1e-9


@dataclass
class PriceWriteResult:
    """Outcome of a price write."""

    inserted: int = 0
    unchanged: int = 0
    restatements: list[tuple[str, str, float, float]] = field(default_factory=list)
    applied_restatements: int = 0

    @property
    def has_restatements(self) -> bool:
        return bool(self.restatements)


def insert_prices(
    conn: sqlite3.Connection,
    rows: Sequence[PriceRow],
    *,
    allow_restate: bool = False,
) -> PriceWriteResult:
    """Insert (trade_date, symbol, close) rows without changing stored closes.

    `prices` is write-once because Alpaca returns split- and dividend-adjusted
    closes, and the adjustment factor for any past date shrinks with every
    subsequent distribution. Re-fetching therefore restates the entire history:
    verified on XLE, where 2020-02-19 carries a factor of 0.3829 against a
    factor of exactly 1.0 today.

    Overwriting would mean a risk figure computed last month no longer
    reproduces from stored inputs, which breaks the determinism constraint. The
    stored close is the input of record. Adjusted prices remain the right
    choice — they measure total return — so the fix is to freeze them, not to
    switch to raw.

    Rows whose close differs from what is stored are reported in
    `restatements` as (trade_date, symbol, stored, incoming) and are **not**
    applied unless `allow_restate=True`. Reporting rather than ignoring is the
    point: silent divergence between the database and the vendor is exactly
    what an auditable system should surface.

    The consequence to accept, and to state in the README: stored history
    slowly diverges from what the vendor currently reports. A reproducible
    number is worth more here than an up-to-date one.
    """
    result = PriceWriteResult()

    for trade_date, symbol, close in rows:
        existing = conn.execute(
            "SELECT close FROM prices WHERE trade_date = ? AND symbol = ?",
            (trade_date, symbol),
        ).fetchone()

        if existing is None:
            conn.execute(
                "INSERT INTO prices (trade_date, symbol, close) VALUES (?, ?, ?)",
                (trade_date, symbol, close),
            )
            result.inserted += 1
            continue

        stored = float(existing["close"])
        if abs(stored - close) <= RESTATEMENT_TOLERANCE:
            result.unchanged += 1
            continue

        result.restatements.append((trade_date, symbol, stored, close))
        if allow_restate:
            conn.execute(
                "UPDATE prices SET close = ? WHERE trade_date = ? AND symbol = ?",
                (close, trade_date, symbol),
            )
            result.applied_restatements += 1

    return result


def upsert_positions(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[str, str, float, float]],
) -> int:
    """Insert or replace (trade_date, symbol, qty, market_value) rows."""
    conn.executemany(
        """
        INSERT INTO positions (trade_date, symbol, qty, market_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (trade_date, symbol) DO UPDATE SET
          qty = excluded.qty,
          market_value = excluded.market_value
        """,
        rows,
    )
    return len(rows)


def upsert_portfolio_pnl(
    conn: sqlite3.Connection,
    trade_date: str,
    total_value: float,
    cash: float,
    daily_pnl: float | None,
    daily_return: float | None,
) -> None:
    """Insert or replace one day of portfolio P&L.

    daily_pnl and daily_return are None on the first day, when there is no
    prior total_value to difference against.
    """
    conn.execute(
        """
        INSERT INTO portfolio_pnl
          (trade_date, total_value, cash, daily_pnl, daily_return)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (trade_date) DO UPDATE SET
          total_value  = excluded.total_value,
          cash         = excluded.cash,
          daily_pnl    = excluded.daily_pnl,
          daily_return = excluded.daily_return
        """,
        (trade_date, total_value, cash, daily_pnl, daily_return),
    )


@dataclass
class RiskWriteResult:
    """Outcome of a risk-estimate write.

    `changed` lists rows whose stored value moved on a re-run, as
    (as_of_date, method, confidence, lookback_days, stored, incoming).

    Re-running a session must reproduce its figures exactly (CLAUDE.md
    constraint 1, and the Phase 2 "done when"). Nothing here blocks an
    overwrite — a deliberate methodology fix should be able to land — but a
    value that moves without a code change is the signal that determinism
    broke, so it is surfaced rather than absorbed silently.
    """

    inserted: int = 0
    unchanged: int = 0
    changed: list[tuple[str, str, float, int, float, float]] = field(
        default_factory=list
    )

    @property
    def has_changes(self) -> bool:
        return bool(self.changed)


def upsert_risk_estimates(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[str, str, str, float, float, float | None, int]],
) -> RiskWriteResult:
    """Insert or refresh risk estimates, reporting any value that moved.

    Each row is
    (as_of_date, applies_to_date, method, confidence, var_amount, es_amount,
    lookback_days), matching the column order of `risk_estimates`.

    `var_amount` is a positive loss magnitude. `es_amount` may be None; the
    column is nullable and not every method supplies one.

    The (as_of_date, method, confidence, lookback_days) key means estimates
    from different windows coexist for the same date rather than overwriting
    each other, so a 60-day and a 250-day figure can be compared directly.

    `applies_to_date` is not part of the key: it is a function of as_of_date
    via the NYSE calendar, so a row cannot disagree with itself about which
    day it predicts.
    """
    result = RiskWriteResult()

    for row in rows:
        as_of, applies_to, method, confidence, var_amount, es_amount, lookback = row

        existing = conn.execute(
            """
            SELECT var_amount FROM risk_estimates
            WHERE as_of_date = ? AND method = ? AND confidence = ?
              AND lookback_days = ?
            """,
            (as_of, method, confidence, lookback),
        ).fetchone()

        if existing is None:
            result.inserted += 1
        else:
            stored = float(existing["var_amount"])
            if abs(stored - var_amount) <= RESTATEMENT_TOLERANCE:
                result.unchanged += 1
            else:
                result.changed.append(
                    (as_of, method, confidence, lookback, stored, var_amount)
                )

        conn.execute(
            """
            INSERT INTO risk_estimates
              (as_of_date, applies_to_date, method, confidence, var_amount,
               es_amount, lookback_days)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (as_of_date, method, confidence, lookback_days)
            DO UPDATE SET
              applies_to_date = excluded.applies_to_date,
              var_amount      = excluded.var_amount,
              es_amount       = excluded.es_amount
            """,
            row,
        )

    return result


def get_return_series(
    conn: sqlite3.Connection,
    as_of_date: str,
    lookback_days: int,
) -> list[float]:
    """Return up to `lookback_days` daily returns ending at `as_of_date`.

    Ordered oldest to newest, though the risk functions sort internally and do
    not depend on it.

    Two filters carry the temporal guarantees:

    - `trade_date <= as_of_date` — an estimate for as_of_date may only use
      information available at that close. Widening this to `<` on a later
      date, or dropping it, is lookahead bias and will not raise.
    - `daily_return IS NOT NULL` — the first stored row has no prior day to
      difference against. The risk functions reject non-finite input rather
      than dropping it, so the trim happens here, where the caller can see
      that a row was skipped for a structural reason and not a data fault.

    Returns fewer than `lookback_days` values when history is short. The
    caller decides whether that is enough; this function does not guess.
    """
    rows = conn.execute(
        """
        SELECT daily_return
        FROM portfolio_pnl
        WHERE trade_date <= ? AND daily_return IS NOT NULL
        ORDER BY trade_date DESC
        LIMIT ?
        """,
        (as_of_date, lookback_days),
    ).fetchall()

    return [float(row["daily_return"]) for row in reversed(rows)]


def upsert_portfolio_metrics(
    conn: sqlite3.Connection,
    as_of_date: str,
    vol_20d: float | None,
    drawdown: float,
    peak_value: float,
) -> None:
    """Insert or replace one day of portfolio-level metrics.

    `vol_20d` is None until 20 returns exist. The column is nullable for that
    reason: a figure computed from a shorter window would be a false claim in a
    column named for a 20-day one.

    `drawdown` is negative or zero, the opposite of the positive-loss convention
    used for var_amount. Both are deliberate and both are in the schema.

    An upsert rather than write-once: these are derived entirely from stored
    rows, so recomputing them cannot lose information the way restating a vendor
    close would.
    """
    conn.execute(
        """
        INSERT INTO portfolio_metrics (as_of_date, vol_20d, drawdown, peak_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (as_of_date) DO UPDATE SET
          vol_20d    = excluded.vol_20d,
          drawdown   = excluded.drawdown,
          peak_value = excluded.peak_value
        """,
        (as_of_date, vol_20d, drawdown, peak_value),
    )


def get_total_value_series(
    conn: sqlite3.Connection,
    as_of_date: str,
) -> list[float]:
    """Return every stored total_value through `as_of_date`, oldest first.

    Unwindowed on purpose. Drawdown is measured against the all-time high, and
    a peak taken over a trailing window instead would understate the decline --
    the direction that flatters the portfolio.

    `trade_date <= as_of_date` carries the same no-lookahead guarantee as
    get_return_series: a metric for a date may not see a value recorded after
    it.
    """
    rows = conn.execute(
        """
        SELECT total_value
        FROM portfolio_pnl
        WHERE trade_date <= ?
        ORDER BY trade_date
        """,
        (as_of_date,),
    ).fetchall()

    return [float(row["total_value"]) for row in rows]


def upsert_risk_contributions(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[str, str, float, float | None, float, str, float, int]],
) -> int:
    """Insert or refresh per-position risk contributions.

    Each row is (as_of_date, symbol, weight, marginal_var, contribution,
    method, confidence, lookback_days), matching the column order of
    `risk_contributions`.

    The method/confidence/lookback_days triple is carried so a contribution row
    always joins back to the `risk_estimates` row it decomposes.

    `marginal_var` is nullable and currently written as None. Marginal VaR is
    the derivative of portfolio VaR with respect to a position's weight -- a
    different quantity from the component contribution stored here, with no
    clean single-day form. An invented figure in that column would be worse
    than an empty one.
    """
    conn.executemany(
        """
        INSERT INTO risk_contributions
          (as_of_date, symbol, weight, marginal_var, contribution, method,
           confidence, lookback_days)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (as_of_date, symbol, method, confidence, lookback_days)
        DO UPDATE SET
          weight       = excluded.weight,
          marginal_var = excluded.marginal_var,
          contribution = excluded.contribution
        """,
        rows,
    )
    return len(rows)


def get_symbol_return_series(
    conn: sqlite3.Connection,
    as_of_date: str,
    lookback_days: int,
) -> dict[str, list[float]]:
    """Per-symbol daily returns from stored closes, ending at `as_of_date`.

    Returns {symbol: [oldest, ..., newest]} with exactly `lookback_days` values
    per symbol, or {} when there is not enough complete history.

    Computed from `prices` rather than read from a table, because per-symbol
    returns are not stored anywhere -- only portfolio-level ones are. Producing
    n returns needs n+1 closes.

    Only dates where **every** symbol has a close are used. A date with partial
    coverage is dropped rather than filled: a missing bar would otherwise
    become a fabricated 0% return for that symbol on that day, which would
    understate its contribution. Dropping keeps every symbol's series aligned to
    the same calendar, which is what makes the k-th worst day the same day for
    all of them.

    `trade_date <= as_of_date` carries the same no-lookahead guarantee as the
    portfolio-level readers.
    """
    rows = conn.execute(
        """
        SELECT trade_date, symbol, close
        FROM prices
        WHERE trade_date <= ?
        ORDER BY trade_date
        """,
        (as_of_date,),
    ).fetchall()

    if not rows:
        return {}

    symbols = {row["symbol"] for row in rows}
    by_date: dict[str, dict[str, float]] = {}
    for row in rows:
        by_date.setdefault(row["trade_date"], {})[row["symbol"]] = float(row["close"])

    complete = [d for d in sorted(by_date) if set(by_date[d]) == symbols]

    # n returns need n+1 closes.
    if len(complete) < lookback_days + 1:
        return {}

    window = complete[-(lookback_days + 1):]
    series: dict[str, list[float]] = {}
    for symbol in symbols:
        closes = [by_date[d][symbol] for d in window]
        series[symbol] = [
            closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))
        ]

    return series


def get_var_amount(
    conn: sqlite3.Connection,
    as_of_date: str,
    method: str,
    confidence: float,
    lookback_days: int,
) -> float | None:
    """Return a stored var_amount, or None when that estimate was not written."""
    row = conn.execute(
        """
        SELECT var_amount FROM risk_estimates
        WHERE as_of_date = ? AND method = ? AND confidence = ?
          AND lookback_days = ?
        """,
        (as_of_date, method, confidence, lookback_days),
    ).fetchone()
    return float(row["var_amount"]) if row is not None else None


def record_run(
    conn: sqlite3.Connection,
    status: str,
    message: str | None = None,
) -> None:
    """Append a heartbeat row.

    GitHub sends no notification when a scheduled workflow fails, so this table
    is the failure signal (CLAUDE.md, "GitHub Actions"). Append-only: a failed
    run must never overwrite the record of an earlier successful one.
    """
    if status not in {"success", "failure"}:
        raise ValueError(f"status must be 'success' or 'failure', got {status!r}")

    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO runs (run_at, status, message) VALUES (?, ?, ?)",
        (run_at, status, message),
    )


def get_previous_total_value(
    conn: sqlite3.Connection,
    trade_date: str,
) -> float | None:
    """Return total_value for the most recent stored date strictly before
    trade_date, or None if there is no earlier row.

    Uses the stored series rather than an assumed calendar step, so a gap in
    history (a missed run, a backfill boundary) is visible to the caller
    instead of being silently papered over.
    """
    row = conn.execute(
        """
        SELECT total_value
        FROM portfolio_pnl
        WHERE trade_date < ?
        ORDER BY trade_date DESC
        LIMIT 1
        """,
        (trade_date,),
    ).fetchone()
    return float(row["total_value"]) if row is not None else None


def latest_successful_run(conn: sqlite3.Connection) -> str | None:
    """Return run_at of the most recent successful run, or None."""
    row = conn.execute(
        """
        SELECT run_at FROM runs
        WHERE status = 'success'
        ORDER BY run_at DESC
        LIMIT 1
        """
    ).fetchone()
    return row["run_at"] if row is not None else None
