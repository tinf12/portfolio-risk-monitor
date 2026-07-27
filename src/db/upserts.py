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
