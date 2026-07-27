"""Idempotent writes.

Every write here is an upsert keyed on the table's primary key. Re-running a
day overwrites that day rather than duplicating it, which is what makes the
daily job safe to retry and the database safe to commit (CLAUDE.md,
"Committing data/risk.db").
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone


def upsert_prices(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[str, str, float]],
) -> int:
    """Insert or replace (trade_date, symbol, close) rows. Returns row count."""
    conn.executemany(
        """
        INSERT INTO prices (trade_date, symbol, close)
        VALUES (?, ?, ?)
        ON CONFLICT (trade_date, symbol) DO UPDATE SET close = excluded.close
        """,
        rows,
    )
    return len(rows)


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
