"""Database schema.

The DDL here is the authority for the table definitions documented in
CLAUDE.md. If the two disagree, that is a bug in one of them.
"""

from __future__ import annotations

import sqlite3

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS prices (
      trade_date TEXT NOT NULL,
      symbol     TEXT NOT NULL,
      close      REAL NOT NULL,
      PRIMARY KEY (trade_date, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
      trade_date   TEXT NOT NULL,
      symbol       TEXT NOT NULL,
      qty          REAL NOT NULL,
      market_value REAL NOT NULL,
      PRIMARY KEY (trade_date, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_pnl (
      trade_date   TEXT PRIMARY KEY,
      total_value  REAL NOT NULL,
      cash         REAL NOT NULL,
      daily_pnl    REAL,
      daily_return REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_estimates (
      as_of_date      TEXT NOT NULL,
      applies_to_date TEXT NOT NULL,
      method          TEXT NOT NULL,
      confidence      REAL NOT NULL,
      var_amount      REAL NOT NULL,
      es_amount       REAL,
      lookback_days   INTEGER NOT NULL,
      PRIMARY KEY (as_of_date, method, confidence, lookback_days)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_metrics (
      as_of_date TEXT PRIMARY KEY,
      vol_20d    REAL,
      drawdown   REAL,
      peak_value REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_contributions (
      as_of_date    TEXT NOT NULL,
      symbol        TEXT NOT NULL,
      weight        REAL NOT NULL,
      marginal_var  REAL,
      contribution  REAL NOT NULL,
      method        TEXT NOT NULL,
      confidence    REAL NOT NULL,
      lookback_days INTEGER NOT NULL,
      PRIMARY KEY (as_of_date, symbol, method, confidence, lookback_days)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
      run_id  INTEGER PRIMARY KEY AUTOINCREMENT,
      run_at  TEXT NOT NULL,
      status  TEXT NOT NULL,
      message TEXT
    )
    """,
    # Breach analysis joins risk_estimates on applies_to_date. Joining on
    # as_of_date is lookahead bias (CLAUDE.md, "Temporal convention"), so the
    # index exists to make the correct join the convenient one.
    """
    CREATE INDEX IF NOT EXISTS idx_risk_estimates_applies_to
      ON risk_estimates (applies_to_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_run_at
      ON runs (run_at DESC)
    """,
)


EXPECTED_TABLES: frozenset[str] = frozenset(
    {
        "prices",
        "positions",
        "portfolio_pnl",
        "risk_estimates",
        "portfolio_metrics",
        "risk_contributions",
        "runs",
    }
)


def schema_exists(conn: sqlite3.Connection) -> bool:
    """True if every expected table is already present."""
    present = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    return EXPECTED_TABLES <= present


def create_schema(conn: sqlite3.Connection) -> None:
    """Create every table and index if it does not already exist.

    Returns without writing when the schema is already complete. This is not a
    micro-optimisation: SQLite bumps the file change counter on any committed
    transaction, so an unconditional DDL-plus-commit rewrites bytes in a
    database that is committed to git. That would produce a binary diff on
    every scheduled run even when no data changed, burying real changes in
    noise and defeating the workflow's "nothing to commit" check.

    Safe to call on every run. Does not migrate or alter existing tables.
    """
    if schema_exists(conn):
        return

    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)
    conn.commit()
