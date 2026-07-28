"""Risk-estimate persistence and the temporal guarantees around it.

The arithmetic is tested in test_var_handcheck.py. What matters here is what
reaches the database: which returns a figure was allowed to see, and which date
it claims to predict.

The lookahead cases are the point. Joining or filtering on the wrong date does
not raise, and it makes the model look excellent (CLAUDE.md, "Temporal
convention"), so it has to be asserted rather than reviewed for.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.db.upserts import (
    get_return_series,
    upsert_portfolio_pnl,
    upsert_risk_estimates,
)

# Four consecutive NYSE sessions. 2026-07-24 is a Friday, so the next trading
# day after it is Monday 2026-07-27 -- a gap a naive "+1 day" would miss.
SESSIONS = ["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-27"]
RETURNS = [-0.01, 0.02, -0.03, 0.04]


@pytest.fixture
def pnl(conn: sqlite3.Connection) -> sqlite3.Connection:
    """A P&L series whose first row has a NULL return, as the real one does."""
    upsert_portfolio_pnl(
        conn,
        trade_date="2026-07-21",
        total_value=100_000.0,
        cash=0.0,
        daily_pnl=None,      # no prior day to difference against
        daily_return=None,
    )
    for session, ret in zip(SESSIONS, RETURNS):
        upsert_portfolio_pnl(
            conn,
            trade_date=session,
            total_value=100_000.0 * (1 + ret),
            cash=0.0,
            daily_pnl=100_000.0 * ret,
            daily_return=ret,
        )
    return conn


class TestReturnSeries:
    def test_excludes_the_null_first_row(self, pnl: sqlite3.Connection) -> None:
        """The risk functions reject non-finite input by design, so the
        structural NULL is trimmed here rather than inside them."""
        assert get_return_series(pnl, "2026-07-27", 10) == RETURNS

    def test_stops_at_as_of_date(self, pnl: sqlite3.Connection) -> None:
        """The lookahead case. An estimate for 2026-07-23 may not see the
        2026-07-24 return, which had not happened at that close."""
        got = get_return_series(pnl, "2026-07-23", 10)
        assert got == [-0.01, 0.02]
        assert -0.03 not in got

    def test_includes_as_of_date_itself(self, pnl: sqlite3.Connection) -> None:
        """The boundary is inclusive: an estimate made at T's close uses T."""
        assert get_return_series(pnl, "2026-07-23", 10)[-1] == 0.02

    def test_takes_the_most_recent_window(self, pnl: sqlite3.Connection) -> None:
        """A short window keeps the newest returns, not the oldest."""
        assert get_return_series(pnl, "2026-07-27", 2) == [-0.03, 0.04]

    def test_oldest_first(self, pnl: sqlite3.Connection) -> None:
        assert get_return_series(pnl, "2026-07-27", 10)[0] == -0.01

    def test_short_history_returns_what_exists(
        self, pnl: sqlite3.Connection
    ) -> None:
        """No padding and no guessing: the caller decides if this is enough."""
        assert len(get_return_series(pnl, "2026-07-22", 250)) == 1

    def test_empty_before_any_history(self, pnl: sqlite3.Connection) -> None:
        assert get_return_series(pnl, "2026-01-01", 250) == []


class TestUpsertRiskEstimates:
    ROW = ("2026-07-24", "2026-07-27", "historical", 0.95, 1234.56, 1500.0, 60)

    def test_inserts(self, conn: sqlite3.Connection) -> None:
        result = upsert_risk_estimates(conn, [self.ROW])
        assert result.inserted == 1
        assert not result.has_changes

    def test_rerun_is_idempotent(self, conn: sqlite3.Connection) -> None:
        """Re-running a session must not duplicate rows or report a change."""
        upsert_risk_estimates(conn, [self.ROW])
        result = upsert_risk_estimates(conn, [self.ROW])

        assert result.unchanged == 1
        assert not result.has_changes
        count = conn.execute("SELECT COUNT(*) AS n FROM risk_estimates").fetchone()
        assert count["n"] == 1

    def test_moved_value_is_reported(self, conn: sqlite3.Connection) -> None:
        """Same inputs, different output means determinism broke. The write
        still lands -- a deliberate fix should be able to -- but it is never
        silent."""
        upsert_risk_estimates(conn, [self.ROW])
        moved = (*self.ROW[:4], 9999.0, self.ROW[5], self.ROW[6])
        result = upsert_risk_estimates(conn, [moved])

        assert result.has_changes
        as_of, method, confidence, window, stored, incoming = result.changed[0]
        assert (stored, incoming) == (1234.56, 9999.0)
        assert (as_of, method, confidence, window) == (
            "2026-07-24", "historical", 0.95, 60,
        )

    def test_windows_coexist(self, conn: sqlite3.Connection) -> None:
        """lookback_days is part of the key, so a 60-day and a 250-day estimate
        for the same date compare rather than overwrite."""
        upsert_risk_estimates(
            conn, [self.ROW, (*self.ROW[:6], 250)]
        )
        rows = conn.execute(
            "SELECT lookback_days FROM risk_estimates ORDER BY lookback_days"
        ).fetchall()
        assert [r["lookback_days"] for r in rows] == [60, 250]

    def test_methods_coexist(self, conn: sqlite3.Connection) -> None:
        parametric = (*self.ROW[:2], "parametric", *self.ROW[3:])
        upsert_risk_estimates(conn, [self.ROW, parametric])
        count = conn.execute("SELECT COUNT(*) AS n FROM risk_estimates").fetchone()
        assert count["n"] == 2

    def test_null_es_is_allowed(self, conn: sqlite3.Connection) -> None:
        """Parametric rows carry no ES: the closed form is a methodology
        decision reserved to the author, so the job stores NULL rather than a
        figure it invented."""
        upsert_risk_estimates(conn, [(*self.ROW[:5], None, 60)])
        row = conn.execute("SELECT es_amount FROM risk_estimates").fetchone()
        assert row["es_amount"] is None

    def test_applies_to_date_is_stored(self, conn: sqlite3.Connection) -> None:
        """The estimate predicts the next trading day, not the one it was
        computed from. Breach tests join on this column."""
        upsert_risk_estimates(conn, [self.ROW])
        row = conn.execute(
            "SELECT as_of_date, applies_to_date FROM risk_estimates"
        ).fetchone()
        assert row["as_of_date"] == "2026-07-24"      # Friday
        assert row["applies_to_date"] == "2026-07-27"  # Monday, not Saturday
