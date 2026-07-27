"""Schema and upsert tests."""

from __future__ import annotations

import sqlite3

import pytest

from src.db.upserts import (
    get_previous_total_value,
    insert_prices,
    latest_successful_run,
    record_run,
    upsert_portfolio_pnl,
    upsert_positions,
)


class TestSchema:
    def test_all_tables_created(self, conn: sqlite3.Connection) -> None:
        names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "prices",
            "positions",
            "portfolio_pnl",
            "risk_estimates",
            "portfolio_metrics",
            "risk_contributions",
            "runs",
        } <= names

    def test_create_schema_is_idempotent(self, conn: sqlite3.Connection) -> None:
        from src.db.schema import create_schema

        insert_prices(conn,[("2024-01-02", "XLK", 100.0)])
        create_schema(conn)
        assert conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 1

    def test_risk_estimates_pk_includes_lookback(
        self, conn: sqlite3.Connection
    ) -> None:
        """Two windows for the same date must coexist, not overwrite.

        This is the whole point of storing lookback_days per estimate: window
        changes are compared, not silently applied to history.
        """
        for lookback in (250, 500):
            conn.execute(
                """
                INSERT INTO risk_estimates
                  (as_of_date, applies_to_date, method, confidence,
                   var_amount, es_amount, lookback_days)
                VALUES ('2024-01-09', '2024-01-10', 'historical', 0.99,
                        1000.0, 1200.0, ?)
                """,
                (lookback,),
            )
        count = conn.execute("SELECT COUNT(*) FROM risk_estimates").fetchone()[0]
        assert count == 2


class TestPricesAreWriteOnce:
    """Adjusted closes are restated by every later distribution. Overwriting
    would silently rewrite the inputs of already-computed risk figures, so a
    stored close is frozen (CLAUDE.md, "Price restatement")."""

    def test_stored_close_is_never_changed(self, conn: sqlite3.Connection) -> None:
        insert_prices(conn, [("2024-01-02", "XLK", 100.0)])
        result = insert_prices(conn, [("2024-01-02", "XLK", 96.0)])

        rows = conn.execute("SELECT * FROM prices").fetchall()
        assert len(rows) == 1
        assert rows[0]["close"] == 100.0
        assert result.inserted == 0
        assert result.applied_restatements == 0

    def test_restatement_is_reported_not_swallowed(
        self, conn: sqlite3.Connection
    ) -> None:
        insert_prices(conn, [("2024-01-02", "XLK", 100.0)])
        result = insert_prices(conn, [("2024-01-02", "XLK", 96.0)])

        assert result.has_restatements
        assert result.restatements == [("2024-01-02", "XLK", 100.0, 96.0)]

    def test_identical_reinsert_is_unchanged_not_restated(
        self, conn: sqlite3.Connection
    ) -> None:
        """A plain re-run must be quiet; only genuine drift is reported."""
        rows = [("2024-01-02", "XLK", 100.0)]
        insert_prices(conn, rows)
        result = insert_prices(conn, rows)

        assert result.unchanged == 1
        assert not result.has_restatements

    def test_float_noise_is_not_a_restatement(
        self, conn: sqlite3.Connection
    ) -> None:
        insert_prices(conn, [("2024-01-02", "XLK", 100.0)])
        result = insert_prices(conn, [("2024-01-02", "XLK", 100.0 + 1e-12)])

        assert result.unchanged == 1
        assert not result.has_restatements

    def test_allow_restate_applies_the_change(
        self, conn: sqlite3.Connection
    ) -> None:
        insert_prices(conn, [("2024-01-02", "XLK", 100.0)])
        result = insert_prices(
            conn, [("2024-01-02", "XLK", 96.0)], allow_restate=True
        )

        assert result.applied_restatements == 1
        stored = conn.execute("SELECT close FROM prices").fetchone()["close"]
        assert stored == 96.0

    def test_gaps_are_filled_while_existing_rows_are_frozen(
        self, conn: sqlite3.Connection
    ) -> None:
        """The re-run case that matters: backfilling a missing day must not
        disturb the days already stored."""
        insert_prices(conn, [("2024-01-02", "XLK", 100.0)])
        result = insert_prices(
            conn,
            [("2024-01-02", "XLK", 96.0), ("2024-01-03", "XLK", 101.0)],
        )

        assert result.inserted == 1
        assert len(result.restatements) == 1
        closes = {
            r["trade_date"]: r["close"]
            for r in conn.execute("SELECT trade_date, close FROM prices")
        }
        assert closes == {"2024-01-02": 100.0, "2024-01-03": 101.0}


class TestUpsertIdempotency:
    def test_positions_overwrite_not_duplicate(
        self, conn: sqlite3.Connection
    ) -> None:
        upsert_positions(conn, [("2024-01-02", "XLK", 10.0, 1000.0)])
        upsert_positions(conn, [("2024-01-02", "XLK", 12.0, 1260.0)])

        rows = conn.execute("SELECT * FROM positions").fetchall()
        assert len(rows) == 1
        assert rows[0]["qty"] == 12.0
        assert rows[0]["market_value"] == 1260.0

    def test_pnl_overwrites_not_duplicates(self, conn: sqlite3.Connection) -> None:
        upsert_portfolio_pnl(conn, "2024-01-02", 100_000.0, 500.0, None, None)
        upsert_portfolio_pnl(conn, "2024-01-02", 100_250.0, 500.0, 250.0, 0.0025)

        rows = conn.execute("SELECT * FROM portfolio_pnl").fetchall()
        assert len(rows) == 1
        assert rows[0]["total_value"] == 100_250.0
        assert rows[0]["daily_pnl"] == 250.0

    def test_rerunning_a_day_leaves_row_count_unchanged(
        self, conn: sqlite3.Connection
    ) -> None:
        """Re-running a session must be safe; the DB is committed to git."""
        rows = [("2024-01-02", sym, 100.0) for sym in ("XLK", "XLF", "XLV")]
        insert_prices(conn,rows)
        before = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        for _ in range(3):
            insert_prices(conn,rows)
        assert conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == before


class TestRuns:
    def test_runs_are_append_only(self, conn: sqlite3.Connection) -> None:
        """A failure must never overwrite the record of an earlier success."""
        record_run(conn, "success", "first")
        record_run(conn, "failure", "second")
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2

    def test_same_second_runs_both_persist(self, conn: sqlite3.Connection) -> None:
        """Timestamps collide at second resolution; the surrogate key fixes it."""
        conn.execute(
            "INSERT INTO runs (run_at, status, message) VALUES (?, 'success', 'a')",
            ("2024-01-02T11:00:00+00:00",),
        )
        conn.execute(
            "INSERT INTO runs (run_at, status, message) VALUES (?, 'failure', 'b')",
            ("2024-01-02T11:00:00+00:00",),
        )
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2

    def test_rejects_unknown_status(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="success"):
            record_run(conn, "partial", "nope")

    def test_latest_successful_run_ignores_failures(
        self, conn: sqlite3.Connection
    ) -> None:
        conn.execute(
            "INSERT INTO runs (run_at, status) VALUES ('2024-01-02T11:00:00', 'success')"
        )
        conn.execute(
            "INSERT INTO runs (run_at, status) VALUES ('2024-01-03T11:00:00', 'failure')"
        )
        assert latest_successful_run(conn) == "2024-01-02T11:00:00"

    def test_latest_successful_run_empty(self, conn: sqlite3.Connection) -> None:
        assert latest_successful_run(conn) is None


class TestPreviousTotalValue:
    def test_returns_most_recent_earlier_day(self, conn: sqlite3.Connection) -> None:
        upsert_portfolio_pnl(conn, "2024-01-02", 100_000.0, 0.0, None, None)
        upsert_portfolio_pnl(conn, "2024-01-03", 100_500.0, 0.0, 500.0, 0.005)
        assert get_previous_total_value(conn, "2024-01-04") == 100_500.0

    def test_none_when_no_history(self, conn: sqlite3.Connection) -> None:
        assert get_previous_total_value(conn, "2024-01-02") is None

    def test_excludes_the_date_itself(self, conn: sqlite3.Connection) -> None:
        """Must be strictly earlier, or a re-run would difference against itself
        and report zero P&L."""
        upsert_portfolio_pnl(conn, "2024-01-02", 100_000.0, 0.0, None, None)
        assert get_previous_total_value(conn, "2024-01-02") is None

    def test_spans_a_gap_in_history(self, conn: sqlite3.Connection) -> None:
        """A missed run leaves a gap; the lookup must still find the last
        stored value rather than assuming a one-day step."""
        upsert_portfolio_pnl(conn, "2024-01-02", 100_000.0, 0.0, None, None)
        assert get_previous_total_value(conn, "2024-01-10") == 100_000.0
