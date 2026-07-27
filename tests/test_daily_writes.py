"""What the daily job is allowed to write.

Both cases here were real bugs found by running the job, not by testing it.
A dry run left rows behind, and a backdated run stored today's live account
equity against a past date — fabricating history that would have fed straight
into the Phase 2 return series.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from src.portfolio.positions import AccountSnapshot
from src.portfolio.spec import SYMBOLS

SESSION = "2026-07-24"  # the "current" completed session in these tests
PAST = "2026-07-01"

PRICES = [(SESSION, s, 100.0) for s in SYMBOLS]
PAST_PRICES = [(PAST, s, 100.0) for s in SYMBOLS]

SNAPSHOT = AccountSnapshot(
    total_value=100_000.0,
    cash=100_000.0,
    positions=(),
)


@pytest.fixture
def job_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the job at a scratch database."""
    db = tmp_path / "job.db"
    monkeypatch.setenv("RISK_DB_PATH", str(db))
    yield db


def _counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(db)
    try:
        return {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("prices", "positions", "portfolio_pnl", "runs")
        }
    finally:
        conn.close()


class TestDryRunWritesNothing:
    def test_no_rows_survive_a_dry_run(self, job_db: Path) -> None:
        from src.jobs.daily import run_daily

        with (
            patch("src.jobs.daily.fetch_daily_closes", return_value=PRICES),
            patch("src.jobs.daily.fetch_account_snapshot", return_value=SNAPSHOT),
            patch("src.jobs.daily.most_recent_completed_session", return_value=SESSION),
            patch("src.jobs.daily.submit_orders"),
        ):
            run_daily(SESSION, dry_run=True)

        assert _counts(job_db) == {
            "prices": 0,
            "positions": 0,
            "portfolio_pnl": 0,
            "runs": 0,
        }

    def test_real_run_does_write(self, job_db: Path) -> None:
        """Control: the same run without dry_run must persist."""
        from src.jobs.daily import run_daily

        with (
            patch("src.jobs.daily.fetch_daily_closes", return_value=PRICES),
            patch("src.jobs.daily.fetch_account_snapshot", return_value=SNAPSHOT),
            patch("src.jobs.daily.most_recent_completed_session", return_value=SESSION),
            patch("src.jobs.daily.submit_orders"),
        ):
            run_daily(SESSION)

        counts = _counts(job_db)
        assert counts["prices"] == len(SYMBOLS)
        assert counts["portfolio_pnl"] == 1
        assert counts["runs"] == 1


class TestBackdatedRunsDoNotFabricateHistory:
    """Account state is a live read with no historical equivalent. Storing it
    against a past date would record equity the account never held on that
    date, and that row would feed the Phase 2 return series."""

    def test_backdated_run_stores_prices_but_not_account_state(
        self, job_db: Path
    ) -> None:
        from src.jobs.daily import run_daily

        with (
            patch("src.jobs.daily.fetch_daily_closes", return_value=PAST_PRICES),
            patch("src.jobs.daily.fetch_account_snapshot", return_value=SNAPSHOT),
            patch("src.jobs.daily.most_recent_completed_session", return_value=SESSION),
            patch("src.jobs.daily.submit_orders"),
        ):
            run_daily(PAST)

        counts = _counts(job_db)
        assert counts["prices"] == len(SYMBOLS)  # historical fact, safe to store
        assert counts["portfolio_pnl"] == 0  # live state, not attributable
        assert counts["positions"] == 0

    def test_current_session_stores_account_state(self, job_db: Path) -> None:
        """Control: the same path on the current session does store it."""
        from src.jobs.daily import run_daily

        with (
            patch("src.jobs.daily.fetch_daily_closes", return_value=PRICES),
            patch("src.jobs.daily.fetch_account_snapshot", return_value=SNAPSHOT),
            patch("src.jobs.daily.most_recent_completed_session", return_value=SESSION),
            patch("src.jobs.daily.submit_orders"),
        ):
            run_daily(SESSION)

        assert _counts(job_db)["portfolio_pnl"] == 1
