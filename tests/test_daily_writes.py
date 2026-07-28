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


class TestWhenOrdersAreSubmitted:
    """The one code path that spends money.

    SESSION (2026-07-24) is a Friday mid-month, so it is not a rebalance date.
    Whether orders go out therefore depends entirely on whether the account
    already holds anything.
    """

    HELD = AccountSnapshot(
        total_value=100_000.0,
        cash=1_000.0,
        positions=tuple((s, 90.0, 9_000.0) for s in SYMBOLS),
    )

    def _run(self, snapshot: AccountSnapshot, session: str = SESSION):
        from src.jobs.daily import run_daily

        with (
            patch("src.jobs.daily.fetch_daily_closes", return_value=PRICES),
            patch("src.jobs.daily.fetch_account_snapshot", return_value=snapshot),
            patch("src.jobs.daily.most_recent_completed_session", return_value=SESSION),
            patch("src.jobs.daily.submit_orders") as submit,
        ):
            run_daily(session)
        return submit

    def test_empty_account_buys_on_any_session(self, job_db: Path) -> None:
        """An empty account does not hold the 11 sleeves the spec requires, so
        waiting for the month's first trading day would leave it in cash for up
        to a month purely because of when it was funded."""
        submit = self._run(SNAPSHOT)  # SNAPSHOT holds no positions
        assert submit.called
        orders = submit.call_args.args[0]
        assert len(orders) == len(SYMBOLS)
        assert all(o.side == "buy" for o in orders)

    def test_held_account_does_not_trade_off_schedule(self, job_db: Path) -> None:
        """The spec is long-only with no discretionary overrides: a portfolio
        already at target must not be touched between rebalance dates."""
        submit = self._run(self.HELD)
        assert not submit.called

    def test_skip_orders_wins_over_an_empty_account(self, job_db: Path) -> None:
        """--skip-orders is the operator's stop button; the empty-account
        trigger must not override it."""
        from src.jobs.daily import run_daily

        with (
            patch("src.jobs.daily.fetch_daily_closes", return_value=PRICES),
            patch("src.jobs.daily.fetch_account_snapshot", return_value=SNAPSHOT),
            patch("src.jobs.daily.most_recent_completed_session", return_value=SESSION),
            patch("src.jobs.daily.submit_orders") as submit,
        ):
            run_daily(SESSION, skip_orders=True)

        assert not submit.called

    def test_dry_run_does_not_submit(self, job_db: Path) -> None:
        """dry_run reaches submit_orders but must pass the flag through, so
        nothing is sent."""
        from src.jobs.daily import run_daily

        with (
            patch("src.jobs.daily.fetch_daily_closes", return_value=PRICES),
            patch("src.jobs.daily.fetch_account_snapshot", return_value=SNAPSHOT),
            patch("src.jobs.daily.most_recent_completed_session", return_value=SESSION),
            patch("src.jobs.daily.submit_orders") as submit,
        ):
            run_daily(SESSION, dry_run=True)

        assert submit.call_args.kwargs["dry_run"] is True
