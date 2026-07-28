"""Replay reconstruction and the no-lookahead guarantee.

The arithmetic cases use two symbols moving in opposite directions, so an
equal-weight portfolio's value is checkable in one line: A at +10% and B at -10%
leaves a 50/50 book exactly flat.

The lookahead case is the important one. `rolling_var_backtest` is where an
estimate could silently be allowed to see the session it is being tested
against, which raises nothing and improves every result.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.db.upserts import insert_prices
from src.risk.var import historical_var
from src.validation.replay import (
    ReplaySeries,
    apply_window_to_weights,
    load_closes,
    replay_series,
    rolling_var_backtest,
    worst_rolling_windows,
)

# Four consecutive NYSE sessions; none is a month's first trading day, so no
# rebalance fires and quantities stay fixed across them.
SESSIONS = ["2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"]


def _closes(*rows: tuple[float, float]) -> dict[str, dict[str, float]]:
    """Build {date: {A, B}} from (a_close, b_close) pairs."""
    return {
        SESSIONS[i]: {"A": a, "B": b} for i, (a, b) in enumerate(rows)
    }


class TestReplaySeries:
    def test_opens_at_the_start_notional(self) -> None:
        series = replay_series(_closes((100.0, 100.0), (100.0, 100.0)))
        assert series.values[0] == pytest.approx(100_000.0)

    def test_flat_prices_give_zero_returns(self) -> None:
        series = replay_series(_closes((100.0, 100.0), (100.0, 100.0)))
        assert series.returns == pytest.approx((0.0,))

    def test_offsetting_moves_cancel(self) -> None:
        """+10% and -10% at equal weight is exactly flat: 500 shares each at
        $100 becomes 500x110 + 500x90 = 100,000."""
        series = replay_series(_closes((100.0, 100.0), (110.0, 90.0)))
        assert series.values[1] == pytest.approx(100_000.0)
        assert series.returns[0] == pytest.approx(0.0)

    def test_common_move_passes_through(self) -> None:
        series = replay_series(_closes((100.0, 100.0), (110.0, 110.0)))
        assert series.values[1] == pytest.approx(110_000.0)
        assert series.returns[0] == pytest.approx(0.10)

    def test_returns_are_one_shorter_than_dates(self) -> None:
        """The first session has no prior close, mirroring the NULL first
        daily_return in portfolio_pnl."""
        series = replay_series(
            _closes((100.0, 100.0), (101.0, 101.0), (102.0, 102.0))
        )
        assert len(series.dates) == 3
        assert len(series.returns) == 2

    def test_weights_drift_between_rebalances(self) -> None:
        """Quantities are held fixed, so a winner's weight grows. This is the
        behaviour being measured, not an approximation of it."""
        closes = _closes((100.0, 100.0), (200.0, 100.0), (200.0, 100.0))
        series = replay_series(closes)
        # 500 shares each: 500*200 + 500*100 = 150,000, of which A is 2/3.
        assert series.values[1] == pytest.approx(150_000.0)

    def test_rebalance_resets_to_equal_weight(self) -> None:
        """2026-08-03 is the first trading day of August (the 1st is a
        Saturday), so holdings reset there and the next session's return
        reflects equal weights again rather than the drifted ones."""
        closes = {
            "2026-07-30": {"A": 100.0, "B": 100.0},
            "2026-07-31": {"A": 200.0, "B": 100.0},   # A doubles; drifted 2:1
            "2026-08-03": {"A": 200.0, "B": 100.0},   # rebalance day
            "2026-08-04": {"A": 220.0, "B": 100.0},   # A +10% post-reset
        }
        series = replay_series(closes)
        # After the reset each sleeve holds 75,000. A gains 10% -> +7,500.
        assert series.values[2] == pytest.approx(150_000.0)
        assert series.values[3] == pytest.approx(157_500.0)
        assert series.returns[2] == pytest.approx(0.05)

    def test_rejects_a_single_session(self) -> None:
        with pytest.raises(ValueError):
            replay_series({SESSIONS[0]: {"A": 100.0, "B": 100.0}})

    def test_rejects_a_missing_symbol(self) -> None:
        closes = _closes((100.0, 100.0), (110.0, 110.0))
        del closes[SESSIONS[1]]["B"]
        with pytest.raises(ValueError):
            replay_series(closes)


class TestLoadCloses:
    def test_drops_dates_with_partial_coverage(
        self, conn: sqlite3.Connection
    ) -> None:
        """A filled gap would become a fabricated 0% return for the missing
        symbol, understating its contribution to every window containing it."""
        insert_prices(
            conn,
            [
                ("2026-07-22", "A", 100.0),
                ("2026-07-22", "B", 100.0),
                ("2026-07-23", "A", 101.0),          # B missing
                ("2026-07-24", "A", 102.0),
                ("2026-07-24", "B", 102.0),
            ],
        )
        got = load_closes(conn)
        assert sorted(got) == ["2026-07-22", "2026-07-24"]

    def test_respects_the_date_range(self, conn: sqlite3.Connection) -> None:
        insert_prices(
            conn,
            [
                (d, s, 100.0)
                for d in ("2026-07-22", "2026-07-23", "2026-07-24")
                for s in ("A", "B")
            ],
        )
        assert sorted(load_closes(conn, start="2026-07-23")) == [
            "2026-07-23", "2026-07-24",
        ]
        assert sorted(load_closes(conn, end="2026-07-23")) == [
            "2026-07-22", "2026-07-23",
        ]

    def test_empty_database(self, conn: sqlite3.Connection) -> None:
        assert load_closes(conn) == {}


class TestWorstRollingWindows:
    @staticmethod
    def _falling_series() -> tuple[ReplaySeries, dict[str, dict[str, float]]]:
        """Ten sessions with one sharp two-day drop in the middle."""
        levels = [100.0, 101.0, 102.0, 103.0, 80.0, 78.0, 79.0, 80.0, 81.0, 82.0]
        dates = [f"2026-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(len(levels))]
        closes = {d: {"A": v, "B": v} for d, v in zip(dates, levels)}
        series = ReplaySeries(
            tuple(dates),
            tuple(v * 1000 for v in levels),
            tuple(
                levels[i] / levels[i - 1] - 1.0 for i in range(1, len(levels))
            ),
        )
        return series, closes

    def test_finds_the_drop(self) -> None:
        series, closes = self._falling_series()
        windows = worst_rolling_windows(series, closes, window=2, top_n=1)
        assert len(windows) == 1
        # 103 -> 78 across sessions 3..5.
        assert windows[0].portfolio_return == pytest.approx(78 / 103 - 1)

    def test_non_overlapping_selection_is_default(self) -> None:
        series, closes = self._falling_series()
        windows = worst_rolling_windows(series, closes, window=2, top_n=3)
        spans = [(w.start_date, w.end_date) for w in windows]
        assert len(spans) == len(set(spans))
        # No selected window may share a session with another.
        starts = [series.dates.index(w.start_date) for w in windows]
        assert all(abs(a - b) > 2 for i, a in enumerate(starts) for b in starts[i + 1:])

    def test_overlapping_allowed_when_asked(self) -> None:
        series, closes = self._falling_series()
        overlapping = worst_rolling_windows(
            series, closes, window=2, top_n=3, non_overlapping=False
        )
        assert len(overlapping) == 3

    def test_carries_per_symbol_returns(self) -> None:
        series, closes = self._falling_series()
        window = worst_rolling_windows(series, closes, window=2, top_n=1)[0]
        assert set(window.symbol_returns) == {"A", "B"}

    def test_empty_when_history_is_shorter_than_the_window(self) -> None:
        series, closes = self._falling_series()
        assert worst_rolling_windows(series, closes, window=50) == []

    def test_rejects_non_positive_window(self) -> None:
        series, closes = self._falling_series()
        with pytest.raises(ValueError):
            worst_rolling_windows(series, closes, window=0)


class TestApplyWindowToWeights:
    def test_weighted_impact(self) -> None:
        from src.validation.replay import StressWindow

        stress = StressWindow(
            start_date="2020-03-04",
            end_date="2020-03-18",
            portfolio_return=-0.25,
            symbol_returns={"A": -0.40, "B": -0.10},
        )
        # 50/50 of -40% and -10% is -25% of 200,000 = -50,000.
        got = apply_window_to_weights(stress, {"A": 0.5, "B": 0.5}, 200_000.0)
        assert got == pytest.approx(-50_000.0)

    def test_negative_is_a_loss(self) -> None:
        from src.validation.replay import StressWindow

        stress = StressWindow("a", "b", -0.25, {"A": -0.40, "B": -0.10})
        assert apply_window_to_weights(stress, {"A": 1.0}, 100.0) < 0

    def test_weights_are_not_renormalised(self) -> None:
        """A portfolio holding cash has sleeve weights summing to under 1.
        Scaling them up would stress a portfolio that is not the one held."""
        from src.validation.replay import StressWindow

        stress = StressWindow("a", "b", -0.25, {"A": -0.40, "B": -0.10})
        half_invested = apply_window_to_weights(
            stress, {"A": 0.25, "B": 0.25}, 200_000.0
        )
        assert half_invested == pytest.approx(-25_000.0)

    def test_rejects_a_weight_without_a_return(self) -> None:
        from src.validation.replay import StressWindow

        stress = StressWindow("a", "b", -0.25, {"A": -0.40})
        with pytest.raises(ValueError):
            apply_window_to_weights(stress, {"A": 0.5, "B": 0.5}, 100.0)


class TestBacktestHasNoLookahead:
    @staticmethod
    def _series_with_one_crash(crash_at: int, n: int = 60) -> ReplaySeries:
        """Calm returns everywhere except a single large loss."""
        returns = [0.001 if i % 2 else -0.001 for i in range(n)]
        returns[crash_at] = -0.20
        dates = tuple(f"d{i:03d}" for i in range(n + 1))
        return ReplaySeries(dates, tuple([100.0] * (n + 1)), tuple(returns))

    def test_estimate_cannot_see_the_session_it_predicts(self) -> None:
        """With a 20-day window, the estimate tested against the crash is built
        from the 20 calm sessions before it, so the crash breaches. An
        implementation that included session t in its own window would size VaR
        off the -20% move and record no breach at all."""
        series = self._series_with_one_crash(crash_at=30)
        breaches, observations, dates = rolling_var_backtest(
            series, 20, historical_var, 0.99
        )
        assert observations == 40
        assert breaches == 1
        assert dates == ["d031"]

    def test_breach_is_recorded_on_the_predicted_session(self) -> None:
        """returns[t] is the move into dates[t+1]. Recording it against
        dates[t] would shift every breach one session early -- the join error
        CLAUDE.md calls out, which raises nothing and flatters the model."""
        series = self._series_with_one_crash(crash_at=25)
        _b, _o, dates = rolling_var_backtest(series, 20, historical_var, 0.99)
        assert dates == ["d026"]

    def test_no_observations_when_history_is_too_short(self) -> None:
        series = self._series_with_one_crash(crash_at=5, n=10)
        assert rolling_var_backtest(series, 20, historical_var, 0.99) == (0, 0, [])

    def test_every_session_after_the_window_is_tested(self) -> None:
        series = self._series_with_one_crash(crash_at=30, n=100)
        _b, observations, _d = rolling_var_backtest(
            series, 30, historical_var, 0.95
        )
        assert observations == 70
