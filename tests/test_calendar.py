"""Calendar tests.

These guard the temporal convention. A wrong next-trading-day silently points
`applies_to_date` at a non-session, and a row that can never breach biases the
Kupiec test toward accepting the model.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.data.calendar import (
    is_first_trading_day_of_month,
    is_trading_day,
    most_recent_completed_session,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
)


class TestNextTradingDay:
    def test_midweek_advances_one_day(self) -> None:
        # Tue 2024-01-09 -> Wed 2024-01-10.
        assert next_trading_day("2024-01-09") == "2024-01-10"

    def test_friday_skips_the_weekend(self) -> None:
        # Fri 2024-01-05 -> Mon 2024-01-08.
        assert next_trading_day("2024-01-05") == "2024-01-08"

    def test_skips_a_holiday(self) -> None:
        # Wed 2024-07-03 -> Fri 2024-07-05. July 4th is a holiday, and a naive
        # weekday step would return it.
        assert next_trading_day("2024-07-03") == "2024-07-05"

    def test_skips_a_holiday_adjacent_weekend(self) -> None:
        # Thu 2024-12-24 -> Thu 2024-12-26; Christmas Day is closed.
        assert next_trading_day("2024-12-24") == "2024-12-26"

    def test_good_friday_is_closed(self) -> None:
        # Thu 2024-03-28 -> Mon 2024-04-01. Good Friday closes the NYSE even
        # though it is a normal business day elsewhere.
        assert next_trading_day("2024-03-28") == "2024-04-01"

    def test_input_need_not_be_a_trading_day(self) -> None:
        # Sat 2024-01-06 -> Mon 2024-01-08.
        assert next_trading_day("2024-01-06") == "2024-01-08"

    def test_result_is_always_a_session(self) -> None:
        day = dt.date(2024, 1, 1)
        while day < dt.date(2025, 1, 1):
            assert is_trading_day(next_trading_day(day))
            day += dt.timedelta(days=1)

    def test_accepts_date_objects(self) -> None:
        assert next_trading_day(dt.date(2024, 1, 5)) == "2024-01-08"


class TestPreviousTradingDay:
    def test_monday_goes_back_to_friday(self) -> None:
        assert previous_trading_day("2024-01-08") == "2024-01-05"

    def test_skips_a_holiday(self) -> None:
        assert previous_trading_day("2024-07-05") == "2024-07-03"

    def test_inverts_next_trading_day(self) -> None:
        for date in ("2024-01-09", "2024-07-03", "2024-03-28", "2024-12-24"):
            assert previous_trading_day(next_trading_day(date)) == date


class TestIsTradingDay:
    @pytest.mark.parametrize(
        "date",
        ["2024-01-02", "2024-07-03", "2024-12-24", "2024-11-29"],
    )
    def test_open_days(self, date: str) -> None:
        assert is_trading_day(date)

    @pytest.mark.parametrize(
        "date",
        [
            "2024-01-01",  # New Year's Day
            "2024-07-04",  # Independence Day
            "2024-03-29",  # Good Friday
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
            "2024-06-19",  # Juneteenth
            "2024-01-06",  # Saturday
            "2024-01-07",  # Sunday
        ],
    )
    def test_closed_days(self, date: str) -> None:
        assert not is_trading_day(date)


class TestFirstTradingDayOfMonth:
    def test_plain_weekday_start(self) -> None:
        # 2024-04-01 was a Monday and the month's first session.
        assert is_first_trading_day_of_month("2024-04-01")

    def test_holiday_start_moves_to_next_session(self) -> None:
        # January 1st is closed, so 2024-01-02 is the rebalance day.
        assert not is_first_trading_day_of_month("2024-01-01")
        assert is_first_trading_day_of_month("2024-01-02")

    def test_weekend_start_moves_to_monday(self) -> None:
        # 2024-06-01 was a Saturday; the first session was Monday the 3rd.
        assert not is_first_trading_day_of_month("2024-06-01")
        assert is_first_trading_day_of_month("2024-06-03")

    def test_second_session_is_not_first(self) -> None:
        assert not is_first_trading_day_of_month("2024-04-02")

    def test_exactly_twelve_rebalances_in_a_year(self) -> None:
        days = trading_days_between("2024-01-01", "2024-12-31")
        assert sum(is_first_trading_day_of_month(d) for d in days) == 12


class TestTradingDaysBetween:
    def test_endpoints_are_inclusive(self) -> None:
        days = trading_days_between("2024-01-02", "2024-01-05")
        assert days == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]

    def test_excludes_holidays_and_weekends(self) -> None:
        days = trading_days_between("2024-07-01", "2024-07-07")
        assert "2024-07-04" not in days
        assert "2024-07-06" not in days
        assert days == ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05"]

    def test_2024_had_252_sessions(self) -> None:
        assert len(trading_days_between("2024-01-01", "2024-12-31")) == 252


class TestMostRecentCompletedSession:
    def test_never_returns_today(self) -> None:
        # The free tier rejects queries touching the last 15 minutes of SIP
        # data, so the job must never request the current session.
        today = dt.date(2024, 1, 10)
        assert most_recent_completed_session(today) == "2024-01-09"

    def test_monday_returns_friday(self) -> None:
        assert most_recent_completed_session(dt.date(2024, 1, 8)) == "2024-01-05"

    def test_after_a_holiday(self) -> None:
        assert most_recent_completed_session(dt.date(2024, 7, 5)) == "2024-07-03"
