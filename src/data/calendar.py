"""NYSE trading calendar.

`applies_to_date` is written at T's close, before T+1 exists in the prices
table, so it cannot be derived from stored data. It comes from here.

Never compute the next trading day as "date + 1 day" or by skipping weekends
alone. That points rows at market holidays. A holiday row can never breach,
which inflates the pass rate and biases the Kupiec test toward accepting the
model (CLAUDE.md, "Temporal convention").
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas_market_calendars as mcal

EXCHANGE = "XNYS"

# Widened well past both ends of the project's data range so the cached schedule
# never needs regenerating mid-run.
_CALENDAR_START = dt.date(2015, 1, 1)
_CALENDAR_END = dt.date(2035, 12, 31)


@lru_cache(maxsize=1)
def _trading_days() -> tuple[dt.date, ...]:
    """All NYSE session dates in the covered range, ascending."""
    calendar = mcal.get_calendar(EXCHANGE)
    schedule = calendar.schedule(
        start_date=_CALENDAR_START.isoformat(),
        end_date=_CALENDAR_END.isoformat(),
    )
    return tuple(d.date() for d in schedule.index)


def _parse(value: str | dt.date) -> dt.date:
    return dt.date.fromisoformat(value) if isinstance(value, str) else value


def is_trading_day(date: str | dt.date) -> bool:
    """True if the NYSE holds a session on this date."""
    return _parse(date) in set(_trading_days())


def next_trading_day(date: str | dt.date) -> str:
    """Return the first NYSE session strictly after `date`, as YYYY-MM-DD.

    This is the value to store in `applies_to_date`. `date` need not itself be
    a trading day.
    """
    target = _parse(date)
    for day in _trading_days():
        if day > target:
            return day.isoformat()
    raise ValueError(
        f"No trading day after {target}; extend _CALENDAR_END in {__name__}"
    )


def previous_trading_day(date: str | dt.date) -> str:
    """Return the last NYSE session strictly before `date`, as YYYY-MM-DD."""
    target = _parse(date)
    for day in reversed(_trading_days()):
        if day < target:
            return day.isoformat()
    raise ValueError(
        f"No trading day before {target}; extend _CALENDAR_START in {__name__}"
    )


def trading_days_between(
    start: str | dt.date,
    end: str | dt.date,
) -> list[str]:
    """Return NYSE sessions from start to end inclusive, as YYYY-MM-DD."""
    lo, hi = _parse(start), _parse(end)
    return [d.isoformat() for d in _trading_days() if lo <= d <= hi]


def most_recent_completed_session(today: dt.date | None = None) -> str:
    """Return the last session whose close is safely available.

    The Alpaca free tier rejects queries touching the last 15 minutes of SIP
    data, so the job always requests bars through the *previous* session rather
    than the current, possibly incomplete, one (CLAUDE.md, "Data constraints").

    `today` is injectable so tests do not depend on the wall clock.
    """
    reference = today if today is not None else dt.datetime.now(dt.UTC).date()
    return previous_trading_day(reference)


def is_first_trading_day_of_month(date: str | dt.date) -> bool:
    """True if `date` is the month's first NYSE session — a rebalance day.

    Derived from the exchange calendar, not the day number, so a month whose
    1st falls on a weekend or holiday rebalances on the correct session.
    """
    target = _parse(date)
    if target not in set(_trading_days()):
        return False
    same_month = [
        d
        for d in _trading_days()
        if d.year == target.year and d.month == target.month
    ]
    return bool(same_month) and same_month[0] == target
