"""Synthetic historical reconstruction.

Live paper trading cannot produce a statistical sample: at 99% confidence the
expected breach count over a few weeks is near zero. History provides the
sample, so validation runs on a portfolio that was never held -- equal weight
across the 11 sleeves, rebalanced on the first trading day of each month, from
the same $100k notional.

This module selects and reconstructs. It does not conclude: no test statistic,
no calibration verdict, no interpretation of what a stress window means. Those
belong to `kupiec.py` and to the README's limitations section.

Two properties of the reconstruction that matter downstream:

- **Price-only.** There is no cash, no dividend, and no execution cost, so the
  series is not comparable with the live `portfolio_pnl` series and the two must
  never be concatenated into one VaR window (CLAUDE.md, "P&L definition").
- **Fractional shares.** The live portfolio floors to whole shares and carries
  the residual as cash. Replay holds exact weights instead, because the cash
  residual would otherwise be a second, unmodelled source of drift in a series
  whose whole purpose is to isolate price behaviour.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.data.calendar import is_first_trading_day_of_month

START_NOTIONAL = 100_000.0


@dataclass(frozen=True)
class ReplaySeries:
    """A reconstructed portfolio history.

    `dates[i]` is the session on which `values[i]` was observed. `returns[i]` is
    the return *into* `dates[i + 1]`, so it is one shorter than the other two --
    the first session has no prior close to difference against, exactly as the
    live series has a NULL first `daily_return`.
    """

    dates: tuple[str, ...]
    values: tuple[float, ...]
    returns: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.dates)


@dataclass(frozen=True)
class StressWindow:
    """A historical stretch selected by how badly the portfolio did in it."""

    start_date: str
    end_date: str
    portfolio_return: float
    symbol_returns: Mapping[str, float]


def load_closes(
    conn: sqlite3.Connection,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, dict[str, float]]:
    """Return {trade_date: {symbol: close}} for dates with complete coverage.

    Dates missing any symbol are dropped rather than filled. A gap-filled close
    would produce a fabricated 0% return for that symbol, which understates its
    contribution to every window containing it. Dropping keeps every symbol on
    the same calendar, which is what makes a rebalance date mean the same thing
    for all of them.
    """
    sql = "SELECT trade_date, symbol, close FROM prices"
    clauses: list[str] = []
    params: list[str] = []
    if start is not None:
        clauses.append("trade_date >= ?")
        params.append(start)
    if end is not None:
        clauses.append("trade_date <= ?")
        params.append(end)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    by_date: dict[str, dict[str, float]] = {}
    for row in conn.execute(sql + " ORDER BY trade_date", params):
        by_date.setdefault(row["trade_date"], {})[row["symbol"]] = float(row["close"])

    if not by_date:
        return {}

    symbols = {s for closes in by_date.values() for s in closes}
    return {d: c for d, c in sorted(by_date.items()) if set(c) == symbols}


def replay_series(
    closes_by_date: Mapping[str, Mapping[str, float]],
    start_notional: float = START_NOTIONAL,
) -> ReplaySeries:
    """Reconstruct an equal-weight, monthly-rebalanced portfolio.

    On the first session, and on the first trading day of every month
    thereafter, holdings are reset so each symbol carries an equal share of
    total value. Between rebalances quantities are held fixed, so weights drift
    with relative performance -- which is the behaviour being measured, not an
    approximation of it.

    Rebalancing uses the same session's closes for both valuation and
    re-entry. That is a simultaneity no live portfolio has, and it is the one
    place this reconstruction flatters itself: a real rebalance executes at
    prices that are not yet known when the decision is made. It does not
    introduce lookahead into the *risk* figures, which only ever read returns
    already realised.

    Args:
        closes_by_date: {trade_date: {symbol: close}}, complete coverage only.
        start_notional: Opening portfolio value.

    Returns:
        A ReplaySeries whose `returns` are one shorter than its `dates`.

    Raises:
        ValueError: If there are fewer than two sessions, or a session is
            missing a symbol present in the first.
    """
    dates = sorted(closes_by_date)
    if len(dates) < 2:
        raise ValueError(f"need at least 2 sessions to replay, got {len(dates)}")

    symbols = sorted(closes_by_date[dates[0]])
    weight = 1.0 / len(symbols)

    values: list[float] = []
    quantities: dict[str, float] = {}
    total = start_notional

    for i, date in enumerate(dates):
        closes = closes_by_date[date]
        missing = set(symbols) - set(closes)
        if missing:
            raise ValueError(f"{date} is missing closes for {sorted(missing)}")

        if quantities:
            total = sum(quantities[s] * closes[s] for s in symbols)

        # First session opens the position; thereafter the spec rebalances on
        # the month's first trading day.
        if i == 0 or is_first_trading_day_of_month(date):
            quantities = {s: total * weight / closes[s] for s in symbols}

        values.append(total)

    returns = tuple(
        values[i] / values[i - 1] - 1.0 for i in range(1, len(values))
    )
    return ReplaySeries(tuple(dates), tuple(values), returns)


def worst_rolling_windows(
    series: ReplaySeries,
    closes_by_date: Mapping[str, Mapping[str, float]],
    window: int = 10,
    top_n: int = 5,
    non_overlapping: bool = True,
) -> list[StressWindow]:
    """Return the worst `window`-session stretches in the replayed history.

    Selection is mechanical: every window is scored by its cumulative return and
    the worst are taken. No date is chosen by hand, which is the point -- picking
    known crises would be hindsight bias, and the code should be able to find
    stress the author did not remember (CLAUDE.md, Phase 3).

    `non_overlapping` drops any window sharing a session with an already-selected
    one. Overlapping windows around a single crash are nearly the same event
    reported several times, which would crowd out genuinely distinct episodes.
    Set it False to see the raw ranking.

    Per-symbol returns are measured across the same span, so a window can be
    replayed against any weighting -- including today's, which is what makes
    these usable as a stress scenario rather than a historical note.
    """
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")

    dates, values = series.dates, series.values
    if len(dates) <= window:
        return []

    scored = [
        (values[i + window] / values[i] - 1.0, i)
        for i in range(len(dates) - window)
    ]
    scored.sort()

    chosen: list[StressWindow] = []
    used: set[int] = set()

    for portfolio_return, i in scored:
        if len(chosen) >= top_n:
            break
        span = range(i, i + window + 1)
        if non_overlapping and any(j in used for j in span):
            continue
        used.update(span)

        start, end = dates[i], dates[i + window]
        first, last = closes_by_date[start], closes_by_date[end]
        chosen.append(
            StressWindow(
                start_date=start,
                end_date=end,
                portfolio_return=portfolio_return,
                symbol_returns={s: last[s] / first[s] - 1.0 for s in sorted(first)},
            )
        )

    return chosen


def apply_window_to_weights(
    stress: StressWindow,
    weights: Mapping[str, float],
    total_value: float,
) -> float:
    """Dollar impact of re-running `stress` against a given weighting.

    Computes sum(w_i * r_i) * total_value using the window's per-symbol returns.
    Positive is a gain, negative a loss -- the natural sign for a P&L figure,
    and deliberately *not* the positive-loss convention used for var_amount,
    which would read strangely on a scenario that happened to be profitable.

    Weights are used as supplied and are not renormalised: a portfolio holding
    cash has sleeve weights summing to less than 1, and scaling them up would
    stress a portfolio that is not the one held.

    Raises:
        ValueError: If a weighted symbol has no return in the window.
    """
    missing = set(weights) - set(stress.symbol_returns)
    if missing:
        raise ValueError(
            f"no window return for {sorted(missing)}; cannot stress those weights"
        )

    weighted = sum(weights[s] * stress.symbol_returns[s] for s in weights)
    return weighted * total_value


def rolling_var_backtest(
    series: ReplaySeries,
    lookback_days: int,
    var_fn,
    confidence: float,
) -> tuple[int, int, list[str]]:
    """Walk the series forward, estimating VaR and testing the next session.

    This is the loop the whole temporal convention exists to protect. At each
    step the estimate uses returns strictly before the session it is tested
    against, and the breach is recorded on the *predicted* date, never the date
    the estimate was computed from. Joining those the other way round would
    raise nothing and would improve the apparent result (CLAUDE.md, "Temporal
    convention").

    Both figures are expressed as fractions of portfolio value, so no scaling by
    total_value is needed: a breach is a next-session loss exceeding the VaR
    fraction estimated from the preceding window.

    Args:
        series: The reconstructed history.
        lookback_days: Window length for each estimate.
        var_fn: A callable (returns, confidence, total_value) -> float, i.e.
            historical_var or parametric_var.
        confidence: Confidence level.

    Returns:
        (breaches, observations, breach_dates). `observations` counts sessions
        actually tested, which is len(returns) - lookback_days.
    """
    returns = series.returns
    if len(returns) <= lookback_days:
        return 0, 0, []

    breaches = 0
    observations = 0
    breach_dates: list[str] = []

    for t in range(lookback_days, len(returns)):
        window = returns[t - lookback_days:t]      # strictly before session t
        var_fraction = var_fn(window, confidence, 1.0)

        realised = returns[t]
        # returns[t] is the return into dates[t + 1]: the session being
        # predicted, and the date a breach belongs to.
        applies_to = series.dates[t + 1]

        observations += 1
        if -realised > var_fraction:
            breaches += 1
            breach_dates.append(applies_to)

    return breaches, observations, breach_dates
