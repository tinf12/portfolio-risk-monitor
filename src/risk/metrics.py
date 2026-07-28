"""Portfolio-level metrics that are not tail estimates.

Volatility and drawdown describe the session they are computed for, rather than
predicting the next one, which is why `portfolio_metrics` is keyed on
as_of_date alone and carries no applies_to_date.

The two take different inputs, and confusing them is the way this goes wrong:
volatility measures the dispersion of *returns*, drawdown measures a position
within the history of *levels*.

Pure functions only: no I/O, no clock reads, no randomness (CLAUDE.md
constraint 1).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

TRADING_DAYS_PER_YEAR = 252


def rolling_volatility(returns: Sequence[float], window: int = 20) -> float | None:
    """Compute annualized volatility from the most recent returns.

    Formula, for the last `window` returns:

        vol = stdev(returns[-window:]) * sqrt(TRADING_DAYS_PER_YEAR)

    `stdev` is the sample (n-1) standard deviation, matching `parametric_var`.
    The population version would report 0.1587 where this gives 0.1629 on a
    20-observation series; either is defensible, using both across one project
    is not.

    The sqrt scaling is why a 1% daily standard deviation is roughly 16%
    annualized: variance accumulates linearly with time, so standard deviation
    accumulates with its square root, and sqrt(252) is 15.8745. The count is 252
    rather than 365 because the return series has entries only for sessions the
    exchange was open.

    Assumptions, both false in practice and both belonging in the README:

    - Returns are independent across days. Volatility clusters instead, so a
      window spanning calm and stressed regimes reports a single number that is
      too high for the calm stretch and too low for the stressed one. This is
      the same assumption that makes VaR understate risk in a crisis.
    - The window is representative of current conditions. A 20-day window turns
      over in a month, so this reacts quickly and is correspondingly noisy.

    Args:
        returns: A sequence of historical daily returns as decimals.
        window: The number of most recent returns to use for the volatility
            estimate. Defaults to 20, matching portfolio_metrics.vol_20d.

    Returns:
        The annualized volatility as a decimal fraction, based on the standard
        deviation of the last `window` returns; or None when fewer than
        `window` returns are available. None rather than a figure from a
        shorter window, which would be a false claim in a column named for a
        20-day one, and rather than NaN, which sorts silently and propagates
        through aggregates.

    Raises:
        ValueError: If `window` is not positive, or any return is non-finite.
    """
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if len(returns) < window:
        return None  # not enough history yet; see Returns above
    for i, r in enumerate(returns):
        if not math.isfinite(r):
            raise ValueError(f"returns[{i}] is {r}; expected a finite number")

    recent = returns[-window:]
    daily = statistics.stdev(recent)

    return daily * math.sqrt(TRADING_DAYS_PER_YEAR)


def current_drawdown(total_values: Sequence[float]) -> tuple[float, float]:
    """Calculate the current drawdown from a series of total values.

    Formula:

        peak     = max(total_values)
        drawdown = (total_values[-1] - peak) / peak

    Negative or zero by construction: the peak is a maximum that includes the
    current value, so the current value can never exceed it, and at a new high
    the two are equal and the result is exactly 0.0. This is the opposite sign
    convention from var_amount, which is a positive loss magnitude. Both are
    deliberate and both are recorded in the schema.

    This is *current* drawdown, not maximum drawdown. On [100, 50, 90] it
    reports -0.1, not -0.5: the portfolio fell 50% at the trough but stands 10%
    below its high now. Maximum drawdown is the worst value this ever reached
    and is a different statistic.

    The peak runs over the whole series passed in, never a trailing window. A
    peak taken over recent history would understate the decline, which is the
    direction that flatters the portfolio.

    Assumption worth stating in the README: the peak is the highest value in
    *stored* history. A decline from a high that predates the series cannot be
    seen, so drawdown is understated for as long as the record is shorter than
    the drawdown being measured.

    Args:
        total_values: A sequence of total portfolio values over time, oldest
            first. Levels, not returns.

    Returns:
        A tuple containing the current drawdown as a decimal fraction (negative
        or zero) and the peak value reached in the series.

    Raises:
        ValueError: If `total_values` is empty. The daily job calls this only
            after writing a total_value row, so an empty series means something
            upstream is broken rather than that history is short.
    """
    if not total_values:
        raise ValueError("total_values is empty; cannot compute drawdown")
    peak = max(total_values)
    current = total_values[-1]
    drawdown = (current - peak) / peak

    return drawdown, peak
