"""Historical expected shortfall (ES), also called conditional VaR.

VaR answers "how bad is a bad day?" — it names a threshold and stops. ES answers
"when it is that bad, how bad is it on average?" — it averages every loss at or
beyond the VaR threshold. So ES sees the shape of the tail, where VaR only sees
its edge, and ES is never smaller than the VaR it extends.

This module is deliberately self-contained: it re-derives the tail cutoff rather
than importing from `src.risk.var`. Both modules must use the same percentile
convention (nearest-rank, see `_tail_count`) or the ES >= VaR relationship can
invert on small samples. That duplication is the price of being able to reason
about each file on its own; the shared convention is the thing to keep in sync.

Pure functions only: no I/O, no clock reads, no randomness. Given the same
returns the result is bit-identical every time (CLAUDE.md constraint 1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Below this many observations the tail is not being estimated, it is being
# guessed. Deliberately permissive: the honest guidance ("99% from 250 days
# rests on 2-3 observations") belongs in the README, not in a hard rejection.
MIN_OBSERVATIONS = 20


def _tail_count(n: int, confidence: float) -> int:
    """How many of the worst observations make up the (1 - confidence) tail.

    Nearest-rank convention: ceil((1 - confidence) * n). At n=100, confidence
    0.95 this is 5 -- the five worst days out of a hundred.

    The `round` is not cosmetic. (1 - 0.95) is 0.05000000000000004 in binary
    floating point, so (1 - 0.95) * 100 is 5.000000000000004 and a bare ceil
    returns 6, silently widening the tail by one observation. Rounding to 9
    decimals before the ceil removes the representation error while staying far
    below any confidence level anyone would actually request.
    """
    return max(1, math.ceil(round((1 - confidence) * n, 9)))


def _validate(returns: Sequence[float], confidence: float, total_value: float) -> None:
    """Reject inputs that would otherwise produce a plausible wrong number.

    Every check here guards against a silent failure rather than a crash. A
    risk figure that is wrong but believable gets committed to the database and
    inherited by the Kupiec test; a raised error gets recorded in `runs` and
    noticed.
    """
    if not returns:
        raise ValueError("returns is empty; cannot estimate a tail from no data")

    if total_value <= 0:
        raise ValueError(f"total_value must be positive, got {total_value}")

    # Strictly exclusive. The failure this really catches is a caller passing 95
    # instead of 0.95: (1 - 95) is negative, and Python's negative indexing
    # would then return a value from the profitable end of the sorted array
    # without raising anything at all.
    if not 0.5 < confidence < 1:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")

    # portfolio_pnl.daily_return is NULL on the first row (no prior day to
    # difference against). Through pandas that NULL arrives as NaN, and NaN
    # breaks sorting silently -- all NaN comparisons are False, so it neither
    # sorts nor raises, and every index past it shifts by one.
    #
    # Rejecting rather than dropping is deliberate: dropping observations here
    # would make the caller's stored `lookback_days` disagree with the number of
    # returns actually used, and that column is part of the primary key of
    # risk_estimates. Silently storing a row that misstates its own window
    # breaks the auditability requirement. The caller trims the leading null.
    for i, r in enumerate(returns):
        if r is None or not math.isfinite(r):
            raise ValueError(f"returns[{i}] is {r}; expected a finite number")

    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {MIN_OBSERVATIONS} returns, got {len(returns)}"
        )


def expected_shortfall(
    returns: Sequence[float],
    confidence: float,
    total_value: float,
) -> float:
    """Mean loss on the worst (1 - confidence) fraction of days, in dollars.

    Formula, for returns sorted ascending and k = ceil((1 - confidence) * n):

        ES = -mean(sorted_returns[:k]) * total_value

    That is: take the k worst daily returns, average them, flip the sign so a
    loss reads as a positive number, and scale by portfolio value.

    Returned as a positive loss magnitude, matching the sign convention used
    for `risk_estimates.var_amount` throughout the project.

    The sign flip is a negation, not abs(). If the window contains no losing
    day the tail mean is positive and the result comes back negative, which
    correctly reads as "no loss at this confidence in this window". abs() would
    report that same window as a large positive loss -- plausible, wrong, and
    undetectable downstream.

    Assumptions, both false in practice and both belonging in the README:

    - Returns are i.i.d. draws from the distribution that governs tomorrow.
      Real returns cluster in volatility, so a window blending calm and
      stressed regimes understates the tail during the stressed ones.
    - The window holds enough tail observations for the mean to be stable. At
      confidence 0.99 the tail is 1% of the window, so 250 days gives 2-3
      observations. ES averages them, which makes it steadier than VaR (which
      depends on one), but it is still a small-sample estimate.

    Estimated from the `daily_return` series in `portfolio_pnl` and scaled by
    current `total_value`, per the single P&L definition in CLAUDE.md. A figure
    computed here from data through T's close is a prediction about T+1; the
    caller is responsible for writing `applies_to_date` from the NYSE calendar.

    Args:
        returns: Daily returns as fractions (-0.02 is a 2% loss). Order does
            not matter -- the function sorts a copy.
        confidence: Confidence level, e.g. 0.95 or 0.99.
        total_value: Portfolio value to scale the result by, in dollars.

    Returns:
        Mean tail loss in dollars, positive for a loss.

    Raises:
        ValueError: On empty input, non-finite returns, non-positive
            total_value, confidence outside (0.5, 1), or fewer than
            MIN_OBSERVATIONS observations.
    """
    _validate(returns, confidence, total_value)

    # sorted() returns a new list, so the caller's sequence is never mutated and
    # the input order cannot affect the result.
    ascending = sorted(returns)

    k = _tail_count(len(ascending), confidence)
    tail = ascending[:k]

    mean_tail_return = sum(tail) / len(tail)
    return -mean_tail_return * total_value
