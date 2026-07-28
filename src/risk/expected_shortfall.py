"""Historical expected shortfall (ES), also called conditional VaR.

VaR answers "how bad is a bad day?" — it names a threshold and stops. ES answers
"when it is that bad, how bad is it on average?" — it averages every loss at or
beyond the VaR threshold. So ES sees the shape of the tail, where VaR only sees
its edge, and ES is never smaller than the VaR it extends.

The tail cutoff and the input guards come from `src.risk._common`, shared with
`src.risk.var`. They have to be one definition rather than two copies: if the
two modules ever disagreed about where the tail starts, the guarantee that
ES >= VaR could invert on small samples, silently and in the direction that
flatters the model.

Pure functions only: no I/O, no clock reads, no randomness. Given the same
returns the result is bit-identical every time (CLAUDE.md constraint 1).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.risk._common import _tail_count, _validate


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
