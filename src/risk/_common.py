"""Shared tail conventions for the risk estimators.

`historical_var`, `parametric_var`, and `expected_shortfall` must agree on two
things or their outputs stop being comparable:

- **What counts as the tail.** ES averages the losses beyond the VaR threshold,
  so if the two disagree about where that threshold sits, the guarantee that
  ES >= VaR can invert on small samples. It would invert silently, and in the
  direction that flatters the model.
- **What counts as a usable input.** A guard applied in one estimator and not
  another means the same window produces a figure from one and an exception
  from the other, for the same date.

Both live here so there is exactly one definition of each, rather than two
copies and a comment asking future readers to keep them in step.

The percentile convention chosen here (nearest-rank, see `_tail_count`) is a
decision, not a detail: it moves the breach count, and therefore the Kupiec
result in Phase 3. It belongs in the README as a stated choice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Below this many observations the tail is not being estimated, it is being
# guessed. Deliberately permissive: the honest caveat ("99% from 250 days rests
# on 2-3 observations") belongs in the README, not in a hard rejection.
MIN_OBSERVATIONS = 20


def _tail_count(n: int, confidence: float) -> int:
    """How many of the worst observations make up the (1 - confidence) tail.

    Nearest-rank convention: ceil((1 - confidence) * n). At n=100 and
    confidence 0.95 this is 5, the five worst days out of a hundred.

    The `round` is not cosmetic. (1 - 0.95) is 0.05000000000000004 in binary
    floating point, so (1 - 0.95) * 100 is 5.000000000000004 and a bare ceil
    returns 6, silently widening the tail by one observation. Rounding to 9
    decimals removes the representation error while staying far below any
    confidence level anyone would actually request.

    Args:
        n: Number of observations in the sample.
        confidence: Confidence level as a decimal, e.g. 0.95 for 95% VaR.

    Returns:
        Count of observations in the tail, always at least 1.
    """
    return max(1, math.ceil(round((1 - confidence) * n, 9)))


def _validate(returns: Sequence[float], confidence: float, total_value: float) -> None:
    """Reject inputs that would produce a plausible but wrong number.

    Every check here guards a silent failure rather than a crash. A risk figure
    that is wrong but believable gets written to risk_estimates and inherited by
    the Kupiec test; a raised error gets recorded in `runs` and noticed.

    Args:
        returns: Periodic returns as decimals.
        confidence: Confidence level as a decimal, e.g. 0.95.
        total_value: Portfolio value the returns apply to.

    Raises:
        ValueError: On empty returns, non-finite observations, non-positive
            total_value, confidence outside (0.5, 1), or fewer than
            MIN_OBSERVATIONS observations.
    """
    if not returns:
        raise ValueError("returns is empty; cannot estimate a tail from no data")

    if total_value <= 0:
        raise ValueError(f"total_value must be positive, got {total_value}")

    # Strictly exclusive, and floored at 0.5. The mistake this really catches is
    # a caller passing the tail probability (0.05) where the confidence level
    # (0.95) belongs: that produces a valid index pointing at a gain, so nothing
    # downstream fails and the number just quietly understates risk.
    if not 0.5 < confidence < 1:
        raise ValueError(f"confidence must be between 0.5 and 1, got {confidence}")

    # portfolio_pnl.daily_return is NULL on the first row (no prior day to
    # difference against), and through pandas that NULL arrives as NaN. NaN
    # breaks sorting silently: every comparison against it is False, so sorted()
    # neither raises nor orders correctly, and every index past it shifts by one.
    #
    # Rejecting rather than dropping is deliberate. Dropping would make the
    # caller's stored lookback_days disagree with the number of returns actually
    # used, and that column is part of the primary key of risk_estimates.
    # Trimming the leading null is the caller's job.
    for i, r in enumerate(returns):
        if r is None or not math.isfinite(r):
            raise ValueError(f"returns[{i}] is {r}; expected a finite number")

    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(f"need at least {MIN_OBSERVATIONS} returns, got {len(returns)}")
