"""Single-period Value at Risk, computed two ways.
 
Sign convention used throughout: VaR comes back as a POSITIVE number
representing a loss. A 95% VaR of 23_000 means "on the worst 5% of days, expect
to lose at least $23,000." A negative result is legitimate and means even the
tail scenario was a gain, so do not wrap these functions in abs(): the sign
carries information."""

from __future__ import annotations 
import math
import statistics
from collections.abc import Sequence

from scipy.stats import norm

# Below this many observations the tail is not being estimated, it is being
# guessed. Deliberately permissive: the honest caveat ("99% from 250 days rests
# on 2-3 observations") belongs in the README, not in a hard rejection.
MIN_OBSERVATIONS = 20


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


def _tail_count(n: int, confidence: float) -> int:
    """How many of the worst observations make up the (1 - confidence) tail.

    Nearest-rank convention: ceil((1 - confidence) * n). At n=100 and
    confidence 0.95 this is 5, the five worst days out of a hundred.

    Args:
        n: Number of observations in the sample.
        confidence: Confidence level as a decimal, e.g. 0.95 for 95% VaR.

    Returns:
        Count of observations in the tail, always at least 1.
    """
    return max(1, math.ceil(round((1 - confidence) * n, 9)))       # returns to 9 decimals to account for floating points


def historical_var(
    returns: Sequence[float],
    confidence: float,
    total_value: float,
) -> float:
    """VaR read straight off the empirical distribution.

    Makes no distributional assumption. This is literally "sort the days and
    look at the cutoff." The tradeoff: it is blind to any scenario absent from
    the lookback window, and can never report a loss worse than the worst
    observation it was handed.

    Args:
        returns: Periodic returns as decimals, e.g. -0.023 for a 2.3% loss.
        confidence: Confidence level as a decimal, e.g. 0.95.
        total_value: Portfolio value the returns apply to.

    Returns:
        Loss magnitude in the same currency units as total_value.

    Raises:
        ValueError: See _validate.
    """
    _validate(returns, confidence, total_value)

    ascending = sorted(returns)                                    # most negative first
    k = _tail_count(len(ascending), confidence)
    worst_at_threshold = ascending[k-1]
    return -worst_at_threshold * total_value                       # Negate to flip a loss (-0.023) into a positive VaR figure.

def parametric_var(
    returns: Sequence[float],
    confidence: float,
    total_value: float,
) -> float:
    """VaR from a normal distribution fitted to the sample's first two moments.

    Smooth, and extrapolates to any confidence level even one the sample is too
    small to support. The cost is the normality assumption: real returns are
    fat-tailed and left-skewed, so this understates risk, and understates it
    worse at 99% than at 95%. Treat it as a cross-check against
    historical_var, not a replacement for it.

    Args:
        returns: Periodic returns as decimals, e.g. -0.023 for a 2.3% loss.
        confidence: Confidence level as a decimal, e.g. 0.95.
        total_value: Portfolio value the returns apply to.

    Returns:
        Loss magnitude in the same currency units as total_value.

    Raises:
        ValueError: See _validate, plus the zero-dispersion case below.
    """
    _validate(returns, confidence, total_value)

    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns)                              # sample stdev, n-1 denominator

    # A normal with zero width has no tail to read a quantile off: the formula
    # would collapse to -mean * total_value and report a VaR near zero, which
    # looks like "no risk" rather than "this input cannot support an estimate".
    # In practice this means every return in the window was identical, which is
    # a data fault (a stale price repeated) rather than a market observation.
    if stdev == 0:
        raise ValueError(
            f"all {len(returns)} returns are identical ({mean}); a normal fit "
            f"has zero dispersion and no tail to estimate"
        )

    z = norm.ppf(confidence)                                       # 1.644854 at 0.95, 2.326348 at 0.99
    return -(mean - z * stdev) * total_value