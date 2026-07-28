"""Single-period Value at Risk, computed two ways.
 
Sign convention used throughout: VaR comes back as a POSITIVE number
representing a loss. A 95% VaR of 23_000 means "on the worst 5% of days, expect
to lose at least $23,000." A negative result is legitimate and means even the
tail scenario was a gain, so do not wrap these functions in abs(): the sign
carries information."""

from __future__ import annotations 
import math
import statistics
from scipy.stats import norm



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


def historical_var(returns, confidence, total_value) -> float:
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
        IndexError: If returns is empty.
    """
    ascending = sorted(returns)                                    # most negative first
    k = _tail_count(len(ascending), confidence)
    worst_at_threshold = ascending[k-1]
    return -worst_at_threshold * total_value                       # Negate to flip a loss (-0.023) into a positive VaR figure.

def parametric_var(returns, confidence, total_value) -> float:
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
        statistics.StatisticsError: If returns has fewer than 2 observations.
    """
    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns)                              # sample stdev, n-1 denominator
    z = norm.ppf(confidence)                                       # 1.644854 at 0.95, 2.326348 at 0.99
    return -(mean - z * stdev) * total_value