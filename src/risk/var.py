from __future__ import annotations 
import math
import statistics
from scipy.stats import norm

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

def historical_var(returns, confidence, total_value) -> float:
    ascending = sorted(returns)
    k = _tail_count(len(ascending), confidence)
    worst_at_threshold = ascending[k-1]
    return -worst_at_threshold * total_value

def parametric_var(returns, confidence, total_value) -> float:
    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns)
    z = norm.ppf(confidence)
    return -(mean - z * stdev) * total_value