from __future__ import annotations
import statistics
import math
from collections.abc import Sequence
from tabnanny import check
TRADING_DAYS_PER_YEAR = 252

def rolling_volatility(returns: Sequence[float], window: int = 20) -> float | None:
    """Compute annualized volatility from the most recent returns.
    Args:
        returns: A sequence of historical daily returns.
        window: The number of most recent returns to use for the volatility estimate.
    Returns:
        The annualized volatility based on the standard deviation of the last
        `window` returns, or None when there is insufficient data.
    """
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if len(returns) < window:
        return None                                                                        # Not enough data to compute rolling volatility
    for i, r in enumerate(returns):
        if not math.isfinite(r):
            raise ValueError(f"returns[{i}] is {r}; expected a finite number")

    recent = returns[-window:]
    daily = statistics.stdev(recent)

    return daily * math.sqrt(TRADING_DAYS_PER_YEAR)                                        # Annualize the volatility

def current_drawdown(total_values: Sequence[float]) -> tuple[float, float]:
    """Calculate the current drawdown from a series of total values.
    Args:
        total_values: A sequence of total portfolio values over time.
    Returns:
        A tuple containing the current drawdown (negative or zero) and the
        peak value reached in the series.
    """
    if not total_values:
        raise ValueError("total_values is empty; cannot compute drawdown")
    peak = max(total_values)
    current = total_values[-1]
    drawdown = (current - peak) / peak                                                     # Drawdown is negative or zero

    return drawdown, peak
    
