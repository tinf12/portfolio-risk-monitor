from __future__ import annotations
from collections.abc import  Mapping, Sequence
from src.risk._common import _tail_count

def historical_contribution(
    returns_by_symbol: Mapping[str, Sequence[float]],
    weights: Mapping[str,float],
    confidence: float,
    total_value: float,
) -> dict:
    """Split a historical VaR figure into per-symbol contributions.

    Finds the day that sets the VaR threshold, then splits that day's loss
    across symbols. The results add up to historical_var() exactly.
    Because it rests on one day, the split is noisy and can shift as new data
    arrives.
   
     Args:
        returns_by_symbol: Symbol to its returns as decimals (-0.023 is a 2.3%
            loss). All series must be the same length and cover the same dates.
        weights: Symbol to portfolio weight, summing to 1.0. Negative means a
            short position.
        confidence: 0.95 for 95% VaR.
        total_value: Portfolio value the weights apply to.
    Returns:
        Symbol to its share of the loss, in the same currency as total_value.
        Positive is a loss; negative means that symbol gained and offset it.
    Raises:
        ValueError: If the symbols do not match, the weights do not sum to 1.0,
            or the series are ragged or empty.
    """

    if len({len(v) for v in returns_by_symbol.values()}) != 1:
        raise ValueError(
            f"ragged history: series lengths "
            f"{ {s: len(v) for s, v in returns_by_symbol.items()} }"
        )
    if set(returns_by_symbol) != set(weights):
        raise ValueError(
            f"symbols differ: returns has {sorted(returns_by_symbol)}, "
            f"weights has {sorted(weights)}"
        )
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {total_weight}")

    
    n = len(next(iter(returns_by_symbol.values())))
    portfolio = [
    sum(returns_by_symbol[s][t] * weights[s] for s in weights)
    for t in range(n)
]
    order = sorted(range(n), key=lambda t: portfolio[t])
    k = _tail_count(n, confidence)
    tail_day = order[k - 1]

    return {
    s: -weights[s] * returns_by_symbol[s][tail_day] * total_value
    for s in weights
}