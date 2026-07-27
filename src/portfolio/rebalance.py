"""Equal-weight rebalance arithmetic.

Pure functions only: given current holdings and prices, compute target
quantities and the orders that close the gap. No API calls, no clock reads, so
the arithmetic is directly testable and deterministic.

The allocation is fixed at 1/11 per sector ETF. This module sizes orders; it
never decides what to hold (CLAUDE.md, "Portfolio spec", "Anti-goals").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.portfolio.spec import SYMBOLS, TARGET_WEIGHT


@dataclass(frozen=True)
class OrderIntent:
    """A whole-share order to move one symbol toward its target weight."""

    symbol: str
    side: str  # 'buy' | 'sell'
    qty: int


def target_quantities(
    total_value: float,
    prices: dict[str, float],
    symbols: tuple[str, ...] = SYMBOLS,
) -> dict[str, int]:
    """Whole-share target quantity for each symbol at equal weight.

    Floors to whole shares. Fractional shares would be marginally more precise
    but complicate order handling and reconciliation for no benefit to the
    measurement, which is the deliverable.

    Flooring leaves a small cash residual, which is expected and is captured in
    `portfolio_pnl.cash`.
    """
    if total_value <= 0:
        raise ValueError(f"total_value must be positive, got {total_value}")

    missing = set(symbols) - set(prices)
    if missing:
        raise ValueError(f"No price for {sorted(missing)}; cannot size orders")

    budget = total_value * TARGET_WEIGHT
    return {symbol: int(math.floor(budget / prices[symbol])) for symbol in symbols}


def compute_orders(
    current_qty: dict[str, float],
    target_qty: dict[str, int],
) -> list[OrderIntent]:
    """Return the orders that move current holdings to target.

    Symbols already at target produce no order. Results are sorted with sells
    first so proceeds are available before buys settle, then by symbol for
    deterministic ordering.
    """
    intents: list[OrderIntent] = []
    for symbol in sorted(target_qty):
        delta = target_qty[symbol] - int(current_qty.get(symbol, 0))
        if delta > 0:
            intents.append(OrderIntent(symbol, "buy", delta))
        elif delta < 0:
            intents.append(OrderIntent(symbol, "sell", -delta))

    intents.sort(key=lambda o: (o.side != "sell", o.symbol))
    return intents
