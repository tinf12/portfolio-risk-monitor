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

# Fraction of equity left uninvested when sizing orders.
#
# Sizing reads the previous close; the order fills at the next open. 1% covers
# a normal overnight gap with room to spare -- the 2026-07-28 entry moved 0.7%
# and overshot into margin on flooring residual alone. It is not enough for a
# genuine gap-up (March 2020 had opens several percent above the prior close),
# which is accepted: an occasional rejected order is a better failure than
# silent leverage, and the next rebalance corrects the drift.
#
# The cost is a permanent ~1% cash drag, which appears in the return series and
# therefore in the VaR estimated from it. That is correct -- it is the P&L
# actually experienced -- but it belongs in the README's limitations.
CASH_BUFFER = 0.01


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
    cash_buffer: float = CASH_BUFFER,
) -> dict[str, int]:
    """Whole-share target quantity for each symbol at equal weight.

    Floors to whole shares. Fractional shares would be marginally more precise
    but complicate order handling and reconciliation for no benefit to the
    measurement, which is the deliverable.

    Sizes against `total_value * (1 - cash_buffer)` rather than the full
    balance. Prices are the previous session's closes -- the only ones available
    when the job runs, since it deliberately never reads today's incomplete
    session -- but the orders fill at the *next* open. On 2026-07-28 that gap
    was about 0.7%, enough to overshoot the balance and leave the account $215
    into margin, which the long-only, no-leverage spec does not permit.

    Flooring alone is not protection: it leaves whatever happens to be left over
    after rounding, roughly half a percent here, which is the same order of
    magnitude as an ordinary day's move. The buffer makes the headroom explicit
    and constant instead of incidental.

    The buffer is a fixed fraction rather than a live quote lookup on purpose:
    sizing must be reproducible from stored inputs, and a quote read at order
    time is neither stored nor repeatable.

    Args:
        total_value: Account equity to allocate.
        prices: Close per symbol, from the session the job is processing.
        symbols: The sleeves to size. Defaults to the 11-sector spec.
        cash_buffer: Fraction of total_value deliberately left uninvested.

    Returns:
        Whole-share target quantity per symbol.

    Raises:
        ValueError: If total_value is not positive, cash_buffer is outside
            [0, 1), or any symbol has no price.
    """
    if total_value <= 0:
        raise ValueError(f"total_value must be positive, got {total_value}")

    if not 0.0 <= cash_buffer < 1.0:
        raise ValueError(f"cash_buffer must be in [0, 1), got {cash_buffer}")

    missing = set(symbols) - set(prices)
    if missing:
        raise ValueError(f"No price for {sorted(missing)}; cannot size orders")

    investable = total_value * (1.0 - cash_buffer)
    budget = investable * TARGET_WEIGHT
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
