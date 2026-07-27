"""Order submission to the Alpaca paper account."""

from __future__ import annotations

import logging

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.data.alpaca_client import get_trading_client, with_retry
from src.portfolio.rebalance import OrderIntent

logger = logging.getLogger(__name__)


def submit_orders(intents: list[OrderIntent], *, dry_run: bool = False) -> int:
    """Submit market orders, returning the number submitted.

    Market-on-open day orders. The job runs outside market hours, so these
    queue for the next session rather than filling immediately — which is why
    positions are always snapshotted before rebalancing, not after.

    `dry_run` logs without submitting, for verifying sizing against a real
    account before letting the scheduled job trade.
    """
    if not intents:
        logger.info("No orders needed; portfolio already at target.")
        return 0

    if dry_run:
        for intent in intents:
            logger.info(
                "[dry run] %s %d %s", intent.side, intent.qty, intent.symbol
            )
        return 0

    client = get_trading_client()
    submitted = 0
    for intent in intents:
        request = MarketOrderRequest(
            symbol=intent.symbol,
            qty=intent.qty,
            side=OrderSide.BUY if intent.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        with_retry(
            lambda r=request: client.submit_order(r),
            description=f"submit {intent.side} {intent.qty} {intent.symbol}",
        )
        submitted += 1
        logger.info("Submitted %s %d %s", intent.side, intent.qty, intent.symbol)

    return submitted
