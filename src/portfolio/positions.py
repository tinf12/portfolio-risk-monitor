"""Position and account snapshots from Alpaca."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.alpaca_client import get_trading_client, with_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountSnapshot:
    """One day's account state.

    `total_value` is Alpaca's reported equity: positions plus cash. This is the
    single P&L definition used for both VaR estimation and breach testing
    (CLAUDE.md, "P&L definition").
    """

    total_value: float
    cash: float
    positions: tuple[tuple[str, float, float], ...]  # (symbol, qty, market_value)

    def position_rows(self, trade_date: str) -> list[tuple[str, str, float, float]]:
        """Shape positions for `upsert_positions`."""
        return [
            (trade_date, symbol, qty, market_value)
            for symbol, qty, market_value in self.positions
        ]


def fetch_account_snapshot() -> AccountSnapshot:
    """Read current equity, cash, and open positions from Alpaca.

    This is a live read with no historical equivalent — Alpaca cannot report
    what the account held on a past date. Positions are therefore raw inputs,
    like prices: reproducibility means "given these stored rows", not
    "re-fetchable" (CLAUDE.md, "Determinism").
    """
    client = get_trading_client()

    account = with_retry(client.get_account, description="fetch account")
    raw_positions = with_retry(
        client.get_all_positions, description="fetch positions"
    )

    positions = tuple(
        sorted(
            (p.symbol, float(p.qty), float(p.market_value)) for p in raw_positions
        )
    )

    snapshot = AccountSnapshot(
        total_value=float(account.equity),
        cash=float(account.cash),
        positions=positions,
    )
    logger.info(
        "Account snapshot: equity=%.2f cash=%.2f positions=%d",
        snapshot.total_value,
        snapshot.cash,
        len(snapshot.positions),
    )
    return snapshot
