"""Alpaca client construction and retry handling.

Credentials come from the environment only: .env locally, GitHub Secrets in CI.
Never hardcode a key (CLAUDE.md, "Conventions").
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0


def _credentials() -> tuple[str, str]:
    load_dotenv()
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. "
            "Copy .env.example to .env locally, or add them as GitHub Secrets."
        )
    return key, secret


def get_trading_client() -> TradingClient:
    """Return a paper-trading client.

    paper=True is not configurable. There is no live-money path in this
    project (CLAUDE.md, "Anti-goals").
    """
    key, secret = _credentials()
    return TradingClient(key, secret, paper=True)


def get_data_client() -> StockHistoricalDataClient:
    """Return a historical market data client."""
    key, secret = _credentials()
    return StockHistoricalDataClient(key, secret)


def with_retry(
    operation: Callable[[], T],
    *,
    description: str,
    max_attempts: int = MAX_ATTEMPTS,
) -> T:
    """Run `operation`, retrying transient failures with exponential backoff.

    Retries exist because scheduled runs are unattended and Alpaca returns
    occasional 5xx and rate-limit responses. The backoff is deterministic (no
    jitter) so a failure is reproducible when investigating a bad run.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - retry then re-raise
            last_error = exc
            if attempt == max_attempts:
                break
            delay = BACKOFF_BASE_SECONDS ** attempt
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.0fs.",
                description,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"{description} failed after {max_attempts} attempts"
    ) from last_error
