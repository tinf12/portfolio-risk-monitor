"""Portfolio definition.

Fixed by CLAUDE.md, "Portfolio spec". This is a requirement, not a default:
do not add, remove, or reweight tickers, and do not introduce an optimizer.
"""

from __future__ import annotations

from typing import Final

SECTOR_ETFS: Final[dict[str, str]] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

SYMBOLS: Final[tuple[str, ...]] = tuple(SECTOR_ETFS)

TARGET_WEIGHT: Final[float] = 1.0 / len(SYMBOLS)

# Earliest date with complete history for all 11 tickers. XLRE launched 2015,
# XLC 2018; starting here avoids a ragged panel (CLAUDE.md, "Universe history").
BACKFILL_START: Final[str] = "2019-01-01"
