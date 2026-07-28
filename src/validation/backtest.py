"""Phase 3 driver: replay, backtest, Kupiec, stress.

Reads stored prices, reconstructs the synthetic portfolio, walks it forward
estimating VaR, counts breaches against the sessions those estimates predicted,
and reports the calibration of each method.

Reporting only. Nothing here writes to the database:

- `risk_estimates` has no column distinguishing a live row from a replayed one,
  and CLAUDE.md is explicit that the two series must never be mixed. Adding a
  `source` column, or a separate table, is a schema decision rather than
  something this script should assume.
- The output is an input to the README, which is written by hand.

Run with:

    python -m src.validation.backtest
    python -m src.validation.backtest --lookback 250 --start 2019-01-02
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping

from src.db.connection import get_connection
from src.risk.var import historical_var, parametric_var
from src.validation.kupiec import KupiecResult, kupiec_pof
from src.validation.replay import (
    ReplaySeries,
    StressWindow,
    apply_window_to_weights,
    load_closes,
    replay_series,
    rolling_var_backtest,
    worst_rolling_windows,
)

logger = logging.getLogger(__name__)

METHODS = {"historical": historical_var, "parametric": parametric_var}
CONFIDENCE_LEVELS = (0.95, 0.99)

STRESS_WINDOW_DAYS = 10
STRESS_TOP_N = 5


def run_backtest(
    series: ReplaySeries,
    lookback_days: int,
    significance: float = 0.05,
) -> dict[tuple[str, float], KupiecResult]:
    """Backtest every method/confidence pair over the replayed series."""
    results: dict[tuple[str, float], KupiecResult] = {}

    for method, var_fn in METHODS.items():
        for confidence in CONFIDENCE_LEVELS:
            breaches, observations, _dates = rolling_var_backtest(
                series, lookback_days, var_fn, confidence
            )
            if observations == 0:
                logger.warning(
                    "No observations for %s at %.2f with a %d-day window.",
                    method, confidence, lookback_days,
                )
                continue
            results[(method, confidence)] = kupiec_pof(
                breaches, observations, confidence, significance
            )

    return results


def equal_weights(symbols: Mapping[str, float] | list[str]) -> dict[str, float]:
    """Equal weight across `symbols`, matching the portfolio spec."""
    names = sorted(symbols)
    return {s: 1.0 / len(names) for s in names}


def _print_calibration(
    results: Mapping[tuple[str, float], KupiecResult],
    lookback_days: int,
) -> None:
    print(f"\nVaR calibration, {lookback_days}-day rolling window")
    print("-" * 78)
    for (method, confidence), result in sorted(results.items()):
        print(f"  {method:<11} {result.summary()}")


def _print_stress(
    windows: list[StressWindow],
    weights: Mapping[str, float],
    total_value: float,
) -> None:
    print(
        f"\nWorst {STRESS_WINDOW_DAYS}-session windows, "
        f"re-run against current weights at ${total_value:,.0f}"
    )
    print("-" * 78)
    for w in windows:
        impact = apply_window_to_weights(w, weights, total_value)
        worst = min(w.symbol_returns.items(), key=lambda kv: kv[1])
        best = max(w.symbol_returns.items(), key=lambda kv: kv[1])
        print(
            f"  {w.start_date} to {w.end_date}: "
            f"replayed {w.portfolio_return:+.2%}, "
            f"today ${impact:+,.0f} ({impact / total_value:+.2%})"
        )
        print(
            f"      worst {worst[0]} {worst[1]:+.2%}, "
            f"best {best[0]} {best[1]:+.2%}"
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Replay-based VaR validation.")
    parser.add_argument("--start", help="First session to replay (YYYY-MM-DD).")
    parser.add_argument("--end", help="Last session to replay (YYYY-MM-DD).")
    parser.add_argument(
        "--lookback",
        type=int,
        default=250,
        help="Rolling window for each VaR estimate. Default 250.",
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="Kupiec rejection threshold. Default 0.05.",
    )
    args = parser.parse_args(argv)

    with get_connection() as conn:
        closes = load_closes(conn, args.start, args.end)

    if not closes:
        print("No complete price history found.", file=sys.stderr)
        return 1

    series = replay_series(closes)
    print(
        f"Replayed {len(series)} sessions, "
        f"{series.dates[0]} to {series.dates[-1]}, "
        f"${series.values[0]:,.0f} -> ${series.values[-1]:,.0f} "
        f"({series.values[-1] / series.values[0] - 1:+.1%})"
    )

    results = run_backtest(series, args.lookback, args.significance)
    _print_calibration(results, args.lookback)

    windows = worst_rolling_windows(
        series, closes, STRESS_WINDOW_DAYS, STRESS_TOP_N
    )
    weights = equal_weights(list(closes[series.dates[-1]]))
    _print_stress(windows, weights, series.values[-1])

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
