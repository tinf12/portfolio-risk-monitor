"""Rebalance arithmetic tests.

Hand-checkable numbers throughout, per CLAUDE.md conventions.
"""

from __future__ import annotations

import pytest

from src.portfolio.rebalance import OrderIntent, compute_orders, target_quantities
from src.portfolio.spec import SYMBOLS, TARGET_WEIGHT


class TestSpec:
    def test_eleven_sectors(self) -> None:
        assert len(SYMBOLS) == 11

    def test_weights_sum_to_one(self) -> None:
        assert TARGET_WEIGHT * 11 == pytest.approx(1.0)


class TestTargetQuantities:
    def test_even_division(self) -> None:
        # $110,000 / 11 = $10,000 per sleeve; at $100 that is 100 shares.
        prices = {sym: 100.0 for sym in SYMBOLS}
        targets = target_quantities(110_000.0, prices)
        assert all(qty == 100 for qty in targets.values())

    def test_floors_to_whole_shares(self) -> None:
        # $10,000 budget at $300 -> 33.33 shares -> 33.
        prices = {sym: 300.0 for sym in SYMBOLS}
        targets = target_quantities(110_000.0, prices)
        assert all(qty == 33 for qty in targets.values())

    def test_flooring_leaves_cash_residual(self) -> None:
        prices = {sym: 300.0 for sym in SYMBOLS}
        targets = target_quantities(110_000.0, prices)
        invested = sum(qty * 300.0 for qty in targets.values())
        assert invested == 33 * 300.0 * 11  # 108,900
        assert 110_000.0 - invested == pytest.approx(1_100.0)

    def test_differing_prices(self) -> None:
        # Each sleeve gets $10,000 regardless of share price.
        prices = dict.fromkeys(SYMBOLS, 100.0)
        prices["XLK"] = 250.0
        prices["XLF"] = 40.0
        targets = target_quantities(110_000.0, prices)
        assert targets["XLK"] == 40  # 10000 / 250
        assert targets["XLF"] == 250  # 10000 / 40
        assert targets["XLV"] == 100

    def test_rejects_missing_price(self) -> None:
        prices = {sym: 100.0 for sym in SYMBOLS if sym != "XLRE"}
        with pytest.raises(ValueError, match="XLRE"):
            target_quantities(110_000.0, prices)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_rejects_nonpositive_value(self, value: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            target_quantities(value, {sym: 100.0 for sym in SYMBOLS})


class TestComputeOrders:
    def test_no_orders_when_at_target(self) -> None:
        current = {sym: 100.0 for sym in SYMBOLS}
        target = {sym: 100 for sym in SYMBOLS}
        assert compute_orders(current, target) == []

    def test_buys_from_empty_portfolio(self) -> None:
        orders = compute_orders({}, {"XLK": 40, "XLF": 250})
        assert orders == [
            OrderIntent("XLF", "buy", 250),
            OrderIntent("XLK", "buy", 40),
        ]

    def test_computes_the_delta_not_the_target(self) -> None:
        orders = compute_orders({"XLK": 30.0}, {"XLK": 40})
        assert orders == [OrderIntent("XLK", "buy", 10)]

    def test_sell_when_above_target(self) -> None:
        orders = compute_orders({"XLK": 55.0}, {"XLK": 40})
        assert orders == [OrderIntent("XLK", "sell", 15)]

    def test_sells_ordered_before_buys(self) -> None:
        # Proceeds should be available before buys settle.
        orders = compute_orders(
            {"XLK": 100.0, "XLF": 10.0}, {"XLK": 40, "XLF": 250}
        )
        assert [o.side for o in orders] == ["sell", "buy"]
        assert orders[0].symbol == "XLK"

    def test_deterministic_ordering(self) -> None:
        current = {"XLV": 5.0, "XLK": 100.0, "XLF": 10.0}
        target = {"XLV": 5, "XLK": 40, "XLF": 250}
        assert compute_orders(current, target) == compute_orders(current, target)

    def test_untouched_symbols_produce_no_order(self) -> None:
        orders = compute_orders({"XLK": 40.0, "XLF": 10.0}, {"XLK": 40, "XLF": 250})
        assert [o.symbol for o in orders] == ["XLF"]
