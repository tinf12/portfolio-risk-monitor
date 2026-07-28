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
        # $110,000 less the 1% buffer is $108,900; / 11 = $9,900 per sleeve;
        # at $100 that is 99 shares.
        prices = {sym: 100.0 for sym in SYMBOLS}
        targets = target_quantities(110_000.0, prices)
        assert all(qty == 99 for qty in targets.values())

    def test_even_division_without_a_buffer(self) -> None:
        # The same arithmetic with the buffer off: $10,000 per sleeve, 100
        # shares. Pins that the buffer is the only thing moving the number.
        prices = {sym: 100.0 for sym in SYMBOLS}
        targets = target_quantities(110_000.0, prices, cash_buffer=0.0)
        assert all(qty == 100 for qty in targets.values())

    def test_floors_to_whole_shares(self) -> None:
        # $9,900 budget at $300 -> 33 shares exactly.
        prices = {sym: 300.0 for sym in SYMBOLS}
        targets = target_quantities(110_000.0, prices)
        assert all(qty == 33 for qty in targets.values())

    def test_flooring_leaves_cash_residual(self) -> None:
        prices = {sym: 300.0 for sym in SYMBOLS}
        targets = target_quantities(110_000.0, prices, cash_buffer=0.0)
        invested = sum(qty * 300.0 for qty in targets.values())
        assert invested == 33 * 300.0 * 11  # 108,900
        assert 110_000.0 - invested == pytest.approx(1_100.0)


    def test_differing_prices(self) -> None:
        # Each sleeve gets $10,000 regardless of share price. Buffer off, so
        # the allocation arithmetic is the only thing under test.
        prices = dict.fromkeys(SYMBOLS, 100.0)
        prices["XLK"] = 250.0
        prices["XLF"] = 40.0
        targets = target_quantities(110_000.0, prices, cash_buffer=0.0)
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


class TestCashBuffer:
    """Sizing reads the previous close; the order fills at the next open.

    The gap between them is what the buffer exists to absorb. On 2026-07-28 the
    entry was sized on 2026-07-27 closes, filled about 0.7% higher, and left the
    account $215 into margin -- which the long-only, no-leverage spec does not
    permit.
    """

    # The actual 2026-07-27 closes the entry was sized from. Real prices matter
    # here: with round numbers, flooring happens to leave a ~1% residual that
    # masks the problem. These leave only 0.507%.
    CLOSES = {
        "XLB": 51.39,
        "XLC": 107.66,
        "XLE": 58.36,
        "XLF": 56.88,
        "XLI": 183.20,
        "XLK": 174.30,
        "XLP": 85.36,
        "XLRE": 45.76,
        "XLU": 45.68,
        "XLV": 163.40,
        "XLY": 110.84,
    }
    EQUITY = 100_000.0

    def _cost(self, targets: dict[str, int], gap: float) -> float:
        """Fill cost if every close moves by `gap` before the open."""
        return sum(q * self.CLOSES[s] * (1 + gap) for s, q in targets.items())

    def test_flooring_alone_left_only_half_a_percent(self) -> None:
        """The precondition for the bug: the residual was the same order of
        magnitude as an ordinary day's move, so it was never real headroom."""
        targets = target_quantities(self.EQUITY, self.CLOSES, cash_buffer=0.0)
        invested = self._cost(targets, 0.0)
        assert invested == pytest.approx(99_493.04, abs=0.01)
        assert (self.EQUITY - invested) / self.EQUITY == pytest.approx(
            0.00507, abs=1e-5
        )

    def test_reproduces_the_overshoot_without_a_buffer(self) -> None:
        """2026-07-28 replayed: sized on closes, filled 0.727% higher."""
        targets = target_quantities(self.EQUITY, self.CLOSES, cash_buffer=0.0)
        assert self._cost(targets, 0.00727) > self.EQUITY

    def test_buffer_absorbs_that_move(self) -> None:
        targets = target_quantities(self.EQUITY, self.CLOSES)
        assert self._cost(targets, 0.00727) < self.EQUITY

    def test_a_large_gap_still_overshoots(self) -> None:
        """1% is not protection against a genuine gap-up. Accepted: a rejected
        order is a better failure than silent leverage, and the next rebalance
        corrects it. Documented rather than engineered around."""
        targets = target_quantities(self.EQUITY, self.CLOSES)
        assert self._cost(targets, 0.04) > self.EQUITY

    def test_leaves_the_buffer_uninvested(self) -> None:
        """$1 prices, so flooring loses nothing and the buffer is the whole
        gap -- isolating the buffer from the rounding residual."""
        prices = {sym: 1.0 for sym in SYMBOLS}
        invested = sum(target_quantities(110_000.0, prices).values())
        assert invested == pytest.approx(108_900.0)

    def test_buffer_scales_with_equity(self) -> None:
        prices = {sym: 1.0 for sym in SYMBOLS}
        for equity in (11_000.0, 110_000.0, 1_100_000.0):
            invested = sum(target_quantities(equity, prices).values())
            assert invested == pytest.approx(equity * 0.99)

    def test_rejects_a_buffer_outside_the_unit_interval(self) -> None:
        prices = {sym: 100.0 for sym in SYMBOLS}
        for bad in (-0.01, 1.0, 1.5):
            with pytest.raises(ValueError):
                target_quantities(110_000.0, prices, cash_buffer=bad)

    def test_zero_buffer_is_allowed(self) -> None:
        """Explicitly opting out is legitimate; it is the default that changed."""
        prices = {sym: 100.0 for sym in SYMBOLS}
        assert target_quantities(110_000.0, prices, cash_buffer=0.0)


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
