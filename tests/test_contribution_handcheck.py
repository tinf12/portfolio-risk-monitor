"""Hand-checkable per-position risk contribution.

Written against `src.risk.contribution`, which is human-authored and does not
exist yet; the module-level importorskip keeps this file green until it does.

Assumed signature, and the one thing to change if you name things differently:

    historical_contribution(
        returns_by_symbol: Mapping[str, Sequence[float]],
        weights: Mapping[str, float],
        confidence: float,
        total_value: float,
    ) -> dict[str, float]        # symbol -> contribution in dollars

The method this encodes
-----------------------
Historical VaR is a specific day: the k-th worst portfolio return in the
window, with k = ceil((1 - confidence) * n). That day is a real, observed
scenario in which each position did something particular. So the natural
decomposition is to ask what each position contributed *on that day*:

    contribution_i = -w_i * r_i(tail day) * total_value

Because the portfolio return on that day is the weighted sum of the position
returns, these contributions sum exactly to portfolio VaR -- no residual, no
scaling fudge. That additivity is the property worth testing, and the reason
the schema says contribution "sums to portfolio VaR across symbols".

This is a methodology choice, not the only one. The main alternative is to
average each position's behaviour across the whole tail rather than reading a
single day, which is steadier but no longer ties to the exact VaR figure being
decomposed. Whichever is chosen belongs in the README, because at a 250-day
window the single-day version rests on one observation.

The fixture
-----------
Three symbols at equal weight (1/3 each), $300,000 portfolio, 25 sessions:

- 20 quiet days: every symbol +0.1%
- 5 loss days, with portfolio returns -0.5%, -1.0%, -1.5%, -2.0%, -3.0%

n = 25, so k = ceil(0.05 * 25) = 2 at 95% and ceil(0.01 * 25) = 1 at 99%.
The 95% tail day is therefore the second-worst (-2.0%) and the 99% tail day is
the worst (-3.0%).

    95% tail day:  A -4.5%   B -2.1%   C +0.6%   -> portfolio -2.0%
    99% tail day:  A -4.5%   B -3.0%   C -1.5%   -> portfolio -3.0%

Contribution is -w * r * V with w = 1/3 and V = 300_000, so each is just
-100_000 * r:

    95%:  A $4,500   B $2,100   C -$600    sum $6,000  = VaR95
    99%:  A $4,500   B $3,000   C $1,500   sum $9,000  = VaR99

C is deliberately positive on the 95% tail day. A position that gains while the
portfolio loses *reduces* the loss, so its contribution is negative. Any
implementation that takes absolute values, or assumes contributions are
non-negative, gets $600 wrong in a way that still sums plausibly.
"""

from __future__ import annotations

import pytest

from src.risk.var import historical_var

contribution = pytest.importorskip(
    "src.risk.contribution", reason="src/risk/contribution.py not written yet"
)

SYMBOLS = ("A", "B", "C")
WEIGHTS = {s: 1 / 3 for s in SYMBOLS}
TOTAL_VALUE = 300_000.0

QUIET_DAYS = 20
_QUIET = {"A": 0.001, "B": 0.001, "C": 0.001}
_LOSS_DAYS = [
    {"A": -0.005, "B": -0.005, "C": -0.005},   # portfolio -0.5%
    {"A": -0.010, "B": -0.010, "C": -0.010},   # portfolio -1.0%
    {"A": -0.015, "B": -0.015, "C": -0.015},   # portfolio -1.5%
    {"A": -0.045, "B": -0.021, "C": 0.006},    # portfolio -2.0%, the 95% tail
    {"A": -0.045, "B": -0.030, "C": -0.015},   # portfolio -3.0%, the 99% tail
]

_DAYS = [_QUIET] * QUIET_DAYS + _LOSS_DAYS

RETURNS_BY_SYMBOL = {s: [day[s] for day in _DAYS] for s in SYMBOLS}
PORTFOLIO_RETURNS = [
    sum(day[s] * WEIGHTS[s] for s in SYMBOLS) for day in _DAYS
]

EXPECTED_95 = {"A": 4_500.0, "B": 2_100.0, "C": -600.0}
EXPECTED_99 = {"A": 4_500.0, "B": 3_000.0, "C": 1_500.0}

EXPECTED_VAR_95 = 6_000.0
EXPECTED_VAR_99 = 9_000.0


class TestFixtureItself:
    """Guards the fixture, so a typo fails here rather than downstream."""

    def test_twenty_five_sessions(self) -> None:
        assert len(PORTFOLIO_RETURNS) == 25
        assert all(len(v) == 25 for v in RETURNS_BY_SYMBOL.values())

    def test_weights_sum_to_one(self) -> None:
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_tail_days_are_where_we_think(self) -> None:
        ascending = sorted(PORTFOLIO_RETURNS)
        assert ascending[0] == pytest.approx(-0.030)  # 99% tail, k=1
        assert ascending[1] == pytest.approx(-0.020)  # 95% tail, k=2

    def test_portfolio_var_matches_the_stated_targets(self) -> None:
        """The contributions are checked against these, so pin them first."""
        assert historical_var(
            PORTFOLIO_RETURNS, 0.95, TOTAL_VALUE
        ) == pytest.approx(EXPECTED_VAR_95)
        assert historical_var(
            PORTFOLIO_RETURNS, 0.99, TOTAL_VALUE
        ) == pytest.approx(EXPECTED_VAR_99)


class TestContributionValues:
    def test_at_95(self) -> None:
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.95, TOTAL_VALUE
        )
        assert got == pytest.approx(EXPECTED_95)

    def test_at_99(self) -> None:
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.99, TOTAL_VALUE
        )
        assert got == pytest.approx(EXPECTED_99)

    def test_covers_every_symbol(self) -> None:
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.95, TOTAL_VALUE
        )
        assert set(got) == set(SYMBOLS)


class TestAdditivity:
    """The property the schema depends on, and the one easiest to break."""

    @pytest.mark.parametrize(
        ("confidence", "expected_var"),
        [(0.95, EXPECTED_VAR_95), (0.99, EXPECTED_VAR_99)],
    )
    def test_sums_to_portfolio_var(
        self, confidence: float, expected_var: float
    ) -> None:
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, confidence, TOTAL_VALUE
        )
        assert sum(got.values()) == pytest.approx(expected_var)

    def test_sums_to_the_var_function_not_just_a_constant(self) -> None:
        """Tie the decomposition to the figure it decomposes, so the two cannot
        drift apart if the percentile convention changes."""
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.95, TOTAL_VALUE
        )
        assert sum(got.values()) == pytest.approx(
            historical_var(PORTFOLIO_RETURNS, 0.95, TOTAL_VALUE)
        )


class TestSignConvention:
    def test_a_hedging_position_contributes_negatively(self) -> None:
        """C gains 0.6% on the 95% tail day, so it offsets part of the loss.
        abs() anywhere in the implementation turns -$600 into +$600 and the sum
        overstates VaR by $1,200 while still looking like a plausible table."""
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.95, TOTAL_VALUE
        )
        assert got["C"] == pytest.approx(-600.0)
        assert got["C"] < 0

    def test_losing_positions_contribute_positively(self) -> None:
        """Same convention as var_amount: a loss is a positive number."""
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.99, TOTAL_VALUE
        )
        assert all(v > 0 for v in got.values())


class TestProperties:
    def test_scales_linearly_with_total_value(self) -> None:
        one = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.95, TOTAL_VALUE
        )
        two = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, WEIGHTS, 0.95, 2 * TOTAL_VALUE
        )
        assert two == pytest.approx({s: 2 * v for s, v in one.items()})

    def test_symbol_order_does_not_matter(self) -> None:
        reordered = {s: RETURNS_BY_SYMBOL[s] for s in reversed(SYMBOLS)}
        assert contribution.historical_contribution(
            reordered, WEIGHTS, 0.95, TOTAL_VALUE
        ) == pytest.approx(
            contribution.historical_contribution(
                RETURNS_BY_SYMBOL, WEIGHTS, 0.95, TOTAL_VALUE
            )
        )

    def test_a_zero_weight_position_contributes_nothing(self) -> None:
        """A position you do not hold cannot contribute risk, whatever it did."""
        weights = {"A": 0.5, "B": 0.5, "C": 0.0}
        got = contribution.historical_contribution(
            RETURNS_BY_SYMBOL, weights, 0.95, TOTAL_VALUE
        )
        assert got["C"] == pytest.approx(0.0)


class TestGuards:
    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        """Weights summing to anything else means the decomposition cannot add
        up to portfolio VaR, so the result would be quietly wrong."""
        with pytest.raises(ValueError):
            contribution.historical_contribution(
                RETURNS_BY_SYMBOL,
                {"A": 0.5, "B": 0.5, "C": 0.5},
                0.95,
                TOTAL_VALUE,
            )

    def test_rejects_symbol_mismatch(self) -> None:
        with pytest.raises(ValueError):
            contribution.historical_contribution(
                RETURNS_BY_SYMBOL, {"A": 0.5, "B": 0.5}, 0.95, TOTAL_VALUE
            )

    def test_rejects_ragged_series(self) -> None:
        """Unequal history lengths mean the k-th worst day is not the same day
        for every symbol, which silently decomposes the wrong scenario."""
        ragged = dict(RETURNS_BY_SYMBOL)
        ragged["C"] = ragged["C"][:-1]
        with pytest.raises(ValueError):
            contribution.historical_contribution(
                ragged, WEIGHTS, 0.95, TOTAL_VALUE
            )
