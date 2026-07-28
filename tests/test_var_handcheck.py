"""Hand-checkable VaR / ES fixture.

The return series here is deliberately arithmetic so every expected value can be
verified with a pencil. It is written against `src.risk.var` and
`src.risk.expected_shortfall`, which are human-authored (see CLAUDE.md "Author
boundary"); the tests skip cleanly until those modules exist.

The series
----------
100 daily returns, exactly:

    r_i = (i - 50) / 10_000     for i = 1 .. 100

so the values are -0.0049, -0.0048, ..., -0.0001, 0.0000, ..., +0.0050,
already in ascending order, spaced 1 bp apart.

    mean = (-0.0049 + 0.0050) / 2 = 0.00005
    sample stdev = 0.0001 * sqrt(100 * 101 / 12) = 0.0029011492

The five worst returns are -0.0049, -0.0048, -0.0047, -0.0046, -0.0045.
The single worst is -0.0049.

Percentile convention
---------------------
The 95% figure is the one place two reasonable implementations disagree:

  * nearest-rank (ceil(0.05 * 100) = 5th smallest)  -> -0.0045
  * linear interpolation (numpy default, index 4.95) -> -0.0044050

`EXPECTED_HISTORICAL_VAR_95` below encodes nearest-rank. If the implementation
uses `np.percentile` (or `np.quantile`) with default interpolation, change this
constant to `INTERPOLATED_HISTORICAL_VAR_95` rather than adjusting the
implementation to fit — either convention is defensible, but the choice has to
be stated in the README and used consistently, because it moves the breach
count in the Kupiec test.

Dollar scaling
--------------
Per CLAUDE.md, VaR is estimated on the `daily_return` series and scaled by
current `total_value`. At a round $100,000 the nearest-rank numbers are exact:

    VaR 95% = 0.0045 * 100_000 = $450.00
    VaR 99% = 0.0049 * 100_000 = $490.00
    ES  95% = 0.0047 * 100_000 = $470.00
"""

from __future__ import annotations

import pytest

var = pytest.importorskip("src.risk.var", reason="src/risk/var.py not written yet")
es = pytest.importorskip(
    "src.risk.expected_shortfall",
    reason="src/risk/expected_shortfall.py not written yet",
)

TOTAL_VALUE = 100_000.0

RETURNS: list[float] = [(i - 50) / 10_000 for i in range(1, 101)]

# --- historical, nearest-rank ---------------------------------------------
EXPECTED_HISTORICAL_VAR_95 = 0.0045  # 5th smallest
EXPECTED_HISTORICAL_VAR_99 = 0.0049  # 1st smallest
INTERPOLATED_HISTORICAL_VAR_95 = 0.0044050  # numpy default, if used instead
INTERPOLATED_HISTORICAL_VAR_99 = 0.00489901  # numpy default, index 0.99

# mean of the 5 worst: (49+48+47+46+45)/5 bp = 47 bp
EXPECTED_ES_95 = 0.0047
# mean of the 1 worst
EXPECTED_ES_99 = 0.0049

# --- parametric ------------------------------------------------------------
MEAN = 0.00005
STDEV = 0.0029011492  # sample (ddof=1)
Z_95 = 1.6448536269514722
Z_99 = 2.3263478740408408
EXPECTED_PARAMETRIC_VAR_95 = 0.0047219658  # -(mean - z95 * stdev)
EXPECTED_PARAMETRIC_VAR_99 = 0.0066990823  # -(mean - z99 * stdev)


class TestSeriesItself:
    """Guards the fixture, so a typo in RETURNS fails here and not downstream."""

    def test_length(self) -> None:
        assert len(RETURNS) == 100

    def test_endpoints(self) -> None:
        assert RETURNS[0] == pytest.approx(-0.0049)
        assert RETURNS[-1] == pytest.approx(0.0050)

    def test_mean(self) -> None:
        assert sum(RETURNS) / len(RETURNS) == pytest.approx(MEAN)

    def test_five_worst(self) -> None:
        assert sorted(RETURNS)[:5] == pytest.approx(
            [-0.0049, -0.0048, -0.0047, -0.0046, -0.0045]
        )


class TestHistoricalVar:
    def test_var_95_is_fifth_worst(self) -> None:
        got = var.historical_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        assert got == pytest.approx(EXPECTED_HISTORICAL_VAR_95 * TOTAL_VALUE)

    def test_var_99_is_worst(self) -> None:
        got = var.historical_var(RETURNS, confidence=0.99, total_value=TOTAL_VALUE)
        assert got == pytest.approx(EXPECTED_HISTORICAL_VAR_99 * TOTAL_VALUE)

    def test_var_is_positive_loss_magnitude(self) -> None:
        """CLAUDE.md: var_amount is stored as a positive number."""
        assert var.historical_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE) > 0

    def test_higher_confidence_is_never_smaller(self) -> None:
        v95 = var.historical_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        v99 = var.historical_var(RETURNS, confidence=0.99, total_value=TOTAL_VALUE)
        assert v99 >= v95

    def test_scales_linearly_with_total_value(self) -> None:
        one = var.historical_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        two = var.historical_var(RETURNS, confidence=0.95, total_value=2 * TOTAL_VALUE)
        assert two == pytest.approx(2 * one)

    def test_order_of_input_does_not_matter(self) -> None:
        """A percentile is order-independent; a bug that slices the tail
        positionally instead of sorting will fail here."""
        shuffled = RETURNS[50:] + RETURNS[:50]
        assert var.historical_var(
            shuffled, confidence=0.95, total_value=TOTAL_VALUE
        ) == pytest.approx(
            var.historical_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        )

    def test_all_positive_returns_gives_no_loss(self) -> None:
        """Degenerate case worth pinning down: with no losing day in the window,
        the 5th-worst return is still positive. Whether that should clamp to 0
        or report a negative VaR is a modelling decision — assert whichever the
        implementation commits to, but commit."""
        gains = [0.001 * i for i in range(1, 101)]
        got = var.historical_var(gains, confidence=0.95, total_value=TOTAL_VALUE)
        assert got <= 0 or got == pytest.approx(0.0)


class TestParametricVar:
    def test_var_95(self) -> None:
        got = var.parametric_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        assert got == pytest.approx(EXPECTED_PARAMETRIC_VAR_95 * TOTAL_VALUE, rel=1e-6)

    def test_var_99(self) -> None:
        got = var.parametric_var(RETURNS, confidence=0.99, total_value=TOTAL_VALUE)
        assert got == pytest.approx(EXPECTED_PARAMETRIC_VAR_99 * TOTAL_VALUE, rel=1e-6)

    def test_exceeds_historical_on_uniform_series(self) -> None:
        """This series is uniform, so it has thinner tails than a normal. The
        normal fit therefore overstates the 95% loss (0.00472 vs 0.0045). If
        this ever inverts on real data, that is the fat-tail result the README
        is supposed to explain, not a test failure."""
        p = var.parametric_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        h = var.historical_var(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        assert p > h


class TestExpectedShortfall:
    def test_es_95_is_mean_of_five_worst(self) -> None:
        got = es.expected_shortfall(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
        assert got == pytest.approx(EXPECTED_ES_95 * TOTAL_VALUE)

    def test_es_99_is_mean_of_one_worst(self) -> None:
        got = es.expected_shortfall(RETURNS, confidence=0.99, total_value=TOTAL_VALUE)
        assert got == pytest.approx(EXPECTED_ES_99 * TOTAL_VALUE)

    def test_es_at_least_var(self) -> None:
        """ES averages the tail beyond VaR, so it can never be the smaller
        number. This is the single most useful sanity check on ES."""
        for confidence in (0.95, 0.99):
            e = es.expected_shortfall(
                RETURNS, confidence=confidence, total_value=TOTAL_VALUE
            )
            v = var.historical_var(
                RETURNS, confidence=confidence, total_value=TOTAL_VALUE
            )
            assert e >= v

    def test_determinism(self) -> None:
        """CLAUDE.md constraint 1: same inputs, same output, every time."""
        calls = [
            es.expected_shortfall(RETURNS, confidence=0.95, total_value=TOTAL_VALUE)
            for _ in range(3)
        ]
        assert calls[0] == calls[1] == calls[2]
