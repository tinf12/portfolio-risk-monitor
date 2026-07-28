"""Hand-checkable volatility and drawdown.

Written against `src.risk.metrics`, which does not exist yet; the module-level
importorskip keeps this file green until it does.

Both series here are chosen so the expected values can be derived with a pencil.

Volatility
----------
Twenty returns alternating +1% and -1%. The mean is exactly zero, so every
deviation is 0.01 and the sample variance is 20 * 0.0001 / 19:

    daily stdev = sqrt(0.0001 * 20 / 19) = 0.0102597835
    annualized  = 0.0102597835 * sqrt(252) = 0.1628690142

The rule of thumb this encodes, worth being able to state out loud: a 1% daily
standard deviation is roughly 16% annualized, because sqrt(252) = 15.8745.

`stdev` is the sample (n-1) standard deviation, matching `parametric_var`. The
population version would give 0.1587 here. Either is defensible; using two
different ones across the project is not.

Drawdown
--------
Levels, not returns -- this is the one function in src/risk that takes
total_value rather than daily_return.

    100_000 -> 120_000 -> 90_000

The peak is 120_000 and the current drawdown is 90_000 / 120_000 - 1 = -0.25.
Negative or zero by convention (see the portfolio_metrics schema comment), which
is the opposite sign convention from var_amount. Both are deliberate.
"""

from __future__ import annotations

import pytest

metrics = pytest.importorskip(
    "src.risk.metrics", reason="src/risk/metrics.py not written yet"
)

# --- volatility ------------------------------------------------------------
ALTERNATING = [0.01, -0.01] * 10  # 20 observations, mean exactly 0
EXPECTED_DAILY_STDEV = 0.0102597835
EXPECTED_ANNUALIZED_VOL = 0.1628690142
SQRT_252 = 15.874507866387544

# --- drawdown --------------------------------------------------------------
PEAK_THEN_FALL = [100_000.0, 120_000.0, 90_000.0]
EXPECTED_DRAWDOWN = -0.25
EXPECTED_PEAK = 120_000.0


class TestSeriesItself:
    """Guards the fixtures, so a typo fails here rather than downstream."""

    def test_twenty_returns(self) -> None:
        assert len(ALTERNATING) == 20

    def test_mean_is_zero(self) -> None:
        assert sum(ALTERNATING) == pytest.approx(0.0)


class TestRollingVolatility:
    def test_annualized_value(self) -> None:
        got = metrics.rolling_volatility(ALTERNATING, window=20)
        assert got == pytest.approx(EXPECTED_ANNUALIZED_VOL)

    def test_is_annualized_not_daily(self) -> None:
        """The failure this catches is a missing sqrt(252): the number would be
        0.0103 instead of 0.1629, off by a factor of 15.87, and nothing would
        raise. The schema documents vol_20d as annualized."""
        got = metrics.rolling_volatility(ALTERNATING, window=20)
        assert got == pytest.approx(EXPECTED_DAILY_STDEV * SQRT_252)
        assert got > 0.1

    def test_defaults_to_twenty_days(self) -> None:
        """CLAUDE.md specifies a 20-day window, and the column is named for it."""
        assert metrics.rolling_volatility(ALTERNATING) == pytest.approx(
            metrics.rolling_volatility(ALTERNATING, window=20)
        )

    def test_uses_only_the_last_window(self) -> None:
        """Older returns outside the window must not affect the result. A
        version that computes over the whole series fails here."""
        noisy_history = [0.5, -0.5, 0.5] + ALTERNATING
        assert metrics.rolling_volatility(noisy_history, window=20) == pytest.approx(
            EXPECTED_ANNUALIZED_VOL
        )

    def test_returns_none_when_history_is_short(self) -> None:
        """Nineteen days is not vol_20d. Storing a shorter window under that
        name would put a false claim in a named column; the column is nullable
        so that None is representable."""
        assert metrics.rolling_volatility(ALTERNATING[:19], window=20) is None

    def test_no_nan_on_short_history(self) -> None:
        """Explicitly not NaN. NaN sorts silently and propagates through
        aggregates; SQL NULL does neither."""
        got = metrics.rolling_volatility(ALTERNATING[:5], window=20)
        assert got is None or got == got  # NaN is the only value failing x == x

    def test_scales_linearly(self) -> None:
        doubled = [r * 2 for r in ALTERNATING]
        assert metrics.rolling_volatility(doubled, window=20) == pytest.approx(
            2 * EXPECTED_ANNUALIZED_VOL
        )

    def test_unchanged_by_a_constant_shift(self) -> None:
        """Volatility measures dispersion, not level. Adding 0.5% to every day
        moves the mean and leaves the spread alone."""
        shifted = [r + 0.005 for r in ALTERNATING]
        assert metrics.rolling_volatility(shifted, window=20) == pytest.approx(
            EXPECTED_ANNUALIZED_VOL
        )

    def test_zero_when_every_return_is_identical(self) -> None:
        """Degenerate but legitimate: no dispersion means no volatility. Unlike
        parametric_var, nothing divides by it here, so 0.0 is returnable rather
        than an error."""
        assert metrics.rolling_volatility([0.001] * 20, window=20) == pytest.approx(0.0)

    def test_rejects_non_positive_window(self) -> None:
        with pytest.raises(ValueError):
            metrics.rolling_volatility(ALTERNATING, window=0)

    def test_rejects_non_finite_returns(self) -> None:
        """Same reasoning as _validate in src/risk/_common.py."""
        with pytest.raises(ValueError):
            metrics.rolling_volatility([float("nan")] + ALTERNATING[1:], window=20)


class TestCurrentDrawdown:
    def test_drawdown_and_peak(self) -> None:
        drawdown, peak = metrics.current_drawdown(PEAK_THEN_FALL)
        assert drawdown == pytest.approx(EXPECTED_DRAWDOWN)
        assert peak == pytest.approx(EXPECTED_PEAK)

    def test_zero_at_a_new_high(self) -> None:
        """At a new high the current value is the peak, so the drawdown is
        exactly zero -- never positive."""
        drawdown, peak = metrics.current_drawdown([100.0, 110.0, 120.0])
        assert drawdown == pytest.approx(0.0)
        assert peak == pytest.approx(120.0)

    def test_never_positive(self) -> None:
        for series in ([100.0, 110.0, 120.0], PEAK_THEN_FALL, [100.0, 50.0]):
            drawdown, _ = metrics.current_drawdown(series)
            assert drawdown <= 0

    def test_peak_is_all_time_not_recent(self) -> None:
        """The peak runs over the whole stored history. Measuring against a
        recent local high understates the drawdown, which is the direction that
        flatters the portfolio."""
        drawdown, peak = metrics.current_drawdown([100.0, 200.0, 150.0, 160.0])
        assert peak == pytest.approx(200.0)
        assert drawdown == pytest.approx(-0.2)

    def test_single_observation(self) -> None:
        """One day of history is its own peak."""
        drawdown, peak = metrics.current_drawdown([100_000.0])
        assert drawdown == pytest.approx(0.0)
        assert peak == pytest.approx(100_000.0)

    def test_measures_the_last_value_not_the_lowest(self) -> None:
        """`drawdown` is current, not maximum. A recovery from the trough
        shrinks it; a max-drawdown statistic would not."""
        drawdown, _ = metrics.current_drawdown([100.0, 50.0, 90.0])
        assert drawdown == pytest.approx(-0.1)

    def test_rejects_empty_series(self) -> None:
        """The daily job calls this only after writing a total_value row, so an
        empty series means something upstream is broken."""
        with pytest.raises(ValueError):
            metrics.current_drawdown([])
