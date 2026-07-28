"""Kupiec proportion-of-failures test, with hand-checkable values.

The anchor cases are chosen so the arithmetic can be done on paper:

- **Perfect calibration.** When x/n lands exactly on p, the null and the
  best-fitting alternative are the same hypothesis, the likelihood ratio is 1,
  and LR = -2*ln(1) = 0 with p-value 1. Any implementation that fails this has
  its numerator and denominator confused.
- **Zero breaches.** The 0*ln(0) term drops out and the statistic collapses to
  LR = -2*n*ln(1-p). At n=250, p=0.01 that is -500*ln(0.99) = 5.0252, whose
  chi-square(1) upper tail is 0.0250. This is the case that actually occurs on
  short windows, so it must return a verdict rather than NaN.
- **A worked middle case.** n=100, p=0.05, x=10 gives

      LR = -2 * [90*ln(0.95) + 10*ln(0.05) - 90*ln(0.90) - 10*ln(0.10)]
         = 4.1309

  which is above the 3.8415 critical value at 5%, so it rejects.
"""

from __future__ import annotations

import math

import pytest

from src.validation.kupiec import kupiec_pof

# chi-square(1) critical value at 5%; the rejection boundary.
CHI2_CRITICAL_5PCT = 3.8414588


class TestPerfectCalibration:
    @pytest.mark.parametrize(
        ("breaches", "observations", "confidence"),
        [(50, 1000, 0.95), (10, 1000, 0.99), (5, 100, 0.95), (25, 500, 0.95)],
    )
    def test_statistic_is_zero(
        self, breaches: int, observations: int, confidence: float
    ) -> None:
        result = kupiec_pof(breaches, observations, confidence)
        assert result.lr_statistic == pytest.approx(0.0, abs=1e-9)
        assert result.p_value == pytest.approx(1.0)
        assert not result.rejected

    def test_statistic_is_never_negative_zero(self) -> None:
        """max(lr, 0.0) returns -0.0 unchanged, which formats as '-0.000' in
        any report. Cosmetic, but it looks like a broken statistic."""
        result = kupiec_pof(10, 1000, 0.99)
        assert math.copysign(1.0, result.lr_statistic) > 0


class TestZeroBreaches:
    def test_hand_checked_statistic(self) -> None:
        """LR = -2 * 250 * ln(0.99) = 5.0252."""
        result = kupiec_pof(0, 250, 0.99)
        assert result.lr_statistic == pytest.approx(-2 * 250 * math.log(0.99))
        assert result.lr_statistic == pytest.approx(5.0252, abs=1e-4)

    def test_rejects_at_five_percent(self) -> None:
        """Zero breaches in 250 sessions is too few for a 99% model: it implies
        risk is overstated, which the test flags in the same way as too many."""
        result = kupiec_pof(0, 250, 0.99)
        assert result.p_value == pytest.approx(0.0250, abs=1e-4)
        assert result.rejected

    def test_returns_a_verdict_not_nan(self) -> None:
        """The 0*ln(0) term must be handled, not evaluated."""
        result = kupiec_pof(0, 60, 0.99)
        assert not math.isnan(result.lr_statistic)
        assert not math.isnan(result.p_value)

    def test_zero_breaches_can_be_consistent_on_a_short_window(self) -> None:
        """Over 60 sessions a 99% model expects 0.6 breaches, so seeing none is
        unremarkable. The test's weak power on small samples is the point."""
        assert not kupiec_pof(0, 60, 0.99).rejected


class TestWorkedMiddleCase:
    def test_statistic(self) -> None:
        expected = -2 * (
            90 * math.log(0.95)
            + 10 * math.log(0.05)
            - 90 * math.log(0.90)
            - 10 * math.log(0.10)
        )
        result = kupiec_pof(10, 100, 0.95)
        assert result.lr_statistic == pytest.approx(expected)
        assert result.lr_statistic == pytest.approx(4.1309, abs=1e-4)

    def test_rejects_because_it_exceeds_the_critical_value(self) -> None:
        result = kupiec_pof(10, 100, 0.95)
        assert result.lr_statistic > CHI2_CRITICAL_5PCT
        assert result.rejected


class TestReportedFields:
    def test_expected_and_observed(self) -> None:
        result = kupiec_pof(26, 1650, 0.99)
        assert result.expected_breaches == pytest.approx(16.5)
        assert result.observed_rate == pytest.approx(26 / 1650)

    def test_summary_states_both_rates(self) -> None:
        text = kupiec_pof(26, 1650, 0.99).summary()
        assert "26 breaches" in text
        assert "1650 sessions" in text
        assert "expected 16.5" in text


class TestRejectionIsTwoSided:
    def test_too_many_breaches_rejects(self) -> None:
        assert kupiec_pof(25, 250, 0.99).rejected

    def test_too_few_breaches_rejects(self) -> None:
        """Overstated risk is a calibration failure too -- capital held against
        a loss the model keeps not experiencing."""
        assert kupiec_pof(0, 500, 0.95).rejected

    def test_significance_threshold_is_honoured(self) -> None:
        """p = 0.0301 at 99% over 1650 sessions: rejected at 5%, accepted at 1%."""
        assert kupiec_pof(26, 1650, 0.99, significance=0.05).rejected
        assert not kupiec_pof(26, 1650, 0.99, significance=0.01).rejected


class TestGuards:
    def test_rejects_zero_observations(self) -> None:
        with pytest.raises(ValueError):
            kupiec_pof(0, 0, 0.99)

    def test_rejects_negative_breaches(self) -> None:
        with pytest.raises(ValueError):
            kupiec_pof(-1, 100, 0.99)

    def test_rejects_breaches_exceeding_observations(self) -> None:
        with pytest.raises(ValueError):
            kupiec_pof(101, 100, 0.99)

    def test_rejects_confidence_outside_unit_interval(self) -> None:
        for bad in (0.0, 1.0, 95):
            with pytest.raises(ValueError):
                kupiec_pof(5, 100, bad)

    def test_all_breaches_is_handled(self) -> None:
        """x = n is the other 0*ln(0) boundary. Absurd in practice, but it must
        produce a number rather than an exception."""
        result = kupiec_pof(100, 100, 0.99)
        assert result.lr_statistic == pytest.approx(-2 * 100 * math.log(0.01))
        assert result.rejected
