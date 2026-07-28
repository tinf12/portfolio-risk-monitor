"""Kupiec proportion-of-failures test for VaR calibration.

The question this answers is narrow and worth stating precisely: **given that a
99% VaR model should be breached on 1% of days, is the number of breaches
actually observed consistent with that rate, or too far from it to be chance?**

It is a likelihood-ratio test. Under the null hypothesis the breach process is
a sequence of independent Bernoulli trials with probability p = 1 - confidence.
The likelihood of the observed data is compared under two parameter values: the
model's claimed p, and the rate actually observed, x/n. If the claimed rate
explains the data nearly as well as the best-fitting rate does, the ratio is
near 1 and the model passes.

    LR_POF = -2 * ln[ (1-p)^(n-x) * p^x  /  (1-x/n)^(n-x) * (x/n)^x ]

Under the null, LR_POF is asymptotically chi-squared with one degree of freedom,
so the p-value is the upper tail of that distribution. One degree of freedom
because a single parameter -- the breach rate -- is being constrained.

What this test does NOT do, and what the README must say plainly:

- **It ignores timing.** Ten breaches spread evenly across four years and ten
  breaches inside one fortnight produce the identical statistic. Clustered
  breaches are the failure mode that actually hurts, because they arrive when
  volatility is high and losses compound. Christoffersen's independence test is
  the standard companion for that, and is not implemented here.
- **It has weak power on small samples.** At 99% over 250 sessions the expected
  breach count is 2.5, and the test will accept almost any count between 0 and
  7. Failing it is informative; passing it is much less so.
- **It assumes each session is an independent trial**, which volatility
  clustering violates -- the same assumption that makes the underlying VaR
  estimate understate risk in stressed periods.

The convention here is that the test *rejects* when the p-value falls below the
significance level, meaning the observed breach rate is inconsistent with the
model's claim. Rejection can come from either direction: too many breaches
(understated risk) or too few (overstated, capital sitting idle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class KupiecResult:
    """Outcome of one proportion-of-failures test.

    `rejected` is the verdict at `significance`; `expected_breaches` is n*p,
    carried so a report can state observed against expected without recomputing
    it and risking a mismatch.
    """

    breaches: int
    observations: int
    confidence: float
    expected_breaches: float
    observed_rate: float
    lr_statistic: float
    p_value: float
    significance: float
    rejected: bool

    def summary(self) -> str:
        """One line suitable for a log or a README table."""
        verdict = "REJECT" if self.rejected else "accept"
        return (
            f"{self.confidence:.0%} VaR: {self.breaches} breaches in "
            f"{self.observations} sessions "
            f"(expected {self.expected_breaches:.1f}, "
            f"rate {self.observed_rate:.2%} vs {1 - self.confidence:.2%}), "
            f"LR={self.lr_statistic:.3f}, p={self.p_value:.4f} -> {verdict}"
        )


def kupiec_pof(
    breaches: int,
    observations: int,
    confidence: float,
    significance: float = 0.05,
) -> KupiecResult:
    """Run the proportion-of-failures test.

    Formula, with p = 1 - confidence, n = observations, x = breaches:

        LR_POF = -2 * [ (n-x)*ln(1-p) + x*ln(p)
                        - (n-x)*ln(1-x/n) - x*ln(x/n) ]

    which is the log form of the likelihood ratio in the module docstring,
    rearranged so no large powers are computed. (1-p)^(n-x) underflows to zero
    for n in the thousands, and the whole statistic would silently become inf;
    summing logs avoids that entirely.

    Two boundary cases need explicit handling, because x*ln(x/n) is 0*ln(0) at
    x=0 and (n-x)*ln(1-x/n) is 0*ln(0) at x=n. Both terms tend to zero, so the
    convention 0*ln(0) = 0 is taken:

        x = 0:  LR = -2 * n * ln(1-p)
        x = n:  LR = -2 * n * ln(p)

    Zero breaches is the case that will actually occur here: a 99% model over a
    short window frequently produces none, and the test must return a verdict
    rather than a NaN.

    Args:
        breaches: Number of sessions where the loss exceeded VaR.
        observations: Number of sessions tested.
        confidence: The model's confidence level, e.g. 0.99.
        significance: Threshold for rejection. 0.05 is conventional.

    Returns:
        A KupiecResult carrying the statistic, p-value, and verdict.

    Raises:
        ValueError: If observations is not positive, breaches is negative or
            exceeds observations, or confidence is outside (0, 1).
    """
    if observations <= 0:
        raise ValueError(f"observations must be positive, got {observations}")
    if breaches < 0:
        raise ValueError(f"breaches cannot be negative, got {breaches}")
    if breaches > observations:
        raise ValueError(
            f"breaches ({breaches}) cannot exceed observations ({observations})"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if not 0.0 < significance < 1.0:
        raise ValueError(f"significance must be in (0, 1), got {significance}")

    p = 1.0 - confidence
    n = observations
    x = breaches
    observed_rate = x / n

    # Log-likelihood under the model's claimed rate.
    log_l_null = (n - x) * math.log(1.0 - p) + x * math.log(p)

    # Log-likelihood under the best-fitting rate, with 0*ln(0) taken as 0.
    if x == 0:
        log_l_alt = 0.0                      # (n-0)*ln(1) + 0*ln(0)
    elif x == n:
        log_l_alt = 0.0                      # 0*ln(0) + n*ln(1)
    else:
        log_l_alt = (n - x) * math.log(1.0 - observed_rate) + x * math.log(
            observed_rate
        )

    lr = -2.0 * (log_l_null - log_l_alt)

    # Floating point can leave the statistic fractionally below zero when the
    # observed rate lands exactly on p. It is non-negative by construction --
    # the alternative hypothesis cannot fit worse than the null it nests.
    # Not max(lr, 0.0): that returns -0.0 unchanged, which formats as "-0.000".
    if lr <= 0.0:
        lr = 0.0

    p_value = float(stats.chi2.sf(lr, df=1))

    return KupiecResult(
        breaches=x,
        observations=n,
        confidence=confidence,
        expected_breaches=n * p,
        observed_rate=observed_rate,
        lr_statistic=lr,
        p_value=p_value,
        significance=significance,
        rejected=p_value < significance,
    )
