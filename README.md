# Portfolio Risk Monitor

Daily risk measurement for a live paper-traded equal-weight portfolio of the 11
SPDR Select Sector ETFs, with calibration validated against seven years of
historical replay.

Each trading day a scheduled job snapshots prices and positions, computes risk
metrics, and stores them against both the date they were computed and the date
they predict. A read-only Streamlit dashboard displays the history.

**The deliverable is the measurement, not the return.** The allocation is
deliberately plain. What is being built and defended here is the risk engine,
its validation against history, and an honest account of where it fails.

---

## The result

Both VaR methods are well calibrated at 95% and both fail at 99%.

Kupiec proportion-of-failures test, 250-day rolling window, 1,650 sessions
tested over 2019–2026:

| Method | Confidence | Observed | Expected | p | Verdict |
|---|---|---|---|---|---|
| Historical | 95% | 5.09% | 5.00% | 0.866 | accept |
| Historical | 99% | 1.58% | 1.00% | 0.030 | **reject** |
| Parametric | 95% | 5.39% | 5.00% | 0.468 | accept |
| Parametric | 99% | 2.24% | 1.00% | <0.001 | **reject** |

Both methods clear 95% and neither clears 99%. Parametric fails roughly twice as
badly, breaching at 2.24% against a 1% claim, because the normal's tail is too
thin for equity returns. Historical fails too, at 1.58%, and normality cannot be
the reason, since that method assumes no distribution at all. Its failure is
small-sample and non-stationarity: three observations sit in the 99% tail of a
250-session window, and a window that has not seen a crash cannot price one.

---

## What it does

- 11 SPDR Select Sector ETFs, equal weight, rebalanced on the first trading day
  of each month, long only, no leverage
- A scheduled job snapshots prices and positions after each close, then computes
  and stores historical and parametric VaR, expected shortfall, 20-day
  volatility, drawdown, and per-position risk contribution
- Every stored figure is reproducible from stored inputs
- Calibration is validated on 1,901 sessions of historical replay, not on the
  live series, which is far too short to prove anything
- A read-only dashboard displays the history and the last successful run

**Deliberately not a strategy.** No return forecasting and no parameter
optimisation; the allocation is fixed so nothing can be tuned to flatter the
risk figures.

---

## Dashboard

**[tinf12-portfolio-risk-monitor.streamlit.app](https://tinf12-portfolio-risk-monitor.streamlit.app/)**

Live trading began 2026-07-28, so the risk panels currently state that their
estimates do not yet exist rather than showing a figure computed from too little
history. Volatility appears after 20 sessions, the 30-day VaR after 30, and the
250-day window not until 2027.

<!-- HUMAN-AUTHORED. Do not fill in automatically.
     Screenshot still to add. The live panels read "not yet computed" for
     anything needing 30 sessions, which is honest but reads as unfinished, so
     mid-September is the earliest a screenshot sells the project. A
     replay-populated view would work sooner but must be captioned explicitly
     as generated data — an uncaptioned fixture screenshot would present
     invented rows as real. -->

<!-- Note: the free tier sleeps after 12 hours without traffic, and a cold start
     shows a bare Streamlit wrapper for around 30 seconds before the app loads.
     Worth knowing before sending the link to anyone. -->

The app opens the database read-only at the SQLite level and imports nothing
from `src.risk`, so it cannot compute or alter a figure. It holds no
credentials: the daily job writes the database, and the dashboard only reads it.

---

## How it works

```
Alpaca (prices + positions)
  → SQLite (write-once prices, upserted account state)
    → risk engine (VaR, ES, vol, drawdown, contribution)
      → Streamlit (read-only)
```

- **GitHub Actions cron, 11:00 UTC on weekdays.** Each run processes the
  previous completed session, never today's incomplete one.
- **SQLite is committed to the repo.** CI is the only writer that commits, since
  git cannot merge a binary.
- **The dashboard cannot write.** Connections open with `mode=ro`, and it
  imports nothing from `src.risk`, so it is structurally incapable of producing
  a figure that is not already a stored row.
- **Failure signalling.** A failed scheduled workflow notifies at most one
  account, and a run that never starts notifies nobody, so the `runs` table and
  the dashboard's staleness indicator are the alarm.

### Temporal convention

The most important detail in the project.

A VaR computed from data through Monday's close is a prediction about Tuesday.
So `as_of_date` is the close through which input data was used, and
`applies_to_date` is the next trading day — the one being predicted.

A breach is a `daily_pnl` loss on date D exceeding the `var_amount` whose
`applies_to_date` is D. Joining on `as_of_date` instead is lookahead bias: it
raises no error and makes the model look excellent.

`applies_to_date` comes from the NYSE calendar, not from adding a day. A naive
weekday step points rows at market holidays, and a holiday row can never breach,
which biases the Kupiec test toward accepting the model.

---

## Methodology

**Nearest-rank percentiles**, `k = max(1, ceil(round((1 − confidence) · n, 9)))`,
defined once in `src/risk/_common.py` and shared by VaR, ES and contribution, since
two definitions of one percentile can silently invert the ES ≥ VaR guarantee. Linear
interpolation is the usual alternative and is less conservative here (−0.0044050
against −0.0045 on the fixture). Both guards are load-bearing: `(1 − 0.95)` is
0.050000000000000044 in floating point, so a bare `ceil` widens the 95% tail from 5
observations to 6 at n=100.

**Historical VaR** is the k-th worst return in the window; assuming no distribution,
it cannot report a loss worse than the worst day it has seen. **Parametric** is
`−(μ − z·σ) · total_value`, z from the inverse normal CDF (1.644854, 2.326348),
subtracting because the loss tail lies left of the mean. Sample standard deviation,
n−1, treating each window as a sample from a longer return process rather than the
population of interest.

**Expected shortfall** is the mean of the same k worst observations, historical rows
only. The normal closed form exists, so the NULL is a decision not to mix an
analytic figure into an empirical column.

**Volatility** is `stdev(last 20 returns) · sqrt(252)`, NULL until 20 returns exist.
**Drawdown** is current, not maximum: on a 100 → 50 → 90 path it reports −10%.

**Contribution** decomposes each historical VaR on the tail day,
`−wᵢ · rᵢ(tail day) · total_value`. Since that day's portfolio return is the weighted
sum of position returns, contributions sum exactly to the VaR implied by those
inputs, by arithmetic rather than approximation. Averaging across the whole tail
would be steadier but would stop tying to the figure being decomposed. **Cash is a
position at constant zero return**, keeping weights at 1.0 without normalising
sleeves not held in isolation.

**Windows of 30 and 250**, both stored, since `lookback_days` is part of the primary
key. 250 is about a trading year; 30 is a floor, since at n=20 the 95% and 99% tails
are both one observation and the estimates would report identical numbers while
presenting as independent. At n=30 they are 2 and 1.

---

## Validation

Calibration is measured on a synthetic reconstruction, not on live trading: at
99% confidence the expected breach count over a few weeks is near zero, so live
data cannot distinguish a working model from a broken one. Replay covers 1,901
sessions from 2019-01-02, equal weight, monthly rebalanced, price-only.

The 250-day results are in [The result](#the-result) above.

### 30-day window

Over 1,870 tested sessions, every method and confidence level rejects:

| Method | Confidence | Observed | Expected | p | Verdict |
|---|---|---|---|---|---|
| Historical | 95% | 6.79% | 5.00% | 0.0007 | **reject** |
| Historical | 99% | 3.85% | 1.00% | <0.001 | **reject** |
| Parametric | 95% | 6.95% | 5.00% | 0.0002 | **reject** |
| Parametric | 99% | 3.05% | 1.00% | <0.001 | **reject** |

The ranking also inverts. At 250 days historical is the better-calibrated
method; at 30 days it is the worse one. A 99% historical estimate over 30
sessions is the single worst day in the window and cannot exceed it, so a calm
month produces an estimate the next volatile session walks straight through.
Parametric at least extrapolates past its own sample.

The 30-day window is therefore a diagnostic, not a calibrated risk figure, and
nothing should be sized off it. It is reported because it reacts to a volatility
change within weeks rather than quarters, which the 250-day window cannot do, and
for that purpose being wrong about the tail matters less than being current.


### What Kupiec does not test

The test counts breaches and ignores their timing entirely: ten breaches inside
one fortnight score identically to ten spread across four years. Clustered
breaches are the failure mode that actually costs money, because they arrive
when volatility is high and losses compound. Christoffersen's independence test
is the standard companion for this and is not implemented here.

The test also has weak power on small samples. At 99% over 250 sessions the
expected breach count is 2.5, and almost any count between 0 and 7 is accepted.
Failing it is informative; passing it is much less so.

---

## Stress scenarios

Windows are ranked mechanically by cumulative replay return, non-overlapping.
No date is chosen by hand, which is what removes hindsight bias from scenario
selection — the code finds stress the author did not remember.

| Window | Replayed | At current weights | Worst sleeve | Best sleeve |
|---|---|---|---|---|
| 2020-03-04 → 03-18 | −25.84% | −$73,247 | XLE −48.69% | XLP −13.15% |
| 2020-02-13 → 02-28 | −12.54% | −$35,406 | XLE −17.48% | XLP −10.29% |
| 2022-06-02 → 06-16 | −12.00% | −$33,866 | XLY −14.58% | XLP −7.35% |
| 2025-03-25 → 04-08 | −11.95% | −$33,637 | XLE −18.00% | XLP −2.96% |
| 2022-09-12 → 09-26 | −11.65% | −$32,857 | XLE −15.33% | XLP −5.90% |

Three observations follow from the table rather than from hindsight:

- The worst window is **twice** the second worst. Tail severity is not smoothly
  distributed.
- It is roughly **15× the 99% one-day VaR**, which is the whole argument for why
  a one-day VaR is not a sufficient risk statement.
- XLE is the worst sleeve in four of five windows; XLP is the best in all five.
  Equal weight is not equal risk.

Nothing follows for the allocation. Equal weight is fixed by specification, and
optimising against a ranked stress table is exactly the hindsight the mechanical
selection was designed to avoid. What follows is a reporting change: a single
portfolio VaR conceals a concentration this table makes obvious, which is the
argument for storing per-position contributions rather than only the headline
figure. Read the other way, XLE's dominance is itself a statement about the
sample, since three of these five windows fall in 2020 and 2022 and both were
energy-led. A longer history might rank a different sleeve first.

---

## Limitations

**The normal assumption understates the tail.** Equity returns depart from
normality through excess kurtosis, volatility clustering and negative skew, all
pushing the same way: σ is pinned down by the quiet centre, then the tail is
extrapolated at a fixed multiplier rather than measured. The error grows with
confidence, which is the 99% rejection above and why parametric fails roughly twice
as badly as historical.

**The sample contains one major stress episode.** History starts in 2019 because
XLC listed mid-2018, so 2008 is out of reach entirely and March 2020 dominates
every tail statistic in this document. The worst stress window is twice the second
worst partly because there is only one crash in the sample available to be worst.

**Small-sample tails.** At 250 sessions the 99% tail is exactly 3 observations; at
30 it is 1, which is why 30-day 99% ES equals 30-day 99% VaR exactly, an average of
one number being that number. Historical VaR also cannot report a loss worse than
the worst day in its window.

**Contributions decompose a portfolio that was never held.** They apply today's
weights to historical returns, while `var_amount` comes from account equity
reflecting the weights as they actually were, plus cash drag, dividends and costs.
On a 60-session series the sums drift +0.55% at 95% and −4.03% at 99%, larger at
99% because that estimate rests on a single day. The job logs the gap every session
rather than rescaling to force a tie, since rescaling would balance the table at
the cost of making every figure in it slightly fictional.

**Drawdown is bounded by stored history**, and is current rather than maximum. A
decline from a peak predating the series cannot be seen, so the figure is
understated until the record outlives the drawdown being measured.

**Volatility assumes day-to-day independence.** sqrt-of-time scaling requires it;
returns cluster instead, the same assumption that makes VaR understate risk in
stressed periods.

**Stored prices diverge from the vendor by design.** Prices are write-once, and
adjusted closes are restated by every subsequent distribution and split, so
re-fetching rewrites history. Reproducibility was chosen over agreement with the
vendor's current numbers.

**Reproducibility means "given the stored rows."** Positions are a live read with
no historical equivalent, so a past session's figures can be recomputed from what
was stored but not reconstructed end to end from the vendor.

**Replay and live series are not comparable** and share no VaR window. Replay is
price-only, excluding dividends and execution costs; the live series includes both.
Concatenating them would splice two different definitions of return.

**The ~1% cash buffer** left when sizing orders is a permanent drag that flows into
the return series and therefore into the VaR estimated from it. Correct, since it
is the P&L actually experienced, but it is there.

**The live sample proves nothing.** The paper account was funded 2026-07-28. At 99%
the expected breach count over a few weeks is near zero, which is why calibration
is measured on replay and the live series demonstrates only that the pipeline runs.


---

## Design decisions

- **Alpaca, not yfinance.** One system supplies both positions and prices, so
  the two never need reconciling. yfinance scrapes unofficial endpoints, is rate
  limited without warning, and breaks when the page structure changes.
- **Write-once prices.** Alpaca returns adjusted closes, and the adjustment
  factor for a past date shrinks with every subsequent distribution — XLE's
  2020-02-19 factor is 0.3829 against 1.0 today. Re-fetching would rewrite
  history and break reproducibility, so a stored close is never changed.
  Divergence from what the vendor currently reports is the accepted cost.
- **SQLite committed to the repo.** Auditability with zero infrastructure. CI is
  the only writer that commits, because git cannot merge a binary.
- **No LLM in the calculation path.** Language models may generate commentary or
  investigate anomalies. They never compute, adjust, or round a risk figure, and
  never choose positions or position sizes.

---

## Setup

Requires Python 3.14 and an Alpaca paper trading account.

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

Streamlit Community Cloud defaults to 3.12, so when deploying the dashboard set
the Python version to 3.14 under Advanced settings. A version mismatch between
local, CI, and the dashboard host is a reproducibility hazard.

Copy `.env.example` to `.env` and fill in your Alpaca paper keys, generated at
app.alpaca.markets under Paper Trading → API Keys. `.env` is gitignored and must
never be committed. In CI the same two values come from GitHub Secrets.

## Usage

Backfill historical prices from 2019, or check coverage without fetching:

```bash
python -m src.data.backfill
python -m src.data.backfill --check-only
```

Run one daily cycle. `--dry-run` performs every read and computation but commits
nothing and submits no orders:

```bash
python -m src.jobs.daily --dry-run
```

<!-- HUMAN-AUTHORED. Do not fill in automatically.
     One line warning that a real run submits live orders to the paper account,
     including opening positions if the account is empty, so --dry-run first is
     the habit. -->

Reproduce the validation results:

```bash
python -m src.validation.backtest --lookback 250
python -m src.validation.backtest --lookback 30
```

Serve the dashboard locally:

```bash
python -m streamlit run dashboard/app.py
```

Run the tests:

```bash
python -m pytest
```

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Plumbing: data, storage, orders, scheduling | Complete |
| 2 | Risk engine: VaR, ES, vol, drawdown, contribution | Complete |
| 3 | Validation: Kupiec, stress replay | Complete |
| 4 | Dashboard and writeup | Dashboard complete, writeup in progress |

Live trading began 2026-07-28. The 30-day risk figures appear once 30 sessions
have accumulated; the 250-day figures are not available until 2027. Until then
the dashboard's risk panels state that the estimates do not yet exist.

## Next Steps

- **Christoffersen's independence test**, to catch the breach clustering Kupiec
  cannot see.
- **A `source` column separating replayed rows from live ones**, so calibration
  results can be stored and displayed rather than recomputed by a script. This
  is currently the only reason the dashboard cannot show the Kupiec figures.
- **Longer live history.** The live series demonstrates the pipeline; it will
  not be a statistical sample for years.
