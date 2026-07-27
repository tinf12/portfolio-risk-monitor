# Portfolio Risk Monitor

Daily risk measurement for a live paper-traded equal-weight portfolio of the 11
SPDR Select Sector ETFs.

Each trading day a scheduled job snapshots prices and positions, computes risk
metrics, and stores them against both the date they were computed and the date
they predict. A read-only Streamlit dashboard displays the history.

**The deliverable is the measurement, not the return.** The allocation is
deliberately plain. What is being built and defended here is the risk engine,
its validation against history, and an honest account of where it fails.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Plumbing: data, storage, orders, scheduling | In progress |
| 2 | Risk engine: VaR, ES, vol, drawdown, contribution | Not started |
| 3 | Validation: Kupiec, stress replay | Not started |
| 4 | Dashboard and writeup | Not started |

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

Backfill historical prices from 2019:

```bash
python -m src.data.backfill
```

Check coverage without fetching:

```bash
python -m src.data.backfill --check-only
```

Run one daily cycle, logging orders instead of submitting them:

```bash
python -m src.jobs.daily --dry-run
```

Run the tests:

```bash
python -m pytest
```

---

## Temporal convention

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

<!-- HUMAN-AUTHORED. Do not fill in automatically.
     Covers: VaR definitions and estimators, the return series they are
     estimated on, lookback choice, ES definition, contribution decomposition,
     and the Kupiec test setup. -->

*To be written alongside Phase 2.*

## Limitations

<!-- HUMAN-AUTHORED. Do not fill in automatically.
     Candidates already known:
     - Historical window starts 2019; the 2008 crisis is out of reach, so the
       sample contains one major stress episode and stress scenarios are drawn
       only from what is present.
     - Parametric VaR assumes normality; equity returns are fat-tailed.
     - Live paper history is far too short for statistical validation; the
       Kupiec sample comes from historical replay.
     - Replay P&L is price-only and excludes dividends and execution costs, so
       it is not directly comparable to the live series.
     - Positions are a live read with no historical equivalent; reproducibility
       means "given the stored rows", not "re-fetchable end to end". -->

*To be written alongside Phase 3.*
