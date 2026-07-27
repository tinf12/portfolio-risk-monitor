# Portfolio Risk Monitor

## What this project is

A daily risk monitoring system for a live paper-traded equity portfolio.

Each trading day after the close, a scheduled job snapshots positions and prices,
computes portfolio risk metrics, and stores them with the date they were computed
and the date they apply to. A read-only dashboard displays the history.

**The deliverable is the measurement, not the return.** The allocation is
deliberately plain. The value of this project is the risk engine, the validation
of that engine against history, and the honest reporting of where the model fails.
Do not propose changes that turn this into a strategy-optimization or
alpha-seeking project.

---

## Non-negotiable constraints

1. **Determinism.** Every stored risk number must be reproducible from stored
   inputs. Same inputs, same output, every time. No randomness, no wall-clock
   dependence, no unpinned data sources in the calculation path.
2. **No LLM in the calculation path.** Language models may generate narrative
   commentary or investigate anomalies. They must never compute, adjust, or
   round a risk figure, and must never choose positions or position sizes.
3. **No lookahead.** A calculation for date T may only use information
   available at T's close. See "Temporal convention" below.
4. **Auditability.** Any figure on the dashboard must be traceable to rows in
   the database, which must be traceable to raw prices and positions.

---

## Author boundary

Read this before writing any code.

**Files the human author writes himself.** Do not write these unless explicitly
asked. When asked to help, prefer reviewing, testing, and explaining over
authoring:

- `src/risk/var.py` — historical and parametric VaR
- `src/risk/expected_shortfall.py`
- `src/risk/contribution.py` — per-position risk contribution
- `src/validation/kupiec.py` — VaR exception testing
- The methodology and limitations sections of `README.md`

These are the parts that must be defensible in a technical interview. If asked
to write one of them, offer instead to: generate a hand-checkable test case,
review an existing implementation for correctness or lookahead bias, or explain
the mechanism behind an unexpected result.

**Files Claude Code can own fully:**

- `.github/workflows/` — all workflow YAML
- `src/data/` — Alpaca client, bar fetching, backfill, retry logic
- `src/db/` — schema, connections, upserts, migrations
- `dashboard/` — all Streamlit code and charts
- `tests/` — test scaffolding (the human specifies expected values)
- `requirements.txt`, tooling config
- `src/validation/replay.py` — synthetic historical position/P&L reconstruction,
  and identifying the worst rolling 10-day windows. This is data plumbing, not
  statistics: it selects and reconstructs, it does not conclude. The
  interpretation of what the stress results mean belongs in the README's
  limitations section, which is human-authored.

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.14 | Local, CI, and Streamlit Cloud must all match |
| Broker + data | Alpaca paper trading (`alpaca-py`) | Free; $100k default balance |
| Compute | pandas, NumPy | |
| Statistics | SciPy | chi-square for Kupiec test |
| Exchange calendar | `pandas-market-calendars` (XNYS) | Required to write `applies_to_date` |
| Storage | SQLite, committed to repo | |
| Scheduler | GitHub Actions cron | Repo must be public |
| Dashboard | Streamlit Community Cloud | |
| Charts | Plotly | |
| Secrets | GitHub Secrets; `.env` locally | `.env` is gitignored |

**Do not use yfinance anywhere in this project.** It scrapes unofficial Yahoo
endpoints, is rate-limited without warning, and breaks when Yahoo changes page
structure. Alpaca is both the broker and the price source, so positions and
prices come from one system and never need reconciling.

---

## Portfolio spec

The 11 SPDR Select Sector ETFs, one per GICS sector:

```
XLK  Technology
XLF  Financials
XLV  Health Care
XLI  Industrials
XLY  Consumer Discretionary
XLP  Consumer Staples
XLE  Energy
XLU  Utilities
XLB  Materials
XLRE Real Estate
XLC  Communication Services
```

- Equal weight, 1/11 each
- Rebalance on the first trading day of each month
- Long only, no leverage, no discretionary overrides

This spec is fixed. Treat it as a requirement, not a starting suggestion.

---

## Data constraints

Verified July 2026. Re-check before relying on any of it.

**Alpaca free tier**
- Real-time data is IEX only (roughly 2% of volume). SIP is delayed 15 minutes.
- Historical SIP data older than 15 minutes is available on the free plan via
  the `feed` parameter. Queries touching the last 15 minutes of SIP will fail.
- Consequence: always request bars through *yesterday's* close, never today's
  incomplete session.
- History depth is roughly 7 years, so 2019 onward is safe.

**Universe history**
- XLRE was created in 2015, XLC in 2018. Historical windows starting 2019 or
  later have complete data for all 11 tickers. Do not backfill earlier without
  handling the ragged start explicitly.
- The 2008 crisis is out of reach. FRED's daily S&P and Dow series are capped at
  10 years by licensing, so that is not a workaround. Stress scenarios must be
  selected from available data (see Phase 3).

**GitHub Actions**
- Free and uncapped on public repositories.
- Cron is UTC. Minimum interval is 5 minutes.
- Scheduled runs are commonly delayed 10 to 30 minutes at peak load. Never
  assume exact execution time.
- GitHub sends no notification when a scheduled workflow fails. The `runs`
  table and a "last successful run" indicator on the dashboard are the
  substitute. Treat this as a requirement, not a nice-to-have.

**Streamlit Community Cloud**
- Roughly 1 GB memory. Apps sleep after 12 hours without traffic.
- Therefore the dashboard **reads only**. It must never compute risk metrics,
  call Alpaca, or place orders. If the app sleeps, nothing should stop.

---

## Database schema

```sql
CREATE TABLE prices (
  trade_date TEXT NOT NULL,
  symbol     TEXT NOT NULL,
  close      REAL NOT NULL,
  PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE positions (
  trade_date   TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  qty          REAL NOT NULL,
  market_value REAL NOT NULL,
  PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE portfolio_pnl (
  trade_date  TEXT PRIMARY KEY,
  total_value REAL NOT NULL,       -- positions market value + cash
  cash        REAL NOT NULL,
  daily_pnl   REAL,                -- total_value(T) - total_value(T-1)
  daily_return REAL                -- daily_pnl / total_value(T-1)
);

CREATE TABLE risk_estimates (
  as_of_date      TEXT NOT NULL,
  applies_to_date TEXT NOT NULL,
  method          TEXT NOT NULL,   -- 'historical' | 'parametric'
  confidence      REAL NOT NULL,   -- 0.95 | 0.99
  var_amount      REAL NOT NULL,   -- positive number, loss magnitude
  es_amount       REAL,
  lookback_days   INTEGER NOT NULL,
  PRIMARY KEY (as_of_date, method, confidence, lookback_days)
);

CREATE TABLE portfolio_metrics (
  as_of_date  TEXT PRIMARY KEY,
  vol_20d     REAL,                -- annualized, sqrt(252) scaling
  drawdown    REAL,                -- current, negative or zero
  peak_value  REAL                 -- running max of total_value
);

CREATE TABLE risk_contributions (
  as_of_date      TEXT NOT NULL,
  symbol          TEXT NOT NULL,
  weight          REAL NOT NULL,
  marginal_var    REAL,
  contribution    REAL NOT NULL,   -- sums to portfolio VaR across symbols
  method          TEXT NOT NULL,
  confidence      REAL NOT NULL,
  lookback_days   INTEGER NOT NULL,
  PRIMARY KEY (as_of_date, symbol, method, confidence, lookback_days)
);

CREATE TABLE runs (
  run_id  INTEGER PRIMARY KEY AUTOINCREMENT,
  run_at  TEXT NOT NULL,
  status  TEXT NOT NULL,           -- 'success' | 'failure'
  message TEXT
);
```

Design notes, all deliberate:

- Prices are long format (one row per date-symbol). Adding tickers requires no
  migration.
- `lookback_days` is part of the `risk_estimates` primary key so estimates from
  different windows coexist for the same date and can be compared, rather than
  one silently overwriting the other.
- `var_amount` is stored as a positive loss magnitude. Be consistent everywhere.
- Dates are `TEXT` in `YYYY-MM-DD`. Use exchange trading dates, not UTC
  timestamps.
- `portfolio_metrics` and `risk_contributions` exist because the dashboard is
  read-only (see Streamlit constraint) and every displayed figure must trace to
  a stored row. Nothing on the dashboard may be computed at render time.
- `risk_contributions` carries the same `method`/`confidence`/`lookback_days`
  triple as `risk_estimates` so a contribution row always joins back to the
  portfolio-level figure it decomposes.
- `runs` uses a surrogate key because two runs can share a timestamp at
  second resolution (a re-run, or a manual trigger racing the schedule).

---

## Temporal convention

This is the most important detail in the project.

A VaR computed from data through **Monday's close** is a prediction about
**Tuesday**. So:

- `as_of_date` = the close through which input data was used
- `applies_to_date` = the next trading day, the one being predicted

A breach is: `portfolio_pnl.daily_pnl` on date D was a loss exceeding
`var_amount` where `applies_to_date = D`.

Joining on `as_of_date` instead of `applies_to_date` is lookahead bias. It will
not raise an error and it will make the model look excellent. Any code touching
breach counts must join on `applies_to_date`, and tests should assert this.

**Determining the next trading day.** `applies_to_date` is written at T's close,
before T+1's data exists, so it cannot be inferred from the `prices` table. It
comes from the NYSE calendar (`pandas-market-calendars`, `XNYS`), which handles
weekends, holidays, and early closes. Never compute it as "date + 1 day" or by
skipping weekends only — that silently produces rows pointing at holidays, and
those rows can never breach, which biases the Kupiec test toward acceptance.

---

## Price restatement

Alpaca returns split- and dividend-adjusted closes. Adjusted is the correct
input for a total-return series and is what this project uses. But the
adjustment factor for any past date **shrinks with every subsequent
distribution**, so re-fetching restates the entire stored history.

Measured on XLE, July 2026:

| Date | Raw | Adjusted | Factor |
|---|---|---|---|
| 2020-02-19 | 54.85 | 21.00 | 0.3829 |
| 2020-03-23 | 23.48 | 9.19 | 0.3914 |
| 2026-07-24 | 59.62 | 59.62 | 1.0000 |

Left alone, this breaks determinism: a VaR computed last month would not
reproduce once its input closes had been silently rewritten, and the committed
database would show a whole-file diff with no code change to explain it.

Therefore:

- **`prices` is write-once.** A stored close is never changed. `insert_prices`
  inserts missing rows only.
- Rows whose incoming close differs from the stored one are **reported, not
  applied**. Silent divergence from the vendor is the thing being prevented, so
  the drift must be visible.
- `src.data.backfill --allow-restate` overwrites deliberately. It invalidates
  reproducibility for every figure already computed from the affected dates, so
  it is a decision, not a routine step.
- The daily job treats any restatement as a **failure**. The session it just
  fetched is new, so a conflicting close means something is wrong upstream, not
  that history moved on.
- `positions` and `portfolio_pnl` remain upserts. They are snapshots of live
  account state, and re-running a session should refresh them.

The accepted consequence, which belongs in the README's limitations: stored
history slowly diverges from what the vendor currently reports. For a risk
system a reproducible number is worth more than an up-to-date one.

---

## P&L definition

`daily_pnl` and the return series used to estimate VaR must be the same series.
If VaR is estimated on position price returns but breaches are measured against
account equity, the two disagree by cash drag and dividends, and the model looks
miscalibrated for a reason that has nothing to do with the model.

So, one definition, used everywhere:

- `total_value` = market value of all positions + cash, as reported by Alpaca.
- `daily_pnl` = `total_value(T) - total_value(T-1)`, where T-1 is the previous
  trading day.
- `daily_return` = `daily_pnl / total_value(T-1)`.
- **VaR is estimated from the `daily_return` series in `portfolio_pnl`**, scaled
  by current `total_value` to produce `var_amount` in dollars.

Consequences to state in the README rather than engineer around:

- Dividends arrive as a one-day positive return with no corresponding price
  move. They appear in both the estimate and the breach test, so they do not
  bias calibration, but they do slightly fatten the right tail.
- Rebalance days include execution costs and slippage in the return. This is
  correct — it is the P&L actually experienced.
- For historical replay (Phase 3), no live equity exists. Replay must construct
  a synthetic `total_value` series from stored prices and hypothetical
  equal-weight positions rebalanced on the first trading day of each month,
  starting from the same $100k notional. Replay P&L is therefore price-only and
  excludes dividends; the live and replay series are **not** directly
  comparable and must never be concatenated into one VaR window.

---

## Repo layout

```
portfolio-risk-monitor/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore                 # must include .env and .venv/
├── .github/workflows/daily.yml
├── data/
│   └── risk.db
├── src/
│   ├── data/                  # Alpaca client, fetch, backfill
│   ├── db/                    # schema, connections, upserts
│   ├── portfolio/             # rebalance logic, order submission
│   ├── risk/                  # VAR, ES, contribution  (human-authored)
│   ├── validation/            # Kupiec, stress scenarios (human-authored)
│   └── jobs/daily.py          # orchestrates the daily run
├── dashboard/
│   └── app.py
└── tests/
```

---

## Milestones

**Phase 1: Plumbing**
Alpaca auth, daily bar fetch for all 11 tickers, SQLite persistence, first paper
orders, scheduled workflow, heartbeat rows in `runs`, 2019-onward backfill.
*Done when:* the workflow runs unattended on schedule and stored positions
reconcile against what the Alpaca API reports.

**Phase 2: Risk engine**
Historical VaR at 95% and 99%. Parametric VaR at 95% and 99%. Expected
shortfall. Rolling 20-day volatility. Drawdown. Per-position risk contribution.
*Done when:* every figure for any date regenerates identically from stored
inputs on a second run.

**Phase 3: Validation and stress**
Kupiec proportion-of-failures test for both VaR methods across the full
historical window (roughly 1,250 trading days from 2019). Stress scenarios.

Two constraints here:

- Validation runs on **historical replay**, not live paper data. At 99%
  confidence you expect one breach per 100 trading days, so a few weeks of live
  trading yields an expected breach count near zero and proves nothing. Live
  trading demonstrates the pipeline works in production; history provides the
  statistical sample.
- **Do not hand-pick crisis dates for stress scenarios.** Have the code identify
  the worst rolling 10-day windows present in the available data and replay
  those against current weights. This removes hindsight bias from scenario
  selection.

*Done when:* the Kupiec result is computed for both methods and the difference
in calibration between them can be stated in one sentence with a reason.

**Phase 4: Dashboard and writeup**
Streamlit app deployed publicly, reading only from SQLite, displaying last
successful run prominently. README documenting methodology, assumptions, and
limitations.
*Done when:* someone who has never seen the repo can understand what it measures
and where it is wrong.

**Phase 5 (optional, only after Phase 4 ships)**
LLM-generated daily risk commentary from computed metrics. Tool-using
investigation when a breach occurs, with output stored and displayed as an
explicitly unverified hypothesis. Claude Haiku 4.5 is $1 per million input
tokens and $5 per million output, so total project cost here is under $2.

This phase is strictly additive. The project is complete and presentable
without it.

---

## Conventions

- Type hints on all function signatures.
- Docstrings on every function in `src/risk/` and `src/validation/` stating the
  formula and the assumptions it depends on.
- No secrets in code, ever. `.env` locally, GitHub Secrets in CI.
- Pin versions in `requirements.txt`.
- Every risk function gets a test with hand-verifiable expected values.
- Prefer explicit and boring over clever. This code needs to be explainable
  out loud months from now.

**Committing `data/risk.db`.** SQLite is a binary blob, so git cannot merge two
versions of it. The rules:

- **CI is the only writer that commits.** The scheduled job pulls with rebase,
  writes, and commits. Local runs must either use a scratch copy or not commit
  the result.
- Re-running a day never duplicates rows. Two write policies apply, and the
  split is deliberate — see "Price restatement" below.
- If a conflict happens anyway, resolve by taking the CI version and re-running
  the backfill locally. Never hand-merge the binary.
- Backfills are run deliberately and committed as their own commit, separate
  from any daily heartbeat commit.

## Anti-goals

Do not add: machine learning return forecasting, strategy parameter
optimization, intraday or high-frequency data, options, leverage, LLM-driven
allocation, or a live-money trading path. Each of these either breaks the
determinism requirement or changes what the project is.
