"""Read-only risk dashboard.

Every figure shown here is a stored row. Nothing is estimated, adjusted, or
rounded at render time: the numbers on screen are the numbers in the database,
which trace back to raw prices and positions (CLAUDE.md, "Auditability").

Two consequences of that, both deliberate:

- The connection is opened read-only at the SQLite level (`mode=ro`), not by
  convention. Streamlit Community Cloud sleeps after 12 hours of no traffic and
  wakes on request, so a dashboard that wrote anything would produce database
  changes at random times driven by whoever happened to visit.
- When a figure has not been computed yet, this says so rather than filling the
  gap. A 20-day volatility needs 20 sessions; until then the column is NULL and
  the panel explains why.

The one derived quantity here is the breach marker on the VaR chart, which
compares two stored columns -- `portfolio_pnl.daily_pnl` against the
`risk_estimates.var_amount` whose `applies_to_date` matches it. That is a join
between stored values rather than a risk calculation, and the join is on
`applies_to_date` because joining on `as_of_date` is lookahead bias.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "risk.db"

# Cache long enough that a page interaction does not re-read the file, short
# enough that a visitor after the daily run sees fresh rows.
CACHE_TTL_SECONDS = 300

POSITIVE = "#2E7D32"
NEGATIVE = "#C62828"
ACCENT = "#1565C0"
MUTED = "#9E9E9E"


st.set_page_config(page_title="Portfolio Risk Monitor", layout="wide")


def _db_path() -> Path:
    import os

    override = os.environ.get("RISK_DB_PATH")
    return Path(override) if override else DEFAULT_DB


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame.

    Opened with mode=ro so the dashboard cannot write even by accident. A
    missing file raises here rather than being silently created, which is what
    `sqlite3.connect` on a plain path would do.
    """
    path = _db_path()
    if not path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def _money(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"${value:,.0f}"


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}%}"


# ---------------------------------------------------------------- run status

def render_run_status() -> None:
    """Last successful run, prominently.

    GitHub sends no reliable notification when a scheduled workflow fails, and
    a workflow can be auto-disabled after 60 days of repository inactivity. A
    stale timestamp here is the substitute for that missing alarm, so it is the
    first thing on the page rather than a footnote (CLAUDE.md, "GitHub
    Actions").
    """
    runs = query(
        "SELECT run_at, status, message FROM runs ORDER BY run_id DESC LIMIT 20"
    )
    if runs.empty:
        st.error("No runs recorded. The daily job has never completed.")
        return

    successes = runs[runs["status"] == "success"]
    if successes.empty:
        st.error("No successful run on record. Every recorded run failed.")
        return

    last = pd.to_datetime(successes.iloc[0]["run_at"])
    age = pd.Timestamp.now(tz="UTC") - last
    hours = age.total_seconds() / 3600

    left, right = st.columns([3, 2])
    with left:
        if hours < 36:
            st.success(f"Last successful run: {last:%Y-%m-%d %H:%M} UTC")
        elif hours < 96:
            st.warning(
                f"Last successful run: {last:%Y-%m-%d %H:%M} UTC "
                f"({hours / 24:.1f} days ago). A weekday gap this long is "
                f"unusual."
            )
        else:
            st.error(
                f"Last successful run: {last:%Y-%m-%d %H:%M} UTC "
                f"({hours / 24:.1f} days ago). The job has likely stopped."
            )
    with right:
        recent_failures = (runs["status"] == "failure").sum()
        if recent_failures:
            st.warning(f"{recent_failures} failure(s) in the last 20 runs.")


# ------------------------------------------------------------ headline stats

def render_headline() -> None:
    pnl = query(
        """
        SELECT trade_date, total_value, cash, daily_pnl, daily_return
        FROM portfolio_pnl ORDER BY trade_date DESC LIMIT 1
        """
    )
    metrics = query(
        """
        SELECT as_of_date, vol_20d, drawdown, peak_value
        FROM portfolio_metrics ORDER BY as_of_date DESC LIMIT 1
        """
    )

    if pnl.empty:
        st.info(
            "No portfolio history yet. The first row appears after the daily "
            "job runs with positions held."
        )
        return

    row = pnl.iloc[0]
    m = metrics.iloc[0] if not metrics.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total value", _money(row["total_value"]), help=f"As of {row['trade_date']}")
    c2.metric(
        "Daily P&L",
        _money(row["daily_pnl"]),
        delta=None if pd.isna(row["daily_return"]) else _pct(row["daily_return"]),
    )
    c3.metric(
        "Drawdown",
        "—" if m is None else _pct(m["drawdown"]),
        help="Current, against the highest value on record — not maximum drawdown.",
    )
    c4.metric(
        "Volatility (20d)",
        "—" if m is None or pd.isna(m["vol_20d"]) else _pct(m["vol_20d"], 1),
        help="Annualised, sqrt(252) scaling. Blank until 20 returns exist.",
    )


# --------------------------------------------------------------- value chart

def render_value_chart() -> None:
    df = query(
        "SELECT trade_date, total_value FROM portfolio_pnl ORDER BY trade_date"
    )
    if len(df) < 2:
        st.info("Portfolio value appears once two sessions have been recorded.")
        return

    peaks = query(
        "SELECT as_of_date, peak_value FROM portfolio_metrics ORDER BY as_of_date"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["trade_date"], y=df["total_value"],
            name="Total value", line={"color": ACCENT, "width": 2},
        )
    )
    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks["as_of_date"], y=peaks["peak_value"],
                name="Running peak",
                line={"color": MUTED, "width": 1, "dash": "dot"},
            )
        )
    fig.update_layout(
        height=340, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        hovermode="x unified", yaxis_title="USD",
        legend={"orientation": "h", "y": 1.1},
    )
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------- VaR chart

def render_var_chart(method: str, confidence: float, window: int) -> None:
    """Predicted loss against realised loss, aligned on the predicted date.

    The join is on `applies_to_date`. An estimate computed from Monday's close
    is a claim about Tuesday, so Tuesday's loss is what tests it. Joining on
    `as_of_date` would compare each estimate to the day it was built from,
    which raises no error and makes the model look far better than it is.
    """
    df = query(
        """
        SELECT r.applies_to_date AS date,
               r.var_amount,
               r.es_amount,
               p.daily_pnl
        FROM risk_estimates r
        LEFT JOIN portfolio_pnl p ON p.trade_date = r.applies_to_date
        WHERE r.method = ? AND r.confidence = ? AND r.lookback_days = ?
        ORDER BY r.applies_to_date
        """,
        (method, confidence, window),
    )
    if df.empty:
        st.info(
            f"No {method} estimates at {confidence:.0%} over {window} days yet. "
            f"They begin once {window} daily returns are on record."
        )
        return

    df["loss"] = -df["daily_pnl"]
    tested = df.dropna(subset=["daily_pnl"])
    breaches = tested[tested["loss"] > tested["var_amount"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tested["date"], y=tested["loss"], name="Realised loss",
            marker_color=[
                NEGATIVE if row.loss > row.var_amount else MUTED
                for row in tested.itertuples()
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["var_amount"], name=f"VaR {confidence:.0%}",
            line={"color": ACCENT, "width": 2},
        )
    )
    if df["es_amount"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["es_amount"], name="Expected shortfall",
                line={"color": ACCENT, "width": 1, "dash": "dash"},
            )
        )
    fig.update_layout(
        height=360, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        hovermode="x unified", yaxis_title="USD (loss positive)",
        legend={"orientation": "h", "y": 1.12}, bargap=0.1,
    )
    st.plotly_chart(fig, width="stretch")

    n = len(tested)
    expected = n * (1 - confidence)
    st.caption(
        f"{len(breaches)} breach(es) in {n} tested session(s); "
        f"{expected:.1f} expected at this confidence. "
        f"Breaches are compared on the predicted session, not the session the "
        f"estimate was computed from. This count is a comparison of stored "
        f"columns, not a calibration test — see the Kupiec results in the README."
    )


# ------------------------------------------------------------- current risk

def render_current_estimates() -> None:
    latest = query("SELECT MAX(as_of_date) AS d FROM risk_estimates")
    if latest.empty or latest.iloc[0]["d"] is None:
        st.info(
            "No risk estimates stored yet. The shortest window needs 30 daily "
            "returns, so these appear roughly six weeks after the first fill."
        )
        return

    as_of = latest.iloc[0]["d"]
    df = query(
        """
        SELECT method, confidence, lookback_days, var_amount, es_amount,
               applies_to_date
        FROM risk_estimates WHERE as_of_date = ?
        ORDER BY lookback_days, confidence, method
        """,
        (as_of,),
    )

    applies_to = df["applies_to_date"].iloc[0]
    st.caption(
        f"Computed from data through **{as_of}**, predicting **{applies_to}**."
    )

    shown = pd.DataFrame({
        "Method": df["method"],
        "Confidence": df["confidence"].map(lambda c: f"{c:.0%}"),
        "Window": df["lookback_days"].map(lambda w: f"{w}d"),
        "VaR": df["var_amount"].map(_money),
        "Expected shortfall": df["es_amount"].map(_money),
    })
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(
        "Expected shortfall is blank for parametric rows: the closed form under "
        "normality is a modelling choice the project has not made, so the "
        "column is left empty rather than filled with an invented figure."
    )


# ------------------------------------------------------------- contributions

def render_contributions(confidence: float, window: int) -> None:
    latest = query(
        """
        SELECT MAX(as_of_date) AS d FROM risk_contributions
        WHERE confidence = ? AND lookback_days = ?
        """,
        (confidence, window),
    )
    if latest.empty or latest.iloc[0]["d"] is None:
        st.info("No contribution rows for this window yet.")
        return

    as_of = latest.iloc[0]["d"]
    df = query(
        """
        SELECT symbol, weight, contribution FROM risk_contributions
        WHERE as_of_date = ? AND confidence = ? AND lookback_days = ?
        ORDER BY contribution DESC
        """,
        (as_of, confidence, window),
    )

    fig = go.Figure(
        go.Bar(
            x=df["contribution"], y=df["symbol"], orientation="h",
            marker_color=[
                NEGATIVE if v > 0 else POSITIVE for v in df["contribution"]
            ],
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(280, 26 * len(df)),
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis_title="Contribution to VaR (USD)",
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"As of {as_of}. Contributions sum to the portfolio VaR implied by the "
        f"same weights and returns. A **negative** bar is a position that rose "
        f"on the tail session and offset part of the loss — that is correct "
        f"output, not a defect. Cash is carried as a zero-return position, so "
        f"it contributes exactly nothing."
    )


# -------------------------------------------------------------------- layout

st.title("Portfolio Risk Monitor")
st.caption(
    "11 SPDR Select Sector ETFs, equal weight, rebalanced on the first trading "
    "day of each month. Every figure below is read from stored rows; nothing is "
    "computed when this page loads."
)

render_run_status()

if not _db_path().exists():
    st.error(f"No database at {_db_path()}.")
    st.stop()

st.divider()
render_headline()

st.subheader("Portfolio value")
render_value_chart()

st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("Value at Risk against realised loss")
    windows = query(
        "SELECT DISTINCT lookback_days FROM risk_estimates ORDER BY lookback_days"
    )
    choices = windows["lookback_days"].tolist() if not windows.empty else [30]

    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Method", ["historical", "parametric"])
    confidence = c2.selectbox("Confidence", [0.95, 0.99], format_func=lambda c: f"{c:.0%}")
    window = c3.selectbox("Window", choices, format_func=lambda w: f"{w} days")
    render_var_chart(method, float(confidence), int(window))

with right:
    st.subheader("Current estimates")
    render_current_estimates()

st.divider()
st.subheader("Risk contribution by position")
c1, c2 = st.columns(2)
contrib_conf = c1.selectbox(
    "Confidence ", [0.95, 0.99], format_func=lambda c: f"{c:.0%}", key="cc"
)
contrib_window = c2.selectbox(
    "Window ", choices, format_func=lambda w: f"{w} days", key="cw"
)
render_contributions(float(contrib_conf), int(contrib_window))

st.divider()
with st.expander("Recent runs"):
    runs = query(
        "SELECT run_at, status, message FROM runs ORDER BY run_id DESC LIMIT 20"
    )
    st.dataframe(runs, hide_index=True, width="stretch")

st.caption(
    "Historical VaR uses the nearest-rank percentile convention. Estimates are "
    "one-session horizons. Calibration is validated on historical replay rather "
    "than on this live series — see the README."
)
