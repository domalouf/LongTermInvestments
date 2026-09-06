"""Rolling-window backtest page — test a strategy across many overlapping
N-year windows spanning all available history, instead of one start/end pick."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti.metrics import FUNDAMENTAL_METRICS, PRICE_METRICS
from lti.ranking import ScreenSpec
from lti.rolling import RollingConfig, run_rolling_backtest

st.set_page_config(page_title="Rolling Backtest", page_icon="🔁", layout="wide")
st.title("🔁 Rolling-window backtest")
st.caption(
    "Reruns the strategy over every overlapping window of each chosen length, spanning all "
    "available price history, instead of a single start/end pick. A strategy that only wins "
    "in one window is a curve-fit, not an edge."
)

PCT_COLS = [
    "mean_cagr", "median_cagr", "worst_cagr", "best_cagr",
    "mean_excess_cagr", "win_rate", "mean_max_drawdown", "worst_max_drawdown",
]


@st.cache_data(show_spinner=True)
def _run(cfg_key: str):
    raw = json.loads(cfg_key)
    spec = ScreenSpec(metrics=raw["metrics"], top_n=raw["top_n"])
    cfg = RollingConfig(
        screen=spec,
        window_years=raw["window_years"],
        step_months=raw["step_months"],
        start=raw["start"] or None,
        end=raw["end"] or None,
        rebalance_month=raw["rebalance_month"],
        market_cap_min=raw["market_cap_min"],
        initial_capital=raw["initial_capital"],
    )
    result = run_rolling_backtest(cfg)
    return result.windows, result.summary, result.warnings


with st.sidebar:
    st.header("Strategy")
    all_metrics = FUNDAMENTAL_METRICS + PRICE_METRICS
    chosen = st.multiselect("Rank by", all_metrics, default=["pe", "debt_to_equity"])
    top_n = st.slider("Top N", 5, 50, 10)
    windows = st.multiselect("Window lengths (years)", [1, 2, 3, 5, 7, 10], default=[3, 5])
    step_months = st.slider("Step between window starts (months)", 1, 24, 12)
    start = st.text_input("Earliest start (blank = all available data)", "")
    end = st.text_input("Latest end (blank = all available data)", "")
    rebal_month = st.slider("Rebalance month", 1, 12, 1)
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    capital = st.number_input("Initial capital ($)", value=100_000.0, step=10_000.0)
    go_btn = st.button("Run rolling backtest", type="primary")

if not chosen:
    st.info("Pick at least one metric.")
    st.stop()
if not windows:
    st.info("Pick at least one window length.")
    st.stop()

cfg_key = json.dumps(
    {
        "metrics": chosen,
        "top_n": top_n,
        "window_years": sorted(windows),
        "step_months": step_months,
        "start": start.strip(),
        "end": end.strip(),
        "rebalance_month": rebal_month,
        "market_cap_min": cap_floor_m * 1e6,
        "initial_capital": capital,
    }
)

if not go_btn:
    st.info("Set the strategy in the sidebar and hit **Run rolling backtest**.")
    st.stop()

try:
    windows_df, summary, warnings = _run(cfg_key)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Summary by window length")
st.dataframe(
    summary.style.format({**{c: "{:.1%}" for c in PCT_COLS}, "mean_sharpe": "{:.2f}"}),
    hide_index=True,
    use_container_width=True,
)

fig = go.Figure()
for years, grp in windows_df.groupby("window_years"):
    fig.add_trace(go.Box(y=grp["port_cagr"], name=f"{years}y", boxpoints="all"))
fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis_title="CAGR per window",
    yaxis_tickformat=".0%",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "`win_rate` = share of windows where the strategy's CAGR beat the benchmark's over that "
    "same window. Windows of one length overlap heavily (a 12-month step on a 5-year window "
    "shares 4 of 5 years with its neighbor), so treat the spread as illustrative, not "
    "independent samples — the survivorship-bias caveat from the single-period backtest still "
    "applies to every window here too."
)

st.subheader("Every window")
st.dataframe(windows_df, hide_index=True, use_container_width=True)
st.download_button(
    "Download windows CSV", windows_df.to_csv(index=False).encode(), "rolling_windows.csv", "text/csv"
)

if warnings:
    with st.expander(f"Warnings ({len(warnings)})"):
        for w in warnings:
            st.text(w)
