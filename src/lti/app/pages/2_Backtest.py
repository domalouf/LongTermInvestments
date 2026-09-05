"""Backtest page — run and visualise an annual-rebalance strategy vs SPY."""

from __future__ import annotations

import json

import lti.config as config

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti.backtest import BacktestConfig, run_backtest
from lti.metrics import FUNDAMENTAL_METRICS, PRICE_METRICS
from lti.ranking import ScreenSpec

st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
st.title("🧪 Strategy backtest")


@st.cache_data(show_spinner=True)
def _run(cfg_key: str):
    import json

    raw = json.loads(cfg_key)
    spec = ScreenSpec(metrics=raw["metrics"], top_n=raw["top_n"], weights=raw.get("weights"))
    cfg = BacktestConfig(
        screen=spec,
        start=raw["start"],
        end=raw["end"],
        rebalance_month=raw["rebalance_month"],
        market_cap_min=raw["market_cap_min"],
        initial_capital=raw["initial_capital"],
    )
    result = run_backtest(cfg)
    return result.equity_curve, result.benchmark_curve, result.holdings, result.period_summary, result.stats, result.warnings


with st.sidebar:
    st.header("Strategy")
    all_metrics = FUNDAMENTAL_METRICS + PRICE_METRICS
    chosen = st.multiselect("Rank by", all_metrics, default=["pe", "debt_to_equity"])
    top_n = st.slider("Top N", 5, 50, 10)
    start = st.text_input("Start", "2011-01-01")
    end = st.text_input("End", "2024-01-01")
    rebal_month = st.slider("Rebalance month", 1, 12, 1)
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    capital = st.number_input("Initial capital ($)", value=100_000.0, step=10_000.0)
    go_btn = st.button("Run backtest", type="primary")

if not chosen:
    st.info("Pick at least one metric.")
    st.stop()

cfg_key = json.dumps(
    {
        "metrics": chosen,
        "top_n": top_n,
        "start": start,
        "end": end,
        "rebalance_month": rebal_month,
        "market_cap_min": cap_floor_m * 1e6,
        "initial_capital": capital,
    }
)

if not go_btn:
    st.info("Set the strategy in the sidebar and hit **Run backtest**.")
    st.stop()

try:
    equity, bench, holdings, period_summary, stats, warnings = _run(cfg_key)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

log_scale = st.toggle("Log scale", value=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy"))
fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="SPY"))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log" if log_scale else "linear")
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy CAGR", f"{stats['port_cagr']:.1%}")
c2.metric("SPY CAGR", f"{stats['bench_cagr']:.1%}")
c3.metric("Max drawdown", f"{stats['port_max_drawdown']:.1%}")
c4.metric("Sharpe", f"{stats['port_sharpe']:.2f}")

st.subheader("Stats")
st.dataframe(
    pd.Series(stats).rename("value").to_frame().reset_index(names="stat"),
    hide_index=True,
    use_container_width=True,
)

n_delisted = int(period_summary["n_delisted"].sum()) if not period_summary.empty else 0
st.warning(
    f"**Survivorship bias:** this backtest can only trade tickers present in the current SEC "
    f"ticker map and in Yahoo Finance. {n_delisted} holding-periods hit a delisting/among the "
    f"selected names; {len(warnings)} warnings were raised. Real-world results for a "
    f"value screen are typically worse than shown."
)

st.subheader("Per-period summary")
st.dataframe(period_summary, hide_index=True, use_container_width=True)

with st.expander("Holdings (every pick, every period)"):
    st.dataframe(holdings, hide_index=True, use_container_width=True)
    st.download_button("Download holdings CSV", holdings.to_csv(index=False).encode(), "holdings.csv", "text/csv")

if warnings:
    with st.expander(f"Warnings ({len(warnings)})"):
        for w in warnings:
            st.text(w)
