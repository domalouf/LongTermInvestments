"""Backtest page — run and visualise an annual-rebalance strategy vs SPY."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti.app import theme
from lti.backtest import BacktestConfig, run_backtest
from lti.metrics import FUNDAMENTAL_METRICS, MAGIC_FORMULA_METRICS, PRICE_METRICS
from lti.ranking import ScreenSpec

theme.header(
    "🧪 Strategy backtest",
    "Buy the top N of a screen, equal-weighted, rebalance once a year, and compare "
    "against SPY.",
    "Read the result with the survivorship warning below firmly in mind: the universe "
    "contains almost no companies that failed, so every screen here looks better than it was.",
)


@st.cache_data(show_spinner=True)
def _run(cfg_key: str):
    import json

    raw = json.loads(cfg_key)
    spec = ScreenSpec(
        metrics=raw["metrics"],
        top_n=raw["top_n"],
        weights=raw.get("weights"),
        filters=raw.get("filters", {}),
    )
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
    magic = st.checkbox(
        "Greenblatt Magic Formula",
        value=False,
        help="EBIT/EV and return on capital, equally weighted, financials and "
        "utilities excluded.",
    )
    chosen = st.multiselect(
        "Rank by",
        all_metrics,
        default=list(MAGIC_FORMULA_METRICS) if magic else ["pe", "debt_to_equity"],
        disabled=magic,
    )
    if magic:
        chosen = list(MAGIC_FORMULA_METRICS)
    top_n = st.slider("Top N", 5, 50, 30 if magic else 10)
    start = st.text_input("Start", "2013-01-01")
    end = st.text_input("End", "2024-01-01")
    rebal_month = st.slider("Rebalance month", 1, 12, 1)
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    capital = st.number_input("Initial capital ($)", value=100_000.0, step=10_000.0)
    excl_fin = st.checkbox("Exclude financials", value=magic)
    excl_util = st.checkbox("Exclude utilities", value=magic)
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
        "filters": {"exclude_financials": excl_fin, "exclude_utilities": excl_util},
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

c1, c2, c3, c4 = st.columns(4)
excess = stats["port_cagr"] - stats["bench_cagr"]
c1.metric("Strategy CAGR", f"{stats['port_cagr']:.1%}", f"{excess:+.1%} vs SPY")
c2.metric("SPY CAGR", f"{stats['bench_cagr']:.1%}")
c3.metric("Max drawdown", f"{stats['port_max_drawdown']:.1%}",
          f"{stats['port_max_drawdown'] - stats['bench_max_drawdown']:+.1%} vs SPY",
          delta_color="inverse")
c4.metric("Sharpe", f"{stats['port_sharpe']:.2f}", f"SPY {stats['bench_sharpe']:.2f}")

st.header("Growth of the initial stake")
log_scale = st.toggle(
    "Log scale", value=False,
    help="On a log axis equal vertical distances are equal *percentage* moves, "
         "which is the honest way to compare two curves over a long window.",
)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=bench.index, y=bench.values, name="SPY", mode="lines",
        line=dict(width=2, color=theme.MUTED),
        hovertemplate="SPY $%{y:,.0f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=equity.index, y=equity.values, name="Strategy", mode="lines",
        line=dict(width=2, color=theme.BLUE),
        hovertemplate="Strategy $%{y:,.0f}<extra></extra>",
    )
)
theme.show(
    fig, height=430, hovermode="x unified",
    yaxis=dict(type="log" if log_scale else "linear", tickprefix="$", title="portfolio value"),
    xaxis_title="",
)

with st.expander("All statistics"):
    # stats mixes floats with the drawdown peak/trough Timestamps, which Arrow
    # can't type as one column — format to text rather than let it guess
    def _fmt(k: str, v) -> str:
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        if not isinstance(v, (int, float)) or v != v:
            return str(v)
        if any(t in k for t in ("cagr", "return", "drawdown", "vol", "rate", "turnover")):
            return f"{v:.2%}"
        return f"{v:,.3f}"

    st.dataframe(
        pd.DataFrame({"stat": list(stats), "value": [_fmt(k, v) for k, v in stats.items()]}),
        hide_index=True,
        width="stretch",
    )

n_delisted = int(period_summary["n_delisted"].sum()) if not period_summary.empty else 0
st.warning(
    f"**Survivorship bias.** This backtest can only trade tickers that appear in today's SEC "
    f"ticker map *and* in Yahoo Finance, so companies that were acquired or went bankrupt are "
    f"largely invisible. Across every holding period here, only **{n_delisted}** position(s) hit "
    f"a delisting — in reality roughly half of US listed companies disappear over a window this "
    f"long. Treat the excess return as an upper bound, especially for a deep-value screen, which "
    f"selects exactly the distressed names that don't come back. ({len(warnings)} warnings raised.)"
)

st.header("Was it consistent, or one good year?")
if not period_summary.empty:
    ps = period_summary.copy()
    ps["year"] = pd.to_datetime(ps["rebalance_date"]).dt.year
    fig2 = go.Figure(
        go.Bar(
            x=ps["year"], y=ps["excess_return"],
            marker_color=[theme.POS if v >= 0 else theme.NEG for v in ps["excess_return"]],
            hovertemplate="%{x}<br>strategy − SPY: %{y:+.1%}<extra></extra>",
        )
    )
    fig2.update_traces(marker_line_width=0, marker_cornerradius=4)
    fig2.update_layout(bargap=0.34)
    theme.zero_line(fig2, axis="y")
    beat = int((ps["excess_return"] > 0).sum())
    theme.note(
        f"The strategy beat SPY in <b>{beat} of {len(ps)}</b> rebalance periods. "
        "With this few periods, a couple of good years can carry the whole CAGR — which is why "
        "the headline number alone shouldn't decide anything."
    )
    theme.show(
        fig2, height=300, legend=False,
        yaxis=dict(tickformat="+.0%", title="strategy − SPY, that period"),
        xaxis=dict(title="", dtick=1),
    )

with st.expander("Per-period detail"):
    st.dataframe(period_summary, hide_index=True, width="stretch")

with st.expander("Holdings (every pick, every period)"):
    st.dataframe(holdings, hide_index=True, width="stretch")
    st.download_button("Download holdings CSV", holdings.to_csv(index=False).encode(), "holdings.csv", "text/csv")

if warnings:
    with st.expander(f"Warnings ({len(warnings)})"):
        for w in warnings:
            st.text(w)
