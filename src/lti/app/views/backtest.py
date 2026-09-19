"""Backtest page — run and visualise an annual-rebalance strategy against SPY and its own universe."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti.app import theme
from lti.backtest import BacktestConfig, rebalance_month_spread, run_backtest
from lti.metrics import FUNDAMENTAL_METRICS, MAGIC_FORMULA_METRICS, PRICE_METRICS
from lti.ranking import ScreenSpec

theme.header(
    "🧪 Strategy backtest",
    "Buy the top N of a screen, equal-weighted, rebalance once a year, and compare against "
    "SPY and against an equal-weighted basket of <i>every</i> stock the screen ranked.",
    "The basket is the fair yardstick: it can only hold today's survivors too, so the gap "
    "between it and the strategy is what the ranking itself added. The gap to SPY also "
    "contains the survivorship bias — see the warning below.",
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _config(cfg_key: str) -> BacktestConfig:
    raw = json.loads(cfg_key)
    spec = ScreenSpec(
        metrics=raw["metrics"],
        top_n=raw["top_n"],
        weights=raw.get("weights"),
        filters=raw.get("filters", {}),
    )
    return BacktestConfig(
        screen=spec,
        start=raw["start"],
        end=raw["end"],
        rebalance_month=raw["rebalance_month"],
        market_cap_min=raw["market_cap_min"],
        initial_capital=raw["initial_capital"],
    )


@st.cache_data(show_spinner="Running the backtest…")
def _run(cfg_key: str):
    result = run_backtest(_config(cfg_key))
    return (
        result.equity_curve, result.benchmark_curve, result.universe_curve,
        result.holdings, result.period_summary, result.stats, result.warnings,
    )


@st.cache_data(show_spinner="Running the strategy once per rebalance month — about a minute…")
def _spread(cfg_key: str) -> pd.DataFrame:
    return rebalance_month_spread(_config(cfg_key))


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
    rebal_month = st.slider(
        "Rebalance month", 1, 12, 4,
        help="April by default: by then nearly every calendar-year company has filed its 10-K, "
             "so the ranking uses last year's numbers. In January it would use the year before's.",
    )
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

# A button is only True on the rerun its click triggers; remember what was run so
# that toggling a chart option below doesn't blank the page. Changing the strategy
# changes the key, which asks for a fresh run.
if go_btn:
    st.session_state["backtest_key"] = cfg_key
if st.session_state.get("backtest_key") != cfg_key:
    st.info("Set the strategy in the sidebar and hit **Run backtest**.")
    st.stop()

try:
    equity, bench, universe, holdings, period_summary, stats, warnings = _run(cfg_key)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Strategy CAGR", f"{stats['port_cagr']:.1%}",
          f"{stats['excess_cagr_vs_univ']:+.1%} vs universe")
c2.metric("Universe CAGR", f"{stats['univ_cagr']:.1%}",
          help="Every stock the screen ranked on each rebalance date, equal-weighted — "
               "what picking at random from the same candidates would have returned.")
c3.metric("SPY CAGR", f"{stats['bench_cagr']:.1%}",
          help=f"The strategy {'beat' if stats['excess_cagr'] >= 0 else 'trailed'} SPY by "
               f"{abs(stats['excess_cagr']):.1%} a year.")
c4.metric("Max drawdown", f"{stats['port_max_drawdown']:.1%}",
          f"{stats['port_max_drawdown'] - stats['bench_max_drawdown']:+.1%} vs SPY",
          delta_color="inverse")
c5.metric("Sharpe", f"{stats['port_sharpe']:.2f}", f"SPY {stats['bench_sharpe']:.2f}")

st.header("Growth of the initial stake")
log_scale = st.toggle(
    "Log scale", value=False,
    help="On a log axis equal vertical distances are equal *percentage* moves, "
         "which is the honest way to compare curves over a long window.",
)
fig = go.Figure()
for curve, name, color in [
    (bench, "SPY", theme.MUTED),
    (universe, "Universe (equal weight)", theme.ORANGE),
    (equity, "Strategy", theme.BLUE),
]:
    fig.add_trace(
        go.Scatter(
            x=curve.index, y=curve.values, name=name, mode="lines",
            line=dict(width=2, color=color),
            hovertemplate=f"{name} $%{{y:,.0f}}<extra></extra>",
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
        if any(t in k for t in ("cagr", "return", "drawdown", "vol", "rate", "turnover", "beat")):
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
    f"long. That flatters the strategy *and* the universe basket alike, which is why the gap "
    f"between those two is the number to trust; the gap to SPY is an upper bound. "
    f"({len(warnings)} warnings raised.)"
)

st.header("Was it consistent, or one good year?")
basis = st.radio(
    "Compare each period against", ["the universe", "SPY"], horizontal=True,
    help="The universe is the fair comparison: same candidates, same survivorship bias.",
)
if not period_summary.empty:
    ps = period_summary.copy()
    ps["year"] = pd.to_datetime(ps["rebalance_date"]).dt.year
    col, label = ("excess_vs_univ", "universe") if basis == "the universe" else ("excess_return", "SPY")
    fig2 = go.Figure(
        go.Bar(
            x=ps["year"], y=ps[col],
            marker_color=[theme.POS if v >= 0 else theme.NEG for v in ps[col]],
            hovertemplate=f"%{{x}}<br>strategy − {label}: %{{y:+.1%}}<extra></extra>",
        )
    )
    fig2.update_traces(marker_line_width=0, marker_cornerradius=4)
    fig2.update_layout(bargap=0.34)
    theme.zero_line(fig2, axis="y")
    beat = int((ps[col] > 0).sum())
    theme.note(
        f"The strategy beat {label} in <b>{beat} of {len(ps)}</b> rebalance periods. "
        "With this few periods, a couple of good years can carry the whole CAGR — which is why "
        "the headline number alone shouldn't decide anything."
    )
    theme.show(
        fig2, height=300, legend=False,
        yaxis=dict(tickformat="+.0%", title=f"strategy − {label}, that period"),
        xaxis=dict(title="", dtick=1),
    )

st.header("Does the rebalance month matter?")
theme.note(
    "With a dozen annual rebalances, <i>when</i> in the year the portfolio turns over can decide "
    "whether a screen beats its universe. Running the same strategy once per month shows how "
    "much of the result above is the screen and how much is the calendar."
)
if st.button("Run all 12 rebalance months"):
    st.session_state["backtest_spread_key"] = cfg_key
if st.session_state.get("backtest_spread_key") == cfg_key:
    spread = _spread(cfg_key)
    if spread.empty:
        st.info("No month produced a result for this date range.")
    else:
        sp = spread.copy()
        sp["month"] = [MONTHS[m - 1] for m in sp["rebalance_month"]]
        fig3 = go.Figure(
            go.Bar(
                x=sp["month"], y=sp["excess_vs_univ"],
                marker_color=[theme.POS if v >= 0 else theme.NEG for v in sp["excess_vs_univ"]],
                customdata=sp[["port_cagr", "univ_cagr", "excess_vs_bench"]].to_numpy(),
                hovertemplate=(
                    "rebalance in %{x}<br>strategy − universe %{y:+.1%} a year"
                    "<br>strategy %{customdata[0]:.1%} · universe %{customdata[1]:.1%}"
                    "<br>vs SPY %{customdata[2]:+.1%}<extra></extra>"
                ),
            )
        )
        fig3.update_traces(marker_line_width=0, marker_cornerradius=4)
        fig3.update_layout(bargap=0.34)
        theme.zero_line(fig3, axis="y")
        ex = sp["excess_vs_univ"]
        theme.note(
            f"Excess CAGR over the universe ranges from <b>{ex.min():+.1%}</b> to "
            f"<b>{ex.max():+.1%}</b> depending on the month (median {ex.median():+.1%}); "
            f"it is positive in <b>{int((ex > 0).sum())} of {len(ex)}</b> months."
        )
        theme.show(
            fig3, height=300, legend=False,
            yaxis=dict(tickformat="+.0%", title="strategy − universe, CAGR"),
            xaxis_title="rebalance month",
        )
        with st.expander("Per-month detail"):
            st.dataframe(sp.drop(columns=["month"]), hide_index=True, width="stretch")

with st.expander("Per-period detail"):
    st.dataframe(period_summary, hide_index=True, width="stretch")

with st.expander("Holdings (every pick, every period)"):
    theme.note("Each pick with the metric values it was ranked on — the first place to look when a result seems too good.")
    st.dataframe(holdings, hide_index=True, width="stretch")
    st.download_button("Download holdings CSV", holdings.to_csv(index=False).encode(), "holdings.csv", "text/csv")

if warnings:
    with st.expander(f"Warnings ({len(warnings)})"):
        for w in warnings:
            st.text(w)
