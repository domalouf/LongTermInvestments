"""Rolling backtest — the same strategy over every N-year window of the history, not one start/end pick."""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from lti.app import theme, widgets
from lti.ranking import ScreenSpec
from lti.rolling import RollingConfig, run_rolling_backtest

theme.header(
    "🔁 Rolling backtest",
    "The same strategy over every window of each length — every 3-year and every 5-year "
    "stretch of the price history, say — instead of one start and end date.",
    "One backtest is one draw from history, and easy to fit to. A strategy that beats its "
    "universe in most windows is worth a closer look; one that wins in a few is a story about "
    "those years.",
)


@st.cache_data(show_spinner="Running one backtest per window — a minute or two…")
def _run(cfg_key: str):
    raw = json.loads(cfg_key)
    cfg = RollingConfig(
        screen=ScreenSpec(metrics=raw["metrics"], top_n=raw["top_n"], min_coverage=raw["min_coverage"]),
        window_years=raw["window_years"],
        step_months=raw["step_months"],
        start=raw["start"] or None,
        end=raw["end"] or None,
        rebalance_month=raw["rebalance_month"],
        market_cap_min=raw["market_cap_min"],
        sell_rank=raw.get("sell_rank"),
        **widgets.friction_kwargs(raw["frictions"]),
    )
    result = run_rolling_backtest(cfg)
    return result.windows, result.summary, result.warnings


with st.sidebar:
    st.header("Strategy")
    chosen, coverage = widgets.rank_by(["pe", "debt_to_equity"])
    top_n = st.slider("Top N", 5, 50, 10)
    sell_rank = widgets.sell_rank(top_n)
    windows = st.multiselect("Window lengths (years)", [1, 2, 3, 5, 7, 10], default=[3, 5])
    step_months = st.slider("Step between window starts (months)", 1, 24, 12)
    start = st.text_input("Earliest start (blank = all the price history)", "")
    end = st.text_input("Latest end (blank = all the price history)", "")
    rebal_month = st.slider(
        "Rebalance month", 1, 12, 4,
        help="April by default, as on the Backtest page: by then calendar-year 10-Ks are in.",
    )
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    fric = widgets.frictions()
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
        "min_coverage": coverage,
        "window_years": sorted(windows),
        "step_months": step_months,
        "start": start.strip(),
        "end": end.strip(),
        "rebalance_month": rebal_month,
        "market_cap_min": cap_floor_m * 1e6,
        "sell_rank": sell_rank,
        "frictions": fric,
    }
)

widgets.run_gate("rolling", go_btn, cfg_key, "Set the strategy in the sidebar and hit **Run rolling backtest**.")

try:
    windows_df, summary, warnings = _run(cfg_key)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

st.header("By window length")
if fric["cost_bps"] > 0 or fric["tax"] is not None:
    theme.note(
        f"Every window after {fric['cost_bps']:g} bps a trade{' and taxes' if fric['tax'] else ''} — the "
        "strategy, its universe and SPY alike. Each window's CAGR before them is in "
        "<i>Every window</i>, as <code>port_cagr_gross</code>."
    )
st.dataframe(
    summary, hide_index=True, width="stretch",
    column_config={
        "window_years": st.column_config.NumberColumn("Years", format="%d"),
        "n_windows": st.column_config.NumberColumn("Windows", format="%d"),
        "median_cagr": st.column_config.NumberColumn("Median CAGR", format="percent"),
        "worst_cagr": st.column_config.NumberColumn("Worst", format="percent"),
        "best_cagr": st.column_config.NumberColumn("Best", format="percent"),
        "mean_excess_vs_univ": st.column_config.NumberColumn(
            "vs universe", format="percent", help="Average CAGR gap to the equal-weighted universe the screen ranked."),
        "win_rate_vs_univ": st.column_config.NumberColumn(
            "Beat universe", format="percent", help="Share of windows the strategy's CAGR beat the universe's."),
        "mean_excess_vs_bench": st.column_config.NumberColumn("vs SPY", format="percent"),
        "win_rate_vs_bench": st.column_config.NumberColumn("Beat SPY", format="percent"),
        "mean_max_drawdown": st.column_config.NumberColumn("Avg max drawdown", format="percent"),
        "worst_max_drawdown": st.column_config.NumberColumn("Worst drawdown", format="percent"),
        "mean_sharpe": st.column_config.NumberColumn("Avg Sharpe", format="%.2f"),
    },
)

fig = go.Figure()
for i, (years, grp) in enumerate(windows_df.groupby("window_years")):
    fig.add_trace(
        go.Box(
            y=grp["excess_cagr_vs_univ"], name=f"{years}-year", boxpoints="all", jitter=0.3,
            marker_color=theme.SERIES[i % len(theme.SERIES)],
            customdata=grp["start"].dt.strftime("%Y-%m-%d"),
            hovertemplate="window from %{customdata}<br>%{y:+.1%} a year vs the universe<extra></extra>",
        )
    )
theme.zero_line(fig, axis="y")
beat = windows_df["beat_universe"].dropna()
theme.note(
    f"Each dot is one window: the strategy's CAGR minus the universe's. It beat the universe in "
    f"<b>{int(beat.sum())} of {len(beat)}</b> windows. Windows of one length overlap heavily — at a "
    "12-month step, neighbouring 5-year windows share four years — so read the spread as "
    "illustrative, not as independent samples. Every window carries the backtest's survivorship "
    "bias, which is why the universe, not SPY, is the yardstick."
)
theme.show(fig, height=420, legend=False, yaxis=dict(tickformat="+.0%", title="strategy − universe, CAGR"),
           xaxis_title="")

with st.expander(f"Every window ({len(windows_df)})"):
    st.dataframe(windows_df, hide_index=True, width="stretch")
    st.download_button(
        "Download windows CSV", windows_df.to_csv(index=False).encode(), "rolling_windows.csv", "text/csv"
    )

widgets.warnings_expander(warnings)
