"""Factor analysis page — which metrics rank stocks by forward return.

For a grid of historical as-of dates it takes a point-in-time snapshot, computes
every metric and each stock's forward return over a fixed horizon, then measures
the cross-sectional correlation (the Information Coefficient) between metric and
forward return. Averaging the per-date ICs isolates stock-selection signal from
market direction.
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from lti.app import theme, widgets
from lti.factor import ALL_METRICS, ICConfig, compute_ic

theme.header(
    "📐 Factor analysis",
    "On each as-of date, the rank correlation across stocks between a metric and its "
    "forward return — the Information Coefficient. Averaging per-date ICs separates "
    "stock-picking signal from market direction.",
    "A <b>negative</b> mean IC means <i>lower</i> values of the metric went with higher returns — "
    "which is what you want to see for <code>pe</code>, <code>pb</code> and <code>debt_to_equity</code>. "
    "This page has far more statistical power than the Backtest page: thousands of names per "
    "date rather than a dozen annual portfolio returns.",
)


@st.cache_data(show_spinner=True)
def _run(cfg_key: str):
    raw = json.loads(cfg_key)
    cfg = ICConfig(
        metrics=raw["metrics"],
        start=raw["start"],
        end=raw["end"] or None,
        horizon_months=raw["horizon"],
        step_months=raw["step"],
        market_cap_min=raw["market_cap_min"],
        require_positive_eps=raw["require_positive_eps"],
        operating_only=raw["operating_only"],
        quantiles=raw["quantiles"],
        method=raw["method"],
    )
    result = compute_ic(cfg)
    return result.summary, result.ic_by_period, result.n_by_period, result.bucket_returns, result.warnings


with st.sidebar:
    st.header("Analysis")
    chosen = st.multiselect("Metrics", ALL_METRICS, default=list(ALL_METRICS))
    start = st.text_input(
        "Start", "2011-04-01",
        help="As-of dates repeat on this day each year. April, like the backtest: by then "
             "calendar-year 10-Ks are in, so each date ranks on last year's numbers.",
    )
    end = st.text_input("End (blank = latest)", "")
    horizon = st.slider("Forward-return horizon (months)", 3, 36, 12, step=3)
    step = st.slider("As-of spacing (months)", 3, 24, 12, step=3)
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    require_pos_eps = st.checkbox("Require positive EPS", value=False)
    operating_only = st.checkbox(
        "Operating companies only", value=True,
        help="Drops commodity and crypto trusts, shells and other filers with no revenue.",
    )
    quantiles = st.slider("Quantile buckets", 3, 10, 5)
    method = st.radio("Correlation", ["spearman", "pearson"], horizontal=True)
    go_btn = st.button("Run analysis", type="primary")

if not chosen:
    st.info("Pick at least one metric in the sidebar.")
    st.stop()

cfg_key = json.dumps(
    {
        "metrics": chosen,
        "start": start,
        "end": end,
        "horizon": horizon,
        "step": step,
        "market_cap_min": cap_floor_m * 1e6,
        "require_positive_eps": require_pos_eps,
        "operating_only": operating_only,
        "quantiles": quantiles,
        "method": method,
    }
)

widgets.run_gate("factor", go_btn, cfg_key, "Set the parameters in the sidebar and hit **Run analysis**.")

try:
    summary, ic_by_period, n_by_period, bucket_returns, warnings = _run(cfg_key)
except (RuntimeError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

if step < horizon:
    st.warning(
        f"As-of spacing ({step}m) is shorter than the horizon ({horizon}m): forward-return "
        "windows overlap, so the plain t-stat is optimistic — read `t_stat_nw`, which allows for it."
    )

st.subheader("Ranking power by metric")
st.caption(
    "`mean_ic` — average cross-sectional correlation · `ic_ir` mean/std · "
    "`t_stat` significance across periods · `t_stat_nw` the same with Newey-West errors, which "
    "allow for overlapping return windows (read this one when spacing < horizon) · "
    "`hit_rate` share of periods with the dominant sign · "
    "`q_spread` top-minus-bottom bucket forward return · `monotonicity` rank corr of bucket index vs return."
)
st.dataframe(
    summary.reset_index(names="metric").style.format(
        {
            "mean_ic": "{:.3f}", "ic_std": "{:.3f}", "ic_ir": "{:.2f}", "t_stat": "{:.2f}", "t_stat_nw": "{:.2f}",
            "hit_rate": "{:.0%}", "avg_n_stocks": "{:.0f}", "q_spread": "{:.1%}", "monotonicity": "{:.2f}",
        },
        na_rep="—",
    ),
    hide_index=True,
    width="stretch",
)

st.header("Mean IC by metric")
theme.note(
    "Further from zero is a stronger ranking signal; the sign says which direction. "
    "Blue bars point the way theory expects for a value metric, red against it."
)
ms = summary.reset_index(names="metric").sort_values("mean_ic")
fig = go.Figure(
    go.Bar(
        x=ms["mean_ic"], y=ms["metric"], orientation="h",
        marker_color=[theme.NEG if v >= 0 else theme.POS for v in ms["mean_ic"]],
        customdata=ms[["t_stat", "n_periods"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>mean IC %{x:.3f}<br>t %{customdata[0]:.2f} over %{customdata[1]:.0f} periods<extra></extra>",
    )
)
theme.bar_marks(fig, color=None)
theme.zero_line(fig, axis="x")
theme.show(fig, height=max(320, 26 * len(ms) + 70), legend=False,
           xaxis_title="mean information coefficient", yaxis_title="")

st.header("IC over time")
theme.note(
    "A metric worth trusting is one whose sign is stable across periods, not one big year. "
    "Compare the bars against the mean line."
)
focus = st.selectbox("Metric", list(ic_by_period.columns), key="ic_focus")
series = ic_by_period[focus].dropna()
if series.empty:
    st.info("No usable periods for this metric.")
else:
    bar = go.Figure(
        go.Bar(
            x=series.index, y=series.values,
            marker_color=[theme.NEG if v >= 0 else theme.POS for v in series.values],
            hovertemplate="%{x|%b %Y}<br>IC %{y:.3f}<extra></extra>",
        )
    )
    theme.bar_marks(bar, color=None)
    theme.zero_line(bar, axis="y")
    bar.add_hline(y=series.mean(), line_width=1.5, line_color=theme.INK_2)
    bar.add_annotation(
        x=1.0, xref="paper", y=series.mean(), yanchor="bottom", xanchor="right",
        text=f"mean {series.mean():.3f}", showarrow=False,
        font=dict(size=10.5, color=theme.INK_2),
    )
    theme.show(bar, height=330, legend=False, yaxis_title=f"{focus} IC", xaxis_title="")

st.header("Forward return by metric quantile")
theme.note(
    "Q1 is the lowest metric value, Qn the highest. A clean staircase means the metric "
    "separates winners from losers across its whole range, not just at one extreme."
)
if bucket_returns.empty:
    st.info("Not enough cross-sectional breadth to form quantile buckets.")
else:
    bfocus = st.selectbox("Metric", list(bucket_returns.index), key="bucket_focus")
    row = bucket_returns.loc[bfocus].dropna()
    bfig = go.Figure(
        go.Bar(
            x=list(row.index), y=row.values,
            hovertemplate="%{x}<br>mean forward return %{y:.1%}<extra></extra>",
        )
    )
    theme.bar_marks(bfig)
    theme.zero_line(bfig, axis="y")
    theme.show(bfig, height=330, legend=False,
               yaxis=dict(tickformat=".1%", title="mean forward return"),
               xaxis_title=f"{bfocus} quantile")

with st.expander("IC by period (table)"):
    st.dataframe(ic_by_period.round(3), width="stretch")
    st.download_button("Download IC-by-period CSV", ic_by_period.to_csv().encode(), "ic_by_period.csv", "text/csv")

widgets.warnings_expander(warnings)

st.info(
    "**Caveats.** Univariate IC ignores that metrics are correlated with each other "
    "(ROE ↔ margins); it does not tell you a metric's *marginal* contribution. "
    "The universe is survivorship-biased — only names in the current SEC ticker map and "
    "in Yahoo are measured — which flatters value and quality signals."
)
