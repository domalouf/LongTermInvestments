"""Factor analysis page — which metrics rank stocks by forward return.

For a grid of historical as-of dates it takes a point-in-time snapshot, computes
every metric and each stock's forward return over a fixed horizon, then measures
the cross-sectional correlation (the Information Coefficient) between metric and
forward return. Averaging the per-date ICs isolates stock-selection signal from
market direction.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lti.factor import ALL_METRICS, ICConfig, compute_ic

st.set_page_config(page_title="Factor Analysis", page_icon="📐", layout="wide")
st.title("📐 Factor analysis")
st.caption(
    "Cross-sectional Information Coefficient (IC): on each as-of date, the rank "
    "correlation across stocks between a metric and its forward return. "
    "**Negative mean IC** ⇒ lower values of the metric went with higher returns "
    "(expected for `pe`, `pb`, `debt_to_equity`)."
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
        quantiles=raw["quantiles"],
        method=raw["method"],
    )
    result = compute_ic(cfg)
    return result.summary, result.ic_by_period, result.n_by_period, result.bucket_returns, result.warnings


with st.sidebar:
    st.header("Analysis")
    chosen = st.multiselect("Metrics", ALL_METRICS, default=list(ALL_METRICS))
    start = st.text_input("Start", "2011-01-01")
    end = st.text_input("End (blank = latest)", "")
    horizon = st.slider("Forward-return horizon (months)", 3, 36, 12, step=3)
    step = st.slider("As-of spacing (months)", 3, 24, 12, step=3)
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    require_pos_eps = st.checkbox("Require positive EPS", value=False)
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
        "quantiles": quantiles,
        "method": method,
    }
)

if not go_btn:
    st.info("Set the parameters in the sidebar and hit **Run analysis**.")
    st.stop()

try:
    summary, ic_by_period, n_by_period, bucket_returns, warnings = _run(cfg_key)
except (RuntimeError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

if step < horizon:
    st.warning(
        f"As-of spacing ({step}m) is shorter than the horizon ({horizon}m): forward-return "
        "windows overlap, so the t-stats below are optimistic. Set spacing ≥ horizon for clean stats."
    )

st.subheader("Ranking power by metric")
st.caption(
    "`mean_ic` — average cross-sectional correlation · `ic_ir` mean/std · "
    "`t_stat` significance across periods · `hit_rate` share of periods with the dominant sign · "
    "`q_spread` top-minus-bottom bucket forward return · `monotonicity` rank corr of bucket index vs return."
)
st.dataframe(
    summary.reset_index(names="metric").style.format(
        {
            "mean_ic": "{:.3f}", "ic_std": "{:.3f}", "ic_ir": "{:.2f}", "t_stat": "{:.2f}",
            "hit_rate": "{:.0%}", "avg_n_stocks": "{:.0f}", "q_spread": "{:.1%}", "monotonicity": "{:.2f}",
        },
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
)

fig = px.bar(
    summary.reset_index(names="metric"),
    x="mean_ic", y="metric", orientation="h",
    color="mean_ic", color_continuous_scale="RdBu", range_color=[-0.15, 0.15],
    title="Mean IC (further from zero = stronger ranking signal)",
)
fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("IC over time")
focus = st.selectbox("Metric", list(ic_by_period.columns), key="ic_focus")
series = ic_by_period[focus].dropna()
if series.empty:
    st.info("No usable periods for this metric.")
else:
    bar = go.Figure()
    bar.add_trace(go.Bar(x=series.index, y=series.values, marker_color=["#c0392b" if v < 0 else "#2c7fb8" for v in series.values]))
    bar.add_hline(y=series.mean(), line_dash="dash", annotation_text=f"mean {series.mean():.3f}")
    bar.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title=f"{focus} — per-period IC")
    st.plotly_chart(bar, use_container_width=True)

st.subheader("Mean forward return by metric quantile")
st.caption("Q1 = lowest metric value, Qn = highest. A monotonic staircase means the metric ranks returns cleanly.")
if bucket_returns.empty:
    st.info("Not enough cross-sectional breadth to form quantile buckets.")
else:
    bfocus = st.selectbox("Metric", list(bucket_returns.index), key="bucket_focus")
    row = bucket_returns.loc[bfocus].dropna()
    bfig = px.bar(x=row.index, y=row.values, title=f"{bfocus} — forward return by bucket", labels={"x": "quantile", "y": "mean forward return"})
    bfig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat=".1%")
    st.plotly_chart(bfig, use_container_width=True)

with st.expander("IC by period (table)"):
    st.dataframe(ic_by_period.round(3), use_container_width=True)
    st.download_button("Download IC-by-period CSV", ic_by_period.to_csv().encode(), "ic_by_period.csv", "text/csv")

if warnings:
    with st.expander(f"Warnings ({len(warnings)})"):
        for w in warnings:
            st.text(w)

st.info(
    "**Caveats.** Univariate IC ignores that metrics are correlated with each other "
    "(ROE ↔ margins); it does not tell you a metric's *marginal* contribution. "
    "The universe is survivorship-biased — only names in the current SEC ticker map and "
    "in Yahoo are measured — which flatters value and quality signals."
)
