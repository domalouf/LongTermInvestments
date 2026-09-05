"""Screener page — rank the point-in-time universe by chosen metrics."""

from __future__ import annotations

import lti.config as config

import pandas as pd
import plotly.express as px
import streamlit as st

from lti import metrics as metrics_mod, pit, prices as prices_mod, ranking
from lti.metrics import FUNDAMENTAL_METRICS, PRICE_METRICS

st.set_page_config(page_title="Screener", page_icon="🔎", layout="wide")
st.title("🔎 Screener")


@st.cache_data(show_spinner=False)
def _load_fund() -> pd.DataFrame:
    from lti.fundamentals import load_fundamentals

    return load_fundamentals()


try:
    fund = _load_fund()
except FileNotFoundError:
    st.error("No fundamentals table. Run `lti build-fundamentals` first.")
    st.stop()

panel = prices_mod.load_adj_close()

with st.sidebar:
    st.header("Screen")
    asof = st.date_input("As of", value=pd.Timestamp.today().date())
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    all_metrics = FUNDAMENTAL_METRICS + PRICE_METRICS
    chosen = st.multiselect("Rank by", all_metrics, default=["pe", "debt_to_equity"])
    top_n = st.slider("Top N", 5, 100, 10)
    require_pos_eps = st.checkbox("Require positive EPS", value=True)

if not chosen:
    st.info("Pick at least one metric in the sidebar.")
    st.stop()

asof_ts = pd.Timestamp(asof)
snap = pit.snapshot_asof(fund, asof_ts)
snap = snap[snap["ticker"].notna()]
snap = metrics_mod.add_fundamental_metrics(snap)

need_price = any(m in PRICE_METRICS for m in chosen) or cap_floor_m > 0
if need_price and not panel.empty:
    price_at = pd.Series(
        {cik: prices_mod.price_on_or_before(panel, t, asof_ts) for cik, t in snap["ticker"].items()}
    )
    shares = snap["shares_outstanding"] if "shares_outstanding" in snap.columns else None
    mcap = price_at * shares if shares is not None else None
    snap = metrics_mod.add_price_metrics(snap, price=price_at, market_cap=mcap)
elif need_price:
    st.warning("No price cache — price metrics and the market-cap filter are unavailable.")

spec = ranking.ScreenSpec(
    metrics=chosen,
    top_n=top_n,
    filters={
        "market_cap_min": cap_floor_m * 1e6 if "market_cap" in snap.columns else None,
        "require_positive_eps": require_pos_eps,
    },
)

try:
    ranked = ranking.rank(snap, spec)
except KeyError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"**{len(ranked)}** companies pass filters · showing top **{min(top_n, len(ranked))}**")

display_cols = [
    c
    for c in ["rank", "ticker", "company", "fiscal_year", "period_end", "filed", "price", "market_cap",
              *chosen, "composite_score"]
    if c in ranked.columns
]
st.dataframe(ranked[display_cols].head(top_n), hide_index=True, use_container_width=True)

st.download_button(
    "Download ranked CSV",
    ranked[display_cols].to_csv(index=False).encode(),
    file_name=f"screen_{asof}.csv",
    mime="text/csv",
)

st.subheader("Metric distributions")
for m in chosen:
    if m in ranked.columns:
        vals = ranked[m].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if not vals.empty:
            q_lo, q_hi = vals.quantile(0.02), vals.quantile(0.98)
            fig = px.histogram(vals[(vals >= q_lo) & (vals <= q_hi)], nbins=40, title=m)
            fig.update_layout(showlegend=False, height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
