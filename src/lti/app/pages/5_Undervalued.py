"""Undervalued dashboard — the widest intrinsic-value-vs-price gaps for a day."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from lti import prices as prices_mod
from lti.valuation import MODELS, ValuationAssumptions, rank_undervalued

st.set_page_config(page_title="Undervalued", page_icon="🎯", layout="wide")
st.title("🎯 Undervalued today")
st.caption(
    "Every model from the Fair-value tab, run across the whole point-in-time universe, "
    "ranked by the gap between blended intrinsic value and the current price. Valuing as "
    "of today against the latest 10-K keeps the split-adjustment problem out of the way."
)


@st.cache_data(show_spinner=False)
def _load_fund() -> pd.DataFrame:
    from lti.fundamentals import load_fundamentals

    return load_fundamentals()


@st.cache_data(show_spinner=True)
def _rank(key: str) -> pd.DataFrame:
    p = json.loads(key)
    a = ValuationAssumptions(
        discount_rate=p["discount_rate"],
        terminal_growth=p["terminal_growth"],
        growth_cap=p["growth_cap"],
    )
    return rank_undervalued(
        _load_fund(),
        prices_mod.load_adj_close(),
        p["asof"],
        assumptions=a,
        market_cap_min=p["market_cap_min"],
        require_positive_eps=p["require_positive_eps"],
        min_models=p["min_models"],
        min_roe=p["min_roe"],
        top_n=p["top_n"],
    )


try:
    _load_fund()
except FileNotFoundError:
    st.error("No fundamentals table. Run `lti build-fundamentals` first.")
    st.stop()

if prices_mod.load_adj_close().empty:
    st.error("No price cache. Run `lti fetch-prices` first.")
    st.stop()

with st.sidebar:
    st.header("Screen")
    asof = st.date_input("As of", value=pd.Timestamp.today().date())
    cap_floor_m = st.number_input("Min market cap ($M)", value=2000.0, step=500.0, min_value=0.0)
    min_models = st.slider("Models that must agree", 2, len(MODELS), 3)
    require_pos_eps = st.checkbox("Require positive EPS", value=True)
    min_roe_pct = st.slider("Min ROE (%) — 0 disables", 0, 30, 0)
    top_n = st.slider("Show top N", 10, 100, 40)
    st.divider()
    st.subheader("Assumptions")
    disc = st.slider("Discount rate", 0.05, 0.15, 0.09, 0.005, format="%.3f")
    term = st.slider("Terminal growth", 0.0, 0.04, 0.025, 0.005, format="%.3f")
    gcap = st.slider("Max growth", 0.05, 0.30, 0.15, 0.01, format="%.2f")

key = json.dumps(
    {
        "asof": str(asof),
        "market_cap_min": cap_floor_m * 1e6,
        "min_models": min_models,
        "require_positive_eps": require_pos_eps,
        "min_roe": (min_roe_pct / 100) if min_roe_pct else None,
        "top_n": top_n,
        "discount_rate": disc,
        "terminal_growth": term,
        "growth_cap": gcap,
    }
)

ranked = _rank(key)
if ranked.empty:
    st.warning("No names pass the filters for this date. Loosen the market-cap floor or model count.")
    st.stop()

up = ranked["fair_value_est_upside"]
c1, c2, c3 = st.columns(3)
c1.metric("Names shown", len(ranked))
c2.metric("Median upside", f"{up.median():.0%}")
if "roe" in ranked.columns:
    c3.metric("With ROE > 15%", int((ranked["roe"] > 0.15).sum()))

upside_cols = [f"{m}_upside" for m in MODELS if f"{m}_upside" in ranked.columns]
base_cols = ["rank", "ticker", "company", "price", "fair_value_est", "fair_value_est_upside", "n_models"]
metric_cols = [c for c in ["pe", "pb", "roe", "net_margin", "debt_to_equity"] if c in ranked.columns]
table = ranked[[c for c in base_cols + upside_cols + metric_cols if c in ranked.columns]].copy()

fmt = {"price": "${:,.2f}", "fair_value_est": "${:,.2f}", "fair_value_est_upside": "{:+.0%}"}
fmt.update({c: "{:+.0%}" for c in upside_cols})
fmt.update({c: "{:.0%}" for c in ("roe", "net_margin") if c in table.columns})
fmt.update({c: "{:.1f}" for c in ("pe", "pb", "debt_to_equity") if c in table.columns})
st.dataframe(table.style.format(fmt, na_rep="—"), hide_index=True, use_container_width=True)
st.download_button(
    "Download CSV", table.to_csv(index=False).encode(), file_name=f"undervalued_{asof}.csv", mime="text/csv"
)

st.subheader("Upside to blended fair value")
head = ranked.head(min(20, len(ranked)))
fig = px.bar(
    head, x="fair_value_est_upside", y="ticker", orientation="h", text="fair_value_est_upside",
    hover_data=["company"], color="n_models", color_continuous_scale="Blues",
)
fig.update_traces(texttemplate="%{text:+.0%}", textposition="outside", cliponaxis=False)
fig.update_layout(
    height=max(320, 24 * len(head) + 90), margin=dict(l=10, r=10, t=20, b=10),
    xaxis_tickformat="+.0%", xaxis_title="", yaxis_title="",
    yaxis=dict(categoryorder="array", categoryarray=list(head["ticker"])[::-1]),
)
st.plotly_chart(fig, use_container_width=True)

if {"roe", "market_cap"} <= set(ranked.columns):
    st.subheader("Cheapness vs quality")
    st.caption("Top-left (high upside, low ROE) is where value traps cluster; top-right is the sweet spot.")
    sc = ranked.dropna(subset=["roe", "fair_value_est_upside"])
    fig2 = px.scatter(
        sc, x="fair_value_est_upside", y="roe", size="market_cap", hover_name="ticker",
        hover_data=["company", "pe"], size_max=40,
    )
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_tickformat="+.0%", yaxis_tickformat=".0%",
                       xaxis_title="upside to fair value", yaxis_title="ROE")
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("How this works / caveats"):
    st.markdown(
        """
- **Universe:** filings known on the as-of date (`filed ≤ date`), one row per company,
  with a ticker, positive revenue, above the market-cap floor.
- **Fair value:** the median of the DCF, Lynch, Graham (× 2), DDM and EPV models that
  produced a number (needs at least the *models that must agree* count). `*_upside`
  columns are each model's fair value ÷ price − 1.
- **Ranking:** by blended upside, descending; upsides above +500% are dropped as
  likely data errors.
- **Watch out for:** a single year's earnings can be inflated by one-off tax items or
  a cyclical peak (egg producers, drillers, homebuilders), which makes a stock look far
  cheaper than its through-cycle earnings power. Check the company on the Stock page.
  The universe is also survivorship-biased — see the README.
- Not investment advice.
"""
    )
