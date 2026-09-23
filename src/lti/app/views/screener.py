"""Screener page — rank the point-in-time universe by chosen metrics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti import pit, prices as prices_mod, ranking
from lti.app import theme, widgets
from lti.metrics import HISTORY_METRICS, LOWER_IS_BETTER, PRICE_METRICS, VALUATION_METRICS
from lti.sectors import DIVISION_SHORT

theme.header(
    "🔎 Screener",
    "Rank the universe as it stood on a chosen date — only filings actually filed by "
    "then are used, so the ranking is one you could have acted on.",
)


fund = widgets.fundamentals()
px = prices_mod.load_price_data()

with st.sidebar:
    st.header("Screen")
    asof = st.date_input("As of", value=pd.Timestamp.today().date())
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    magic = st.checkbox(
        "Greenblatt Magic Formula",
        value=False,
        help="Rank on EBIT/EV and return on capital, equally weighted, "
        "with financials and utilities excluded — the screen from "
        "*The Little Book that Beats the Market*.",
    )
    chosen, coverage = widgets.rank_by(["pe", "debt_to_equity"], magic=magic)
    top_n = st.slider("Top N", 5, 100, 30 if magic else 10)
    require_pos_eps = st.checkbox("Require positive EPS", value=True)

    st.caption("Exclusions")
    operating_only = st.checkbox(
        "Operating companies only",
        value=True,
        help="Drops commodity and crypto trusts (gold, silver, oil, bitcoin), shells and other "
             "filers with no revenue. Trusts report the mark-to-market on what they hold as "
             "earnings, which otherwise puts them at the top of any cheapness ranking.",
    )
    excl_fin = st.checkbox(
        "Exclude financials (SIC 6000-6799)",
        value=magic,
        help="Return on capital is meaningless for a bank: its balance sheet is its product.",
    )
    excl_util = st.checkbox(
        "Exclude utilities (SIC 4900-4999)",
        value=magic,
        help="Regulated utilities earn an allowed return on a rate base, so ranking "
        "them on capital efficiency measures the regulator.",
    )

if not chosen:
    st.info("Pick at least one metric in the sidebar.")
    st.stop()

asof_ts = pd.Timestamp(asof)
if px.close.empty:
    # fundamentals-only ranking still works; anything priced needs the backfill
    snap = pit.company_snapshot(fund, asof_ts, px.splits, operating_only=operating_only)
    if any(m in PRICE_METRICS + HISTORY_METRICS + VALUATION_METRICS for m in chosen) or cap_floor_m > 0:
        st.warning(
            "No split-adjusted price cache — price metrics and the market-cap filter are "
            "unavailable. Run `lti fetch-prices`."
        )
else:
    # with history: the fair-value table below runs on normalized earnings
    snap = pit.priced_snapshot(fund, asof_ts, px, operating_only=operating_only, with_history=True)

spec = ranking.ScreenSpec(
    metrics=chosen,
    top_n=top_n,
    min_coverage=coverage,
    filters={
        "market_cap_min": cap_floor_m * 1e6 if "market_cap" in snap.columns else None,
        "require_positive_eps": require_pos_eps,
        "exclude_financials": excl_fin,
        "exclude_utilities": excl_util,
    },
)

try:
    ranked = ranking.rank(snap, spec)
except KeyError as exc:
    st.error(str(exc))
    st.stop()

ranked_metrics = [m for m in chosen if m in ranked.columns]

s = st.columns([1, 1, 1, 1.7])  # the sector's name needs the room
s[0].metric("Pass the filters", f"{len(ranked):,}",
            help=f"The table below shows the top {min(top_n, len(ranked))}.")
s[1].metric("Ranked on", f"{len(ranked_metrics)} metric{'s' if len(ranked_metrics) != 1 else ''}")
if "market_cap" in ranked.columns and ranked["market_cap"].notna().any():
    s[2].metric("Median pick size", theme.money(ranked["market_cap"].head(top_n).median()))
if "sector" in ranked.columns:
    shown = ranked["sector"].head(top_n)
    top_sector = shown.mode()
    if len(top_sector):
        name = str(top_sector.iloc[0])
        s[3].metric("Most common sector", DIVISION_SHORT.get(name, name),
                    help=f"{name}: {int((shown == name).sum())} of the top {len(shown)}.")
    else:
        s[3].metric("Most common sector", "—")

# per-metric percentile rank (0% = best) so a multi-metric composite is legible
pct_cols: list[str] = []
if len(ranked_metrics) > 1:
    for m in ranked_metrics:
        col = f"{m}_pctile"
        ranked[col] = ranked[m].rank(pct=True, ascending=m in LOWER_IS_BETTER)
        pct_cols.append(col)

display_cols = [
    c
    for c in ["rank", "ticker", "company", "sector", "form", "fiscal_year", "filed", "price", "market_cap",
              *ranked_metrics, *pct_cols, "composite_score"]
    if c in ranked.columns
]
top = ranked[display_cols].head(top_n)

col_cfg = {
    "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
    "ticker": st.column_config.TextColumn("Ticker", width="small"),
    "company": st.column_config.TextColumn("Company", width="medium"),
    "sector": st.column_config.TextColumn("Sector", width="medium"),
    "form": st.column_config.TextColumn(
        "From", width="small",
        help="10-K: the fiscal year. 10-Q: the trailing twelve months to the latest quarter — see `lti build-quarterly`."),
    "fiscal_year": st.column_config.NumberColumn("FY", format="%d"),
    "filed": st.column_config.DateColumn("Filed", format="YYYY-MM-DD"),
    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
    "market_cap": st.column_config.NumberColumn("Market cap", format="compact"),
    "composite_score": st.column_config.NumberColumn(
        "Score", format="percent", help="Mean percentile rank across the chosen metrics; lower is better."
    ),
}
col_cfg.update({m: st.column_config.NumberColumn(m, format="%.2f") for m in ranked_metrics})
col_cfg.update({c: st.column_config.NumberColumn(c.replace("_pctile", " %ile"), format="percent")
                for c in pct_cols})
st.dataframe(top, hide_index=True, width="stretch", column_config=col_cfg)

st.download_button(
    "Download ranked CSV",
    ranked[display_cols].to_csv(index=False).encode(),
    file_name=f"screen_{asof}.csv",
    mime="text/csv",
)

st.header("Ranked metric values")
theme.note(
    "Each pick's actual value for every metric it was ranked on, against the universe median. "
    "A pick sitting near the median on one metric is being carried by the others."
)
for m in ranked_metrics:
    picks = top[["ticker", m]].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if picks.empty:
        continue
    median = ranked[m].replace([float("inf"), float("-inf")], pd.NA).dropna().median()
    picks = picks.iloc[::-1]  # rank 1 on top

    st.subheader(m)
    fig = go.Figure(
        go.Bar(
            x=picks[m], y=picks["ticker"], orientation="h",
            text=[f"{v:,.2f}" for v in picks[m]],
            textposition="outside", cliponaxis=False,
            textfont=dict(color=theme.INK_2, size=11),
            hovertemplate=f"<b>%{{y}}</b><br>{m} %{{x:,.3f}}<extra></extra>",
        )
    )
    theme.bar_marks(fig)
    if pd.notna(median):
        fig.add_vline(x=median, line_width=1, line_color=theme.MUTED)
        fig.add_annotation(
            x=median, y=1.0, yref="paper", yanchor="bottom", xanchor="left",
            text=f"  universe median {median:,.2f}", showarrow=False,
            font=dict(size=10.5, color=theme.MUTED),
        )
    theme.show(fig, height=max(250, 26 * len(picks) + 80), xaxis_title=m, yaxis_title="")

    with st.expander(f"Where the picks sit in the {m} distribution"):
        vals = ranked[m].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if not vals.empty:
            q_lo, q_hi = vals.quantile(0.02), vals.quantile(0.98)
            trimmed = vals[(vals >= q_lo) & (vals <= q_hi)]
            hist = go.Figure(
                go.Histogram(
                    x=trimmed, nbinsx=40,
                    hovertemplate=f"{m} %{{x}}<br>%{{y}} companies<extra></extra>",
                )
            )
            hist.update_traces(marker_color=theme.GRID, marker_line_width=0)
            for v in picks[m]:
                if q_lo <= v <= q_hi:
                    hist.add_vline(x=v, line_width=1.5, line_color=theme.BLUE, opacity=0.85)
            theme.show(
                hist, height=270, legend=False, bargap=0.04,
                xaxis_title=f"{m} (2nd–98th percentile)", yaxis_title="companies",
            )
            theme.note("Each blue line is one of the picks above; the grey histogram is the whole universe.")

st.header("Fair-value estimates")
if "price" not in ranked.columns or not ranked["price"].notna().any():
    st.caption("Needs the price cache — pick a price metric or set a market-cap floor so prices load.")
else:
    from lti.valuation import ValuationAssumptions, add_valuation_models

    fv1, fv2 = st.columns(2)
    rate_kw = widgets.rates(asof_ts, fv1)
    gcap = fv2.slider("Max growth", 0.05, 0.30, 0.15, 0.01, format="%.2f")
    picks_snap = ranked.head(top_n)
    basis = "normalized" if "eps_norm" in picks_snap.columns else "latest"
    v = add_valuation_models(
        picks_snap, picks_snap["price"],
        assumptions=ValuationAssumptions(**rate_kw, growth_cap=gcap),
        basis=basis,
    )
    from lti.valuation import MODELS

    cols = ["ticker", "price", "eps_norm", "profit_years", "est_growth", "fair_value_est", "fair_value_est_upside"]
    cols += [f"{m}_upside" for m in MODELS if f"{m}_upside" in v.columns]
    fv_table = v[[c for c in cols if c in v.columns]].copy()
    fmt = {"price": "${:,.2f}", "eps_norm": "${:,.2f}", "profit_years": "{:.0f}",
           "est_growth": "{:.0%}", "fair_value_est": "${:,.2f}"}
    fmt.update({c: "{:+.0%}" for c in fv_table.columns if c.endswith("_upside")})
    # a static table: the interactive grid draws a missing value as "None" whatever the
    # Styler says, and a model that produced no value is common enough to matter here
    st.table(fv_table.set_index("ticker").style.format(fmt, na_rep="—"))
    st.caption(
        "`*_upside` = model fair value ÷ price − 1. Blended `fair_value_est` is the median of the "
        "models that produced a number. The models run on normalized earnings — `eps_norm`, the "
        "median of the last five years' EPS — with growth from the five-year revenue trend, "
        "clipped to [0, max]; `profit_years` is how many of those years were profitable."
    )
