"""Undervalued today — the widest intrinsic-value-vs-price gaps, and how much to trust each."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lti.config as config
from lti import prices as prices_mod
from lti.app import theme
from lti.valuation import MODELS, ValuationAssumptions, rank_undervalued

MODEL_LABELS = {
    "dcf_value": "Two-stage DCF",
    "lynch_fair_value": "Peter Lynch",
    "graham_number": "Graham number",
    "graham_intrinsic": "Graham revised",
    "ddm_value": "Dividend discount",
    "epv_value": "Earnings power",
}

theme.header(
    "🎯 Undervalued today",
    "Six intrinsic-value models run across every company with a filing known on the "
    "as-of date, ranked by the gap between blended fair value and the traded price. "
    "The models disagree by design — the <b>models agreeing</b> column is how much weight "
    "the blend deserves.",
)


@st.cache_data(show_spinner=False)
def _load_fund() -> pd.DataFrame:
    from lti.fundamentals import load_fundamentals

    return load_fundamentals()


@st.cache_data(show_spinner="Valuing the universe…")
def _rank(key: str) -> pd.DataFrame:
    p = json.loads(key)
    return rank_undervalued(
        _load_fund(),
        prices_mod.load_price_data(),
        p["asof"],
        assumptions=ValuationAssumptions(
            discount_rate=p["discount_rate"],
            terminal_growth=p["terminal_growth"],
            growth_cap=p["growth_cap"],
        ),
        market_cap_min=p["market_cap_min"],
        require_positive_eps=p["require_positive_eps"],
        min_models=p["min_models"],
        min_roe=p["min_roe"],
        top_n=None,  # slice in the page so the sliders below don't re-run this
    )


try:
    _load_fund()
except FileNotFoundError:
    st.error("No fundamentals table. Run `lti build-fundamentals` first.")
    st.stop()
if not config.get_paths().close_parquet.exists():
    st.error("No split-adjusted price cache. Run `lti fetch-prices` first.")
    st.stop()

# --- filters ---------------------------------------------------------------

with st.sidebar:
    st.header("Universe")
    asof = st.date_input("As of", value=pd.Timestamp.today().date())
    cap_floor_m = st.number_input(
        "Min market cap ($M)", value=2000.0, step=500.0, min_value=0.0,
        help="Smaller companies dominate any value screen and are the least tradeable.",
    )
    require_pos_eps = st.checkbox("Require positive EPS", value=True)
    min_roe_pct = st.slider("Min ROE (%)", 0, 30, 0, help="0 disables the filter.")
    excl_fin = st.checkbox(
        "Exclude financials", value=True,
        help="Graham, DDM and EPV all assume an operating business; a bank's balance "
             "sheet is its product, so their fair values there are not meaningful.",
    )

    st.header("Confidence")
    min_models = st.slider(
        "Models that must agree", 2, len(MODELS), 3,
        help="A name is only ranked if at least this many models produced a number.",
    )
    top_n = st.slider("Show top N", 10, 100, 40)

    with st.expander("Valuation assumptions"):
        disc = st.slider("Discount rate", 0.05, 0.15, 0.09, 0.005, format="%.3f")
        term = st.slider("Terminal growth", 0.0, 0.04, 0.025, 0.005, format="%.3f")
        gcap = st.slider("Max growth", 0.05, 0.30, 0.15, 0.01, format="%.2f")

ranked_all = _rank(
    json.dumps(
        {
            "asof": str(asof),
            "market_cap_min": cap_floor_m * 1e6,
            "min_models": min_models,
            "require_positive_eps": require_pos_eps,
            "min_roe": (min_roe_pct / 100) if min_roe_pct else None,
            "discount_rate": disc,
            "terminal_growth": term,
            "growth_cap": gcap,
        }
    )
)

if ranked_all.empty:
    st.warning("No names pass these filters. Loosen the market-cap floor or the model count.")
    st.stop()

if excl_fin and "is_financial" in ranked_all.columns:
    ranked_all = ranked_all[~ranked_all["is_financial"].fillna(False).astype(bool)]
if ranked_all.empty:
    st.warning("Every name was excluded. Turn off the financials filter or loosen the others.")
    st.stop()

ranked_all = ranked_all.drop(columns=["rank"], errors="ignore")
ranked_all.insert(0, "rank", np.arange(1, len(ranked_all) + 1))
ranked = ranked_all.head(top_n).copy()

# --- headline ---------------------------------------------------------------

up = ranked["fair_value_est_upside"]
best = ranked.iloc[0]
# No `delta=` on these — an up-arrow would imply improvement where the second line
# is just context, so the qualifier goes in the label instead.
k = st.columns(4)
k[0].metric("Names found", f"{len(ranked_all):,}",
            help=f"Passing every filter. The table below shows the top {len(ranked)}.")
k[1].metric("Median upside", f"{up.median():+.0%}", help="Across the names shown.")
k[2].metric(f"Widest gap · {best['ticker']}", f"{best['fair_value_est_upside']:+.0%}",
            help=str(best["company"]))
k[3].metric(
    f"All {len(MODELS)} models agree",
    f"{int((ranked['n_models'] == len(MODELS)).sum())} of {len(ranked)}",
    help="Names where every model produced a fair value — the blend is most trustworthy here.",
)

# --- the list ---------------------------------------------------------------

st.header("The list")

base = ["rank", "ticker", "company", "sector", "price", "fair_value_est",
        "upside_pct", "n_models"]
quality = [c for c in ["pe", "roe", "net_margin", "debt_to_equity", "market_cap"] if c in ranked.columns]

table = ranked.copy()
# ProgressColumn formats the raw number, so hand it whole percentage points
table["upside_pct"] = table["fair_value_est_upside"] * 100
table = table[[c for c in base + quality if c in table.columns]]
for c in table.columns:  # nullable dtypes render as the literal "None"
    if table[c].dtype == object or str(table[c].dtype) == "string":
        table[c] = table[c].astype(object).where(table[c].notna(), "—")
    elif pd.api.types.is_numeric_dtype(table[c]):
        table[c] = table[c].astype("float64")

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    height=min(620, 36 * len(table) + 40),
    column_config={
        "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "company": st.column_config.TextColumn("Company", width="medium"),
        "sector": st.column_config.TextColumn("Sector", width="medium"),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "fair_value_est": st.column_config.NumberColumn("Fair value", format="$%.2f"),
        "upside_pct": st.column_config.ProgressColumn(
            "Upside", format="%+.0f%%", min_value=0.0,
            max_value=float(max(up.max() * 100, 1.0)),
            help="Blended fair value ÷ price − 1. The bar is relative to the widest gap on screen.",
        ),
        "n_models": st.column_config.NumberColumn(
            "Models", format="%d", help=f"How many of the {len(MODELS)} models produced a number."
        ),
        "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
        "roe": st.column_config.NumberColumn("ROE", format="percent"),
        "net_margin": st.column_config.NumberColumn("Net margin", format="percent"),
        "debt_to_equity": st.column_config.NumberColumn("D/E", format="%.1f"),
        "market_cap": st.column_config.NumberColumn("Market cap", format="compact"),
    },
)

st.download_button(
    "Download this list as CSV",
    ranked.to_csv(index=False).encode(),
    file_name=f"undervalued_{asof}.csv",
    mime="text/csv",
)

# --- upside chart -----------------------------------------------------------

st.header("Widest gaps")
head = ranked.head(min(15, len(ranked))).iloc[::-1]
fig = go.Figure(
    go.Bar(
        x=head["fair_value_est_upside"], y=head["ticker"], orientation="h",
        text=[f"{v:+.0%}" for v in head["fair_value_est_upside"]],
        textposition="outside", cliponaxis=False,
        textfont=dict(color=theme.INK_2, size=11),
        customdata=np.stack([head["company"], head["n_models"], head["price"], head["fair_value_est"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b> — %{customdata[0]}<br>"
            "price $%{customdata[2]:.2f} → fair value $%{customdata[3]:.2f}<br>"
            "%{x:+.0%} upside · %{customdata[1]} models<extra></extra>"
        ),
    )
)
theme.bar_marks(fig)
theme.show(
    fig,
    height=max(300, 27 * len(head) + 70),
    xaxis=dict(tickformat="+.0%", title="upside to blended fair value",
               range=[0, float(head["fair_value_est_upside"].max()) * 1.16]),
    yaxis_title="",
)

# --- value trap check -------------------------------------------------------

if {"roe", "market_cap"} <= set(ranked.columns):
    st.header("Cheap for a reason?")
    theme.note(
        "A wide gap on a business that earns nothing is usually the market being right, not wrong. "
        "Names to the <b>lower right</b> are cheap and unprofitable — the classic value trap. "
        "Bubble size is market cap."
    )
    sc = ranked.dropna(subset=["roe", "fair_value_est_upside"]).copy()
    sc["_size"] = sc["market_cap"].fillna(sc["market_cap"].median()) if "market_cap" in sc else 1.0

    fig2 = go.Figure(
        go.Scatter(
            x=sc["fair_value_est_upside"], y=sc["roe"], mode="markers",
            marker=dict(
                size=sc["_size"], sizemode="area",
                sizeref=2.0 * sc["_size"].max() / (42.0**2), sizemin=7,
                color=theme.BLUE, opacity=0.75,
                line=dict(width=2, color=theme.SURFACE),  # 2px surface ring on overlap
            ),
            customdata=np.stack([sc["ticker"], sc["company"], sc["market_cap"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                "%{x:+.0%} upside · ROE %{y:.0%}<extra></extra>"
            ),
        )
    )
    theme.zero_line(fig2, axis="y")
    fig2.add_hline(y=0.15, line_width=1, line_color=theme.AXIS, layer="below")
    fig2.add_annotation(
        x=1.0, xref="paper", y=0.15, yanchor="bottom", xanchor="right",
        text="ROE 15%", showarrow=False, font=dict(size=10.5, color=theme.MUTED),
    )
    theme.show(
        fig2, height=430,
        xaxis=dict(tickformat="+.0%", title="upside to blended fair value"),
        yaxis=dict(tickformat=".0%", title="return on equity"),
    )

# --- per-name model spread --------------------------------------------------

st.header("Do the models agree?")
theme.note(
    "Every model's fair value for one company, against what it trades at now. "
    "A tight cluster well above the price is the case worth trusting; one model far out on "
    "its own is usually a growth or dividend assumption doing the work."
)

pick = st.selectbox(
    "Company",
    ranked["ticker"].tolist(),
    format_func=lambda t: f"{t} — {ranked.loc[ranked['ticker'] == t, 'company'].iloc[0]}",
    label_visibility="collapsed",
)
row = ranked[ranked["ticker"] == pick].iloc[0]
price = float(row["price"])

vals = [(MODEL_LABELS.get(m, m), float(row[m])) for m in MODELS if m in row.index and pd.notna(row[m])]
vals.sort(key=lambda t: t[1])

if not vals:
    st.info("No model produced a fair value for this company.")
else:
    labels = [v[0] for v in vals]
    amounts = [v[1] for v in vals]
    fig3 = go.Figure(
        go.Bar(
            x=amounts, y=labels, orientation="h",
            text=[f"${v:,.2f}" for v in amounts],
            textposition="outside", cliponaxis=False,
            textfont=dict(color=theme.INK_2, size=11),
            marker_color=[theme.POS if v >= price else theme.NEG for v in amounts],
            hovertemplate="%{y}<br>fair value $%{x:,.2f}<extra></extra>",
        )
    )
    fig3.update_traces(marker_line_width=0, marker_cornerradius=4)
    fig3.update_layout(bargap=0.42)
    fig3.add_vline(x=price, line_width=1.5, line_color=theme.INK_2)
    fig3.add_annotation(
        x=price, y=1.0, yref="paper", yanchor="bottom", xanchor="left",
        text=f"  price ${price:,.2f}", showarrow=False,
        font=dict(size=11, color=theme.INK_2),
    )
    theme.show(
        fig3, height=max(300, 40 * len(vals) + 100), legend=False,
        xaxis=dict(tickprefix="$", title="fair value per share", rangemode="tozero",
                   range=[0, max(max(amounts), price) * 1.22]),
        yaxis_title="",
    )

    d = st.columns(4)
    d[0].metric("Price", f"${price:,.2f}")
    d[1].metric("Blended fair value", f"${row['fair_value_est']:,.2f}")
    d[2].metric("Upside", f"{row['fair_value_est_upside']:+.0%}")
    d[3].metric("Models agreeing", f"{int(row['n_models'])} of {len(MODELS)}")

st.divider()
st.page_link("views/stock.py", label="Check a name's history on the Stock page", icon="🔬")

with st.expander("How this works, and where it misleads"):
    st.markdown(
        f"""
**Universe.** Filings known on the as-of date (`filed ≤ date`), one row per company,
with a ticker, positive revenue, not a commodity or crypto trust, and above the
market-cap floor. Each filing's EPS and share count are restated for every split since
it was filed, and the price is the split-adjusted close — so a company that split 10:1
after its last 10-K is valued on its real earnings per share, not ten times them.

**Fair value.** The median of the {len(MODELS)} models that produced a number: two-stage
DCF, Peter Lynch, Graham number, Graham revised, dividend discount and earnings power.
Upside is fair value ÷ price − 1. Gaps above +500% are dropped as data errors.

**Where this misleads.**
- One year's earnings can be a cyclical peak — drillers, homebuilders, shipping,
  egg producers — which makes the stock look far cheaper than its through-cycle
  earnings power. Every model here extrapolates from a single 10-K.
- Growth defaults to a one-year figure clipped to the *max growth* assumption. That is
  crude; a multi-year CAGR is the honest input.
- The universe is survivorship-biased: delisted companies have no ticker, so they are
  invisible here. See the Data health page.
- These are assumption-sensitive estimates, **not investment advice**.
"""
    )
