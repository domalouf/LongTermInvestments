"""Undervalued today — the widest intrinsic-value-vs-price gaps among steady earners, and how much to trust each."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lti.config as config
from lti import prices as prices_mod, stock as stock_mod
from lti.app import theme
from lti.history import HISTORY_YEARS
from lti.valuation import MIN_MODELS, MODELS, ValuationAssumptions, rank_undervalued

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
    "Six intrinsic-value models run on each company's <b>normalized</b> earnings — the median "
    f"of its last {HISTORY_YEARS} years, not the latest one alone — ranked by the gap between "
    "blended fair value and the traded price. Only steady earners qualify: profitable now and "
    f"in most of the last {HISTORY_YEARS} years.",
    "A list of candidates to research, not a buy list: backtested (the <code>fair_value_upside</code> "
    "metric on the Backtest page), the widest gaps have trailed the average steady earner — "
    "usually the market knows why a stock is cheap. The models all start from the same earnings, "
    "so their agreement says little; what helps is how many years were profitable, how much of "
    "the profit turned into cash, and whether the latest year sits far from the norm.",
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
        min_profit_years=p["min_profit_years"],
        min_roe=p["min_roe"],
        exclude_financials=p["exclude_financials"],
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
    excl_fin = st.checkbox(
        "Exclude financials", value=True,
        help="Banks, insurers, REITs and business development companies. Graham, DDM and EPV "
             "all assume an operating business; a lender's balance sheet is its product.",
    )
    min_roe_pct = st.slider("Min ROE (%)", 0, 30, 0, help="0 disables the filter.")

    st.header("Consistency")
    min_profit_years = st.slider(
        f"Profitable in at least … of the last {HISTORY_YEARS} years", 0, HISTORY_YEARS, 4,
        help="A year counts only if EPS, net income and operating income were all positive — "
             "a one-off gain on an operating loss doesn't make a profitable year.",
    )
    require_pos_eps = st.checkbox(
        "Profitable now", value=True,
        help="Positive EPS in the latest year as well as on the normalized basis. Off, the list "
             "admits companies valued on better years they may not get back.",
    )
    top_n = st.slider("Show top N", 10, 100, 40)

    with st.expander("Valuation assumptions"):
        disc = st.slider("Discount rate", 0.05, 0.15, 0.09, 0.005, format="%.3f")
        term = st.slider("Terminal growth", 0.0, 0.04, 0.025, 0.005, format="%.3f")
        gcap = st.slider("Max growth", 0.05, 0.30, 0.15, 0.01, format="%.2f")
        min_models = st.slider(
            "Models that must produce a value", 2, len(MODELS), MIN_MODELS,
            help="A technical minimum for the blend, not a vote of confidence: four of the six "
                 "models multiply the same normalized EPS.",
        )

ranked_all = _rank(
    json.dumps(
        {
            "asof": str(asof),
            "market_cap_min": cap_floor_m * 1e6,
            "min_models": min_models,
            "min_profit_years": min_profit_years,
            "require_positive_eps": require_pos_eps,
            "exclude_financials": excl_fin,
            "min_roe": (min_roe_pct / 100) if min_roe_pct else None,
            "discount_rate": disc,
            "terminal_growth": term,
            "growth_cap": gcap,
        }
    )
)

if ranked_all.empty:
    st.warning("No names pass these filters. Loosen the market-cap floor or the consistency requirements.")
    st.stop()

ranked = ranked_all.head(top_n).copy()

# --- headline ---------------------------------------------------------------

up = ranked["fair_value_est_upside"]
best = ranked.iloc[0]
steady = int((ranked["profit_years"] >= ranked["history_years"]).sum()) if "profit_years" in ranked else 0
# No `delta=` on these — an up-arrow would imply improvement where the second line
# is just context, so the qualifier goes in the label instead.
k = st.columns(4)
k[0].metric("Names found", f"{len(ranked_all):,}",
            help=f"Passing every filter. The table below shows the top {len(ranked)}.")
k[1].metric("Median upside", f"{up.median():+.0%}", help="Across the names shown.")
k[2].metric(f"Widest gap · {best['ticker']}", f"{best['fair_value_est_upside']:+.0%}",
            help=str(best["company"]))
k[3].metric(
    "Profitable every year", f"{steady} of {len(ranked)}",
    help=f"Names profitable in every one of their last {HISTORY_YEARS} years on file.",
)

# --- the list ---------------------------------------------------------------

st.header("The list")

base = ["rank", "ticker", "company", "sector", "price", "fair_value_est", "upside_pct"]
evidence = ["pe", "pe_norm", "dividend_yield", "profit_years", "eps_vs_norm", "revenue_cagr",
            "fcf_conversion", "debt_to_equity", "market_cap"]

table = ranked.copy()
# ProgressColumn formats the raw number, so hand it whole percentage points
table["upside_pct"] = table["fair_value_est_upside"] * 100
table = table[[c for c in base + evidence if c in table.columns]]
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
        "pe": st.column_config.NumberColumn("P/E", format="%.1f", help="On the latest year's EPS."),
        "pe_norm": st.column_config.NumberColumn(
            "P/E (norm.)", format="%.1f",
            help=f"On normalized EPS — the median of the last {HISTORY_YEARS} years.",
        ),
        "dividend_yield": st.column_config.NumberColumn(
            "Dividend yield", format="percent",
            help="The last twelve months' payments over today's price. A blank column means the "
                 "name pays nothing — cheap and paying is a different proposition from cheap alone.",
        ),
        "profit_years": st.column_config.NumberColumn(
            "Profitable yrs", format="%d",
            help=f"Of the last {HISTORY_YEARS} on file; every reported figure positive.",
        ),
        "eps_vs_norm": st.column_config.NumberColumn(
            "Latest vs norm", format="%.2f×",
            help="Latest EPS ÷ normalized EPS. Well above 1 is a peak or a one-off; "
                 "well below, a trough the fair value assumes the company climbs out of.",
        ),
        "revenue_cagr": st.column_config.NumberColumn(
            "Revenue growth", format="percent",
            help=f"Annual revenue trend over the last {HISTORY_YEARS} years — the models' growth input.",
        ),
        "fcf_conversion": st.column_config.NumberColumn(
            "Cash conversion", format="percent",
            help="Free cash flow ÷ net income, summed over the years: how much of the profit was cash.",
        ),
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
        customdata=np.stack([head["company"], head["price"], head["fair_value_est"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b> — %{customdata[0]}<br>"
            "price $%{customdata[1]:.2f} → fair value $%{customdata[2]:.2f}<br>"
            "%{x:+.0%} upside<extra></extra>"
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

if {"eps_vs_norm", "market_cap"} <= set(ranked.columns):
    st.header("Cheap for a reason?")
    theme.note(
        "The fair values assume earnings return to their five-year norm. Names <b>below the line</b> "
        "earn less than that today — the upside is a bet on recovery, and a business in lasting "
        "decline never collects it. Above the line the latest year beats the norm, so the "
        "normalized value is the cautious one. Bubble size is market cap."
    )
    sc = ranked.dropna(subset=["eps_vs_norm", "fair_value_est_upside"]).copy()
    sc = sc[sc["eps_vs_norm"] > 0]  # a log axis: a loss-making latest year can't sit on it
    sc["_size"] = sc["market_cap"].fillna(sc["market_cap"].median())

    fig2 = go.Figure(
        go.Scatter(
            x=sc["fair_value_est_upside"], y=sc["eps_vs_norm"], mode="markers",
            marker=dict(
                size=sc["_size"], sizemode="area",
                sizeref=2.0 * sc["_size"].max() / (42.0**2), sizemin=7,
                color=theme.BLUE, opacity=0.75,
                line=dict(width=2, color=theme.SURFACE),  # 2px surface ring on overlap
            ),
            customdata=np.stack([sc["ticker"], sc["company"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                "%{x:+.0%} upside · latest EPS %{y:.2f}× the norm<extra></extra>"
            ),
        )
    )
    fig2.add_hline(y=1.0, line_width=1, line_color=theme.AXIS, layer="below")
    fig2.add_annotation(
        x=1.0, xref="paper", y=0.0, yanchor="bottom", xanchor="right",
        text="latest year = norm", showarrow=False, font=dict(size=10.5, color=theme.MUTED),
    )
    theme.show(
        fig2, height=430,
        xaxis=dict(tickformat="+.0%", title="upside to blended fair value"),
        yaxis=dict(type="log", title="latest EPS ÷ normalized EPS"),
    )

# --- one company: the fair value and the earnings behind it -------------------

st.header("How the fair value is built")
theme.note(
    "Each model's fair value for one company against its price, and the earnings history the "
    "models are fed. Four of the six models multiply the same normalized EPS, so a tight "
    "cluster mostly restates that one number — the history chart is the better evidence."
)

pick = st.selectbox(
    "Company",
    ranked["ticker"].tolist(),
    format_func=lambda t: f"{t} — {ranked.loc[ranked['ticker'] == t, 'company'].iloc[0]}",
    label_visibility="collapsed",
)
row = ranked[ranked["ticker"] == pick].iloc[0]
cik = int(ranked.index[ranked["ticker"] == pick][0])
price = float(row["price"])

left, right = st.columns(2)
with left:
    st.subheader("Model values")
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

with right:
    st.subheader("Earnings history")
    annual = stock_mod.annual_fundamentals(_load_fund(), cik)
    annual = annual[annual["filed"] <= pd.Timestamp(asof)].tail(10)
    if annual.empty or "eps" not in annual.columns or not annual["eps"].notna().any():
        st.info("No EPS history on file.")
    else:
        splits = stock_mod.splits_for(prices_mod.load_splits(), str(row["ticker"]))
        eps_hist = annual["eps"] / stock_mod._split_divisor(annual["filed"], splits)
        in_window = np.arange(len(annual)) >= len(annual) - HISTORY_YEARS
        fig4 = go.Figure(
            go.Bar(
                x=annual["fiscal_year"], y=eps_hist,
                marker_color=[theme.BLUE if w else theme.GRID for w in in_window],
                hovertemplate="FY%{x}<br>EPS $%{y:,.2f} (today's share basis)<extra></extra>",
            )
        )
        fig4.update_traces(marker_line_width=0, marker_cornerradius=3)
        fig4.update_layout(bargap=0.3)
        theme.zero_line(fig4, axis="y")
        if pd.notna(row.get("eps_norm")):
            fig4.add_hline(y=float(row["eps_norm"]), line_width=1.5, line_color=theme.ORANGE)
            fig4.add_annotation(
                x=0.0, xref="paper", y=float(row["eps_norm"]), yanchor="bottom", xanchor="left",
                text=f"normalized ${row['eps_norm']:,.2f}", showarrow=False,
                font=dict(size=10.5, color=theme.ORANGE),
            )
        theme.show(fig4, height=max(300, 40 * len(vals) + 100) if vals else 320, legend=False,
                   yaxis=dict(tickprefix="$", title="EPS"), xaxis=dict(title="", dtick=1))
        theme.note(
            f"Blue: the {HISTORY_YEARS} years in the norm (grey ones are older). EPS restated onto "
            "today's share count, as filed at the time."
        )

d = st.columns(5)
d[0].metric("Price", f"${price:,.2f}")
d[1].metric("Blended fair value", f"${row['fair_value_est']:,.2f}")
d[2].metric("Upside", f"{row['fair_value_est_upside']:+.0%}")
d[3].metric("Profitable years", f"{int(row['profit_years'])} of {int(row['history_years'])}")
d[4].metric("Cash conversion", f"{row['fcf_conversion']:.0%}" if pd.notna(row.get("fcf_conversion")) else "—")

st.divider()
st.page_link("views/stock.py", label="Check a name's history on the Stock page", icon="🔬")

with st.expander("How this works, and where it misleads"):
    st.markdown(
        f"""
**Universe.** Filings known on the as-of date (`filed ≤ date`), one row per company,
with a ticker, positive revenue, not a commodity or crypto trust, and above the
market-cap floor. Each filing's EPS and share count are restated for every split since
it was filed, and the price is the split-adjusted close.

**Normalized earnings.** Each company's last {HISTORY_YEARS} fiscal years, as they were
filed by the as-of date: normalized EPS is the median of those years' EPS, normalized free
cash flow the median of theirs, and growth the revenue trend across them. A median, so one
year — a cyclical peak, a tax-asset release, a merger gain — can't move it.

**Fair value.** The median of the {len(MODELS)} models that produced a number: two-stage
DCF (on normalized free cash flow), Peter Lynch, Graham number, Graham revised and earnings
power (on normalized EPS), and dividend discount (on the current dividend). Upside is fair
value ÷ price − 1; gaps above +500% are dropped as data errors.

**Where this misleads.**
- A business in lasting decline: normalized earnings assume the last {HISTORY_YEARS}
  years are a fair guide, and the market may know better. Check *Latest vs norm*.
- Longer cycles than {HISTORY_YEARS} years — a whole commodity boom can fit inside the window.
- The universe is survivorship-biased: delisted companies have no ticker, so they are
  invisible here. See the Data health page.
- These are assumption-sensitive estimates, **not investment advice**.
"""
    )
