"""Stock page — one company's price and annual fundamentals over time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti import prices as prices_mod, stock as stock_mod
from lti.app import theme

theme.header(
    "🔬 Stock detail",
    "One company's price, annual fundamentals and fair-value estimates. This is the page "
    "to open before acting on anything the Undervalued screen turned up.",
)


@st.cache_data(show_spinner=False)
def _load_fund() -> pd.DataFrame:
    from lti.fundamentals import load_fundamentals

    return load_fundamentals()


try:
    fund = _load_fund()
except FileNotFoundError:
    st.error("No fundamentals table. Run `lti build-fundamentals` first.")
    st.stop()

px = prices_mod.load_price_data()
panel = px.adj  # total return: the price chart

with st.sidebar:
    st.header("Company")
    symbol = st.text_input("Ticker", value="MSFT").upper().strip()
    log_price = st.toggle("Log price axis", value=True)
    show_filings = st.toggle("Mark 10-K filing dates", value=True)
    adjust_splits = st.toggle("Split-adjust EPS / book value", value=True)

if not symbol:
    st.info("Enter a ticker in the sidebar.")
    st.stop()

cik, sym = stock_mod.resolve(fund, symbol)
if cik is None:
    st.error(f"'{symbol}' not found in the fundamentals table (10-K filers only).")
    st.stop()

annual = stock_mod.annual_fundamentals(fund, cik)
name = stock_mod.company_name(fund, cik) or symbol
psym = stock_mod.price_symbol(fund, cik, panel, sym)
splits = stock_mod.splits_for(px.splits, psym) if (adjust_splits and psym) else pd.Series(dtype="float64")
paid = stock_mod.dividends_for(px.dividends, psym) if psym else pd.Series(dtype="float64")
# valuation runs on the split-adjusted close: the adjusted one sits below the traded
# price by every dividend since, which would drag the early multiples down
has_close = bool(psym) and psym in px.close.columns

# The last twelve months of dividends, and the same window five years back — read
# off the latest close, so the tab below and the fair-value models agree.
last_close = px.close[psym].dropna() if has_close else pd.Series(dtype="float64")
div_asof = last_close.index.max() if len(last_close) else (paid.index.max() if not paid.empty else None)
ttm_dps = dps_5y_ago = 0.0
if div_asof is not None and not paid.empty:
    ttm_dps = float(paid[paid.index > div_asof - pd.DateOffset(months=12)].sum())
    dps_5y_ago = float(
        paid[(paid.index > div_asof - pd.DateOffset(months=72))
             & (paid.index <= div_asof - pd.DateOffset(months=60))].sum()
    )
div_growth_5y = (ttm_dps / dps_5y_ago) ** 0.2 - 1 if ttm_dps > 0 and dps_5y_ago > 0 else None

st.subheader(f"{symbol} — {name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Annual filings", len(annual))
if not annual.empty:
    c2.metric("Fiscal years", f"{int(annual['fiscal_year'].min())}–{int(annual['fiscal_year'].max())}")
if psym:
    ps = panel[psym].dropna()
    c3.metric("Price history", f"{ps.index.min().date()} → {ps.index.max().date()}")
    if len(ps) > 1:
        c4.metric("Total price return", f"{ps.iloc[-1] / ps.iloc[0] - 1:.0%}")
else:
    st.caption("No price history cached for this ticker — run `lti fetch-prices`.")

if annual.empty and not psym:
    st.warning(f"No fundamentals and no price history for {symbol}.")
    st.stop()

price_tab, income_tab, margin_tab, bs_tab, cf_tab, div_tab, val_tab, fv_tab, raw_tab = st.tabs(
    ["Price", "Income", "Margins & returns", "Balance sheet", "Cash flow",
     "Dividends", "Valuation", "Fair value", "Raw data"]
)

with price_tab:
    if not psym:
        st.info("No price history cached for this ticker.")
    else:
        s = panel[psym].dropna()
        fig = go.Figure(
            go.Scatter(
                x=s.index, y=s.values, name="Adjusted close", mode="lines",
                line=dict(width=2, color=theme.BLUE),
                hovertemplate="$%{y:,.2f}<extra></extra>",
            )
        )
        if show_filings and not annual.empty:
            for d in annual["filed"].dropna():
                fig.add_vline(x=d, line_width=1, line_color=theme.AXIS, layer="below")
        theme.show(
            fig, height=460, hovermode="x unified", legend=False,
            yaxis=dict(type="log" if log_price else "linear", tickprefix="$",
                       title="adjusted close"),
            xaxis_title="",
        )
        theme.note(
            "Adjusted close, so splits and dividends are already in the line. "
            "The vertical hairlines mark 10-K filing dates — the moments the fundamentals "
            "on the other tabs actually became public."
        )


def _year_bar(df: pd.DataFrame, items: list[str], ylabel: str = "$B"):
    """Grouped annual bars, one categorical slot per line item."""
    present = [c for c in items if c in df.columns and df[c].notna().any()]
    if not present:
        st.info("No data for this view.")
        return
    plot = df[["fiscal_year", *present]].copy()
    fig = go.Figure()
    for i, col in enumerate(present):
        fig.add_trace(
            go.Bar(
                x=plot["fiscal_year"], y=plot[col] / 1e9, name=col.replace("_", " "),
                marker_color=theme.SERIES[i % len(theme.SERIES)], marker_line_width=0,
                marker_cornerradius=3,
                hovertemplate=f"{col.replace('_', ' ')}<br>FY%{{x}}: $%{{y:,.2f}}B<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    theme.show(fig, height=400, yaxis_title=ylabel, xaxis=dict(title="", dtick=1))


with income_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        st.subheader("Income statement")
        _year_bar(annual, stock_mod.INCOME_ITEMS)
        if "eps" in annual.columns and annual["eps"].notna().any():
            st.subheader("Reported EPS")
            fig = go.Figure(
                go.Bar(x=annual["fiscal_year"], y=annual["eps"],
                       hovertemplate="FY%{x}: $%{y:,.2f}<extra></extra>")
            )
            theme.bar_marks(fig)
            theme.zero_line(fig, axis="y")
            theme.show(fig, height=300, legend=False,
                       yaxis=dict(tickprefix="$", title="EPS as filed"),
                       xaxis=dict(title="", dtick=1))
            theme.note(
                "As filed, on the share count of the day — not restated for later splits. "
                "The Valuation and Fair value tabs do restate it."
            )

with margin_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        present = [m for m in stock_mod.MARGIN_METRICS if m in annual.columns and annual[m].notna().any()]
        if present:
            st.subheader("Margins & return on equity")
            fig = go.Figure()
            for i, col in enumerate(present):
                fig.add_trace(
                    go.Scatter(
                        x=annual["fiscal_year"], y=annual[col], name=col.replace("_", " "),
                        mode="lines+markers", line=dict(width=2, color=theme.SERIES[i % len(theme.SERIES)]),
                        marker=dict(size=8, line=dict(width=2, color=theme.SURFACE)),
                        hovertemplate=f"{col.replace('_', ' ')}<br>FY%{{x}}: %{{y:.1%}}<extra></extra>",
                    )
                )
            theme.zero_line(fig, axis="y")
            theme.show(fig, height=420, hovermode="x unified",
                       yaxis=dict(tickformat=".0%", title=""), xaxis=dict(title="", dtick=1))
            theme.note(
                "Gross margin is unreliable here: the SEC standardizer sets it equal to revenue "
                "whenever it can't find a cost-of-revenue line. A flat 100% means missing data, "
                "not a perfect business."
            )
        else:
            st.info("No margin metrics available.")

with bs_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        st.subheader("Balance sheet")
        _year_bar(annual, stock_mod.BALANCE_ITEMS)
        if "debt_to_equity" in annual.columns and annual["debt_to_equity"].notna().any():
            st.subheader("Liabilities ÷ equity")
            fig = go.Figure(
                go.Scatter(
                    x=annual["fiscal_year"], y=annual["debt_to_equity"], mode="lines+markers",
                    line=dict(width=2, color=theme.BLUE),
                    marker=dict(size=8, line=dict(width=2, color=theme.SURFACE)),
                    hovertemplate="FY%{x}: %{y:.2f}×<extra></extra>",
                )
            )
            theme.show(fig, height=300, legend=False,
                       yaxis=dict(ticksuffix="×", title=""), xaxis=dict(title="", dtick=1))
            theme.note("Total liabilities, not just interest-bearing debt — see the README.")

with cf_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        st.subheader("Cash flow")
        _year_bar(annual, stock_mod.CASHFLOW_ITEMS)
        theme.note(
            "<code>cfo</code> operating cash flow · <code>capex</code> capital expenditure "
            "(as reported) · <code>free_cash_flow</code> = cfo − |capex|."
        )

with div_tab:
    if not psym:
        st.info("No price history cached for this ticker, so no dividend history either.")
    elif paid.empty:
        since = last_close.index.min().date() if len(last_close) else "the cache begins"
        st.info(
            f"{symbol} has paid no dividend since {since}. Everything it earns it keeps, "
            "reinvests or spends on buybacks."
        )
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Paid last 12 months", f"${ttm_dps:,.2f}", help="Per share, on today's share count.")
        if len(last_close):
            d2.metric("Yield", f"{ttm_dps / float(last_close.iloc[-1]):.2%}",
                      help="Last twelve months' dividends over the latest split-adjusted close.")
        if div_growth_5y is not None:
            d3.metric("5-year growth", f"{div_growth_5y:+.1%}",
                      help="CAGR of the trailing-twelve-month payment.")
        d4.metric("Payments on record", f"{len(paid):,}", help=f"Since {paid.index.min().date()}.")

        st.subheader("Dividends per share, by calendar year")
        by_year = stock_mod.dividends_by_year(paid)
        fig = go.Figure(
            go.Bar(
                x=by_year["year"], y=by_year["dividends"],
                hovertemplate="%{x}: $%{y:,.2f} per share<extra></extra>",
            )
        )
        theme.bar_marks(fig, theme.BLUE)
        theme.show(fig, height=330, legend=False,
                   yaxis=dict(tickprefix="$", title="per share", rangemode="tozero"),
                   xaxis=dict(title="", dtick=1))
        theme.note(
            "What a holder actually received, restated onto today's share count — so a payment "
            "from before a 4:1 split shows as a quarter of the cheque that arrived. The current "
            "year is part-way through. The price chart already contains all of this: the adjusted "
            "close is what these dividends compound to if they were reinvested."
        )

with val_tab:
    val = stock_mod.valuation_history(annual, px.close, psym, splits=splits) if has_close else pd.DataFrame()
    if val.empty:
        st.info(
            "Valuation history needs fundamentals with EPS / book value and a split-adjusted "
            "price series — run `lti fetch-prices` if this ticker hasn't been backfilled."
        )
    else:
        if not adjust_splits:
            theme.note(
                "Split adjustment is off: if the company split, the P/E and P/B below jump at "
                "the split date and the level before it is wrong."
            )
        for col, label in [("pe", "Price / earnings"), ("pb", "Price / book")]:
            if col not in val.columns or not val[col].notna().any():
                continue
            v = val.dropna(subset=[col])
            med = v[col].median()
            st.subheader(label)
            fig = go.Figure(
                go.Scatter(
                    x=v["date"], y=v[col], name=label, mode="lines",
                    line=dict(width=2, color=theme.BLUE),
                    hovertemplate="%{y:.1f}×<extra></extra>",
                )
            )
            fig.add_hline(y=med, line_width=1.5, line_color=theme.MUTED)
            fig.add_annotation(
                x=1.0, xref="paper", y=med, yanchor="bottom", xanchor="right",
                text=f"median {med:.1f}×", showarrow=False,
                font=dict(size=10.5, color=theme.MUTED),
            )
            theme.show(fig, height=330, legend=False, hovermode="x unified",
                       yaxis=dict(ticksuffix="×", title=""), xaxis_title="")
        theme.note(
            "Trailing multiple: price on each date ÷ the EPS or book value from the most recent "
            "10-K as of that date, restated onto today's share count. Stretches of negative "
            "earnings are dropped from P/E rather than plotted as meaningless negatives."
        )

with fv_tab:
    from lti.valuation import MODELS, ValuationAssumptions, add_valuation_models, historical_cagr

    if annual.empty or not psym:
        st.info("Fair-value models need annual fundamentals and a cached price.")
    else:
        latest = annual.iloc[[-1]].copy()
        latest.index = pd.Index([cik], name="cik")
        cur_price = float((px.close if has_close else panel)[psym].dropna().iloc[-1])

        # restate the latest per-share figures onto today's share count
        div = 1.0
        if not splits.empty:
            fdate = pd.Timestamp(latest["filed"].iloc[0])
            future = splits[splits.index > fdate]
            div = float(future.prod()) if len(future) else 1.0
        for c in ("eps", "book_value_per_share"):
            if c in latest.columns:
                latest[c] = latest[c] / div
        if "shares_outstanding" in latest.columns:
            latest["shares_outstanding"] = latest["shares_outstanding"] * div

        # normalized earnings: the median of the last few years, each on today's share basis
        from lti.history import HISTORY_YEARS, summarize_history

        norm = summarize_history(annual.tail(HISTORY_YEARS), px.splits)
        for c in norm.columns:
            latest[c] = norm[c].iloc[0] if len(norm) else np.nan
        shares_now = latest["shares_outstanding"] if "shares_outstanding" in latest.columns else np.nan
        latest["fcf_ps_norm"] = latest["fcf_norm"] / shares_now

        # the models prefer real payments to the cash-flow tag; hand them the
        # same trailing-twelve-month figure a snapshot would carry
        if not paid.empty:
            latest["dps_ttm"] = ttm_dps
            if div_growth_5y is not None:
                latest["dividend_growth_5y"] = div_growth_5y

        cagr_eps = historical_cagr(annual, "eps", 5)
        cagr_rev = historical_cagr(annual, "revenues", 5)

        with st.expander("Assumptions", expanded=True):
            eps_latest, eps_norm = latest["eps"].iloc[0], latest["eps_norm"].iloc[0]
            basis_label = st.radio(
                "Earnings basis",
                [f"Normalized — median of {HISTORY_YEARS} years (${eps_norm:,.2f})" if pd.notna(eps_norm)
                 else f"Normalized — median of {HISTORY_YEARS} years (n/a)",
                 f"Latest year (${eps_latest:,.2f})" if pd.notna(eps_latest) else "Latest year (n/a)"],
                horizontal=True,
                help="One year's earnings can be a peak, a trough or a one-off; the models "
                     "multiply whichever you pick. EPS restated onto today's share count.",
            )
            basis = "normalized" if basis_label.startswith("Normalized") else "latest"
            a1, a2, a3 = st.columns(3)
            disc = a1.slider("Discount rate", 0.05, 0.15, 0.09, 0.005, format="%.3f")
            term = a2.slider("Terminal growth", 0.0, 0.04, 0.025, 0.005, format="%.3f")
            years = a3.slider("DCF window (years)", 5, 15, 10)

            opts = []
            if pd.notna(cagr_rev):
                opts.append((f"Revenue CAGR 5y ({cagr_rev:.1%})", float(cagr_rev)))
            if pd.notna(cagr_eps):
                opts.append((f"EPS CAGR 5y ({cagr_eps:.1%})", float(cagr_eps)))
            opts.append(("Custom", None))
            labels = [o[0] for o in opts]
            pick = st.radio("Growth rate", labels, horizontal=True)
            g_val = dict(opts)[pick]
            if g_val is None:
                g_val = st.slider("Custom growth", 0.0, 0.30, 0.08, 0.01, format="%.2f")
            st.caption(
                "Growth feeds the DCF, Lynch and Graham-revised models (each caps it further). "
                "5-year CAGR needs positive EPS / revenue at both ends."
            )

        assumptions = ValuationAssumptions(
            discount_rate=disc, terminal_growth=term, dcf_years=years, fixed_growth=g_val
        )
        v = add_valuation_models(latest, pd.Series({cik: cur_price}), assumptions=assumptions, basis=basis)
        row = v.iloc[0]

        present = [m for m in MODELS if m in v.columns and pd.notna(row[m])]
        if not present:
            st.warning("No model could produce a value (needs positive EPS, FCF or dividends).")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Current price", f"${cur_price:,.2f}")
            m2.metric("Blended fair value", f"${row['fair_value_est']:,.2f}")
            m3.metric("Upside to fair value", f"{row['fair_value_est_upside']:.0%}")

            st.subheader("Estimated fair value per share")
            names = present + ["blended"]
            amounts = [float(row[m]) for m in present] + [float(row["fair_value_est"])]
            order = sorted(range(len(names)), key=lambda i: amounts[i])
            names = [names[i].replace("_", " ") for i in order]
            amounts = [amounts[i] for i in order]

            fig = go.Figure(
                go.Bar(
                    x=amounts, y=names, orientation="h",
                    text=[f"${v:,.2f}" for v in amounts],
                    textposition="outside", cliponaxis=False,
                    textfont=dict(color=theme.INK_2, size=11),
                    marker_color=[theme.POS if v >= cur_price else theme.NEG for v in amounts],
                    hovertemplate="%{y}<br>$%{x:,.2f} per share<extra></extra>",
                )
            )
            fig.update_traces(marker_line_width=0, marker_cornerradius=4)
            fig.update_layout(bargap=0.4)
            fig.add_vline(x=cur_price, line_width=1.5, line_color=theme.INK_2)
            fig.add_annotation(
                x=cur_price, y=1.0, yref="paper", yanchor="bottom", xanchor="left",
                text=f"  price ${cur_price:,.2f}", showarrow=False,
                font=dict(size=11, color=theme.INK_2),
            )
            theme.show(
                fig, height=max(300, 40 * len(names) + 80), legend=False,
                xaxis=dict(tickprefix="$", title="fair value per share", rangemode="tozero",
                           range=[0, max(max(amounts), cur_price) * 1.22]),
                yaxis_title="",
            )
            theme.note(
                "Blue bars sit above today's price, red below. The spread between models "
                "<i>is</i> the uncertainty — a tight cluster is worth more than a high median."
            )

            table = pd.DataFrame(
                {
                    "model": present,
                    "fair_value": [row[m] for m in present],
                    "upside_vs_price": [row[f"{m}_upside"] for m in present],
                }
            )
            st.dataframe(
                table.style.format({"fair_value": "${:,.2f}", "upside_vs_price": "{:+.0%}"}),
                hide_index=True, width="stretch",
            )
            g_used = row.get("est_growth")
            dy_used = row.get("dividend_yield")
            bits = [f"{'normalized' if basis == 'normalized' else 'latest-year'} earnings",
                    f"growth **{g_used:.1%}**" if pd.notna(g_used) else None,
                    f"dividend yield **{dy_used:.1%}**" if dy_used is not None and pd.notna(dy_used) else None,
                    f"discount rate **{disc:.1%}**"]
            st.caption("Inputs: " + " · ".join(b for b in bits if b) + ". "
                       "DCF uses FCF/share; EPV capitalises EPS with no growth; Graham number "
                       "= √(22.5·EPS·BVPS); Lynch fair P/E = growth% + yield%; DDM is Gordon growth.")
            st.warning(
                "Rough estimates, sensitive to the assumptions above — normalized earnings assume "
                "the last few years are a fair guide to the next. Not investment advice."
            )

with raw_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        show = annual.drop(columns=[c for c in ["tickers_all", "adsh", "form"] if c in annual.columns])
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button(
            "Download annual fundamentals CSV",
            annual.to_csv(index=False).encode(),
            file_name=f"{symbol}_annual_fundamentals.csv",
            mime="text/csv",
        )
