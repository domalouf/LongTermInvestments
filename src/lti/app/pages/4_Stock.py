"""Stock page — one company's price and annual fundamentals over time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lti import prices as prices_mod, stock as stock_mod

st.set_page_config(page_title="Stock", page_icon="🔬", layout="wide")
st.title("🔬 Stock detail")


@st.cache_data(show_spinner=False)
def _load_fund() -> pd.DataFrame:
    from lti.fundamentals import load_fundamentals

    return load_fundamentals()


@st.cache_data(show_spinner=False)
def _splits(symbol: str) -> pd.Series:
    """Split history for one ticker from yfinance (cached; best-effort)."""
    try:
        import yfinance as yf

        s = yf.Ticker(symbol).splits
        if s is None or len(s) == 0:
            return pd.Series(dtype="float64")
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s[s > 0]
    except Exception:  # noqa: BLE001 - offline / delisted / rate-limited
        return pd.Series(dtype="float64")


try:
    fund = _load_fund()
except FileNotFoundError:
    st.error("No fundamentals table. Run `lti build-fundamentals` first.")
    st.stop()

panel = prices_mod.load_adj_close()

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
splits = _splits(psym) if (adjust_splits and psym) else pd.Series(dtype="float64")

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

price_tab, income_tab, margin_tab, bs_tab, cf_tab, val_tab, raw_tab = st.tabs(
    ["Price", "Income", "Margins & returns", "Balance sheet", "Cash flow", "Valuation", "Raw data"]
)

with price_tab:
    if not psym:
        st.info("No price history cached for this ticker.")
    else:
        s = panel[psym].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Adj. close", line=dict(width=1.5)))
        if show_filings and not annual.empty:
            for d in annual["filed"].dropna():
                fig.add_vline(x=d, line_width=1, line_dash="dot", line_color="rgba(128,128,128,0.4)")
        fig.update_layout(
            height=460, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_type="log" if log_price else "linear", yaxis_title="Adjusted close ($)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Adjusted close (splits + dividends). Dotted lines mark 10-K filing dates.")


def _year_bar(df: pd.DataFrame, items: list[str], title: str):
    present = [c for c in items if c in df.columns and df[c].notna().any()]
    if not present:
        st.info("No data for this view.")
        return
    plot = df[["fiscal_year", *present]].copy()
    plot[present] = plot[present] / 1e9
    melted = plot.melt("fiscal_year", var_name="item", value_name="value")
    fig = px.bar(melted, x="fiscal_year", y="value", color="item", barmode="group", title=title)
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="$B", xaxis_title="", legend_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


with income_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        _year_bar(annual, stock_mod.INCOME_ITEMS, "Income statement ($B)")
        if "eps" in annual.columns and annual["eps"].notna().any():
            fig = px.bar(annual, x="fiscal_year", y="eps", title="Reported EPS ($, as filed)")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="$")
            st.plotly_chart(fig, use_container_width=True)

with margin_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        present = [m for m in stock_mod.MARGIN_METRICS if m in annual.columns and annual[m].notna().any()]
        if present:
            melted = annual[["fiscal_year", *present]].melt("fiscal_year", var_name="metric", value_name="value")
            fig = px.line(melted, x="fiscal_year", y="value", color="metric", markers=True,
                          title="Margins & return on equity")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                              yaxis_tickformat=".0%", xaxis_title="", legend_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No margin metrics available.")

with bs_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        _year_bar(annual, stock_mod.BALANCE_ITEMS, "Balance sheet ($B)")
        if "debt_to_equity" in annual.columns and annual["debt_to_equity"].notna().any():
            fig = px.line(annual, x="fiscal_year", y="debt_to_equity", markers=True,
                          title="Liabilities / equity")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="×")
            st.plotly_chart(fig, use_container_width=True)

with cf_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        _year_bar(annual, stock_mod.CASHFLOW_ITEMS, "Cash flow ($B)")
        st.caption("`cfo` operating cash flow · `capex` capital expenditure (as reported) · "
                   "`free_cash_flow` = cfo − |capex|.")

with val_tab:
    val = stock_mod.valuation_history(annual, panel, psym or "", splits=splits) if psym else pd.DataFrame()
    if val.empty:
        st.info("Valuation history needs both cached prices and fundamentals with EPS / book value.")
    else:
        if splits.empty and psym:
            st.caption("No split history loaded — P/E and P/B are distorted across any stock split.")
        for col, label in [("pe", "Price / earnings"), ("pb", "Price / book")]:
            if col not in val.columns or not val[col].notna().any():
                continue
            v = val.dropna(subset=[col])
            med = v[col].median()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v["date"], y=v[col], name=label, line=dict(width=1.5)))
            fig.add_hline(y=med, line_dash="dash", annotation_text=f"median {med:.1f}")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=label, yaxis_title="×")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Trailing multiple: price on each date ÷ the EPS / book value from the most "
                   "recent 10-K as of that date, restated to today's share count. Negative-earnings "
                   "stretches are dropped from P/E.")

with raw_tab:
    if annual.empty:
        st.info("No fundamentals for this ticker.")
    else:
        show = annual.drop(columns=[c for c in ["tickers_all", "adsh", "form"] if c in annual.columns])
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.download_button(
            "Download annual fundamentals CSV",
            annual.to_csv(index=False).encode(),
            file_name=f"{symbol}_annual_fundamentals.csv",
            mime="text/csv",
        )
