"""Portfolio — your own account: what it holds, what it cost, and the same money in SPY."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti import portfolio, prices as prices_mod
from lti.app import theme, widgets

theme.header(
    "💼 Portfolio",
    "Your own account, from a ledger of what you put in, bought and sold: what it holds, what it "
    "cost, and what the same money — moved on the same days — would have made in SPY.",
    "The journal scores decisions; this scores the money. Every deposit buys SPY in a shadow "
    "account and every withdrawal sells it, so the gap is what your choices were worth in dollars. "
    "The same runs for each position and for the stocks you picked against your funds — the numbers "
    "behind how much belongs in picks and how much in an index.",
)


@st.cache_data(ttl=600, show_spinner=False)
def _px() -> prices_mod.PriceData:
    return prices_mod.load_price_data()


@st.cache_data(show_spinner=False)
def _stocks() -> set[str] | None:
    fund = widgets.fundamentals_or_none()
    return portfolio.stock_tickers(fund) if fund is not None else None


def _signed(v: float) -> str:
    """Dollars with the sign in front: +$1,234, −$567."""
    return f"{'−' if v < 0 else '+'}${abs(v):,.0f}"


ledger = portfolio.load_ledger()

with st.expander("Log a transaction", expanded=ledger.empty):
    with st.form("new_transaction", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        action = c1.selectbox("Action", portfolio.ACTIONS)
        day = c2.date_input("Date", value=pd.Timestamp.today().date())
        ticker = c3.text_input("Ticker", help="For a buy or sell — or income from one holding.").upper().strip()
        c4, c5, c6, c7 = st.columns(4)
        shares = c4.number_input("Shares", min_value=0.0, value=0.0, step=1.0, format="%.4f",
                                 help="As the broker showed them that day; a later split is handled.")
        price = c5.number_input("Price per share", min_value=0.0, value=0.0, step=0.01,
                                help="0 takes that day's close from the price cache.")
        amount = c6.number_input("Amount", value=0.0, step=100.0,
                                 help="For a deposit, a withdrawal, or income — negative for an account fee.")
        fees = c7.number_input("Fees", min_value=0.0, value=0.0, step=1.0)
        kind = st.radio(
            "Counts as", ["auto", "stock", "fund"], horizontal=True,
            help="Auto: a US company that files 10-Ks is a stock you picked; anything else — an index "
                 "fund, a bond ETF, a gold trust — a fund. Mark a foreign stock by hand.",
        )
        note = st.text_input("Note")
        if st.form_submit_button("Log it", type="primary"):
            try:
                e = portfolio.add_transaction(
                    action, ticker=ticker or None, shares=shares or None, price=price or None,
                    amount=amount or None, fees=fees, kind=None if kind == "auto" else kind, note=note,
                    date=day, px=_px(),
                )
                st.success(f"Logged {e['id']}: {e['action']} on {e['date']}.")
                ledger = portfolio.load_ledger()
            except ValueError as exc:
                st.error(str(exc))
    st.caption(
        "Dividends on tickers in the price cache are credited on their own. A buy beyond the cash on "
        "hand counts the difference as new money that day, so logging only trades works too. "
        "Also on the CLI: `lti portfolio-add`, `lti portfolio`."
    )

if ledger.empty:
    st.info("Nothing logged yet. Start with a deposit, or just the buys.")
    st.stop()

acct = portfolio.replay(ledger, _px(), stocks=_stocks())
s = acct.summary
years = s["years"]

k = st.columns(5)
k[0].metric("Value", f"${s['value']:,.0f}", help=f"Cash ${s['cash']:,.0f} of it.")
k[1].metric(
    "Money in, net", f"${s['net_deposits']:,.0f}",
    help="Deposits less withdrawals"
         + (f", with ${s['implicit_deposits']:,.0f} counted from buys beyond the cash on hand." if s["implicit_deposits"] else "."),
)
k[2].metric("Gain", _signed(s["gain"]),
            f"{s['gain'] / s['net_deposits']:+.1%}" if s["net_deposits"] > 0 else None)
k[3].metric(
    "Against SPY", _signed(s["vs_spy"]) if pd.notna(s["vs_spy"]) else "—",
    help=f"The same money, on the same days, in SPY would be worth ${s['spy_same_flows']:,.0f} now."
    if pd.notna(s["spy_same_flows"]) else "SPY isn't in the price cache.",
)
if years >= 1 and pd.notna(s["money_weighted"]):
    k[4].metric("Money-weighted return", f"{s['money_weighted']:+.1%} a year",
                f"SPY {s['spy_money_weighted']:+.1%}" if pd.notna(s["spy_money_weighted"]) else None,
                help="The rate your deposits and withdrawals, and what's there now, work out to (XIRR). "
                     "SPY's is the same flows in SPY.")
else:
    k[4].metric("Money-weighted return", "—", help="Shown after a year: a few months' return, annualized, "
                                                    "says more about the calendar than about you.")

st.header("The account against the same money in SPY")
d = acct.daily
fig = go.Figure()
for col, name, color, dash, shape in [
    ("net_deposits", "Money in, net", theme.ORANGE, "dot", "hv"),
    ("spy", "The same money in SPY", theme.MUTED, "solid", "linear"),
    ("value", "Your account", theme.BLUE, "solid", "linear"),
]:
    if d[col].notna().any():
        fig.add_trace(go.Scatter(
            x=d.index, y=d[col], name=name, mode="lines", line=dict(width=2, color=color, dash=dash, shape=shape),
            hovertemplate=f"{name} $%{{y:,.0f}}<extra></extra>",
        ))
theme.show(fig, height=400, hovermode="x unified", yaxis=dict(tickprefix="$", title=""), xaxis_title="")
if pd.notna(s["time_weighted_pa"]) and years >= 1:
    theme.note(
        f"Time-weighted, which takes the timing of your deposits out: <b>{s['time_weighted_pa']:+.1%}</b> a year "
        f"against SPY's {s['spy_return_pa']:+.1%} over the same {years:.1f} years. The money-weighted figure "
        "above counts the timing in — both are right, they answer different questions."
    )

st.header("Stocks you picked, against funds")
sl = acct.sleeves
st.dataframe(
    sl, hide_index=True, width="stretch",
    column_config={
        "sleeve": st.column_config.TextColumn(""),
        "value": st.column_config.NumberColumn("Value", format="$%.0f"),
        "weight": st.column_config.NumberColumn("Of the account", format="percent"),
        "money_in": st.column_config.NumberColumn(
            "Money in, net", format="$%.0f", help="What went into its buys, less what came out of sales and dividends."),
        "money_weighted": st.column_config.NumberColumn("Money-weighted return", format="percent"),
        "spy_same_flows": st.column_config.NumberColumn(
            "Same money in SPY", format="$%.0f", help="Every buy of these a buy of SPY, every sale and dividend a sale."),
        "vs_spy": st.column_config.NumberColumn("Against SPY", format="$%+.0f"),
    },
)
picks = sl[sl["sleeve"] == "Stocks you picked"]
if not picks.empty and pd.notna(picks["vs_spy"].iloc[0]):
    ahead = picks["vs_spy"].iloc[0] >= 0
    theme.note(
        f"The stocks you picked are <b>${abs(picks['vs_spy'].iloc[0]):,.0f} {'ahead of' if ahead else 'behind'}</b> "
        "what the same buys and sales in SPY would have made. "
        + ("Keep it up for years before reading much into it." if ahead else
           "If that holds for years, it's the case for moving more of the money into the index.")
    )

held = acct.holdings[acct.holdings["shares"] > 0].sort_values("value", ascending=False)
closed = acct.holdings[acct.holdings["shares"] == 0]
st.header("Holdings")
cols = {
    "ticker": st.column_config.TextColumn("Ticker", width="small"),
    "kind": st.column_config.TextColumn("Kind", width="small"),
    "shares": st.column_config.NumberColumn("Shares", format="%.4g", help="On today's share basis."),
    "avg_cost": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
    "value": st.column_config.NumberColumn("Value", format="$%.0f"),
    "weight": st.column_config.NumberColumn("Weight", format="percent"),
    "unrealized": st.column_config.NumberColumn("Unrealized", format="$%+.0f"),
    "unrealized_pct": st.column_config.NumberColumn("", format="percent"),
    "realized": st.column_config.NumberColumn("Realized", format="$%+.0f"),
    "income": st.column_config.NumberColumn("Dividends", format="$%.0f"),
    "money_weighted": st.column_config.NumberColumn("Money-weighted", format="percent"),
    "vs_spy": st.column_config.NumberColumn(
        "Against SPY", format="$%+.0f", help="Against the same buys, sales and dividends in SPY instead."),
    "first_bought": st.column_config.DateColumn("First bought", format="YYYY-MM-DD"),
}
order = list(cols)
st.dataframe(held[[c for c in order if c in held.columns]], hide_index=True, width="stretch", column_config=cols)
if not closed.empty:
    with st.expander(f"Closed positions ({len(closed)})"):
        st.dataframe(closed[[c for c in order if c in closed.columns]], hide_index=True, width="stretch",
                     column_config=cols)

with st.expander(f"Transactions ({len(ledger)})"):
    st.dataframe(ledger.sort_values(["date", "id"], ascending=False), hide_index=True, width="stretch")
    st.download_button("Download CSV", ledger.to_csv(index=False).encode(), "portfolio.csv", "text/csv")
    st.caption("Stored in `data/track/portfolio.jsonl`, one line per transaction; the page only ever "
               "appends, so fix a mistake by editing that file. The nightly job copies `data/track/` "
               "off the machine when `LTI_TRACK_BACKUP` is set.")

widgets.warnings_expander(acct.warnings)
