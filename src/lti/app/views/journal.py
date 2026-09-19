"""Decision journal — log what you decide and why, then see how each decision has aged."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lti import journal
from lti.app import theme

theme.header(
    "✍️ Decision journal",
    "Write down each decision — buy, sell, pass — with why, and what would prove you wrong. "
    "Each one is scored against SPY from the day you made it: a buy is right if the stock beat "
    "the market since, a sell or a pass if it lagged.",
    "Greenblatt found that people running the Magic Formula themselves trailed the automated "
    "version: they overrode picks they didn't like and quit after bad years. The journal is how "
    "you find out whether your own judgement adds anything. Entries are never edited — a later "
    "look is a new <b>review</b> entry.",
)

entries = journal.load_journal()

with st.expander("Log a decision", expanded=entries.empty):
    with st.form("new_entry", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker").upper().strip()
        action = c2.selectbox("Action", journal.ACTIONS)
        conviction = c3.slider("Conviction", 1, 5, 3)
        thesis = st.text_area("Why", placeholder="The reason, in a sentence or two.")
        change = st.text_area("What would change my mind", placeholder="The evidence that would make this wrong.")
        c4, c5, c6 = st.columns(3)
        fair_value = c4.number_input("Fair value / share (optional)", min_value=0.0, value=0.0, step=1.0)
        review_by = c5.date_input("Review by", value=(pd.Timestamp.today() + pd.DateOffset(years=1)).date())
        ids = entries.loc[entries["action"] != "review", "id"].tolist() if not entries.empty else []
        refers_to = c6.selectbox("Reviewing (for a review)", ["—", *ids])
        if st.form_submit_button("Log it", type="primary"):
            try:
                e = journal.add_entry(
                    ticker, action, thesis, change_my_mind=change,
                    fair_value=fair_value or None, conviction=conviction, review_by=review_by,
                    refers_to=None if refers_to == "—" else refers_to,
                )
                st.success(f"Logged {e['id']}: {e['action']} {e['ticker']} at ${e['price']:,.2f}.")
                entries = journal.load_journal()
            except ValueError as exc:
                st.error(str(exc))

if entries.empty:
    st.info("No decisions logged yet.")
    st.stop()

out = journal.outcomes(entries)
scored = out["right_so_far"].dropna()
k = st.columns(4)
k[0].metric("Decisions logged", f"{int((out['action'] != 'review').sum())}")
k[1].metric("Right so far", f"{int(scored.sum())} of {len(scored)}" if len(scored) else "—",
            help="Against SPY since the decision: buys and adds that beat it, sells, trims and passes that lagged.")
k[2].metric("Reviews due", f"{int(out['review_due'].sum())}")
k[3].metric("Oldest decision", str(out["date"].min().date()))

due = out[out["review_due"]]
if not due.empty:
    st.warning("Due for a second look: " + ", ".join(f"{r.ticker} ({r.action}, {r.date.date()})" for r in due.itertuples()))

view = out.sort_values("date", ascending=False)[
    ["date", "ticker", "action", "conviction", "price", "price_now", "since", "spy_since", "vs_spy",
     "right_so_far", "review_due", "thesis", "change_my_mind", "fair_value", "id"]
].copy()
view["right_so_far"] = view["right_so_far"].map({1.0: "✓", 0.0: "✗"}).fillna("")
st.dataframe(
    view, hide_index=True, width="stretch",
    column_config={
        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "price": st.column_config.NumberColumn("Price then", format="$%.2f"),
        "price_now": st.column_config.NumberColumn("Now", format="$%.2f"),
        "since": st.column_config.NumberColumn("Since", format="percent", help="Total return, dividends included."),
        "spy_since": st.column_config.NumberColumn("SPY since", format="percent"),
        "vs_spy": st.column_config.NumberColumn("vs SPY", format="percent"),
        "right_so_far": st.column_config.TextColumn("Right?", width="small"),
        "review_due": st.column_config.CheckboxColumn("Review due"),
        "fair_value": st.column_config.NumberColumn("Fair value", format="$%.2f"),
    },
)
st.caption("Stored in `data/track/journal.jsonl`, one line per entry — also `lti journal` / `lti journal-add` on the CLI.")
