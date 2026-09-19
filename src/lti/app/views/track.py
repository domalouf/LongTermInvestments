"""Track record — what each tracked strategy held, day by day, and how it has done since."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti import track
from lti.app import theme

LABELS = {s.name: s.label for s in track.STRATEGIES} | {track.BENCHMARK: "SPY"}

theme.header(
    "📒 Track record",
    "Every night the tracked strategies write down what they'd hold that day — once, never "
    "revised — and this page measures what those holdings went on to return from the next "
    "day's close, against SPY and against the whole universe recorded the same day.",
    "The one test here free of hindsight and of survivorship bias: nothing on record could have "
    "been chosen knowing what came next, and a company that later fails stays in it. It takes "
    "time — read nothing into a few months.",
)


@st.cache_data(ttl=600, show_spinner=False)
def _records() -> pd.DataFrame:
    return track.load_records()


@st.cache_data(ttl=600, show_spinner="Scoring the record…")
def _scored(n_rows: int, last_day: str):  # the arguments only key the cache
    records = track.load_records()
    return track.summary(track.performance(records)), track.paper_curves(records)


records = _records()
strategies = pd.DataFrame(
    [{"strategy": s.label, "why it's tracked": s.why} for s in track.STRATEGIES]
)

if records.empty:
    st.info("Nothing on record yet. `lti track-record` writes the first day; the nightly job runs it.")
    st.dataframe(strategies, hide_index=True, width="stretch")
    st.stop()

days = records["record_date"].drop_duplicates().sort_values()
latest = days.iloc[-1]
k = st.columns(4)
k[0].metric("Tracking since", str(days.iloc[0].date()))
k[1].metric("Days on record", f"{len(days):,}")
k[2].metric("Latest record", str(latest.date()))
k[3].metric("Strategies", f"{records['strategy'].nunique()}")

summary, curves = _scored(len(records), str(latest.date()))

st.header("How the records have done")
if summary.empty:
    first_due = (days.iloc[0] + pd.DateOffset(months=1)).date()
    st.info(f"No record has reached its first horizon yet — the first one-month results arrive after {first_due}.")
else:
    horizons = sorted(summary["horizon"].unique())
    h = st.radio("Horizon (months)", horizons, horizontal=True, index=len(horizons) - 1)
    view = summary[summary["horizon"] == h].copy()
    view["strategy"] = view["strategy"].map(LABELS)
    theme.note(
        f"Each record's holdings over the {h} month(s) after it, averaged across records. "
        "<b>vs universe</b> is the fair comparison — same day, same candidates. <b>t</b> uses one "
        "record a month with Newey-West errors, since daily records share most of their returns; "
        "below about 2 it can't be told from luck."
    )
    st.dataframe(
        view[["strategy", "records", "months", "ret", "vs_universe", "vs_spy", "beat_universe", "t_vs_universe"]],
        hide_index=True, width="stretch",
        column_config={
            "strategy": st.column_config.TextColumn("Strategy", width="medium"),
            "records": st.column_config.NumberColumn("Records", format="%d"),
            "months": st.column_config.NumberColumn("Months", format="%d"),
            "ret": st.column_config.NumberColumn("Return", format="percent"),
            "vs_universe": st.column_config.NumberColumn("vs universe", format="percent"),
            "vs_spy": st.column_config.NumberColumn("vs SPY", format="percent"),
            "beat_universe": st.column_config.NumberColumn("Beat universe", format="percent"),
            "t_vs_universe": st.column_config.NumberColumn("t", format="%.1f"),
        },
    )
    bars = view[view["strategy"] != LABELS["universe"]]
    fig = go.Figure(
        go.Bar(
            x=bars["vs_universe"], y=bars["strategy"], orientation="h",
            marker_color=[theme.POS if v >= 0 else theme.NEG for v in bars["vs_universe"]],
            hovertemplate="%{y}<br>%{x:+.1%} vs the universe<extra></extra>",
        )
    )
    fig.update_traces(marker_line_width=0, marker_cornerradius=4)
    fig.update_layout(bargap=0.4)
    theme.zero_line(fig, axis="x")
    theme.show(fig, height=max(240, 44 * len(bars) + 80), legend=False,
               xaxis=dict(tickformat="+.1%", title=f"average {h}-month return minus the universe's"), yaxis_title="")

if len(curves) > 1:
    st.header("Following each one")
    theme.note("Growth of $1 moving into each month's first record at the next close — the portfolio "
              "you'd hold rebalancing monthly.")
    fig2 = go.Figure()
    for i, col in enumerate(curves.columns):
        color = theme.MUTED if col == track.BENCHMARK else theme.SERIES[i % len(theme.SERIES)]
        fig2.add_trace(go.Scatter(x=curves.index, y=curves[col], name=LABELS.get(col, col), mode="lines+markers",
                                  line=dict(width=2, color=color),
                                  hovertemplate=f"{LABELS.get(col, col)} $%{{y:.3f}}<extra></extra>"))
    theme.show(fig2, height=400, hovermode="x unified", yaxis=dict(tickprefix="$"), xaxis_title="")

st.header("What's on record")
pick = st.selectbox("Strategy", [s.name for s in track.STRATEGIES], format_func=lambda n: LABELS[n])
latest_rows = records[(records["record_date"] == latest) & (records["strategy"] == pick)]
st.caption(f"{len(latest_rows)} holdings on {latest.date()}, equal-weighted.")
st.dataframe(latest_rows[["rank", "ticker", "company", "close", "score"]], hide_index=True, width="stretch")
with st.expander("Why each strategy is tracked"):
    st.dataframe(strategies, hide_index=True, width="stretch")
