"""Backtest page — run and visualise an annual-rebalance strategy against SPY and its own universe."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti import attribution
from lti.app import theme, widgets
from lti.backtest import BacktestConfig, rebalance_month_spread, run_backtest
from lti.ranking import ScreenSpec

theme.header(
    "🧪 Strategy backtest",
    "Buy the top N of a screen, equal-weighted, rebalance once a year, and compare against "
    "SPY and against an equal-weighted basket of <i>every</i> stock the screen ranked — all "
    "three after trading costs, and after tax if the account is taxable.",
    "The basket is the fair yardstick: it can only hold today's survivors too, so the gap "
    "between it and the strategy is what the ranking itself added. The gap to SPY also "
    "contains the survivorship bias — see the warning below.",
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _config(cfg_key: str) -> BacktestConfig:
    raw = json.loads(cfg_key)
    spec = ScreenSpec(
        metrics=raw["metrics"],
        top_n=raw["top_n"],
        weights=raw.get("weights"),
        filters=raw.get("filters", {}),
        min_coverage=raw.get("min_coverage", 1.0),
    )
    return BacktestConfig(
        screen=spec,
        start=raw["start"],
        end=raw["end"],
        rebalance_month=raw["rebalance_month"],
        market_cap_min=raw["market_cap_min"],
        initial_capital=raw["initial_capital"],
        sell_rank=raw.get("sell_rank"),
        industry_cap=raw.get("industry_cap"),
        quarterly=raw.get("quarterly", True),
        **widgets.friction_kwargs(raw["frictions"]),
    )


@st.cache_data(show_spinner="Running the backtest…")
def _run(cfg_key: str):
    result = run_backtest(_config(cfg_key))
    return (
        result.equity_curve, result.benchmark_curve, result.universe_curve,
        result.holdings, result.period_summary, result.stats, result.warnings,
        result.equity_curve_gross,
    )


@st.cache_data(show_spinner="Running the strategy once per rebalance month — about a minute…")
def _spread(cfg_key: str) -> pd.DataFrame:
    return rebalance_month_spread(_config(cfg_key))


with st.sidebar:
    st.header("Strategy")
    magic = st.checkbox(
        "Greenblatt Magic Formula",
        value=False,
        help="EBIT/EV and return on capital, equally weighted, financials and "
        "utilities excluded.",
    )
    chosen, coverage = widgets.rank_by(["pe", "debt_to_equity"], magic=magic)
    top_n = st.slider("Top N", 5, 50, 30 if magic else 10)
    sell_rank = widgets.sell_rank(top_n)
    industry_cap = widgets.industry_cap()
    use_quarterly = st.checkbox(
        "Use 10-Q filings", value=True,
        help="Rank each company on its latest quarter — trailing twelve months from its 10-Qs — rather "
             "than waiting up to a year for the next 10-K. Off ranks on 10-Ks alone, as before. Needs "
             "`lti build-quarterly` (or a fresh `lti build-fundamentals`).",
    )
    start = st.text_input("Start", "2013-01-01")
    end = st.text_input("End", "2024-01-01")
    rebal_month = st.slider(
        "Rebalance month", 1, 12, 4,
        help="April by default: by then nearly every calendar-year company has filed its 10-K, "
             "so the ranking uses last year's numbers. In January it would use the year before's.",
    )
    cap_floor_m = st.number_input("Min market cap ($M)", value=500.0, step=100.0, min_value=0.0)
    capital = st.number_input("Initial capital ($)", value=100_000.0, step=10_000.0)
    excl_fin = st.checkbox("Exclude financials", value=magic)
    excl_util = st.checkbox("Exclude utilities", value=magic)
    fric = widgets.frictions()
    go_btn = st.button("Run backtest", type="primary")

if not chosen:
    st.info("Pick at least one metric.")
    st.stop()

cfg_key = json.dumps(
    {
        "metrics": chosen,
        "top_n": top_n,
        "start": start,
        "end": end,
        "rebalance_month": rebal_month,
        "market_cap_min": cap_floor_m * 1e6,
        "initial_capital": capital,
        "filters": {"exclude_financials": excl_fin, "exclude_utilities": excl_util},
        "min_coverage": coverage,
        "sell_rank": sell_rank,
        "industry_cap": industry_cap,
        "quarterly": use_quarterly,
        "frictions": fric,
    }
)

widgets.run_gate("backtest", go_btn, cfg_key, "Set the strategy in the sidebar and hit **Run backtest**.")

try:
    equity, bench, universe, holdings, period_summary, stats, warnings, equity_gross = _run(cfg_key)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if period_summary.empty:
    # nothing was ever bought, so every statistic below is NaN and the charts are
    # blank — the warnings are the whole answer, one per rebalance date
    st.error(
        "**No rebalance date produced a portfolio.** Nothing was bought, so there is nothing "
        "to measure. The usual cause is filters that empty the universe — a market-cap floor "
        "above every company, or a metric the fundamentals table doesn't carry for anyone. "
        "Each warning below names a date and the reason."
    )
    widgets.warnings_expander(warnings, expanded=True)
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Strategy CAGR", f"{stats['port_cagr']:.1%}",
          f"{stats['excess_cagr_vs_univ']:+.1%} vs universe")
c2.metric("Universe CAGR", f"{stats['univ_cagr']:.1%}",
          help="Every stock the screen ranked on each rebalance date, equal-weighted — "
               "what picking at random from the same candidates would have returned.")
c3.metric("SPY CAGR", f"{stats['bench_cagr']:.1%}",
          help=f"The strategy {'beat' if stats['excess_cagr'] >= 0 else 'trailed'} SPY by "
               f"{abs(stats['excess_cagr']):.1%} a year.")
c4.metric("Max drawdown", f"{stats['port_max_drawdown']:.1%}",
          f"{stats['port_max_drawdown'] - stats['bench_max_drawdown']:+.1%} vs SPY",
          delta_color="inverse")
c5.metric("Sharpe", f"{stats['port_sharpe']:.2f}", f"SPY {stats['bench_sharpe']:.2f}")

has_frictions = fric["cost_bps"] > 0 or fric["tax"] is not None
if has_frictions:
    theme.note(
        f"All three after {fric['cost_bps']:g} bps a trade{' and taxes' if fric['tax'] else ''}. "
        f"Before: strategy {stats['port_cagr_gross']:.1%}, universe {stats['univ_cagr_gross']:.1%}, "
        f"SPY {stats['bench_cagr_gross']:.1%} — see <i>What trading and taxes took</i> below."
    )

st.header("Growth of the initial stake")
log_scale = st.toggle(
    "Log scale", value=False,
    help="On a log axis equal vertical distances are equal *percentage* moves, "
         "which is the honest way to compare curves over a long window.",
)
fig = go.Figure()
lines = [
    (bench, "SPY", theme.MUTED, "solid"),
    (universe, "Universe (equal weight)", theme.ORANGE, "solid"),
    (equity, "Strategy", theme.BLUE, "solid"),
]
if has_frictions:
    lines.append((equity_gross, "Strategy before costs & taxes", theme.BLUE, "dot"))
for curve, name, color, dash in lines:
    fig.add_trace(
        go.Scatter(
            x=curve.index, y=curve.values, name=name, mode="lines",
            line=dict(width=2 if dash == "solid" else 1.5, color=color, dash=dash),
            hovertemplate=f"{name} $%{{y:,.0f}}<extra></extra>",
        )
    )
theme.show(
    fig, height=430, hovermode="x unified",
    yaxis=dict(type="log" if log_scale else "linear", tickprefix="$", title="portfolio value"),
    xaxis_title="",
)

with st.expander("All statistics"):
    # stats mixes floats with the drawdown peak/trough Timestamps, which Arrow
    # can't type as one column — format to text rather than let it guess
    def _fmt(k: str, v) -> str:
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            return str(v)
        if any(t in k for t in ("cagr", "return", "drawdown", "vol", "rate", "turnover", "beat", "_pa", "share")):
            return f"{v:.2%}"
        return f"{v:,.3f}"

    st.dataframe(
        pd.DataFrame({"stat": list(stats), "value": [_fmt(k, v) for k, v in stats.items()]}),
        hide_index=True,
        width="stretch",
    )

if has_frictions:
    st.header("What trading and taxes took")
    taxed = fric["tax"] is not None
    table = pd.DataFrame(
        [
            {
                "portfolio": name,
                "before": stats[f"{leg}_cagr_gross"],
                "after": stats[f"{leg}_cagr"],
                "lost": stats[f"{leg}_cagr"] - stats[f"{leg}_cagr_gross"],
                "costs": stats[f"{leg}_costs_pa"],
                "taxes": stats[f"{leg}_taxes_pa"],
                "sold": stats[f"{leg}_cagr_liquidated"],
            }
            for leg, name in [("port", "Strategy"), ("univ", "Universe"), ("bench", "SPY")]
        ]
    )
    if not taxed:
        table = table.drop(columns=["taxes", "sold"])

    def pct(label: str, **kw):
        return st.column_config.NumberColumn(label, format="percent", **kw)

    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "portfolio": st.column_config.TextColumn(""),
            "before": pct("CAGR before"),
            "after": pct("CAGR after"),
            "lost": pct("Lost a year"),
            "costs": pct("Costs a year", help="Paid to trade at each rebalance, as a share of the portfolio."),
            "taxes": pct("Taxes a year", help="On dividends and realized gains, as a share of the portfolio."),
            "sold": pct("Sold at the end", help="The CAGR had everything been sold on the last day, "
                                                "paying the tax on gains not yet realized."),
        },
    )
    turnover = period_summary["turnover"].iloc[1:].median() if len(period_summary) > 1 else float("nan")
    notes = []
    if turnover == turnover:
        notes.append(
            f"A typical rebalance replaced <b>{turnover:.0%}</b> of the strategy (one-way turnover), each "
            f"dollar of it paying {fric['cost_bps']:g} bps to sell and again to buy. The universe pays the "
            "same rate on its own, smaller, turnover, so the gap to it is what the ranking adds after "
            "paying for the trading it takes."
        )
        if sell_rank:
            kept = period_summary["n_held_over"].iloc[1:].median()
            notes.append(
                f"The buffer kept a median <b>{kept:.0f} of {top_n}</b> holdings at each rebalance: a holding "
                f"stays until it drops out of the top {sell_rank}."
            )
        elif len(period_summary) > 1:
            notes.append(
                "A buffer (<i>Sell a holding once it drops out of the top</i> …) keeps names that slip just "
                "below the cut-off, which is the cheapest way to trade less."
            )
    if taxed:
        share = stats["port_short_term_share"]
        rates = fric["tax"]
        if share == share:
            line = (
                f"<b>{share:.0%}</b> of the strategy's realized gains were short-term, taxed at "
                f"{rates['short_term']:.0%} rather than {rates['long_term']:.0%}."
            )
            if not fric["hold_past_one_year"] and share > 0:
                line += (
                    " The rebalance lands on or just short of the one-year mark in most years — tick "
                    "<i>Sell only after a full year</i> to see what waiting a few days longer is worth."
                )
            notes.append(line)
        notes.append(
            "SPY is bought once and never sold, so apart from its dividends none of its gain is taxed "
            "until you sell. <i>Sold at the end</i> puts all three on the same footing."
        )
    theme.note(" ".join(notes))

n_delisted = int(period_summary["n_delisted"].sum()) if not period_summary.empty else 0
st.warning(
    f"**Survivorship bias.** This backtest can only trade tickers that appear in today's SEC "
    f"ticker map *and* in Yahoo Finance, so companies that were acquired or went bankrupt are "
    f"largely invisible. Across every holding period here, only **{n_delisted}** position(s) hit "
    f"a delisting — in reality roughly half of US listed companies disappear over a window this "
    f"long. That flatters the strategy *and* the universe basket alike, which is why the gap "
    f"between those two is the number to trust; the gap to SPY is an upper bound. "
    f"({len(warnings)} warnings raised.)"
)

st.header("Was it consistent, or one good year?")
basis = st.radio(
    "Compare each period against", ["the universe", "SPY"], horizontal=True,
    help="The universe is the fair comparison: same candidates, same survivorship bias.",
)
if not period_summary.empty:
    ps = period_summary.copy()
    ps["year"] = pd.to_datetime(ps["rebalance_date"]).dt.year
    col, label = ("excess_vs_univ", "universe") if basis == "the universe" else ("excess_return", "SPY")
    fig2 = go.Figure(
        go.Bar(
            x=ps["year"], y=ps[col],
            marker_color=[theme.POS if v >= 0 else theme.NEG for v in ps[col]],
            hovertemplate=f"%{{x}}<br>strategy − {label}: %{{y:+.1%}}<extra></extra>",
        )
    )
    theme.bar_marks(fig2, color=None)
    theme.zero_line(fig2, axis="y")
    beat = int((ps[col] > 0).sum())
    theme.note(
        f"The strategy beat {label} in <b>{beat} of {len(ps)}</b> rebalance periods. "
        "With this few periods, a couple of good years can carry the whole CAGR — which is why "
        "the headline number alone shouldn't decide anything."
    )
    theme.show(
        fig2, height=300, legend=False,
        yaxis=dict(tickformat="+.0%", title=f"strategy − {label}, that period"),
        xaxis=dict(title="", dtick=1),
    )

if not period_summary.empty and period_summary["top_industry_share"].notna().any():
    st.header("Was it one theme?")
    ps = period_summary.dropna(subset=["top_industry_share"]).copy()
    ps["year"] = pd.to_datetime(ps["rebalance_date"]).dt.year
    fig_ind = go.Figure(
        go.Bar(
            x=ps["year"], y=ps["top_industry_share"], customdata=ps["top_industry"],
            hovertemplate="%{x}: %{customdata}, %{y:.0%} of the picks<extra></extra>",
        )
    )
    theme.bar_marks(fig_ind)
    top = float(ps["top_industry_share"].max())
    if industry_cap:
        # drawn to stand apart from the gridlines, with headroom above so it isn't the frame's top edge
        fig_ind.add_hline(y=industry_cap, line_width=1.5, line_dash="dash", line_color=theme.ORANGE)
        fig_ind.add_annotation(
            x=1.0, xref="paper", y=industry_cap, yanchor="bottom", xanchor="right",
            text=f"cap {industry_cap:.0%}", showarrow=False, font=dict(size=10.5, color=theme.ORANGE),
        )
        top = max(top, industry_cap)
    common = ps["top_industry"].mode()
    theme.note(
        f"The largest industry in the portfolio at each rebalance — on average <b>{ps['top_industry_share'].mean():.0%}"
        f"</b> of it, most often {common.iloc[0] if len(common) else '—'}. A screen can rank well on every "
        "metric and still be one bet on one industry. "
        + (
            f"The dashed line is the cap: at most {industry_cap:.0%} in any one."
            if industry_cap
            else "<i>Most in one industry</i> in the sidebar caps it."
        )
    )
    theme.show(
        fig_ind, height=280, legend=False,
        yaxis=dict(tickformat=".0%", title="largest industry's share", range=[0, top * 1.18]),
        xaxis=dict(title="", dtick=1),
    )

st.header("Skill or style?")
factors = attribution.load_factors()  # small; read fresh so a fetch shows up on the next run
if factors.empty:
    theme.note(
        "A screen that beats its universe may just own smaller, cheaper or more profitable companies — tilts "
        "an index fund can buy. A regression on the Fama-French factors separates the two: run "
        "<code>lti fetch-factors</code> to cache them."
    )
else:
    model = st.radio(
        "Factors", list(attribution.MODELS), index=list(attribution.MODELS).index("ff5_mom"),
        format_func=attribution.MODEL_LABELS.get, horizontal=True,
    )
    fit = attribution.attribute(
        SimpleNamespace(equity_curve=equity, universe_curve=universe, benchmark_curve=bench), factors, model
    )
    edge = "Strategy − universe"
    if fit.n[edge] > 0:
        loads = fit.coef[edge].drop("alpha")
        tvals = fit.t[edge].drop("alpha")
        fig_f = go.Figure(
            go.Bar(
                y=[attribution.TERM_LABELS[f] for f in loads.index], x=loads, orientation="h",
                marker_color=[theme.POS if v >= 0 else theme.NEG for v in loads],
                customdata=tvals, hovertemplate="%{y}: %{x:+.2f} (t %{customdata:.1f})<extra></extra>",
            )
        )
        theme.bar_marks(fig_f, color=None)
        theme.zero_line(fig_f, axis="x")
        alpha, t_alpha = fit.coef.at["alpha", edge], fit.t.at["alpha", edge]
        gap = stats.get("excess_cagr_vs_univ", float("nan"))
        biggest = loads.abs().idxmax()
        verdict = (
            "indistinguishable from none — the edge is the tilts" if abs(t_alpha) < 2
            else "the part the tilts don't explain"
        )
        theme.note(
            f"Against its universe the strategy returned <b>{gap:+.1%}</b> a year. The factors leave "
            f"<b>{alpha:+.1%}</b> a year of that unexplained (t {t_alpha:.1f}): {verdict}. Its biggest tilt "
            f"against the universe is <b>{attribution.TERM_LABELS[biggest].lower()}</b> ({loads[biggest]:+.2f}). "
            f"SPY loads {fit.coef.at['mkt_rf', 'SPY']:.2f} on the market, a check that the returns line up "
            "with the factors."
        )
        theme.show(fig_f, height=60 + 38 * len(loads), legend=False,
                   xaxis=dict(title="strategy − universe: loading on each factor"), yaxis=dict(title="", autorange="reversed"))
    with st.expander("All four regressions"):
        st.dataframe(attribution.as_text(fit), width="stretch")
        span = f"{fit.months[0]:%b %Y} to {fit.months[1]:%b %Y}" if fit.months else "no overlapping months"
        st.caption(
            f"Monthly returns, {span}, on {attribution.MODEL_LABELS[model]} from Ken French's data library; "
            "Newey-West t-stats. The strategy and universe can only hold survivors, while the factors come from "
            "every stock, so their own alphas carry survivorship bias — strategy − universe largely nets it out. "
            "The curves are after the costs and taxes set in the sidebar."
        )
    widgets.warnings_expander(fit.warnings)

st.header("Does the rebalance month matter?")
theme.note(
    "With a dozen annual rebalances, <i>when</i> in the year the portfolio turns over can decide "
    "whether a screen beats its universe. Running the same strategy once per month shows how "
    "much of the result above is the screen and how much is the calendar."
)
if widgets.ran_with("backtest_spread", st.button("Run all 12 rebalance months"), cfg_key):
    spread = _spread(cfg_key)
    if spread.empty:
        st.info("No month produced a result for this date range.")
    else:
        sp = spread.copy()
        sp["month"] = [MONTHS[m - 1] for m in sp["rebalance_month"]]
        fig3 = go.Figure(
            go.Bar(
                x=sp["month"], y=sp["excess_vs_univ"],
                marker_color=[theme.POS if v >= 0 else theme.NEG for v in sp["excess_vs_univ"]],
                customdata=sp[["port_cagr", "univ_cagr", "excess_vs_bench"]].to_numpy(),
                hovertemplate=(
                    "rebalance in %{x}<br>strategy − universe %{y:+.1%} a year"
                    "<br>strategy %{customdata[0]:.1%} · universe %{customdata[1]:.1%}"
                    "<br>vs SPY %{customdata[2]:+.1%}<extra></extra>"
                ),
            )
        )
        theme.bar_marks(fig3, color=None)
        theme.zero_line(fig3, axis="y")
        ex = sp["excess_vs_univ"]
        theme.note(
            f"Excess CAGR over the universe ranges from <b>{ex.min():+.1%}</b> to "
            f"<b>{ex.max():+.1%}</b> depending on the month (median {ex.median():+.1%}); "
            f"it is positive in <b>{int((ex > 0).sum())} of {len(ex)}</b> months."
        )
        theme.show(
            fig3, height=300, legend=False,
            yaxis=dict(tickformat="+.0%", title="strategy − universe, CAGR"),
            xaxis_title="rebalance month",
        )
        with st.expander("Per-month detail"):
            st.dataframe(sp.drop(columns=["month"]), hide_index=True, width="stretch")

with st.expander("Per-period detail"):
    returns_note = " After costs and taxes." if has_frictions else ""
    st.dataframe(
        period_summary, hide_index=True, width="stretch",
        column_config={
            "rebalance_date": st.column_config.DateColumn("Bought", format="YYYY-MM-DD"),
            "exit_date": st.column_config.DateColumn("Held until", format="YYYY-MM-DD"),
            "n_selected": st.column_config.NumberColumn("Picks"),
            "n_universe": st.column_config.NumberColumn("Ranked", help="Companies the screen ranked that day."),
            "n_delisted": st.column_config.NumberColumn(
                "Delisted", help="Picks whose prices stopped before the period ended."),
            "n_held_over": st.column_config.NumberColumn(
                "Kept", help="Picks already held from the rebalance before, so not bought again."),
            "top_industry": st.column_config.TextColumn("Top industry", help="Fama-French 12, from SIC codes."),
            "top_industry_share": st.column_config.NumberColumn("Its share", format="percent"),
            "port_return": st.column_config.NumberColumn("Strategy", format="percent", help="Over the period." + returns_note),
            "univ_return": st.column_config.NumberColumn("Universe", format="percent", help="Over the period." + returns_note),
            "bench_return": st.column_config.NumberColumn("SPY", format="percent", help="Over the period." + returns_note),
            "excess_return": st.column_config.NumberColumn("Strategy − SPY", format="percent"),
            "excess_vs_univ": st.column_config.NumberColumn("Strategy − universe", format="percent"),
            "port_return_gross": st.column_config.NumberColumn("Strategy before costs & taxes", format="percent"),
            "turnover": st.column_config.NumberColumn(
                "Turnover", format="percent", help="Share of the portfolio replaced at this rebalance, one way: a full swap is 100%."),
            "costs": st.column_config.NumberColumn(
                "Costs", format="percent", help="Paid to trade, as a share of what the strategy was worth going in."),
            "taxes": st.column_config.NumberColumn(
                "Taxes", format="percent", help="On gains and dividends, as a share of what the strategy was worth going in."),
            "gains_short_term": st.column_config.NumberColumn(
                "Short-term gains", format="percent", help="Realized at this rebalance, net of losses, as a share of the portfolio."),
            "gains_long_term": st.column_config.NumberColumn(
                "Long-term gains", format="percent", help="Realized at this rebalance, net of losses, as a share of the portfolio."),
        },
    )

with st.expander("Holdings (every pick, every period)"):
    theme.note("Each pick with the metric values it was ranked on — the first place to look when a result seems too good.")
    st.dataframe(holdings, hide_index=True, width="stretch")
    st.download_button("Download holdings CSV", holdings.to_csv(index=False).encode(), "holdings.csv", "text/csv")

widgets.warnings_expander(warnings)
