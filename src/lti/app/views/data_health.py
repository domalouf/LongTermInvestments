"""Data health — what's built, how much of it there is, and what's missing."""

from __future__ import annotations

import lti.config as config

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lti.app import theme

theme.header(
    "🩺 Data health",
    "Which artifacts exist on disk, how far the fundamentals reach, and how much "
    "of the universe has prices. Every gap here names the <code>lti</code> command that fills it.",
)

paths = config.get_paths()
if config.is_smoke():
    st.warning("Running in **smoke mode** (`LTI_SMOKE=1`) — a limited data subset.")

ARTIFACTS = [
    ("SEC parquet dir", paths.sec_parquet, "lti update --force"),
    ("SEC index DB", paths.sec_db, "lti update --force"),
    ("Standardized statements", paths.concat_std_bs, "lti pipeline"),
    ("Fundamentals table", paths.fundamentals_parquet, "lti build-fundamentals"),
    ("CIK → ticker map", paths.cik_ticker_parquet, "lti refresh-tickers"),
    ("Price cache (total return)", paths.adj_close_parquet, "lti fetch-prices"),
    ("Price cache (split-adjusted)", paths.close_parquet, "lti fetch-prices"),
    ("Split history", paths.splits_parquet, "lti fetch-prices"),
    ("Dividend history", paths.dividends_parquet, "lti fetch-prices"),
]

present = [(n, p, c) for n, p, c in ARTIFACTS if p.exists()]
missing = [(n, p, c) for n, p, c in ARTIFACTS if not p.exists()]

c1, c2, c3 = st.columns(3)
c1.metric("Artifacts present", f"{len(present)}/{len(ARTIFACTS)}")
try:
    from lti import sec_update

    c2.metric("Latest SEC quarter", sec_update.latest_quarter() or "—")
except Exception:  # noqa: BLE001 — the index simply may not be built yet
    c2.metric("Latest SEC quarter", "—")

try:
    from lti import prices

    _px = prices.load_price_data()
    _panel = _px.adj
    c3.metric("Tickers priced", f"{_panel.shape[1]:,}" if not _panel.empty else "0")
except Exception:  # noqa: BLE001
    _px = None
    _panel = pd.DataFrame()
    c3.metric("Tickers priced", "—")

if missing:
    st.error(
        "**Missing:** "
        + " · ".join(f"{n} → `{c}`" for n, _, c in missing)
    )

with st.expander(f"Artifact paths ({len(present)} present)"):
    st.dataframe(
        pd.DataFrame(
            [{"artifact": n, "present": "✅" if p.exists() else "❌", "path": str(p)}
             for n, p, _ in ARTIFACTS]
        ),
        hide_index=True,
        width="stretch",
    )

# --- fundamentals -----------------------------------------------------------

st.header("Fundamentals")
try:
    from lti import fundamentals

    fund = fundamentals.load_fundamentals()
    _universe = set(fundamentals.price_universe(fund))
    with_ticker = fund.loc[fund["ticker"].notna(), "cik"].nunique()
    all_cik = fund["cik"].nunique()

    m = st.columns(4)
    m[0].metric("Filings", f"{len(fund):,}")
    m[1].metric("Companies", f"{all_cik:,}")
    # not a delta — a green up-arrow on a shortfall reads exactly backwards
    m[2].metric("With a ticker", f"{with_ticker:,}",
                help=f"{all_cik - with_ticker:,} companies have no ticker in the SEC's map.")
    m[3].metric("Fiscal years", f"{int(fund['fiscal_year'].min())}–{int(fund['fiscal_year'].max())}")

    theme.note(
        "The gap between <b>companies</b> and <b>with a ticker</b> is the survivorship-bias hole: "
        "delisted companies have no entry in the SEC's current <code>company_tickers.json</code>, "
        "so a backtest can only ever trade the survivors."
    )

    by_year = fund.groupby("fiscal_year").size()
    fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values, hovertemplate="FY%{x}<br>%{y:,} filings<extra></extra>"))
    theme.bar_marks(fig)
    st.subheader("Filings per fiscal year")
    theme.show(fig, height=260, xaxis_title="fiscal year", yaxis_title="10-K filings")

    st.subheader("Coverage of the fields the screens depend on")
    tickered = fund[fund["ticker"].notna()]
    fields = [
        ("operating_income_reported", "EBIT (as tagged by the filer)"),
        ("total_debt", "Interest-bearing debt"),
        ("ppe_net", "Net PP&E"),
        ("shares_outstanding", "Shares outstanding"),
        ("eps", "EPS"),
        ("sic", "SIC / sector"),
    ]
    rows = [
        {"field": label, "coverage": tickered[col].notna().mean()}
        for col, label in fields
        if col in tickered.columns
    ]
    cov = pd.DataFrame(rows).sort_values("coverage")
    bar = go.Figure(
        go.Bar(
            x=cov["coverage"], y=cov["field"], orientation="h",
            text=[f"{v:.0%}" for v in cov["coverage"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.1%} of filings with a ticker<extra></extra>",
        )
    )
    theme.bar_marks(bar)
    theme.show(
        bar, height=max(230, 34 * len(cov) + 60),
        xaxis=dict(tickformat=".0%", range=[0, 1.09], title="share of filings with a ticker"),
        yaxis_title="",
    )
    theme.note(
        "A field at less than 100% shrinks the universe for any screen that ranks on it — "
        "<code>ranking.rank()</code> drops rows with a missing value."
    )
    if "shares_source" in tickered.columns:
        src = tickered["shares_source"].value_counts(normalize=True)
        parts = [
            ("reported", "as reported"),
            ("balance_sheet", "from the balance sheet"),
            ("implied", "net income ÷ EPS"),
            ("conflict", "refused — sources a unit error apart"),
            ("missing", "missing"),
        ]
        theme.note(
            "Where the share counts come from: "
            + " · ".join(f"<b>{src.get(k, 0.0):.1%}</b> {label}" for k, label in parts)
            + ". Many filers footnote their weighted-average shares, which the SEC data sets "
            "don't carry, so the balance sheet fills in — every count cross-checked "
            "(<code>rawtags.reconcile_shares</code>)."
        )
except FileNotFoundError:
    _universe = set()
    st.info("No fundamentals table yet. Run `lti build-fundamentals` (add `--smoke` for a quick subset).")

# --- prices -----------------------------------------------------------------

st.header("Price cache")
if _panel.empty:
    st.info("No prices cached yet. Run `lti fetch-prices`.")
else:
    p = st.columns(6)
    p[0].metric("Tickers", f"{_panel.shape[1]:,}")
    p[1].metric("From", str(_panel.index.min().date()))
    p[2].metric("To", str(_panel.index.max().date()))
    n_close = _px.close.shape[1] if _px is not None else 0
    p[3].metric("Split-adjusted", f"{n_close:,}",
                help="Tickers with a split-adjusted close — the price every valuation uses.")
    p[4].metric("Splits on record", f"{len(_px.splits):,}" if _px is not None else "—",
                help="Used to restate each 10-K's EPS and share count onto today's share basis.")
    p[5].metric("Dividends on record", f"{len(_px.dividends):,}" if _px is not None else "—",
                help="Cash paid per share, ex-date by ex-date — what the yield and the DDM run on.")

    # only companies fetch-prices would fill: warrants, preferreds and tickers
    # without a share count sit in the cache too, but no screen can price them,
    # and a ticker Yahoo has stopped serving ("no_data") can't be backfilled
    _meta = prices._load_meta()
    _gone = set(_meta.loc[_meta["status"] == "no_data", "ticker"]) if not _meta.empty else set()
    unbackfilled = (
        len((_universe & set(_panel.columns)) - set(_px.close.columns) - _gone) if _px is not None else 0
    )
    if unbackfilled:
        st.error(
            f"**{unbackfilled:,} companies have no split-adjusted close.** They are left out of "
            "every screen, backtest and valuation — pairing a back-adjusted price with an "
            "as-reported 10-K makes any company that split later look cheap. "
            "Run `lti fetch-prices` to backfill them."
        )

    last = pd.to_datetime(pd.Series(_panel.apply(lambda c: c.last_valid_index())))
    still_trading = (last >= _panel.index.max() - pd.Timedelta(days=30)).sum()
    st.warning(
        f"**{still_trading:,} of {len(last):,} cached tickers ({still_trading / len(last):.1%}) "
        "are still trading today.** In a real 13-year window roughly half of US listed "
        "companies disappear, so this cache is effectively survivors-only and every "
        "backtest result is optimistic — most of all for deep-value screens, which "
        "select exactly the distressed names that don't come back."
    )
    try:
        from lti import prices as _pm

        meta = _pm._load_meta()
        if not meta.empty:
            with st.expander("Fetch status by ticker"):
                st.dataframe(
                    meta["status"].value_counts().rename_axis("status").reset_index(name="tickers"),
                    hide_index=True,
                )
    except Exception:  # noqa: BLE001
        pass
