"""Streamlit entry point — data-status dashboard.

Run with:  streamlit run src/lti/app/Home.py
"""

from __future__ import annotations

import lti.config as config

import pandas as pd
import streamlit as st

st.set_page_config(page_title="LongTermInvestments", page_icon="📈", layout="wide")

st.title("📈 LongTermInvestments")
st.caption("SEC fundamentals screener & strategy backtester")

paths = config.get_paths()
smoke = config.is_smoke()
if smoke:
    st.warning("Running in **smoke mode** (`LTI_SMOKE=1`) — limited data subset.")


def _exists(p) -> str:
    return "✅" if p.exists() else "❌"


st.header("Data artifacts")
rows = [
    ("SEC parquet dir", str(paths.sec_parquet), _exists(paths.sec_parquet)),
    ("SEC index DB", str(paths.sec_db), _exists(paths.sec_db)),
    ("Concatenated standardized BS", str(paths.concat_std_bs), _exists(paths.concat_std_bs)),
    ("fundamentals parquet", str(paths.fundamentals_parquet), _exists(paths.fundamentals_parquet)),
    ("cik→ticker map", str(paths.cik_ticker_parquet), _exists(paths.cik_ticker_parquet)),
    ("price cache", str(paths.adj_close_parquet), _exists(paths.adj_close_parquet)),
]
st.dataframe(
    pd.DataFrame(rows, columns=["artifact", "path", "present"]),
    hide_index=True,
    use_container_width=True,
)

try:
    from lti import sec_update

    st.write("**Latest SEC quarter:**", sec_update.latest_quarter() or "—")
except Exception as exc:  # noqa: BLE001
    st.info(f"SEC index not ready: {exc}")

st.header("Fundamentals coverage")
try:
    from lti import fundamentals

    fund = fundamentals.load_fundamentals()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("rows", f"{len(fund):,}")
    c2.metric("distinct CIK", f"{fund['cik'].nunique():,}")
    c3.metric("CIK with ticker", f"{fund.loc[fund['ticker'].notna(), 'cik'].nunique():,}")
    c4.metric("fiscal years", f"{int(fund['fiscal_year'].min())}–{int(fund['fiscal_year'].max())}")
    st.caption(
        "The gap between *distinct CIK* and *CIK with ticker* is the survivorship-bias hole: "
        "delisted companies have no ticker in the SEC's current `company_tickers.json`."
    )
    st.bar_chart(fund.groupby("fiscal_year").size().rename("filings"))
except FileNotFoundError:
    st.info("No fundamentals table yet. Run `lti build-fundamentals` (add `--smoke` for a quick subset).")

st.header("Price cache")
try:
    from lti import prices

    panel = prices.load_adj_close()
    if panel.empty:
        st.info("No prices cached yet. Run `lti fetch-prices`.")
    else:
        st.write(f"{panel.shape[1]:,} tickers · {panel.index.min().date()} → {panel.index.max().date()}")
        meta = prices._load_meta()
        if not meta.empty:
            st.dataframe(meta["status"].value_counts().rename_axis("status").reset_index(name="count"), hide_index=True)
except Exception as exc:  # noqa: BLE001
    st.info(f"Price cache not ready: {exc}")

st.divider()
st.page_link("pages/1_Screener.py", label="→ Screener", icon="🔎")
st.page_link("pages/2_Backtest.py", label="→ Backtest", icon="🧪")
st.page_link("pages/3_Factor_Analysis.py", label="→ Factor analysis", icon="📐")
st.page_link("pages/4_Stock.py", label="→ Stock detail", icon="🔬")
st.page_link("pages/5_Undervalued.py", label="→ Undervalued today", icon="🎯")
st.page_link("pages/6_Rolling_Backtest.py", label="→ Rolling backtest", icon="🔁")
