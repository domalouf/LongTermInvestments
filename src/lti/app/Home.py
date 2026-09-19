"""Streamlit entry point.

Run with:  streamlit run src/lti/app/Home.py

This script owns the page config and the navigation; each view lives in
``views/`` and renders itself. The order below is the order in the sidebar, and
"Undervalued today" leads because it is the point of the app.

The views deliberately sit in ``views/`` rather than ``pages/``: Streamlit
auto-routes any ``pages/`` directory by filename, which would serve a view
directly and skip this script — losing the grouped navigation and the shared
stylesheet along with it.
"""

from __future__ import annotations

import streamlit as st

from lti.app import theme

theme.configure_page("LongTermInvestments", "🎯")

nav = st.navigation(
    {
        "Find something to buy": [
            st.Page("views/undervalued.py", title="Undervalued today", icon="🎯", default=True),
            st.Page("views/stock.py", title="Stock detail", icon="🔬"),
        ],
        "Test an idea": [
            st.Page("views/screener.py", title="Screener", icon="🔎"),
            st.Page("views/backtest.py", title="Backtest", icon="🧪"),
            st.Page("views/factor_analysis.py", title="Factor analysis", icon="📐"),
        ],
        "Keep score": [
            st.Page("views/track.py", title="Track record", icon="📒"),
            st.Page("views/journal.py", title="Decision journal", icon="✍️"),
        ],
        "Housekeeping": [
            st.Page("views/data_health.py", title="Data health", icon="🩺"),
        ],
    }
)
nav.run()
