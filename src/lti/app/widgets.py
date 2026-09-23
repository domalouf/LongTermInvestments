"""Page furniture more than one view needs.

:mod:`lti.app.theme` owns what the app looks like; this module owns the few
controls and blocks the pages would otherwise each keep their own copy of — the
metric picker three pages share, the turnover buffer and the cost and tax
settings both backtest pages take, the "hit Run first" gate, the warnings fold,
the model explainer, and the one cached read of the fundamentals table.

The cache matters: every page that calls :func:`fundamentals` shares a single
copy of the table rather than holding one apiece.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lti.frictions import DEFAULT_COST_BPS, TaxRates
from lti.metrics import ALL_METRICS, MAGIC_FORMULA_METRICS
from lti.valuation import MODEL_DOCS, MODELS


@st.cache_data(show_spinner=False)
def _fundamentals() -> pd.DataFrame:
    from lti.fundamentals import load_fundamentals

    return load_fundamentals()


def fundamentals() -> pd.DataFrame:
    """The fundamentals table — or a clear message and a stopped page without one."""
    try:
        return _fundamentals()
    except FileNotFoundError:
        st.error("No fundamentals table. Run `lti build-fundamentals` first.")
        st.stop()


def rank_by(default: list[str], *, magic: bool = False) -> tuple[list[str], float]:
    """Which metrics to rank on, and the share of them a company must have to rank.

    Returns the pair a :class:`lti.ranking.ScreenSpec` wants. ``magic`` pins the
    choice to Greenblatt's two and locks the picker; the coverage slider only
    appears once there are enough metrics for partial coverage to mean anything.
    """
    chosen = st.multiselect(
        "Rank by",
        ALL_METRICS,
        default=list(MAGIC_FORMULA_METRICS) if magic else default,
        disabled=magic,
    )
    if magic:
        chosen = list(MAGIC_FORMULA_METRICS)
    if len(chosen) <= 2:
        return chosen, 1.0
    pct = st.slider(
        "Rank companies with at least … % of the metrics", 50, 100, 100, step=10,
        help="100% needs every metric, which shrinks the universe as the list grows. Lower, "
             "a company ranks on the average of the metrics it has.",
    )
    return chosen, pct / 100


def sell_rank(top_n: int) -> int | None:
    """The buffer against turnover: how far a holding may slip before it's sold.

    ``None`` when there is no buffer — a holding goes the moment it leaves the top N.
    """
    rank = st.slider(
        "Sell a holding once it drops out of the top", top_n, 4 * top_n, top_n,
        help=f"At {top_n}, no buffer: whatever leaves the top {top_n} is sold, however narrowly. "
             "Higher, a holding stays until it falls out of this many, and only the places that "
             "frees are refilled — so a name drifting from 28th to 33rd isn't sold and bought back "
             "a year later. Twice the top N is a common choice.",
    )
    return rank if rank > top_n else None


def frictions() -> dict:
    """The sidebar's trading-cost and tax settings, as plain JSON for a cache key.

    :func:`friction_kwargs` turns them back into config fields.
    """
    st.subheader("Costs and taxes")
    cost = st.number_input(
        "Trading cost (bps, each way)", value=DEFAULT_COST_BPS, min_value=0.0, step=5.0,
        help="What a dollar traded loses to the spread and any commission, in hundredths of a "
             "percent: 10 bps on a buy and 10 on the sale. 0 gives the gross backtest.",
    )
    taxable = st.toggle(
        "Taxable account", value=False,
        help="Off: an IRA or 401(k), where nothing is taxed until withdrawal. On: dividends are taxed "
             "as they arrive and gains when a sale realizes them — the strategy, its universe and SPY alike.",
    )
    out = {"cost_bps": cost, "tax": None, "hold_past_one_year": False}
    if taxable:
        d = TaxRates()
        rates = {
            "short_term": st.number_input("Short-term gains tax (%)", 0.0, 60.0, round(d.short_term * 100, 2), 1.0,
                                          help="Held a year or less: taxed as income."),
            "long_term": st.number_input("Long-term gains tax (%)", 0.0, 60.0, round(d.long_term * 100, 2), 1.0),
            "dividends": st.number_input("Dividend tax (%)", 0.0, 60.0, round(d.dividends * 100, 2), 1.0),
        }
        out["tax"] = {k: v / 100 for k, v in rates.items()}
        out["hold_past_one_year"] = st.checkbox(
            "Sell only after a full year", value=False,
            help="An annual rebalance on the first trading day of the month lands on or just short of the "
                 "one-year mark in most years, so the gains are short-term. This waits until a year and a "
                 "day have passed, which drifts the rebalance a few days later each year.",
        )
    return out


def friction_kwargs(raw: dict) -> dict:
    """:func:`frictions`' JSON as :class:`lti.backtest.BacktestConfig` fields."""
    return dict(
        cost_bps=raw["cost_bps"],
        tax=TaxRates(**raw["tax"]) if raw.get("tax") else None,
        hold_past_one_year=raw.get("hold_past_one_year", False),
    )


def ran_with(name: str, clicked: bool, cfg_key: str) -> bool:
    """Whether the ``name`` Run button has been pressed for *this* configuration.

    A button reads True only on the rerun its own click triggers, so the
    configuration it was pressed for is remembered: toggling a chart option
    further down the page doesn't blank it, while changing the strategy asks for
    a fresh run.
    """
    state = f"{name}_run_key"
    if clicked:
        st.session_state[state] = cfg_key
    return st.session_state.get(state) == cfg_key


def run_gate(name: str, clicked: bool, cfg_key: str, prompt: str) -> None:
    """:func:`ran_with`, holding the rest of the page behind ``prompt`` until it is."""
    if not ran_with(name, clicked, cfg_key):
        st.info(prompt)
        st.stop()


def warnings_expander(warnings: list[str], expanded: bool = False) -> None:
    """A run's warnings, folded away — or open, where they are the whole answer."""
    if warnings:
        with st.expander(f"Warnings ({len(warnings)})", expanded=expanded):
            for w in warnings:
                st.text(w)


def model_notes(
    models: list[str] = MODELS,
    arithmetic: dict[str, str] | None = None,
    label: str = "",
) -> None:
    """Each valuation model's equation and caveats, rendered from :data:`MODEL_DOCS`.

    ``arithmetic`` (from :func:`lti.valuation.explain`) adds one company's own
    numbers as the first bullet, under ``label``.
    """
    for m in models:
        doc = MODEL_DOCS[m]
        bullets = [f"- **{label}:** {arithmetic[m]}"] if arithmetic and m in arithmetic else []
        bullets += [
            f"- **At the default assumptions:** {doc.at_defaults}",
            f"- **No value when:** {doc.silent}",
            f"- **Where it misleads:** {doc.misleads}",
        ]
        st.markdown(f"**{doc.label}** — `{doc.formula}`")
        st.markdown(f"{doc.idea} It takes {doc.inputs}\n\n" + "\n".join(bullets))
        st.markdown("")
