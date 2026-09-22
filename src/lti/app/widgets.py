"""Page furniture more than one view needs.

:mod:`lti.app.theme` owns what the app looks like; this module owns the few
controls and blocks the pages would otherwise each keep their own copy of — the
metric picker three pages share, the "hit Run first" gate, the warnings fold,
the model explainer, and the one cached read of the fundamentals table.

The cache matters: every page that calls :func:`fundamentals` shares a single
copy of the table rather than holding one apiece.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

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


def warnings_expander(warnings: list[str]) -> None:
    """A run's warnings, folded away."""
    if warnings:
        with st.expander(f"Warnings ({len(warnings)})"):
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
