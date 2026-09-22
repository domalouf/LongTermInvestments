"""Intrinsic-value models — DCF, Peter Lynch, Graham, DDM, EPV.

Each model turns a company's fundamentals + a price into an estimated fair value
per share; ``*_upside`` is ``fair_value / price - 1``. All functions are
vectorised over pandas Series (index = cik, the same as a fundamentals
snapshot), return ``NaN`` where the model does not apply (negative earnings,
no dividend, …), and never raise on bad inputs.

Six equations, and each one answers a different question:

===================  ==========================================  ======================
model                equation                                    what it needs
===================  ==========================================  ======================
two-stage DCF        Σ FCF·(1+g)ᵗ/(1+r)ᵗ + terminal/(1+r)ᴺ       FCF per share
Peter Lynch          EPS · (growth% + dividend yield%)           EPS, growth
Graham number        √(22.5 · EPS · BVPS)                        EPS, book value
Graham revised       EPS · (8.5 + 2g) · 4.4/Y                    EPS, growth
dividend discount    D₀·(1+g) / (r − g)                          dividends paid
earnings power       EPS / r                                     EPS
===================  ==========================================  ======================

Each function's docstring works its equation through in full — where the
constants come from, what the model takes on faith, when it declines to answer.
:data:`MODEL_DOCS` carries the same explanations as data, for the pages and the
published HTML to render, and :func:`explain` prints one company's own numbers
in place of the symbols.

:func:`add_valuation_models` runs all six and blends them into
``fair_value_est``, the *median* of those that produced a number — a median so
that one model's extreme can't set the answer. Note that four of the six
multiply the same EPS, so six values agreeing is not six independent opinions;
the DCF (cash flow) and the DDM (dividends actually paid) are the two carrying
separate evidence.

The models run on one of two earnings bases:

``normalized`` (the default)
    the median of the last five years' EPS and free cash flow, with growth
    from the five-year revenue trend (:mod:`lti.history`). One year's earnings
    can be a cyclical peak or a one-off, and every model multiplies it — so on
    the latest year alone, the most "undervalued" names were mostly Lyft's
    tax-asset release, Uniti's merger gain and Cal-Maine's egg-price year.
``latest``
    the latest 10-K alone, growth from its one-year change — what this
    module did before, kept for comparison.

Dividends are the exception to "from the filings": the models take the
trailing twelve months of actual payments when the frame carries them, since
the cash-flow tag that would otherwise supply them is sparse and stale.

Growth is clipped to ``[0, growth_cap]`` either way; pass an explicit
``growth`` Series (e.g. :func:`historical_cagr`) to override it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lti.metrics import _col, _safe_div

if TYPE_CHECKING:
    from lti.prices import PriceData

# A blended fair value needs at least this many models behind it, and an upside
# beyond MAX_UPSIDE is treated as a data error rather than a bargain.
MIN_MODELS = 3
MAX_UPSIDE = 5.0

BASES = ("normalized", "latest")

# fair-value columns added by add_valuation_models, in display order
MODELS = [
    "dcf_value",
    "lynch_fair_value",
    "graham_number",
    "graham_intrinsic",
    "ddm_value",
    "epv_value",
]


@dataclass
class ValuationAssumptions:
    discount_rate: float = 0.09      # required return / cost of equity
    terminal_growth: float = 0.025   # perpetual growth after the explicit window
    dcf_years: int = 10              # length of the explicit DCF window
    growth_cap: float = 0.15         # cap on the extrapolated growth rate
    lynch_growth_cap: float = 0.25   # Lynch fair P/E tops out here (+ div yield)
    bond_yield: float = 0.045        # AAA yield for Graham's revised formula
    fixed_growth: float | None = None  # override the per-company growth estimate

    def __post_init__(self) -> None:
        # terminal growth must stay below the discount rate for a finite value
        self.terminal_growth = min(self.terminal_growth, self.discount_rate - 0.01)


@dataclass(frozen=True)
class ModelDoc:
    """One model explained in one place.

    The Stock page, the Undervalued page, the published ``index.html`` and the
    README all render these instead of keeping their own wording, so an
    explanation can't drift away from the equation it describes.
    """

    label: str        # display name
    formula: str      # the equation, in symbols
    idea: str         # the reasoning the equation encodes, in one sentence
    inputs: str       # what goes in, and where each input comes from
    at_defaults: str  # what the equation works out to under ValuationAssumptions()
    silent: str       # when it declines to produce a number (NaN)
    misleads: str     # where the equation breaks down


# Keyed by fair-value column, in MODELS (display) order. Every multiple quoted
# in ``at_defaults`` is at ValuationAssumptions() — r 9%, terminal 2.5%, N 10,
# growth capped at 15%, AAA yield 4.5%.
MODEL_DOCS: dict[str, ModelDoc] = {
    "dcf_value": ModelDoc(
        label="Two-stage DCF",
        formula=(
            "value = Σ FCF·(1+g)^t / (1+r)^t  for t = 1…N,  "
            "plus  [FCF·(1+g)^N·(1+g_term) / (r − g_term)] / (1+r)^N"
        ),
        idea=(
            "A share is worth the cash the business will hand its owners, with cash further out "
            "worth less than cash now."
        ),
        inputs=(
            "free cash flow per share — the normalized median FCF over today's share count, or "
            "operating cash flow − capex from the latest 10-K — grown at g (clipped to [0, growth_cap]) "
            "for N years, then forever at g_term, all discounted at the required return r."
        ),
        at_defaults=(
            "ten years of cash plus a perpetuity: about 19× today's FCF per share for a 5% grower, "
            "13× for a flat one."
        ),
        silent="FCF per share ≤ 0 — a business burning cash has no cash to discount.",
        misleads=(
            "The terminal lump is most of the answer — ≈57% of it for a 5% grower — so the value is "
            "largely a bet on the perpetuity, and a point off r or onto g_term moves it more than the "
            "whole first stage does. It also values today's share count: future dilution is invisible."
        ),
    ),
    "lynch_fair_value": ModelDoc(
        label="Peter Lynch",
        formula="value = EPS × (growth% + dividend yield%)",
        idea=(
            "Lynch's rule of thumb: a growth company is fairly priced at a P/E equal to its growth rate "
            "(PEG = 1), and a dividend is return you collect without needing growth, so the yield adds to it."
        ),
        inputs=(
            "EPS (normalized or latest-year), the growth estimate in percentage points (clipped to "
            "[0, lynch_growth_cap]) and the trailing dividend yield in percentage points."
        ),
        at_defaults=(
            "a fair P/E of growth + yield: 14× earnings for a 12% grower yielding 2%, 25× at the "
            "25% growth cap, and for a flat non-payer, nothing at all."
        ),
        silent="EPS ≤ 0.",
        misleads=(
            "It is a heuristic, not a valuation: no discount rate, no horizon, no balance sheet. Growth is "
            "the whole answer, which makes it the most sensitive of the six to a growth estimate that here "
            "comes from a five-year revenue trend rather than the analyst forecasts Lynch was reading. A "
            "profitable company with no growth and no dividend fairly values at $0 — the rule showing its "
            "limits rather than a data error, and a $0 that still counts as a model in the blend."
        ),
    ),
    "graham_number": ModelDoc(
        label="Graham number",
        formula="value = √(22.5 × EPS × book value per share) = √(15·EPS × 1.5·BVPS)",
        idea=(
            "Graham's two defensive-investor screens — pay no more than 15× earnings and no more than "
            "1.5× book — combined into one ceiling that lets a company be dearer on one if it is cheaper "
            "on the other (P/E × P/B ≤ 22.5)."
        ),
        inputs="EPS and book value per share (equity ÷ shares outstanding).",
        at_defaults=(
            "the geometric mean of '15× earnings' and '1.5× book': 15× EPS for a company earning 10% on "
            "its book value, and less than that for one earning more on less book."
        ),
        silent="EPS or book value per share ≤ 0 — the product under the root has to be positive.",
        misleads=(
            "It is a ceiling for a defensive buy, not an estimate of worth, and it is half a book-value "
            "measure: it understates asset-light businesses whose R&D and brands are expensed rather than "
            "capitalised, and flatters ones carrying goodwill from acquisitions that didn't work."
        ),
    ),
    "graham_intrinsic": ModelDoc(
        label="Graham revised",
        formula="value = EPS × (8.5 + 2g) × 4.4 / Y      (g in percentage points, Y = AAA yield in percent)",
        idea=(
            "Graham's 1962 multiple table: 8.5× earnings for a company expected to grow not at all, plus "
            "2 more turns of P/E for every point of annual growth, the whole thing rescaled for interest "
            "rates — 4.4% was the AAA corporate yield when he wrote it down."
        ),
        inputs="EPS, the growth estimate in percentage points (clipped to [0, 20]) and bond_yield (floored at 0.5%).",
        at_defaults=(
            "at a 4.5% AAA yield the rate term is 0.98, so essentially Graham's own table: 8.3× earnings at "
            "no growth, 18.1× at 5%, 37.6× at the 15% growth cap."
        ),
        silent="EPS ≤ 0.",
        misleads=(
            "Graham later disowned it as too crude, and it is close to linear in g — two turns of P/E per "
            "growth point is generous at the top of the range, which is what the growth cap is holding back. "
            "The rate term also cuts both ways: it marks every company up as bond yields fall."
        ),
    ),
    "ddm_value": ModelDoc(
        label="Dividend discount",
        formula="value = D1 / (r − g) = dividend × (1 + g) / (r − g)",
        idea=(
            "Gordon growth: value only the cash actually paid out, as a perpetuity growing at g and "
            "discounted at the required return r."
        ),
        inputs=(
            "the last twelve months of dividends actually paid (from the price cache, not the sparse "
            "cash-flow tag), grown at the dividend's own five-year rate where there is one — a company can "
            "grow revenue and hold its payout flat — and clipped to [0, terminal_growth]."
        ),
        at_defaults="15.8× the trailing dividend (growth capped at 2.5%, discounted at 9%).",
        silent="no dividend — which is most of the market, and why the blend needs a minimum model count.",
        misleads=(
            "Capping perpetual growth at the terminal rate is what keeps the formula finite as g approaches r, "
            "but it values a genuine dividend grower as if it grew 2.5% forever, so this is usually the lowest "
            "of the six. It also ignores retained earnings and buybacks: a company returning its cash by "
            "repurchase is worth nothing here."
        ),
    ),
    "epv_value": ModelDoc(
        label="Earnings power",
        formula="value = EPS / r",
        idea=(
            "Greenwald's Earnings Power Value: growth is the least knowable input, so drop it and capitalise "
            "what the business earns if it never grows at all."
        ),
        inputs="EPS (normalized or latest-year) and the discount rate r.",
        at_defaults="a flat 11.1× earnings (1 ÷ 9%) — 14.3× at a 7% required return, 8.3× at 12%.",
        silent="EPS ≤ 0.",
        misleads=(
            "Greenwald's version normalizes earnings across a full cycle and adjusts for maintenance capex, "
            "excess cash and one-offs; this one capitalises reported EPS as it stands. Being a single "
            "multiple, it is also a pure restatement of the discount rate you chose."
        ),
    ),
}


# --- individual models --------------------------------------------------


def graham_number(eps: pd.Series, bvps: pd.Series) -> pd.Series:
    """√(22.5 · EPS · book value per share) — Graham's ceiling for a defensive buy.

    The *Intelligent Investor* asks a defensive buyer for two things at once: no
    more than 15× earnings and no more than 1.5× book value. Multiply the two
    limits and the constant is 22.5 — the equation is just that pair of screens
    rearranged, so a company may be dearer on one where it is cheaper on the
    other::

        price ≤ 15 · EPS   and   price ≤ 1.5 · BVPS
        → price² ≤ 22.5 · EPS · BVPS
        → price ≤ √(15·EPS × 1.5·BVPS)

    Which makes the result the geometric mean of the two ceilings, not their
    average: it lands at 15× EPS for a company earning 10% on book, and below
    that for one earning more on less book — why this is usually the lowest of
    the six for an asset-light business.

    NaN unless both inputs are positive; with either negative the product under
    the root is meaningless rather than small.
    """
    x = 22.5 * eps * bvps
    return np.sqrt(x.where(x > 0))


def graham_intrinsic(eps: pd.Series, growth: pd.Series, bond_yield: float) -> pd.Series:
    """Graham's revised formula: ``EPS · (8.5 + 2g) · 4.4 / Y``.

    Three pieces. ``8.5`` is the P/E Graham put on a company expected to grow
    not at all. ``2g`` adds two turns of that multiple for every point of
    expected annual growth (``g`` in *percentage points* — 5 for 5%, not 0.05),
    which is where nearly all the variation comes from. ``4.4 / Y`` rescales the
    lot for interest rates: 4.4% was the AAA corporate yield in 1962, ``Y`` is
    today's in percent, so the same earnings are worth less when bonds pay more.

    Growth is clipped to [0, 20] points and ``Y`` floored at 0.5%, which is what
    keeps a runaway growth estimate or a zero yield from producing an absurd
    multiple. At the default 4.5% yield the rate term is 0.98, leaving roughly
    Graham's own table: 8.3× earnings at no growth, 18.1× at 5%, 37.6× at 15%.

    NaN on non-positive EPS. Graham himself came to think the formula too
    crude to rely on; it is here as one vote among six, not as an answer.
    """
    g_pts = (growth.fillna(0.0) * 100).clip(0, 20)
    y_pct = max(bond_yield, 0.005) * 100
    return eps.where(eps > 0) * (8.5 + 2 * g_pts) * 4.4 / y_pct


def lynch_fair_value(
    eps: pd.Series, growth: pd.Series, dividend_yield: pd.Series | None, growth_cap: float
) -> pd.Series:
    """Peter Lynch: ``fair value = EPS · (g% + dividend yield%)``.

    *One Up on Wall Street*'s rule of thumb — a growth company is fairly priced
    when its P/E equals its growth rate (PEG = 1) — with the dividend yield
    added, since a payout is return that arrives whether or not the growth does.
    A 12% grower yielding 2% earns a fair P/E of 14; on $3 of EPS that is $42.

    Both rates are in percentage points, and growth is clipped to
    [0, ``growth_cap``] — here ``lynch_growth_cap`` (25%), looser than the cap
    the other models get, so it only binds when you hand the models a growth
    rate of your own. The estimate they use by default is already clipped lower.

    NaN on non-positive EPS. A profitable company with no growth and no
    dividend fairly values at $0: the rule says a business going nowhere and
    paying nothing out is worth nothing, which is the rule showing its limits
    rather than a data error.
    """
    g_pts = (growth.fillna(0.0).clip(0, growth_cap)) * 100
    dy_pts = (dividend_yield.fillna(0.0) * 100) if dividend_yield is not None else 0.0
    return eps.where(eps > 0) * (g_pts + dy_pts)


def two_stage_dcf(fcf_per_share: pd.Series, growth: pd.Series, a: ValuationAssumptions) -> pd.Series:
    """Present value of ``dcf_years`` of free cash flow per share, plus a perpetuity.

    Stage one walks the cash forward and discounts it back, year by year: each
    year's FCF/share is the last one grown at ``g`` (clipped to
    [0, ``growth_cap``]), and year ``t``'s cash is worth ``1 / (1 + r)^t`` of
    its face value today. Stage two assumes the business then settles into
    growing at ``terminal_growth`` forever, and capitalises that with Gordon
    growth — ``FCF_N · (1 + g_term) / (r − g_term)`` — a lump sitting at year
    ``N``, so it is discounted back ``N`` years as well::

        Σ FCF·(1+g)^t / (1+r)^t   for t = 1…N
        + [FCF_N·(1+g_term) / (r − g_term)] / (1+r)^N

    ``ValuationAssumptions`` forces ``terminal_growth`` at least a point below
    ``discount_rate``; without that the denominator goes to zero and the value
    to infinity. At the defaults the whole thing comes to about 19× current
    FCF/share for a 5% grower and 13× for a flat one — of which the terminal
    lump is ≈57% and ≈51% respectively. That is the model's weak point: most of
    the answer is the part furthest out and least knowable, and it moves more
    on a point of ``r`` than on the entire first stage.

    NaN where FCF per share is not positive: there is no cash to discount, and
    projecting a burn forward would only produce a confident negative.
    """
    r, gt, n = a.discount_rate, a.terminal_growth, a.dcf_years
    g1 = growth.fillna(0.0).clip(0, a.growth_cap)

    pv = pd.Series(0.0, index=fcf_per_share.index)
    cf = fcf_per_share.astype("float64").copy()
    for t in range(1, n + 1):
        cf = cf * (1 + g1)
        pv = pv + cf / (1 + r) ** t
    terminal = cf * (1 + gt) / (r - gt)
    pv = pv + terminal / (1 + r) ** n
    return pv.where(fcf_per_share > 0)


def ddm_value(dps: pd.Series, growth: pd.Series, a: ValuationAssumptions) -> pd.Series:
    """Gordon growth on the dividend: ``D1 / (r − g)``, where ``D1 = D0 · (1 + g)``.

    The sum of a dividend growing at ``g`` forever, discounted at ``r``, closes
    to that one term. ``D0`` is the trailing twelve months of payments;
    grow it one year to get next year's, then capitalise the stream at the
    spread between the required return and the growth rate — a spread of 6.5
    points at the defaults, so 15.8× the current dividend.

    That spread is the whole model, and it is why growth is clipped to
    [0, ``terminal_growth``]: as ``g`` approaches ``r`` the denominator goes to
    zero and the value to infinity, and a dividend cannot outgrow its discount
    rate forever in any case. The cap is conservative on purpose — it values a
    genuine dividend grower as if it grew 2.5% a year — which is why this is
    usually the lowest of the six.

    NaN for a non-payer, which is most of the market: the model has nothing to
    value, and 0 would be a claim rather than an abstention. Buybacks and
    retained earnings are invisible to it either way.
    """
    g = growth.fillna(0.0).clip(0.0, a.terminal_growth)
    d1 = dps * (1 + g)
    return (d1 / (a.discount_rate - g)).where(dps > 0)


def epv_value(eps: pd.Series, discount_rate: float) -> pd.Series:
    """Earnings Power Value — no-growth capitalised earnings, ``EPS / r``.

    Greenwald's argument is that growth is the least knowable input in any
    valuation, so the sturdier question is what the business is worth if it
    never grows at all: a perpetuity of today's earnings, which is ``EPS / r``.
    That is a single multiple — 11.1× at the default 9% required return, 14.3×
    at 7%, 8.3× at 12% — so the number says as much about the discount rate you
    chose as about the company.

    Simplified from the textbook version, which normalizes earnings over a full
    cycle and adjusts for maintenance capex, excess cash and one-offs; on the
    ``normalized`` basis the five-year median EPS stands in for the first of
    those and nothing stands in for the rest. NaN on non-positive EPS.
    """
    return eps.where(eps > 0) / discount_rate


# --- growth helpers ----------------------------------------------------


def _estimate_growth(df: pd.DataFrame, a: ValuationAssumptions, basis: str = "normalized") -> pd.Series:
    if a.fixed_growth is not None:
        return pd.Series(float(a.fixed_growth), index=df.index)
    cols = ("revenue_cagr",) if basis == "normalized" else ("eps_growth_1y", "revenue_growth_1y")
    g = pd.Series(np.nan, index=df.index)
    for col in cols:
        if col in df.columns:
            g = g.fillna(df[col])
    return g.clip(lower=0.0, upper=a.growth_cap)


def historical_cagr(annual: pd.DataFrame, column: str = "eps", years: int = 5) -> float:
    """Trailing CAGR of ``column`` from an annual frame (sorted by period), using
    the last ``years`` steps. NaN if either endpoint is non-positive."""
    if annual.empty or column not in annual.columns:
        return float("nan")
    s = annual.sort_values("period_end")[column].dropna()
    if len(s) < 2:
        return float("nan")
    s = s.iloc[-(years + 1):]
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    n = len(s) - 1
    if first <= 0 or last <= 0 or n <= 0:
        return float("nan")
    return (last / first) ** (1 / n) - 1


# --- explaining one company's number -----------------------------------


def _money(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{float(x) * 100:.{digits}f}%"


def explain(model: str, row: pd.Series, assumptions: ValuationAssumptions | None = None) -> str:
    """``model``'s equation with one company's own numbers substituted into it.

    ``row`` is a row of an :func:`add_valuation_models` frame — the ``*_used``
    columns it carries are the inputs that model actually ran on, after every
    clip. Returns a line like ``√(22.5 × EPS $3.05 × BVPS $12.40) = $29.18``;
    an input the frame doesn't carry shows as ``—``, and a model that stayed
    silent says why, in the words of its :data:`MODEL_DOCS` entry.

    Presentation lives here rather than in the pages so the Stock page, the
    Undervalued page and the published HTML show the same arithmetic.
    """
    if model not in MODEL_DOCS:
        raise KeyError(f"unknown model {model!r} — expected one of {list(MODEL_DOCS)}")
    a = assumptions or ValuationAssumptions()
    doc = MODEL_DOCS[model]
    value = row.get(model)

    eps, bvps = row.get("eps_used"), row.get("bvps_used")
    fcf_ps, dps = row.get("fcf_ps_used"), row.get("dps_used")
    g = row.get("est_growth")
    g = 0.0 if pd.isna(g) else float(g)
    dy = row.get("dividend_yield")
    dy = 0.0 if dy is None or pd.isna(dy) else float(dy)
    g_ddm = row.get("ddm_growth_used")
    if g_ddm is None or pd.isna(g_ddm):
        g_ddm = min(max(g, 0.0), a.terminal_growth)

    if model == "dcf_value":
        lhs = (f"{a.dcf_years} years of FCF/share {_money(fcf_ps)} growing {_pct(min(max(g, 0.0), a.growth_cap))}, "
               f"then {_pct(a.terminal_growth)} forever, discounted at {_pct(a.discount_rate)}")
    elif model == "lynch_fair_value":
        g_l = min(max(g, 0.0), a.lynch_growth_cap) * 100
        lhs = f"EPS {_money(eps)} × (growth {g_l:.1f} + yield {dy * 100:.1f})"
    elif model == "graham_number":
        lhs = f"√(22.5 × EPS {_money(eps)} × BVPS {_money(bvps)})"
    elif model == "graham_intrinsic":
        lhs = (f"EPS {_money(eps)} × (8.5 + 2 × {min(max(g, 0.0), 0.20) * 100:.1f}) "
               f"× 4.4 / {a.bond_yield * 100:.1f}")
    elif model == "ddm_value":
        lhs = (f"dividend {_money(dps)} × (1 + {_pct(g_ddm)}) / "
               f"({_pct(a.discount_rate)} − {_pct(g_ddm)})")
    else:  # epv_value
        lhs = f"EPS {_money(eps)} / {_pct(a.discount_rate)}"

    if value is None or pd.isna(value):
        return f"{lhs} — no value: {doc.silent}"
    return f"{lhs} = {_money(value)}"


# --- public API ------------------------------------------------------


def add_valuation_models(
    df: pd.DataFrame,
    price: pd.Series,
    *,
    assumptions: ValuationAssumptions | None = None,
    growth: pd.Series | None = None,
    basis: str = "normalized",
) -> pd.DataFrame:
    """Add every model's fair value + ``*_upside``, a blended ``fair_value_est``
    (median of the models that produced a number) and ``fair_value_est_upside``.

    ``price`` is indexed like ``df`` (by cik). ``basis`` picks the earnings the
    models run on (see the module docstring); ``normalized`` needs the
    :func:`lti.history.add_history` columns.

    The blend is a median rather than a mean so that one model's extreme can't
    set the answer — but the models are not six independent opinions: four of
    them (Lynch, both Grahams, EPV) multiply the same EPS, so their agreement
    mostly restates that one number. The DCF runs on cash flow and the DDM on
    dividends actually paid; those two are the ones carrying separate evidence.

    Alongside the values it records the inputs they ran on — ``eps_used``,
    ``bvps_used``, ``fcf_ps_used``, ``dps_used``, ``est_growth``,
    ``ddm_growth_used`` and ``dividend_yield`` — so a fair value can be checked
    against the numbers that produced it. :func:`explain` renders exactly that,
    one model at a time.
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, not {basis!r}")
    if basis == "normalized" and "eps_norm" not in df.columns:
        raise KeyError("normalized valuation needs the history columns — see lti.history.add_history")
    a = assumptions or ValuationAssumptions()
    df = df.copy()
    price = price.reindex(df.index).astype("float64")
    df["price"] = price

    shares = df["shares_outstanding"] if "shares_outstanding" in df.columns else None
    eps_col = "eps_norm" if basis == "normalized" else "eps"
    eps = _col(df, eps_col)

    bvps = _col(df, "book_value_per_share")
    if "book_value_per_share" not in df.columns and {"equity", "shares_outstanding"} <= set(df.columns):
        bvps = _safe_div(df["equity"], df["shares_outstanding"])

    if basis == "normalized":
        fcf_ps = df["fcf_ps_norm"] if "fcf_ps_norm" in df.columns else None
    else:
        fcf_ps = (
            _safe_div(df["free_cash_flow"], shares)
            if "free_cash_flow" in df.columns and shares is not None
            else None
        )
    # Dividends per share: the last twelve months of actual payments when the
    # frame carries them (``dps_ttm``, from :func:`lti.pit.priced_snapshot`),
    # else the cash-flow statement's total over the share count — a tag only a
    # quarter of filers report, up to a year stale, preferred lumped in with
    # common.
    if "dps_ttm" in df.columns:
        dps = df["dps_ttm"].astype("float64")
    elif "dividends_paid" in df.columns and shares is not None:
        dps = _safe_div(df["dividends_paid"].abs(), shares)
    else:
        dps = None

    if growth is None:
        growth = _estimate_growth(df, a, basis)
    growth = growth.reindex(df.index)
    df["est_growth"] = growth

    div_yield = _safe_div(dps, price) if dps is not None else None
    if div_yield is not None:
        df["dividend_yield"] = div_yield

    # The inputs as fed to the models, whichever basis and columns they came
    # from — an audit trail for the values below, and what explain() reads.
    df["eps_used"] = eps
    df["bvps_used"] = bvps
    df["fcf_ps_used"] = fcf_ps if fcf_ps is not None else np.nan
    df["dps_used"] = dps if dps is not None else np.nan

    df["graham_number"] = graham_number(eps, bvps)
    df["graham_intrinsic"] = graham_intrinsic(eps, growth, a.bond_yield)
    df["lynch_fair_value"] = lynch_fair_value(eps, growth, div_yield, a.lynch_growth_cap)
    df["epv_value"] = epv_value(eps, a.discount_rate)
    if fcf_ps is not None:
        df["dcf_value"] = two_stage_dcf(fcf_ps, growth, a)
    if dps is not None:
        # Gordon growth grows the *dividend*, so use the dividend's own rate
        # where one is known — a company can grow revenue and hold its payout
        # flat. The earnings-side estimate stands in otherwise, and the model
        # caps either at the terminal rate.
        d_growth = growth
        if "dividend_growth_5y" in df.columns:
            d_growth = df["dividend_growth_5y"].reindex(df.index).fillna(growth)
        df["ddm_growth_used"] = d_growth.clip(lower=0.0, upper=a.terminal_growth)
        df["ddm_value"] = ddm_value(dps, d_growth, a)

    present = [m for m in MODELS if m in df.columns]
    df["fair_value_est"] = df[present].replace([np.inf, -np.inf], np.nan).median(axis=1)
    df["n_models"] = df[present].replace([np.inf, -np.inf], np.nan).notna().sum(axis=1)

    for m in [*present, "fair_value_est"]:
        df[f"{m}_upside"] = _safe_div(df[m], price) - 1.0

    return df


def add_fair_value_metrics(snap: pd.DataFrame, assumptions: ValuationAssumptions | None = None) -> pd.DataFrame:
    """``fair_value_upside`` (normalized earnings) and ``fair_value_upside_1y``
    (the latest year alone) as rankable metrics.

    Both follow the Undervalued page's credibility rules: a blend of at least
    :data:`MIN_MODELS` models, and no upside beyond :data:`MAX_UPSIDE` — above
    that it's a data error, and a ranking would put it first.
    """
    out = snap.copy()
    for name, basis in (("fair_value_upside", "normalized"), ("fair_value_upside_1y", "latest")):
        if basis == "normalized" and "eps_norm" not in out.columns:
            continue
        v = add_valuation_models(snap, snap["price"], assumptions=assumptions, basis=basis)
        upside = v["fair_value_est_upside"]
        out[name] = upside.where((v["n_models"] >= MIN_MODELS) & (upside <= MAX_UPSIDE))
    return out


def rank_undervalued(
    fund: pd.DataFrame,
    px: "PriceData",
    asof,
    *,
    assumptions: ValuationAssumptions | None = None,
    market_cap_min: float = 1_000_000_000.0,
    require_positive_eps: bool = True,
    min_models: int = MIN_MODELS,
    min_profit_years: int | None = 4,
    min_upside: float = 0.0,
    max_upside: float | None = MAX_UPSIDE,
    min_roe: float | None = None,
    exclude_financials: bool = True,
    top_n: int | None = 50,
    basis: str = "normalized",
) -> pd.DataFrame:
    """The most undervalued names known at ``asof``, ranked by blended upside.

    Split-correct priced snapshot of operating companies with their last five
    years of filings (:func:`lti.pit.priced_snapshot`) → every valuation model
    on the chosen earnings ``basis`` → keep companies profitable in at least
    ``min_profit_years`` of those years (a stable earner rather than one good
    year), with at least ``min_models`` models producing a number and a blended
    upside in ``(min_upside, max_upside]`` (the upper bound drops data errors),
    then sort by ``fair_value_est_upside`` descending.

    ``require_positive_eps`` asks for positive EPS both in the latest year and
    on the basis the models run on: a company losing money today isn't
    undervalued on the strength of its better years. Financials — and business
    development companies, which have no SIC — are out by default: Graham, DDM
    and EPV all assume an operating business, and a lender's balance sheet is
    its product.
    """
    from lti import pit

    asof = pd.Timestamp(asof)
    snap = pit.priced_snapshot(fund, asof, px, with_history=True)
    if snap.empty:
        return snap

    snap = snap[snap["price"].notna() & (snap["price"] > 0)]
    if market_cap_min and "market_cap" in snap.columns:
        snap = snap[snap["market_cap"] >= market_cap_min]
    if require_positive_eps:
        for col in ("eps", "eps_norm" if basis == "normalized" else "eps"):
            if col in snap.columns:
                snap = snap[snap[col] > 0]
    if exclude_financials:
        from lti import sectors

        snap = sectors.drop_financials(snap)
    if min_profit_years is not None and "profit_years" in snap.columns:
        snap = snap[snap["profit_years"] >= min_profit_years]
    if min_roe is not None and "roe" in snap.columns:
        snap = snap[snap["roe"] >= min_roe]
    if snap.empty:
        return snap

    v = add_valuation_models(snap, snap["price"], assumptions=assumptions, basis=basis)
    v = v[(v["n_models"] >= min_models) & v["fair_value_est_upside"].notna()]
    v = v[v["fair_value_est_upside"] > min_upside]
    if max_upside is not None:
        v = v[v["fair_value_est_upside"] <= max_upside]
    if v.empty:
        return v

    v = v.sort_values("fair_value_est_upside", ascending=False)
    v.insert(0, "rank", np.arange(1, len(v) + 1))
    return v.head(top_n) if top_n else v
