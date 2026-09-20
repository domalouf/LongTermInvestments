"""Intrinsic-value models — DCF, Peter Lynch, Graham, DDM, EPV.

Each model turns a company's fundamentals + a price into an estimated fair value
per share; ``*_upside`` is ``fair_value / price - 1``. All functions are
vectorised over pandas Series (index = cik, the same as a fundamentals
snapshot), return ``NaN`` where the model does not apply (negative earnings,
no dividend, …), and never raise on bad inputs.

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

from lti.metrics import _safe_div

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


# --- individual models --------------------------------------------------


def graham_number(eps: pd.Series, bvps: pd.Series) -> pd.Series:
    """√(22.5 · EPS · book value per share) — Graham's ceiling for a defensive buy
    (22.5 = 15× earnings × 1.5× book)."""
    x = 22.5 * eps * bvps
    return np.sqrt(x.where(x > 0))


def graham_intrinsic(eps: pd.Series, growth: pd.Series, bond_yield: float) -> pd.Series:
    """Graham's revised formula: EPS · (8.5 + 2g) · 4.4 / Y, with g in percentage
    points (capped at 20) and Y the AAA bond yield in percent."""
    g_pts = (growth.fillna(0.0) * 100).clip(0, 20)
    y_pct = max(bond_yield, 0.005) * 100
    return eps.where(eps > 0) * (8.5 + 2 * g_pts) * 4.4 / y_pct


def lynch_fair_value(
    eps: pd.Series, growth: pd.Series, dividend_yield: pd.Series | None, growth_cap: float
) -> pd.Series:
    """Peter Lynch: a fair P/E equals the earnings growth rate in percent, plus
    the dividend yield (``fair value = EPS · (g% + div_yield%)``)."""
    g_pts = (growth.fillna(0.0).clip(0, growth_cap)) * 100
    dy_pts = (dividend_yield.fillna(0.0) * 100) if dividend_yield is not None else 0.0
    return eps.where(eps > 0) * (g_pts + dy_pts)


def two_stage_dcf(fcf_per_share: pd.Series, growth: pd.Series, a: ValuationAssumptions) -> pd.Series:
    """PV of ``dcf_years`` of FCF/share growing at ``growth`` (capped), plus a
    Gordon terminal value at ``terminal_growth``, discounted at ``discount_rate``."""
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
    """Gordon growth: ``D1 / (r - g)``. Perpetual dividend growth is capped at the
    terminal rate — near ``r`` the formula explodes and is meaningless. Payers only."""
    g = growth.fillna(0.0).clip(0.0, a.terminal_growth)
    d1 = dps * (1 + g)
    return (d1 / (a.discount_rate - g)).where(dps > 0)


def epv_value(eps: pd.Series, discount_rate: float) -> pd.Series:
    """Earnings Power Value — no-growth capitalised earnings, ``EPS / r``."""
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
    eps = df[eps_col] if eps_col in df.columns else pd.Series(np.nan, index=df.index)

    bvps = df["book_value_per_share"] if "book_value_per_share" in df.columns else None
    if bvps is None and {"equity", "shares_outstanding"} <= set(df.columns):
        bvps = _safe_div(df["equity"], df["shares_outstanding"])
    if bvps is None:
        bvps = pd.Series(np.nan, index=df.index)

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

        if "is_financial" in snap.columns:
            snap = snap[~snap["is_financial"].fillna(False).astype(bool)]
        if "sic" in snap.columns:  # BDCs and other investment companies carry no SIC
            snap = snap[~sectors.is_investment_company(snap["sic"])]
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
