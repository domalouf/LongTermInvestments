"""Intrinsic-value models — DCF, Peter Lynch, Graham, DDM, EPV.

Each model turns a company's fundamentals + a price into an estimated fair value
per share; ``*_upside`` is ``fair_value / price - 1``. All functions are
vectorised over pandas Series (index = cik, the same as a fundamentals
snapshot), return ``NaN`` where the model does not apply (negative earnings,
no dividend, …), and never raise on bad inputs.

Growth is the weak link: the flat fundamentals table only carries one year of
lag, so by default we use ``eps_growth_1y`` (then ``revenue_growth_1y``) clipped
to ``[0, growth_cap]``. Pass an explicit ``growth`` Series (e.g. a multi-year
CAGR from :func:`historical_cagr`) for anything more serious.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lti.metrics import _safe_div

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


def _estimate_growth(df: pd.DataFrame, a: ValuationAssumptions) -> pd.Series:
    if a.fixed_growth is not None:
        return pd.Series(float(a.fixed_growth), index=df.index)
    g = pd.Series(np.nan, index=df.index)
    for col in ("eps_growth_1y", "revenue_growth_1y"):
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
) -> pd.DataFrame:
    """Add every model's fair value + ``*_upside``, a blended ``fair_value_est``
    (median of the models that produced a number) and ``fair_value_upside``.

    ``price`` is indexed like ``df`` (by cik).
    """
    a = assumptions or ValuationAssumptions()
    df = df.copy()
    price = price.reindex(df.index).astype("float64")
    df["price"] = price

    shares = df["shares_outstanding"] if "shares_outstanding" in df.columns else None
    eps = df["eps"] if "eps" in df.columns else pd.Series(np.nan, index=df.index)

    bvps = df["book_value_per_share"] if "book_value_per_share" in df.columns else None
    if bvps is None and {"equity", "shares_outstanding"} <= set(df.columns):
        bvps = _safe_div(df["equity"], df["shares_outstanding"])
    if bvps is None:
        bvps = pd.Series(np.nan, index=df.index)

    fcf_ps = (
        _safe_div(df["free_cash_flow"], shares)
        if "free_cash_flow" in df.columns and shares is not None
        else None
    )
    dps = (
        _safe_div(df["dividends_paid"].abs(), shares)
        if "dividends_paid" in df.columns and shares is not None
        else None
    )

    if growth is None:
        growth = _estimate_growth(df, a)
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
        df["ddm_value"] = ddm_value(dps, growth, a)

    present = [m for m in MODELS if m in df.columns]
    df["fair_value_est"] = df[present].replace([np.inf, -np.inf], np.nan).median(axis=1)
    df["n_models"] = df[present].replace([np.inf, -np.inf], np.nan).notna().sum(axis=1)

    for m in [*present, "fair_value_est"]:
        df[f"{m}_upside"] = _safe_div(df[m], price) - 1.0

    return df


def rank_undervalued(
    fund: pd.DataFrame,
    panel: pd.DataFrame,
    asof,
    *,
    assumptions: ValuationAssumptions | None = None,
    market_cap_min: float = 1_000_000_000.0,
    require_positive_eps: bool = True,
    min_models: int = 3,
    min_upside: float = 0.0,
    max_upside: float | None = 5.0,
    min_roe: float | None = None,
    top_n: int | None = 50,
) -> pd.DataFrame:
    """The most undervalued names known at ``asof``, ranked by blended upside.

    Point-in-time snapshot → fundamental + price metrics as of ``asof`` → every
    valuation model → keep rows where at least ``min_models`` produced a number
    and the blended upside is in ``(min_upside, max_upside]`` (the upper bound
    drops data errors), then sort by ``fair_value_est_upside`` descending.

    Valuing as of today against the latest 10-K sidesteps the split-adjustment
    problem that distorts historical multiples: today's adjusted close equals the
    real price and the filing's per-share figures are on a matching basis. A
    split between the last filing and ``asof`` is the residual risk.
    """
    from lti import metrics as metrics_mod, pit, prices as prices_mod

    asof = pd.Timestamp(asof)
    snap = pit.snapshot_asof(fund, asof)
    snap = snap[snap["ticker"].notna()]
    if snap.empty:
        return snap

    snap = metrics_mod.add_fundamental_metrics(snap)
    price_at = pd.Series(
        {cik: prices_mod.price_on_or_before(panel, t, asof) for cik, t in snap["ticker"].items()}
    )
    shares = snap["shares_outstanding"] if "shares_outstanding" in snap.columns else None
    mcap = price_at * shares if shares is not None else None
    snap = metrics_mod.add_price_metrics(snap, price=price_at, market_cap=mcap)

    snap = snap[snap["price"].notna() & (snap["price"] > 0)]
    if "revenues" in snap.columns:  # drop commodity / ETF trusts that file 10-Ks
        snap = snap[snap["revenues"] > 0]
    if market_cap_min and "market_cap" in snap.columns:
        snap = snap[snap["market_cap"] >= market_cap_min]
    if require_positive_eps and "eps" in snap.columns:
        snap = snap[snap["eps"] > 0]
    if min_roe is not None and "roe" in snap.columns:
        snap = snap[snap["roe"] >= min_roe]
    if snap.empty:
        return snap

    v = add_valuation_models(snap, snap["price"], assumptions=assumptions)
    v = v[(v["n_models"] >= min_models) & v["fair_value_est_upside"].notna()]
    v = v[v["fair_value_est_upside"] > min_upside]
    if max_upside is not None:
        v = v[v["fair_value_est_upside"] <= max_upside]
    if v.empty:
        return v

    v = v.sort_values("fair_value_est_upside", ascending=False)
    v.insert(0, "rank", np.arange(1, len(v) + 1))
    return v.head(top_n) if top_n else v
