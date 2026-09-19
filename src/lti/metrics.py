"""Per-stock metric computation.

Fundamental metrics depend only on the ``fundamentals`` frame. Price metrics
(P/E, P/B, earnings yield, market cap) additionally need a price and an as-of
date, so they are computed separately — the backtest and screener pass in the
price that was known at the relevant date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNDAMENTAL_METRICS = [
    "debt_to_equity",
    "current_ratio",
    "roe",
    "roic",
    "net_margin",
    "gross_margin",
    "revenue_growth_1y",
    "eps_growth_1y",
    "fcf_margin",
    "op_profitability",
    "cash_profitability",
    "accruals",
]

PRICE_METRICS = [
    "pe",
    "pb",
    "earnings_yield",
    "ebit_ev",
    "peg",
    "fcf_yield",
    "ebitda_ev",
    "sales_ev",
    "book_to_market",
    "shareholder_yield",
    "altman_z",
    "momentum_12_1",  # needs the price history, so lti.pit adds it
]

# From several years of filings rather than the latest one (lti.history): the
# normalized versions of the valuation ratios, and how consistent the business is.
HISTORY_METRICS = [
    "pe_norm",
    "earnings_yield_norm",
    "fcf_yield_norm",
    "profit_years",
    "revenue_cagr",
    "fcf_conversion",
    "roic_median",
    "share_growth",
    "asset_growth",
    "f_score",
]

# Blended intrinsic-value upside (lti.valuation): on normalized earnings, and on
# the latest year alone — what the Undervalued page ranked on before.
VALUATION_METRICS = ["fair_value_upside", "fair_value_upside_1y"]

# lower value = "better" (used as the default sort direction in ranking)
LOWER_IS_BETTER = {"pe", "pb", "debt_to_equity", "peg", "pe_norm", "accruals", "share_growth", "asset_growth"}


def needs_history(names) -> bool:
    """Whether any of ``names`` (metrics or filters) needs the multi-year history."""
    return any(n in HISTORY_METRICS or n in VALUATION_METRICS or n == "min_profit_years" for n in names)

# Greenblatt's Magic Formula: rank on these two, equally weighted.
MAGIC_FORMULA_METRICS = ["ebit_ev", "roic"]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def ebit(df: pd.DataFrame) -> pd.Series:
    """Operating income, taken only from what the filer actually tagged.

    The standardized income statement always carries an operating income, but
    when the filer never reported one the standardizer derives it — and for
    filers whose income statement doesn't follow the usual shape (homebuilders,
    PEOs, integrated oil) the derivation is badly wrong, typically landing near
    100% of revenue. Those errors all point the same way: they inflate EBIT, so
    they sort straight to the top of an EBIT-based screen. A smaller universe
    beats a universe whose top names are artefacts.

    So EBIT comes from ``operating_income_reported`` (the raw
    ``OperatingIncomeLoss`` tag, see :mod:`lti.rawtags`), which covers about 76%
    of filings; where the filer did tag it the standardizer agrees to within 1%
    on 99.5% of rows, so nothing is lost by preferring it. Rows without the tag
    get NaN. A value above revenue is refused as well — filers mis-tag too.
    """
    source = (
        "operating_income_reported"
        if "operating_income_reported" in df.columns
        else "operating_income"
    )
    out = df[source].astype("float64")

    revenues = df.get("revenues")
    if revenues is None:
        return out
    return out.where(~(out > revenues).fillna(False))


def add_fundamental_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    equity_pos = df["equity"].where(df["equity"] > 0) if "equity" in df else pd.Series(np.nan, index=df.index)

    if {"liabilities", "equity"} <= set(df.columns):
        df["debt_to_equity"] = _safe_div(df["liabilities"], equity_pos)
    if {"assets_current", "liabilities_current"} <= set(df.columns):
        df["current_ratio"] = _safe_div(df["assets_current"], df["liabilities_current"])
    if {"net_income", "equity"} <= set(df.columns):
        df["roe"] = _safe_div(df["net_income"], equity_pos)
    if {"net_income", "revenues"} <= set(df.columns):
        df["net_margin"] = _safe_div(df["net_income"], df["revenues"])
    if {"gross_profit", "revenues"} <= set(df.columns):
        df["gross_margin"] = _safe_div(df["gross_profit"], df["revenues"])
    if {"free_cash_flow", "revenues"} <= set(df.columns):
        df["fcf_margin"] = _safe_div(df["free_cash_flow"], df["revenues"])
    if {"revenues", "revenues_prev"} <= set(df.columns):
        df["revenue_growth_1y"] = _safe_div(df["revenues"], df["revenues_prev"].where(df["revenues_prev"] > 0)) - 1.0
    if {"eps", "eps_prev"} <= set(df.columns):
        # only meaningful when both years are positive
        both_pos = (df["eps"] > 0) & (df["eps_prev"] > 0)
        df["eps_growth_1y"] = np.where(both_pos, df["eps"] / df["eps_prev"] - 1.0, np.nan)

    if {"equity", "shares_outstanding"} <= set(df.columns):
        df["book_value_per_share"] = _safe_div(df["equity"], df["shares_outstanding"])

    df = add_capital_metrics(df)
    df = add_quality_metrics(df)

    return df


def add_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Profitability and earnings quality, each scaled by total assets.

    ``op_profitability``
        operating income / assets — Novy-Marx's gross profitability, on
        operating income because gross profit is unreliable in this data.
    ``cash_profitability``
        operating cash flow / assets (Ball et al., 2016): profitability that
        accruals can't flatter.
    ``accruals``
        (net income − operating cash flow) / assets (Sloan, 1996): the part of
        earnings that didn't arrive as cash. Lower is better.
    """
    if "assets" not in df.columns:
        return df
    df = df.copy()
    assets = df["assets"].where(df["assets"] > 0)
    if "ebit" not in df.columns and "operating_income" in df.columns:
        df["ebit"] = ebit(df)
    if "ebit" in df.columns:
        df["op_profitability"] = _safe_div(df["ebit"], assets)
    if "cfo" in df.columns:
        df["cash_profitability"] = _safe_div(df["cfo"], assets)
        if "net_income" in df.columns:
            df["accruals"] = _safe_div(df["net_income"] - df["cfo"], assets)
    return df


def add_capital_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Greenblatt's return on capital: EBIT / (net working capital + net fixed assets).

    Return on *capital* rather than on equity: the denominator is the money the
    business actually has to tie up, so it doesn't reward leverage the way ROE
    does and doesn't break when buybacks push book equity negative.

    Following *The Little Book that Beats the Market*:

    * working capital excludes cash (which isn't capital the operation needs) and
      credits back short-term debt (which is financing, not an operating payable);
    * a negative working-capital position is floored at zero — customer float is
      a financing benefit, not negative invested capital;
    * fixed assets are net PP&E only, excluding goodwill: what a previous acquirer
      overpaid is not capital this business needs to earn its return on.

    Companies whose invested capital works out to zero or less get NaN rather than
    a flattering infinite return.
    """
    df = df.copy()
    needed = {"operating_income", "assets_current", "liabilities_current", "ppe_net"}
    if not needed <= set(df.columns):
        return df

    cash = df["cash"].fillna(0.0) if "cash" in df.columns else 0.0
    current_debt = df["debt_current"].fillna(0.0) if "debt_current" in df.columns else 0.0

    working_capital = (df["assets_current"] - cash) - (df["liabilities_current"] - current_debt)
    invested_capital = working_capital.clip(lower=0.0) + df["ppe_net"]

    df["ebit"] = ebit(df)
    df["invested_capital"] = invested_capital.where(invested_capital > 0)
    df["roic"] = _safe_div(df["ebit"], df["invested_capital"])
    return df


def add_price_metrics(
    df: pd.DataFrame,
    price: pd.Series,
    market_cap: pd.Series | None = None,
) -> pd.DataFrame:
    """``price`` / ``market_cap`` are indexed the same way as ``df`` (typically by cik)."""
    df = df.copy()
    price = price.reindex(df.index)
    df["price"] = price

    if "eps" in df.columns:
        eps_pos = df["eps"].where(df["eps"] > 0)
        df["pe"] = _safe_div(price, eps_pos)
        df["earnings_yield"] = _safe_div(df["eps"], price)
        # PEG: P/E over the earnings growth rate in percentage points (>0 only)
        if "eps_growth_1y" in df.columns:
            growth_pts = (df["eps_growth_1y"] * 100).where(df["eps_growth_1y"] > 0)
            df["peg"] = _safe_div(df["pe"], growth_pts)
    if "book_value_per_share" in df.columns:
        bvps_pos = df["book_value_per_share"].where(df["book_value_per_share"] > 0)
        df["pb"] = _safe_div(price, bvps_pos)

    if market_cap is not None:
        df["market_cap"] = market_cap.reindex(df.index)
    elif "shares_outstanding" in df.columns:
        df["market_cap"] = price * df["shares_outstanding"]

    df = add_enterprise_value(df)
    df = add_yield_metrics(df)

    return df


def add_yield_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """The value composite's other yields, and the Altman Z-score.

    ``fcf_yield`` free cash flow / market cap · ``ebitda_ev`` (EBIT + D&A) / EV ·
    ``sales_ev`` revenue / EV · ``book_to_market`` equity / market cap ·
    ``shareholder_yield`` (dividends + buybacks) / market cap — what the company
    handed back; a payout that isn't tagged counts as none, as long as the
    cash-flow statement is there.

    ``altman_z`` is Altman's (1968) distress score, 1.2·WC/TA + 1.4·RE/TA +
    3.3·EBIT/TA + 0.6·MVE/TL + 1.0·Sales/TA; below ~1.8 is the distress zone.
    """
    if "market_cap" not in df.columns:
        return df
    df = df.copy()
    mcap = df["market_cap"].where(df["market_cap"] > 0)
    ev = df["enterprise_value"] if "enterprise_value" in df.columns else pd.Series(np.nan, index=df.index)

    if "free_cash_flow" in df.columns:
        df["fcf_yield"] = _safe_div(df["free_cash_flow"], mcap)
    if {"ebit", "dep_amort"} <= set(df.columns):
        df["ebitda_ev"] = _safe_div(df["ebit"] + df["dep_amort"], ev)
    if "revenues" in df.columns:
        df["sales_ev"] = _safe_div(df["revenues"], ev)
    if "equity" in df.columns:
        df["book_to_market"] = _safe_div(df["equity"].where(df["equity"] > 0), mcap)
    if "cfo" in df.columns:
        paid = sum(
            df[c].abs().fillna(0.0) if c in df.columns else 0.0 for c in ("dividends_paid", "buybacks")
        )
        df["shareholder_yield"] = _safe_div(paid, mcap).where(df["cfo"].notna())

    need = {"assets", "assets_current", "liabilities_current", "retained_earnings", "ebit", "liabilities", "revenues"}
    if need <= set(df.columns):
        ta = df["assets"].where(df["assets"] > 0)
        df["altman_z"] = (
            1.2 * (df["assets_current"] - df["liabilities_current"]) / ta
            + 1.4 * df["retained_earnings"] / ta
            + 3.3 * df["ebit"] / ta
            + 0.6 * _safe_div(mcap, df["liabilities"].where(df["liabilities"] > 0))
            + 1.0 * df["revenues"] / ta
        ).replace([np.inf, -np.inf], np.nan)
    return df


def add_enterprise_value(df: pd.DataFrame) -> pd.DataFrame:
    """Enterprise value and Greenblatt's earnings yield, EBIT / EV.

    EV is what it would cost to buy the whole business: the equity plus the debt
    you inherit, less the cash you get to keep. Unlike P/E this is indifferent to
    how the company is financed, so a debt-laden company and a debt-free one with
    the same operating earnings are compared on the same basis.

    ``total_debt`` is left NaN by :mod:`lti.rawtags` when a filing's debt could
    not be identified, and that NaN deliberately propagates here — quoting no
    yield beats quoting one that assumes away the debt. Negative enterprise
    values (net cash above market cap) are dropped as well; the ratio stops
    meaning anything there.
    """
    if not {"market_cap", "total_debt"} <= set(df.columns):
        return df

    df = df.copy()
    cash = df["cash"].fillna(0.0) if "cash" in df.columns else 0.0
    ev = df["market_cap"] + df["total_debt"] - cash
    df["enterprise_value"] = ev.where(ev > 0)

    if "operating_income" in df.columns:
        if "ebit" not in df.columns:
            df["ebit"] = ebit(df)
        df["ebit_ev"] = _safe_div(df["ebit"], df["enterprise_value"])
    return df
