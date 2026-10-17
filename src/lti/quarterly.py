"""Trailing-twelve-month rows from 10-Q filings: fundamentals as fresh as the last quarter.

Every screen here ran on 10-Ks alone, so a company's numbers were up to fifteen
months old by the time the next annual report arrived — an April screen ranked
on December's year, an October screen on the same one, six months staler. The
SEC data sets carry every 10-Q too, and the standardization pipeline already
keeps them; this turns each one into a row shaped exactly like an annual one:

* **Flows** (income statement, cash flow) are trailing twelve months::

      TTM = last fiscal year + this year to date − the same period a year earlier

  e.g. at a Q2 10-Q: FY2023 + H1 2024 − H1 2023. The year-to-date figures are the
  filing's own (``qtrs`` = 1, 2 or 3 — the cash-flow statement only ever reports
  year to date); the other two come from the last 10-K and the 10-Q a year
  earlier, both filed before this one, so nothing is known before it could be.
* **The balance sheet** — and the debt, fixed assets and share counts read from
  the raw SEC files — is the quarter-end one.
* **Shares** are reconciled like an annual row's (:func:`lti.rawtags.reconcile_shares`),
  from the quarter's weighted average, EPS and net income; **EPS** is then
  TTM net income over that count, since EPS from different filings can sit on
  different share bases.
* **Operating income** counts as reported (``operating_income_reported``, what
  EBIT/EV and ROIC use) only when the company's last 10-K tagged it — a filer
  that tags it annually tags it quarterly; one that doesn't gets the same NaN.
* **A year ago** (``revenues_prev``, ``eps_prev`` …): the TTM row of the 10-Q a
  year earlier, so growth compares like with like.

A 10-Q is only turned into a row when the arithmetic holds: the last 10-K ends
3, 6 or 9 months before it (matching the quarters its year-to-date covers), the
year-earlier 10-Q exists, and revenue, net income, operating cash flow and
equity all come out. Anything else is dropped, and point-in-time snapshots fall
back to the 10-K, exactly as before. The rows carry ``form = "10-Q"`` and
``basis = "ttm"``; ``lti.pit.annual`` strips them where one row per fiscal year
is what's wanted (the five-year history, the Stock page).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lti import rawtags, sectors

LOGGER = logging.getLogger(__name__)

# flows that are summed into a trailing twelve months, by the standardized tag
IS_FLOWS = {
    "Revenues": "revenues",
    "CostOfRevenue": "cost_of_revenue",
    "GrossProfit": "gross_profit",
    "OperatingExpenses": "operating_expenses",
    "OperatingIncomeLoss": "operating_income",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxExpenseBenefit": "pretax_income",
    "AllIncomeTaxExpenseBenefit": "income_tax",
    "IncomeLossFromContinuingOperations": "income_continuing",
    "IncomeLossFromDiscontinuedOperationsNetOfTax": "income_discontinued",
    "ProfitLoss": "profit_loss",
    "NetIncomeLossAttributableToNoncontrollingInterest": "net_income_nci",
    "NetIncomeLoss": "net_income",
}
CF_FLOWS = {
    "NetCashProvidedByUsedInOperatingActivities": "cfo",
    "NetCashProvidedByUsedInInvestingActivities": "cfi",
    "NetCashProvidedByUsedInFinancingActivities": "cff",
    "DepreciationDepletionAndAmortization": "dep_amort",
    "ShareBasedCompensation": "stock_comp",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "PaymentsForRepurchaseOfCommonStock": "buybacks",
    "PaymentsOfDividends": "dividends_paid",
    "ProceedsFromIssuanceOfDebt": "debt_issued",
    "RepaymentsOfDebt": "debt_repaid",
}
FLOWS = [*IS_FLOWS.values(), *CF_FLOWS.values()]
# what a row must have to stand in for the 10-K
REQUIRED = ["revenues", "net_income", "cfo", "equity"]
# how far a period end may sit from where the quarter count puts it
TOLERANCE_DAYS = 25
_QUARTER_DAYS = 365.25 / 4


def _dates(s: pd.Series) -> pd.Series:
    """SEC ``YYYYMMDD`` dates, whether they arrive as ints, strings or floats."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    digits = pd.to_numeric(s, errors="coerce").astype("Int64").astype("string")
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


def _by_qtrs(df: pd.DataFrame, tag_map: dict[str, str], adsh: pd.Index, qtrs: set[int]) -> pd.DataFrame:
    """One row per ``(adsh, qtrs)`` of a standardized statement: main registrant, renamed tags."""
    tags = [t for t in tag_map if t in df.columns]
    sub = df.loc[df["adsh"].isin(adsh) & df["qtrs"].isin(qtrs), ["adsh", "coreg", "ddate", "qtrs", *tags]]
    if "coreg" in sub.columns:
        sub = sub[sub["coreg"].isin(["", None]) | sub["coreg"].isna()]
    sub = sub.assign(_nan=sub[tags].isna().sum(axis=1)).sort_values(["adsh", "qtrs", "_nan"])
    sub = sub.drop_duplicates(["adsh", "qtrs"]).drop(columns=["coreg", "_nan"])
    return sub.rename(columns=tag_map)


def _quarter_frame(bs_raw, is_raw, cf_raw, idx: pd.DataFrame) -> pd.DataFrame:
    """Every 10-Q as one row: its year to date, its own quarter, and its balance sheet."""
    q = idx[idx["form"] == "10-Q"].drop_duplicates("adsh", keep="last").set_index("adsh")
    if q.empty:
        return pd.DataFrame()
    inc = _by_qtrs(is_raw, {**IS_FLOWS, "OutstandingShares": "shares_q", "EarningsPerShare": "eps_q"}, q.index, {1, 2, 3})
    cfl = _by_qtrs(cf_raw, CF_FLOWS, q.index, {1, 2, 3})
    from lti.fundamentals import BS_MAP

    bal = _by_qtrs(bs_raw, BS_MAP, q.index, {0}).drop(columns="qtrs").set_index("adsh")

    # the year to date covers as many quarters as the longest period either statement
    # reports — the cash-flow statement is year-to-date only, so it settles a Q2 or Q3
    # income statement that shows just the three months
    n = pd.concat([inc[["adsh", "qtrs"]], cfl[["adsh", "qtrs"]]]).groupby("adsh")["qtrs"].max()
    ytd_is = inc.merge(n.rename("n"), left_on="adsh", right_index=True).query("qtrs == n").set_index("adsh")
    ytd_cf = cfl.merge(n.rename("n"), left_on="adsh", right_index=True).query("qtrs == n").set_index("adsh")
    own = inc[inc["qtrs"] == 1].set_index("adsh").reindex(columns=["shares_q", "eps_q", "net_income"]).rename(
        columns={"net_income": "net_income_q"}
    )

    out = pd.DataFrame(index=n.index)
    out["n"] = n
    out = out.join(ytd_is[[c for c in IS_FLOWS.values() if c in ytd_is.columns]])
    out = out.join(ytd_cf[[c for c in CF_FLOWS.values() if c in ytd_cf.columns]])
    out = out.join(own).join(bal.drop(columns=["ddate"], errors="ignore"))
    ddate = _dates(ytd_is["ddate"]).combine_first(_dates(bal["ddate"])) if "ddate" in bal.columns else _dates(ytd_is["ddate"])
    out["period_end"] = ddate.reindex(out.index)
    ident = q[["cik", "name", "filed", "period"]].rename(columns={"name": "company"})
    out = out.join(ident, how="inner")
    out["period_end"] = out["period_end"].fillna(_dates(out["period"]))
    out["filed"] = _dates(out["filed"])
    out = out.drop(columns="period").dropna(subset=["cik", "period_end", "filed"]).reset_index()
    out["cik"] = out["cik"].astype("int64")
    return out


def _ttm(quarters: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """TTM flows for each 10-Q, and the checks that make them trustworthy."""
    # each 10-K as first filed, which always precedes the next year's 10-Qs
    cols = ["cik", "period_end", "filed", *[c for c in ("operating_income_reported", *FLOWS) if c in annual.columns]]
    years = (
        annual.sort_values("filed")
        .drop_duplicates(["cik", "period_end"])
        .loc[:, cols]
        .rename(columns=lambda c: c if c == "cik" else f"fy_{c}")
    )
    q = quarters.sort_values("period_end")
    q = pd.merge_asof(
        q, years.sort_values("fy_period_end"), left_on="period_end", right_on="fy_period_end",
        by="cik", direction="backward", allow_exact_matches=False,
    )
    gap = (q["period_end"] - q["fy_period_end"]).dt.days
    q["fy_ok"] = ((gap - q["n"] * _QUARTER_DAYS).abs() <= TOLERANCE_DAYS) & (q["fy_filed"] <= q["filed"])

    # the same quarter a year earlier: same cik and year-to-date length, period a year back
    prior = q[["cik", "n", "period_end", "filed", *[f for f in FLOWS if f in q.columns]]].rename(
        columns=lambda c: c if c in ("cik", "n") else f"py_{c}"
    )
    q["_target"] = q["period_end"] - pd.Timedelta(days=365)
    q = pd.merge_asof(
        q.sort_values("_target"), prior.sort_values("py_period_end"), left_on="_target", right_on="py_period_end",
        by=["cik", "n"], direction="nearest", tolerance=pd.Timedelta(days=TOLERANCE_DAYS),
    ).drop(columns="_target")
    q["py_ok"] = q["py_period_end"].notna() & (q["py_filed"] < q["filed"])

    ok = q["fy_ok"] & q["py_ok"]
    for f in FLOWS:
        if f in q.columns and f"fy_{f}" in q.columns and f"py_{f}" in q.columns:
            q[f] = (q[f"fy_{f}"] + q[f] - q[f"py_{f}"]).where(ok)
        else:
            q[f] = np.nan
    return q


def build_quarterly(
    bs_raw: pd.DataFrame,
    is_raw: pd.DataFrame,
    cf_raw: pd.DataFrame,
    idx: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    cik_map: pd.DataFrame | None = None,
    sic_map: pd.DataFrame | None = None,
    raw_bs: pd.DataFrame | None = None,
    raw_shares: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """TTM rows for every 10-Q the arithmetic holds for; see the module docstring.

    ``bs_raw``/``is_raw``/``cf_raw`` are the standardized statements, ``idx`` the
    SEC index (``adsh, cik, name, form, filed, period``) and ``annual`` the 10-K
    table :func:`lti.fundamentals.build_fundamentals` just built. The rest are the
    same lookups the annual build merges in, and are skipped when not given.
    """
    from lti.fundamentals import add_free_cash_flow

    quarters = _quarter_frame(bs_raw, is_raw, cf_raw, idx)
    if quarters.empty:
        return quarters
    q = _ttm(quarters, annual)

    # counted as reported only where the last 10-K reported it too
    if "fy_operating_income_reported" in q.columns:
        q["operating_income_reported"] = q["operating_income"].where(q["fy_operating_income_reported"].notna())

    if cik_map is not None:
        q = q.merge(cik_map[["cik", "ticker", "tickers_all"]], on="cik", how="left")
    if sic_map is not None:
        q = q.merge(sic_map, on="adsh", how="left")
        q = sectors.add_sector_columns(q)
    if raw_bs is not None:
        q = q.merge(raw_bs.drop(columns=["ddate_raw"], errors="ignore"), on="adsh", how="left")
        q = rawtags.add_debt_provenance(q)

    # shares from the quarter's own figures, cross-checked as for a 10-K
    q["net_income_ttm"] = q["net_income"]
    q = q.assign(shares_outstanding=q["shares_q"], eps=q["eps_q"], net_income=q["net_income_q"])
    if raw_shares is not None:
        q = q.merge(raw_shares.drop(columns=["shares_wavg"], errors="ignore"), on="adsh", how="left")
    q = rawtags.reconcile_shares(q)
    q["net_income"] = q["net_income_ttm"]
    shares = q["shares_outstanding"].where(q["shares_outstanding"] > 0)
    q["eps"] = q["net_income"] / shares
    q["eps_source"] = np.where(q["eps"].notna(), "ttm", "missing")
    q["eps_reported"] = np.nan  # a single quarter's EPS, which no metric should take for a year's

    q = add_free_cash_flow(q)
    if {"total_debt", "cash"} <= set(q.columns):
        q["net_debt"] = q["total_debt"] - q["cash"].fillna(0.0)

    q = q[q.reindex(columns=REQUIRED).notna().all(axis=1) & (q["revenues"] >= 0)].copy()

    # a year ago: the TTM row of the 10-Q a year earlier, where it made one
    prev = q[["cik", "n", "period_end", "revenues", "net_income", "eps", "equity", "filed"]].rename(
        columns={"period_end": "prev_period_end", "revenues": "revenues_prev", "net_income": "net_income_prev",
                 "eps": "eps_prev", "equity": "equity_prev", "filed": "filed_prev"}
    )
    q["_target"] = q["period_end"] - pd.Timedelta(days=365)
    q = pd.merge_asof(
        q.sort_values("_target"), prev.sort_values("prev_period_end"), left_on="_target", right_on="prev_period_end",
        by=["cik", "n"], direction="nearest", tolerance=pd.Timedelta(days=TOLERANCE_DAYS),
    )

    q["form"] = "10-Q"
    q["basis"] = "ttm"
    q["ttm_quarters"] = q["n"]
    q["fiscal_year"] = q["period_end"].dt.year.astype("int16")
    drop = [c for c in q.columns if c.startswith(("fy_", "py_")) or c in (
        "_target", "prev_period_end", "n", "fy_ok", "py_ok", "net_income_ttm", "shares_q", "eps_q", "net_income_q")]
    q = q.drop(columns=drop)
    keep = list(dict.fromkeys([*[c for c in annual.columns if c in q.columns], "basis", "ttm_quarters"]))
    return q[keep].sort_values(["cik", "period_end", "filed"]).reset_index(drop=True)
