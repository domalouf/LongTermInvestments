"""Build the flat ``fundamentals`` table that every downstream module consumes.

One row per company-filing (one 10-K). Every filing is kept, including
restatements / 10-K/A — point-in-time filtering happens later in :mod:`lti.pit`,
so duplicate ``(cik, period_end)`` rows are expected and intentional.
"""

from __future__ import annotations

import gc
import logging

import lti.config as config
from lti import sec_update, tickers

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# --- standardized tag -> flat column name -----------------------------------

BS_MAP = {
    "Assets": "assets",
    "AssetsCurrent": "assets_current",
    "Cash": "cash",
    "AssetsNoncurrent": "assets_noncurrent",
    "Liabilities": "liabilities",
    "LiabilitiesCurrent": "liabilities_current",
    "LiabilitiesNoncurrent": "liabilities_noncurrent",
    "Equity": "equity",
    "HolderEquity": "equity_holders",
    "RetainedEarnings": "retained_earnings",
    "AdditionalPaidInCapital": "additional_paid_in_capital",
    "TreasuryStockValue": "treasury_stock",
    "LiabilitiesAndEquity": "liabilities_and_equity",
}

IS_MAP = {
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
    "OutstandingShares": "shares_outstanding",
    "EarningsPerShare": "eps",
}

CF_MAP = {
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

_SMOKE_DEFAULT_QUARTERS = [
    f"{y}q{q}.zip" for y in (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)
    for q in (1, 2, 3, 4)
]

_ID_COLS = ["adsh", "coreg", "report", "ddate", "qtrs"]


# --- loading standardized statement frames ----------------------------------


def _standardized_result_df(path) -> pd.DataFrame:
    from secfsdstools.f_standardize.standardizing import StandardizedBag

    return StandardizedBag.load(str(path)).result_df


def _smoke_standardized_frames(quarters: list[str]) -> dict[str, pd.DataFrame]:
    """Standardize a handful of quarters in-memory via ZipCollector (no full pipeline)."""
    from secfsdstools.e_collector.zipcollecting import ZipCollector
    from secfsdstools.f_standardize.bs_standardize import BalanceSheetStandardizer
    from secfsdstools.f_standardize.cf_standardize import CashFlowStandardizer
    from secfsdstools.f_standardize.is_standardize import IncomeStatementStandardizer
    from secfsdstools.u_usecases.bulk_loading import default_postloadfilter

    available = set(sec_update.all_quarter_zip_names())
    use = [q for q in quarters if q in available]
    missing = sorted(set(quarters) - available)
    if missing:
        LOGGER.warning("smoke: %d requested quarters not downloaded yet: %s", len(missing), missing)
    if not use:
        raise RuntimeError("smoke: none of the requested quarters are available; run `lti update` first")

    out: dict[str, pd.DataFrame] = {}
    for stmt, standardizer_cls in (
        ("BS", BalanceSheetStandardizer),
        ("IS", IncomeStatementStandardizer),
        ("CF", CashFlowStandardizer),
    ):
        LOGGER.info("smoke: collecting %s for %d quarters", stmt, len(use))
        raw = ZipCollector.get_zip_by_names(
            names=use,
            forms_filter=["10-K"],
            stmt_filter=[stmt],
            post_load_filter=default_postloadfilter,
        ).collect()
        joined = raw.join()
        standardizer = standardizer_cls()
        standardizer.process(joined.pre_num_df)
        out[stmt] = standardizer.get_standardize_bag().result_df.copy()
        del raw, joined, standardizer
        gc.collect()
    return out


# --- per-statement flattening ---------------------------------------------


def _flatten_statement(df: pd.DataFrame, tag_map: dict[str, str], keep_qtrs: set[int], label: str) -> pd.DataFrame:
    """One row per ``adsh``: rename tags, keep only the wanted ``qtrs``, dedupe."""
    present_tags = [t for t in tag_map if t in df.columns]
    cols = _ID_COLS + present_tags
    sub = df.loc[df["qtrs"].isin(keep_qtrs), [c for c in cols if c in df.columns]].copy()

    # main co-registrant only (MainCoregRawFilter already applied, but be defensive)
    if "coreg" in sub.columns:
        sub = sub[sub["coreg"].isin(["", None]) | sub["coreg"].isna()]

    value_cols = present_tags
    sub["_nan_count"] = sub[value_cols].isna().sum(axis=1)
    sub = (
        sub.sort_values(["adsh", "_nan_count"])
        .drop_duplicates("adsh", keep="first")
        .drop(columns=["_nan_count", "coreg", "report", "qtrs"], errors="ignore")
    )
    sub = sub.rename(columns={t: tag_map[t] for t in present_tags})
    sub = sub.rename(columns={"ddate": f"period_end_{label}"})
    return sub.reset_index(drop=True)


def _add_prior_year(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = {"revenues": "revenues_prev", "net_income": "net_income_prev", "eps": "eps_prev", "equity": "equity_prev"}
    have = {src: dst for src, dst in lag_cols.items() if src in df.columns}
    if not have:
        return df

    latest = (
        df.sort_values(["cik", "period_end", "filed"])
        .drop_duplicates(["cik", "period_end"], keep="last")
        .loc[:, ["cik", "period_end", *have.keys()]]
        .copy()
    )
    latest = latest.sort_values(["cik", "period_end"])
    for src, dst in have.items():
        latest[dst] = latest.groupby("cik")[src].shift(1)
    prev = latest.drop(columns=list(have.keys()))
    return df.merge(prev, on=["cik", "period_end"], how="left")


# --- public API ----------------------------------------------------------


def build_fundamentals(smoke: bool = False, quarters: list[str] | None = None) -> "str":
    """Build ``fundamentals.parquet`` (or ``fundamentals.smoke.parquet``).

    Returns the output path as a string.
    """
    paths = config.get_paths()

    if smoke:
        frames = _smoke_standardized_frames(quarters or _SMOKE_DEFAULT_QUARTERS)
        bs_raw, is_raw, cf_raw = frames["BS"], frames["IS"], frames["CF"]
    else:
        LOGGER.info("loading concatenated standardized bags")
        bs_raw = _standardized_result_df(paths.concat_std_bs)
        is_raw = _standardized_result_df(paths.concat_std_is)
        cf_raw = _standardized_result_df(paths.concat_std_cf)

    bs = _flatten_statement(bs_raw, BS_MAP, keep_qtrs={0}, label="bs")
    is_ = _flatten_statement(is_raw, IS_MAP, keep_qtrs={4}, label="is")
    cf = _flatten_statement(cf_raw, CF_MAP, keep_qtrs={4}, label="cf")
    del bs_raw, is_raw, cf_raw
    gc.collect()

    merged = is_.merge(bs, on="adsh", how="outer").merge(cf, on="adsh", how="outer")

    # attach identity from the index
    idx = sec_update.index_dataframe()
    idx = idx.loc[idx["form"] == "10-K", ["adsh", "cik", "name", "form", "filed", "period"]].copy()
    idx = idx.rename(columns={"name": "company"})
    idx = idx.sort_values("filed").drop_duplicates("adsh", keep="last")
    merged = merged.merge(idx, on="adsh", how="inner")

    # canonical period end: prefer IS, then BS, then CF, then the index period
    pe = merged.get("period_end_is")
    for alt in ("period_end_bs", "period_end_cf"):
        if alt in merged.columns:
            pe = pe.fillna(merged[alt]) if pe is not None else merged[alt]
    merged["period_end"] = pd.to_datetime(pe, format="%Y%m%d", errors="coerce")
    merged.loc[merged["period_end"].isna(), "period_end"] = pd.to_datetime(
        merged["period"], format="%Y%m%d", errors="coerce"
    )
    merged["filed"] = pd.to_datetime(merged["filed"], format="%Y%m%d", errors="coerce")
    merged = merged.drop(columns=["period_end_is", "period_end_bs", "period_end_cf", "period"], errors="ignore")

    merged = merged.dropna(subset=["cik", "period_end"])
    merged["cik"] = merged["cik"].astype("int64")
    merged["fiscal_year"] = merged["period_end"].dt.year.astype("int16")
    merged = merged[merged["fiscal_year"] >= config.get_universe_config().start_year]

    # attach ticker (many NaN expected — survivorship hole)
    cik_map = tickers.get_cik_ticker_map()[["cik", "ticker", "tickers_all"]]
    merged = merged.merge(cik_map, on="cik", how="left")

    # convenience columns
    if "capex" in merged.columns and "cfo" in merged.columns:
        merged["free_cash_flow"] = merged["cfo"] - merged["capex"].abs()

    merged = _add_prior_year(merged)

    # stable column order
    front = [
        "cik", "ticker", "tickers_all", "company", "adsh", "form",
        "fiscal_year", "period_end", "filed",
    ]
    ordered = front + [c for c in merged.columns if c not in front]
    merged = merged[ordered].sort_values(["cik", "period_end", "filed"]).reset_index(drop=True)

    out_path = paths.fundamentals_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    LOGGER.info("wrote %d rows -> %s", len(merged), out_path)
    return str(out_path)


def load_fundamentals(smoke: bool | None = None) -> pd.DataFrame:
    paths = config.get_paths()
    path = paths.fundamentals_parquet
    if smoke is True:
        path = paths.derived_dir / "fundamentals.smoke.parquet"
    elif smoke is False:
        path = paths.derived_dir / "fundamentals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `lti build-fundamentals`")
    return pd.read_parquet(path)


def coverage_report(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-column non-null coverage plus cik / ticker / per-year counts."""
    if df is None:
        df = load_fundamentals()

    n = len(df)
    non_null = (df.notna().sum() / n * 100).round(1).rename("pct_non_null")
    report = non_null.to_frame()
    report["n_non_null"] = df.notna().sum()

    summary = pd.DataFrame(
        {
            "metric": ["rows", "distinct_cik", "distinct_cik_with_ticker", "distinct_ticker", "fiscal_year_min", "fiscal_year_max"],
            "value": [
                n,
                df["cik"].nunique(),
                df.loc[df["ticker"].notna(), "cik"].nunique(),
                df["ticker"].nunique(),
                int(df["fiscal_year"].min()),
                int(df["fiscal_year"].max()),
            ],
        }
    )
    print(summary.to_string(index=False))
    print()
    print("rows per fiscal_year:")
    print(df.groupby("fiscal_year").size().to_string())
    print()
    print("column coverage (%):")
    print(report.sort_values("pct_non_null", ascending=False).to_string())
    return report
