"""CIK <-> ticker mapping from the SEC's ``company_tickers.json``.

Note: that file lists only *currently listed* issuers, so delisted companies get
no ticker. This is the project's main survivorship-bias hole (see README).
"""

from __future__ import annotations

import lti.config as config

import pandas as pd
import requests

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def collapse_ticker_list(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per CIK from the SEC's ticker list, kept in the file's order.

    The SEC lists every company's main security first — Prudential's common
    stock PRU sits near position 390, its baby bonds PFH, PRH and PRS after
    9,600 — so the first-listed ticker becomes ``ticker``, and all of them are
    kept comma-joined in ``tickers_all``. Picking by the look of the symbol
    instead (no punctuation, then shortest, then alphabetical) chose PFH over
    PRU and DTB over DTE: $25-par debt priced against the company's earnings,
    which put both at the top of the undervalued list at a P/E of 1.5.
    """
    rows = rows.assign(_pos=range(len(rows)))
    return (
        rows.sort_values(["cik", "_pos"])
        .groupby("cik", as_index=False)
        .agg(
            ticker=("ticker", "first"),
            tickers_all=("ticker", lambda s: ",".join(sorted(set(s.dropna())))),
            company=("company", "first"),
        )[["cik", "ticker", "tickers_all", "company"]]
    )


def refresh_cik_ticker_map() -> pd.DataFrame:
    """Download ``company_tickers.json`` and write ``cik_ticker.parquet``
    (one row per CIK — see :func:`collapse_ticker_list`)."""
    resp = requests.get(
        _COMPANY_TICKERS_URL,
        headers={"User-Agent": config.user_agent()},
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()

    rows = pd.DataFrame(raw.values() if isinstance(raw, dict) else raw)
    rows = rows.rename(columns={"cik_str": "cik", "title": "company"})
    rows["cik"] = rows["cik"].astype("int64")
    rows["ticker"] = rows["ticker"].astype("string").str.upper()
    rows["company"] = rows["company"].astype("string")
    grouped = collapse_ticker_list(rows)

    path = config.get_paths().cik_ticker_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(path, index=False)
    return grouped


def get_cik_ticker_map(refresh: bool = False) -> pd.DataFrame:
    path = config.get_paths().cik_ticker_parquet
    if refresh or not path.exists():
        return refresh_cik_ticker_map()
    return pd.read_parquet(path)
