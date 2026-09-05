"""CIK <-> ticker mapping from the SEC's ``company_tickers.json``.

Note: that file lists only *currently listed* issuers, so delisted companies get
no ticker. This is the project's main survivorship-bias hole (see README).
"""

from __future__ import annotations

import lti.config as config

import pandas as pd
import requests

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _pick_primary(tickers: list[str]) -> str:
    """The most common-stock-like ticker for a multi-class issuer: prefer symbols
    with no punctuation (preferreds / baby bonds carry ``-``/``.``), then the
    shortest, then alphabetical. e.g. ``AMJB,JPM,JPM-PC`` -> ``JPM``."""
    clean = [t for t in tickers if t and not any(c in t for c in "-.$/^")]
    pool = clean or [t for t in tickers if t]
    return min(pool, key=lambda t: (len(t), t))


def refresh_cik_ticker_map() -> pd.DataFrame:
    """Download ``company_tickers.json`` and write ``cik_ticker.parquet``.

    Multi-class issuers (several tickers for one CIK) collapse to one row: the
    most common-stock-like symbol becomes ``ticker`` (see :func:`_pick_primary`);
    all are kept comma-joined in ``tickers_all``.
    """
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

    grouped = (
        rows.sort_values(["cik", "ticker"])
        .groupby("cik", as_index=False)
        .agg(
            tickers_all=("ticker", lambda s: sorted(set(s.dropna()))),
            company=("company", "first"),
        )
    )
    grouped["ticker"] = grouped["tickers_all"].apply(_pick_primary).astype("string")
    grouped["tickers_all"] = grouped["tickers_all"].apply(",".join)
    grouped = grouped[["cik", "ticker", "tickers_all", "company"]]

    path = config.get_paths().cik_ticker_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(path, index=False)
    return grouped


def get_cik_ticker_map(refresh: bool = False) -> pd.DataFrame:
    path = config.get_paths().cik_ticker_parquet
    if refresh or not path.exists():
        return refresh_cik_ticker_map()
    return pd.read_parquet(path)


def ticker_for_cik(cik: int) -> str | None:
    df = get_cik_ticker_map()
    hit = df.loc[df["cik"] == int(cik), "ticker"]
    if hit.empty or pd.isna(hit.iloc[0]):
        return None
    return str(hit.iloc[0])
