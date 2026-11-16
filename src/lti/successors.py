"""Join a company's filings across a change of SEC registrant.

The SEC's ticker list maps *today's* registrants to tickers. When a company
reorganizes under a new holding company it gets a new CIK and the old one goes
quiet: Google became Alphabet in 2015, Walgreen became Walgreens Boots Alliance,
Disney re-registered when it bought Fox, Cigna when it bought Express Scripts.
The stock never stopped trading and the ticker never changed, but every filing
under the old CIK is left without a ticker — so, to every backtest here, Google
did not exist before 2015. That isn't survivorship (the company survived); it is
a hole in the survivors, and unlike the real one it can be closed from the data
already on disk.

A filing's XBRL instance document is named for its filer — ``goog-20131231.xml``
— and by convention the prefix is the ticker. :func:`build_instance_prefixes`
reads that prefix for every registrant, and :func:`find_successors` links an old
CIK with no ticker to the current registrant trading under its prefix when the
handover also *looks* like one company carrying on:

* the successor's first 10-K comes within :data:`MAX_GAP_DAYS` after the
  predecessor's last (and not long before it) — a ticker reissued years later to
  an unrelated company fails this; and
* the two filings report revenue within a factor of :data:`MAX_REVENUE_RATIO` of
  each other — a small company inheriting a famous symbol fails this; and
* both are in the same two-digit SIC industry, where both have a code.

:func:`apply_successors` then files the predecessor's rows under the successor's
CIK and ticker, keeping the original in ``cik_reported``. Companies that really
did leave the market — acquired, taken private, bankrupt — are not linked to
anything: Yahoo has no prices for them, and :mod:`lti.survivorship` measures what
their absence does.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import lti.config as config
from lti import rawtags

LOGGER = logging.getLogger(__name__)

MAX_GAP_DAYS = 550  # the successor's first 10-K, after the predecessor's last
MAX_OVERLAP_DAYS = 60  # ... or just before it; two registrants filing side by side are two companies
MAX_REVENUE_RATIO = 2.0
SUCCESSOR_COLUMNS = ["cik", "successor_cik", "ticker", "tickers_all", "last_filed", "successor_first_filed", "revenue_ratio"]


def _paths():
    d = config.get_paths().derived_dir
    return d / "instance_prefix.parquet", d / "cik_successors.parquet"


def _read_quarter_instances(qdir) -> pd.DataFrame | None:
    sub = qdir / "sub.txt.parquet"
    if not sub.exists():
        return None
    import pyarrow.parquet as pq

    if pq.ParquetFile(sub).metadata.num_rows == 0:
        return None
    df = pd.read_parquet(sub, columns=["adsh", "cik", "form", "instance"])
    return df[df["form"].isin(["10-K", "10-Q"])]


def prefix_of(instance: pd.Series) -> pd.Series:
    """The filer's prefix in an instance document's name: ``goog-20131231.xml`` → ``GOOG``.

    Only the conventional shape counts — one to five letters, a hyphen, a date —
    so ``form10k_htm.xml`` and the like give NA rather than a made-up symbol.
    """
    return instance.astype("string").str.extract(r"^([A-Za-z]{1,5})-\d{8}", expand=False).str.upper()


def build_instance_prefixes(force: bool = False) -> pd.DataFrame:
    """``cik -> prefix``: the prefix most of each registrant's 10-K and 10-Q instance
    documents carry. Cached to ``instance_prefix.parquet``."""
    out_path, _ = _paths()
    if not force and rawtags._cache_is_current(out_path):
        return pd.read_parquet(out_path)
    filings = rawtags._scan_quarters(_read_quarter_instances, "instance names")
    filings = filings.assign(prefix=prefix_of(filings["instance"])).dropna(subset=["prefix"])
    counts = filings.groupby(["cik", "prefix"]).size().rename("n").reset_index()
    top = counts.sort_values(["cik", "n", "prefix"], ascending=[True, False, True]).drop_duplicates("cik")
    return rawtags._write_cache(top[["cik", "prefix", "n"]].astype({"cik": "int64"}), out_path)


def find_successors(annual: pd.DataFrame, cik_map: pd.DataFrame, prefixes: pd.DataFrame) -> pd.DataFrame:
    """Old registrants whose business carries on under a listed one (:data:`SUCCESSOR_COLUMNS`).

    ``annual`` is the 10-K table as built (``cik, filed, period_end, revenues``),
    ``cik_map`` the SEC's current list (``cik, ticker, tickers_all``) and
    ``prefixes`` :func:`build_instance_prefixes`.
    """
    listed = cik_map.dropna(subset=["ticker"])
    by_ticker = {}
    for row in listed.itertuples(index=False):
        for t in str(row.tickers_all or row.ticker).split(","):
            by_ticker.setdefault(t.strip().upper(), row)  # first-listed registrant wins a shared symbol
    rows = annual.sort_values(["cik", "period_end", "filed"])
    last = rows.drop_duplicates("cik", keep="last").set_index("cik")
    first = rows.drop_duplicates("cik", keep="first").set_index("cik")
    listed_ciks = set(listed["cik"])

    out = []
    for p in prefixes.itertuples(index=False):
        if p.cik in listed_ciks or p.cik not in last.index or p.prefix not in by_ticker:
            continue
        succ = by_ticker[p.prefix]
        if succ.cik == p.cik or succ.cik not in first.index:
            continue
        gap = (first.at[succ.cik, "filed"] - last.at[p.cik, "filed"]).days
        ratio = first.at[succ.cik, "revenues"] / last.at[p.cik, "revenues"] if last.at[p.cik, "revenues"] else np.nan
        if not (-MAX_OVERLAP_DAYS <= gap <= MAX_GAP_DAYS):
            continue
        if not (pd.notna(ratio) and 1 / MAX_REVENUE_RATIO <= ratio <= MAX_REVENUE_RATIO):
            continue
        if "sic" in rows.columns:
            a, b = last.at[p.cik, "sic"], first.at[succ.cik, "sic"]
            if pd.notna(a) and pd.notna(b) and int(a) // 100 != int(b) // 100:
                continue
        out.append(
            {
                "cik": int(p.cik), "successor_cik": int(succ.cik), "ticker": succ.ticker,
                "tickers_all": succ.tickers_all, "last_filed": last.at[p.cik, "filed"],
                "successor_first_filed": first.at[succ.cik, "filed"], "revenue_ratio": float(ratio),
            }
        )
    return pd.DataFrame(out, columns=SUCCESSOR_COLUMNS)


def build_successors(annual: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """:func:`find_successors` on the tables on disk, written to ``cik_successors.parquet``."""
    from lti import tickers

    _, out_path = _paths()
    found = find_successors(annual, tickers.get_cik_ticker_map(), build_instance_prefixes(force=force))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    found.to_parquet(out_path, index=False)
    LOGGER.info("successors: linked %d old registrants to a listed successor -> %s", len(found), out_path)
    return found


def load_successors() -> pd.DataFrame:
    _, path = _paths()
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=SUCCESSOR_COLUMNS)


def apply_successors(fund: pd.DataFrame, successors: pd.DataFrame) -> pd.DataFrame:
    """File each predecessor's rows under its successor's CIK and ticker.

    ``cik_reported`` keeps the registrant that actually filed each row. A
    predecessor row for a fiscal year the successor also reported is dropped —
    the successor's is the one filed later. Rows that already have a ticker are
    never touched.
    """
    if successors.empty or fund.empty or "cik" not in fund.columns:
        return fund
    link = successors.drop_duplicates("cik").set_index("cik")
    old = fund["cik"].isin(link.index) & fund["ticker"].isna()
    out = fund.assign(cik_reported=fund["cik"])
    if not old.any():
        return out
    to = out.loc[old, "cik"].map(link["successor_cik"])
    out.loc[old, "ticker"] = out.loc[old, "cik"].map(link["ticker"]).to_numpy()
    if "tickers_all" in out.columns:
        out.loc[old, "tickers_all"] = out.loc[old, "cik"].map(link["tickers_all"]).to_numpy()
    out.loc[old, "cik"] = to.to_numpy()
    out["cik"] = out["cik"].astype(fund["cik"].dtype)
    # one row per company and period: where both registrants reported it, keep the successor's
    keys = [c for c in ("cik", "form", "period_end") if c in out.columns]
    moved = pd.Series(old.to_numpy(), index=out.index)
    dupe = out.duplicated(keys, keep=False)
    out = out[~(dupe & moved)]
    sort = [c for c in ("cik", "period_end", "filed") if c in out.columns]
    return out.sort_values(sort, kind="stable").reset_index(drop=True)
