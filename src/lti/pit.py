"""Point-in-time snapshots of the fundamentals table (no look-ahead)."""

from __future__ import annotations

import pandas as pd


def snapshot_asof(
    fund: pd.DataFrame,
    asof: pd.Timestamp,
    max_staleness_days: int = 550,
) -> pd.DataFrame:
    """The fundamentals an investor could have known at ``asof``.

    1. keep only filings with ``filed <= asof``
    2. per ``cik``, take the most recent ``period_end``; if several filings cover
       that period (restatement / 10-K/A), keep the one with the latest ``filed``
    3. drop rows whose ``period_end`` is older than ``asof - max_staleness_days``
       (company stopped filing / went dark)
    4. one row per cik, indexed by cik
    """
    asof = pd.Timestamp(asof)
    known = fund[fund["filed"] <= asof]
    if known.empty:
        return known.set_index("cik") if "cik" in known.columns else known

    known = known.sort_values(["cik", "period_end", "filed"])
    latest = known.drop_duplicates("cik", keep="last").copy()

    cutoff = asof - pd.Timedelta(days=max_staleness_days)
    latest = latest[latest["period_end"] >= cutoff]

    latest["asof"] = asof
    latest["days_since_period_end"] = (asof - latest["period_end"]).dt.days
    return latest.set_index("cik")
