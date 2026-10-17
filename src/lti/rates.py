"""Interest rates at a date: what the valuation models discount at.

Three of the six models discount at a required return — the DCF, the dividend
discount, earnings power — and Graham's revised formula scales by the AAA
corporate yield. Held at 9% and 4.5% for every date, a backtest values April
2013, when the 10-year Treasury paid 1.8%, at the same rates as October 2023,
when it paid nearly 5%: rates nobody could have used then, and a fair-value
ranking scored against a world that didn't exist.

``lti fetch-rates`` caches two daily FRED series — the 10-year Treasury
(``DGS10``) and Moody's seasoned AAA corporate yield (``DAAA``) — in
``data/prices/rates.parquet``, as fractions (0.018, not 1.8).
:func:`lti.valuation.market_assumptions` then values each date at its own
rates: a discount rate of the Treasury plus an equity risk premium, and the AAA
yield for Graham. Both are market yields rather than estimates, so there is
nothing to revise later, and a day's yield is known by its close: reading them
as of a rebalance date looks at nothing a screen couldn't have.
"""

from __future__ import annotations

import io
import logging

import lti.config as config

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# cache column -> FRED series id
FRED_SERIES = {"treasury_10y": "DGS10", "aaa": "DAAA"}
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# How old a reading may be and still count as the rate on a date: a long
# weekend, a holiday, a publishing delay — not a gap in the series.
MAX_STALENESS_DAYS = 14


def empty_rates() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in FRED_SERIES}, index=pd.DatetimeIndex([], name="date")
    )


def parse_fred_csv(text: str) -> pd.Series:
    """One FRED series from its CSV download, as a fraction per date.

    The file is two columns, a date and the value in percent. FRED has named the
    date column both ``DATE`` and ``observation_date``, and marked a missing day
    both ``.`` and blank, so neither is relied on.
    """
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] < 2:
        raise ValueError("not a FRED series CSV: expected a date column and a value column")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce") / 100.0
    out = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]), name="date"))
    return out.dropna().sort_index()


def fetch_rates() -> pd.DataFrame:
    """Download every series in :data:`FRED_SERIES` and cache them; returns the table.

    A series that fails to download keeps what the cache already had for it,
    with a warning, rather than wiping it; if none downloads, nothing is written.
    """
    import requests

    cached = load_rates()
    series = {}
    failed = []
    for col, sid in FRED_SERIES.items():
        try:
            resp = requests.get(
                FRED_CSV.format(series=sid), headers={"User-Agent": config.user_agent()}, timeout=60
            )
            resp.raise_for_status()
            series[col] = parse_fred_csv(resp.text)
            LOGGER.info("%s (%s): %d days to %s", col, sid, len(series[col]), series[col].index.max().date())
        except Exception as exc:  # noqa: BLE001 — one series down shouldn't lose the other
            LOGGER.warning("could not fetch %s (%s): %s — keeping the cached series", col, sid, exc)
            series[col] = cached[col].dropna() if col in cached.columns else pd.Series(dtype="float64")
            failed.append(sid)
    if len(failed) == len(FRED_SERIES):
        raise RuntimeError(f"could not fetch any of {', '.join(failed)} from FRED — the cache is unchanged")
    table = pd.DataFrame(series).sort_index()
    table.index.name = "date"
    table = table.reindex(columns=list(FRED_SERIES))
    path = config.get_paths().rates_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)
    return table


def load_rates() -> pd.DataFrame:
    """The cached rates, or an empty table when ``lti fetch-rates`` hasn't run."""
    path = config.get_paths().rates_parquet
    if not path.exists():
        return empty_rates()
    return pd.read_parquet(path)


def rates_asof(rates: pd.DataFrame, date, max_staleness_days: int = MAX_STALENESS_DAYS) -> pd.Series:
    """Each rate's last reading on or before ``date``, NaN where there is none that recent."""
    date = pd.Timestamp(date)
    out = pd.Series(np.nan, index=list(FRED_SERIES), dtype="float64")
    for col in FRED_SERIES:
        if col not in rates.columns:
            continue
        s = rates[col].dropna()
        s = s[s.index <= date]
        if not s.empty and (date - s.index[-1]).days <= max_staleness_days:
            out[col] = float(s.iloc[-1])
    return out
