"""yfinance adjusted-close cache.

Stored as a wide parquet panel (``data/prices/adj_close.parquet``): a
``DatetimeIndex`` of trading days, one column per ticker. Fetching is incremental
and resumable — ``_prices_meta.parquet`` records per-ticker status so re-runs skip
tickers already covered.
"""

from __future__ import annotations

import logging
import time

import lti.config as config

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_DEFAULT_START = "2008-01-01"


def _make_bar(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="tkr", dynamic_ncols=True)
    except Exception:  # noqa: BLE001 - tqdm optional / non-tty
        return None


def _bar_advance(bar, n: int, postfix: str | None = None) -> None:
    if bar is None:
        return
    if postfix:
        bar.set_postfix_str(postfix)
    bar.update(n)


def _bar_close(bar) -> None:
    if bar is not None:
        bar.close()


# --- cache IO --------------------------------------------------------------


def _load_panel() -> pd.DataFrame:
    path = config.get_paths().adj_close_parquet
    if not path.exists():
        return pd.DataFrame()
    panel = pd.read_parquet(path)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def _save_panel(panel: pd.DataFrame) -> None:
    path = config.get_paths().adj_close_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = panel.sort_index()
    panel.index.name = "date"
    panel.to_parquet(path)


def _load_meta() -> pd.DataFrame:
    path = config.get_paths().prices_meta_parquet
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "status", "rows", "first_date", "last_date", "last_fetch"])
    return pd.read_parquet(path)


def _save_meta(meta: pd.DataFrame) -> None:
    path = config.get_paths().prices_meta_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(path, index=False)


# --- fetching -------------------------------------------------------------


def _download(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:  # single ticker
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]
    close.index = pd.to_datetime(close.index)
    return close.dropna(how="all")


def fetch_prices(
    tickers: list[str],
    start: str = _DEFAULT_START,
    end: str | None = None,
    batch_size: int = 40,
    force: bool = False,
    pause: float = 1.0,
) -> pd.DataFrame:
    """Fetch adjusted close for ``tickers`` not already cached; upsert the panel."""
    tickers = sorted({t.upper().strip() for t in tickers if t and isinstance(t, str)})
    panel = _load_panel()
    meta = _load_meta()

    done = set()
    if not force and not meta.empty:
        done = set(meta.loc[meta["status"] == "ok", "ticker"]) | set(
            meta.loc[meta["status"] == "no_data", "ticker"]
        )
    todo = [t for t in tickers if t not in done]
    LOGGER.info("prices: %d requested, %d already cached, %d to fetch", len(tickers), len(tickers) - len(todo), len(todo))

    meta_rows = {r["ticker"]: dict(r) for _, r in meta.iterrows()}

    bar = _make_bar(total=len(todo), desc="prices")
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        try:
            close = _download(batch, start, end)
        except Exception as exc:  # noqa: BLE001 - resumable, record and move on
            LOGGER.warning("prices: batch failed (%s); marking error", exc)
            for t in batch:
                meta_rows[t] = {"ticker": t, "status": "error", "rows": 0, "first_date": None, "last_date": None, "last_fetch": pd.Timestamp.utcnow()}
            _bar_advance(bar, len(batch))
            continue

        for t in batch:
            series = close[t].dropna() if t in close.columns else pd.Series(dtype="float64")
            if series.empty:
                meta_rows[t] = {"ticker": t, "status": "no_data", "rows": 0, "first_date": None, "last_date": None, "last_fetch": pd.Timestamp.utcnow()}
                continue
            panel = panel.join(series.rename(t), how="outer") if not panel.empty else series.rename(t).to_frame()
            meta_rows[t] = {
                "ticker": t,
                "status": "ok",
                "rows": int(series.notna().sum()),
                "first_date": series.index.min(),
                "last_date": series.index.max(),
                "last_fetch": pd.Timestamp.utcnow(),
            }

        _save_panel(panel)
        _save_meta(pd.DataFrame(meta_rows.values()))
        n_ok = sum(1 for r in meta_rows.values() if r.get("status") == "ok")
        _bar_advance(bar, len(batch), postfix=f"{n_ok} ok")
        if pause and i + batch_size < len(todo):
            time.sleep(pause)

    _bar_close(bar)
    _save_panel(panel)
    _save_meta(pd.DataFrame(meta_rows.values()))
    return panel


def refresh_prices(
    tickers: list[str] | None = None,
    lookback_days: int = 7,
    batch_size: int = 40,
    pause: float = 1.0,
) -> pd.DataFrame:
    """Top up already-cached tickers with the last ``lookback_days`` of bars.

    The cheap daily counterpart to :func:`fetch_prices`: it never adds new
    tickers and only re-downloads a short recent window, overwriting the overlap
    so recent dividend/split re-adjustments are picked up. Run ``fetch-prices``
    (not this) after the universe grows.

    ``tickers`` restricts the refresh to a subset; ``None`` refreshes every
    ticker currently marked ``ok`` in the cache.
    """
    panel = _load_panel()
    meta = _load_meta()
    if panel.empty or meta.empty:
        LOGGER.info("prices: cache empty; run `lti fetch-prices` first")
        return panel

    cached = meta.loc[meta["status"] == "ok", "ticker"].tolist()
    if tickers is not None:
        want = {t.upper().strip() for t in tickers if isinstance(t, str)}
        cached = [t for t in cached if t in want]
    cached = sorted(set(cached) & set(panel.columns))
    if not cached:
        LOGGER.info("prices: no cached tickers to refresh")
        return panel

    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    LOGGER.info("prices: refreshing %d tickers from %s", len(cached), start)
    meta_rows = {r["ticker"]: dict(r) for _, r in meta.iterrows()}

    bar = _make_bar(total=len(cached), desc="refresh")
    updated = 0
    for i in range(0, len(cached), batch_size):
        batch = cached[i : i + batch_size]
        try:
            close = _download(batch, start, None)
        except Exception as exc:  # noqa: BLE001 - resumable, skip and move on
            LOGGER.warning("prices: refresh batch failed (%s); skipping", exc)
            _bar_advance(bar, len(batch))
            continue

        for t in batch:
            series = close[t].dropna() if t in close.columns else pd.Series(dtype="float64")
            if series.empty:
                continue
            new_idx = series.index.difference(panel.index)
            if len(new_idx):
                panel = panel.reindex(panel.index.union(new_idx))
            panel.loc[series.index, t] = series
            updated += 1
            row = meta_rows.get(t, {"ticker": t, "status": "ok"})
            last = pd.Timestamp(row["last_date"]) if row.get("last_date") is not None else None
            row["last_date"] = series.index.max() if last is None or series.index.max() > last else last
            row["rows"] = int(panel[t].notna().sum())
            row["last_fetch"] = pd.Timestamp.utcnow()
            meta_rows[t] = row

        _save_panel(panel)
        _save_meta(pd.DataFrame(meta_rows.values()))
        _bar_advance(bar, len(batch), postfix=f"{updated} upd")
        if pause and i + batch_size < len(cached):
            time.sleep(pause)

    _bar_close(bar)
    _save_panel(panel)
    _save_meta(pd.DataFrame(meta_rows.values()))
    LOGGER.info("prices: refreshed %d/%d tickers", updated, len(cached))
    return panel


# --- reads --------------------------------------------------------------


def load_adj_close(tickers: list[str] | None = None) -> pd.DataFrame:
    panel = _load_panel()
    if tickers is not None:
        cols = [t.upper() for t in tickers if t.upper() in panel.columns]
        panel = panel[cols]
    return panel


def price_on_or_before(panel: pd.DataFrame, ticker: str, date: pd.Timestamp, window_days: int = 7) -> float | None:
    """Last valid adj close at/just before ``date`` (within ``window_days``)."""
    ticker = ticker.upper()
    if ticker not in panel.columns:
        return None
    date = pd.Timestamp(date)
    series = panel[ticker].dropna()
    series = series[series.index <= date]
    if series.empty:
        return None
    if (date - series.index[-1]).days > window_days:
        return None
    return float(series.iloc[-1])


def prices_asof(
    panel: pd.DataFrame,
    tickers: "pd.Series | list[str]",
    date: pd.Timestamp,
    window_days: int = 7,
) -> pd.Series:
    """:func:`price_on_or_before` for many tickers at once.

    Same answer, one vectorised pass instead of a Python loop per ticker — which
    is the difference between a screener page that renders and one that appears
    to hang, since every page prices the whole universe on load.

    ``tickers`` may be a Series (the result keeps its index, so it aligns with a
    snapshot indexed by cik) or a plain list (the result is indexed by ticker).
    """
    want = pd.Series(tickers) if not isinstance(tickers, pd.Series) else tickers
    out = pd.Series(np.nan, index=want.index, dtype="float64")

    sub = panel.loc[: pd.Timestamp(date)]
    if sub.empty or panel.empty:
        return out

    arr = sub.to_numpy(dtype="float64", na_value=np.nan)
    valid = ~np.isnan(arr)
    any_valid = valid.any(axis=0)
    # index of the last non-NaN row per column
    last_row = valid.shape[0] - 1 - valid[::-1].argmax(axis=0)

    cols = np.arange(arr.shape[1])
    last_price = np.where(any_valid, arr[last_row, cols], np.nan)
    index_ns = sub.index.to_numpy(dtype="datetime64[ns]")
    age_days = (np.datetime64(pd.Timestamp(date), "ns") - index_ns[last_row]) / np.timedelta64(1, "D")
    fresh = any_valid & (age_days <= window_days)

    by_ticker = pd.Series(np.where(fresh, last_price, np.nan), index=sub.columns)
    symbols = want.astype("string").str.upper()
    return pd.Series(by_ticker.reindex(symbols).to_numpy(), index=want.index, dtype="float64")


def forward_return(
    panel: pd.DataFrame, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> tuple[float | None, bool]:
    """``(total_return, delisted)`` over ``[start_date, end_date]`` from adj close.

    If the series ends before ``end_date`` the return is computed to the last
    available price and ``delisted`` is ``True``.
    """
    ticker = ticker.upper()
    entry = price_on_or_before(panel, ticker, start_date)
    if entry is None or entry <= 0:
        return None, False

    series = panel[ticker].dropna()
    end_date = pd.Timestamp(end_date)
    at_or_before_end = series[series.index <= end_date]
    if at_or_before_end.empty:
        return None, False

    exit_price = float(at_or_before_end.iloc[-1])
    delisted = (end_date - at_or_before_end.index[-1]).days > 15
    return exit_price / entry - 1.0, delisted


def missing_report(tickers: list[str]) -> pd.DataFrame:
    """Which of ``tickers`` have no / short price history in the cache."""
    meta = _load_meta()
    wanted = pd.DataFrame({"ticker": sorted({t.upper() for t in tickers if isinstance(t, str)})})
    merged = wanted.merge(meta, on="ticker", how="left")
    merged["status"] = merged["status"].fillna("not_fetched")
    return merged
