"""yfinance price cache: total-return prices, split-adjusted prices and split events.

Four artifacts under ``data/prices/``:

``adj_close.parquet``
    Close adjusted for splits *and* dividends — a total-return series. It is
    what returns are measured with, and never what a company is valued at:
    every dividend paid since a date lowers the adjusted price *on* that date,
    so pairing it with that day's 10-K flatters past dividend payers.
``close.parquet``
    Close adjusted for splits only: the traded price, restated onto today's
    share count. This is the price to value a company at, once the 10-K's
    per-share figures are put on the same basis
    (:func:`lti.pit.restate_per_share`).
``splits.parquet``
    One row per split (``ticker, date, ratio``) — what does that restating.
``dividends.parquet``
    One row per dividend (``ticker, date, amount``) — the cash paid per share,
    on the same restated share basis as ``close.parquet``, so a yield is one
    divided by the other. The adjusted close already *contains* these; this
    table is what lets a screen rank on them, and what the dividend-discount
    model values a payer on.

Both panels are wide: a ``DatetimeIndex`` of trading days, one column per
ticker. Fetching is incremental and resumable — ``_prices_meta.parquet``
records per-ticker status so re-runs skip tickers already covered.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import lti.config as config

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_DEFAULT_START = "2008-01-01"

# How closely a fresh download must match the cached bars it overlaps before it
# is spliced on. Yahoo rewrites a ticker's whole history on every split (both
# panels) and every dividend (the adjusted close); either shows up far above this.
_REBASE_RTOL = 1e-4


def _empty_events(value: str) -> pd.DataFrame:
    """An empty ``ticker, date, <value>`` event table."""
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="string"),
            "date": pd.Series(dtype="datetime64[ns]"),
            value: pd.Series(dtype="float64"),
        }
    )


def empty_splits() -> pd.DataFrame:
    return _empty_events("ratio")


def empty_dividends() -> pd.DataFrame:
    return _empty_events("amount")


@dataclass(frozen=True)
class PriceData:
    """The price cache as one object: what to value at, and what to measure returns with."""

    adj: pd.DataFrame     # split- and dividend-adjusted close -> returns
    close: pd.DataFrame   # split-adjusted close -> valuation
    splits: pd.DataFrame  # ticker, date, ratio
    # ticker, date, amount. Defaulted: a caller that only cares about prices
    # (most tests, anything pre-dividends) builds a PriceData from three frames.
    dividends: pd.DataFrame = field(default_factory=empty_dividends)

    @property
    def empty(self) -> bool:
        return self.adj.empty


class _NoBar:
    """Stands in for tqdm where it can't be imported, so callers needn't check."""

    def set_postfix_str(self, text: str) -> None: ...
    def update(self, n: int) -> None: ...
    def close(self) -> None: ...


def _make_bar(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="tkr", dynamic_ncols=True)
    except Exception:  # noqa: BLE001 - tqdm optional / non-tty
        return _NoBar()


# --- cache IO --------------------------------------------------------------


def _write_parquet(df: pd.DataFrame, path: Path, index: bool) -> None:
    """Write via a temp file + rename, so a reader (the app, mid-refresh) never
    sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=index)
    os.replace(tmp, path)


def _load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    panel = pd.read_parquet(path)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def _save_panel(panel: pd.DataFrame, path: Path) -> None:
    panel = panel.sort_index()
    panel.index.name = "date"
    _write_parquet(panel, path, index=True)


def _load_events(path: Path, value: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_events(value)
    ev = pd.read_parquet(path)
    ev["date"] = pd.to_datetime(ev["date"])
    return ev


def _save_events(ev: pd.DataFrame, path: Path) -> None:
    ev = ev.sort_values(["ticker", "date"]).reset_index(drop=True)
    _write_parquet(ev, path, index=False)


def load_splits() -> pd.DataFrame:
    return _load_events(config.get_paths().splits_parquet, "ratio")


def load_dividends() -> pd.DataFrame:
    return _load_events(config.get_paths().dividends_parquet, "amount")


def _load_meta() -> pd.DataFrame:
    path = config.get_paths().prices_meta_parquet
    if not path.exists():
        return pd.DataFrame(
            columns=["ticker", "status", "rows", "dividends", "first_date", "last_date", "last_fetch"]
        )
    return pd.read_parquet(path)


def _save_meta(meta: pd.DataFrame) -> None:
    _write_parquet(meta, config.get_paths().prices_meta_parquet, index=False)


def _save_all(
    adj: pd.DataFrame,
    close: pd.DataFrame,
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
    meta_rows: dict,
) -> None:
    paths = config.get_paths()
    _save_panel(adj, paths.adj_close_parquet)
    _save_panel(close, paths.close_parquet)
    _save_events(splits, paths.splits_parquet)
    _save_events(dividends, paths.dividends_parquet)
    _save_meta(pd.DataFrame(meta_rows.values()))


def _upsert_columns(panel: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Replace ``new``'s columns in ``panel`` wholesale — for a full-history fetch."""
    if new.empty:
        return panel
    if panel.empty:
        return new.sort_index()
    return panel.drop(columns=new.columns, errors="ignore").join(new, how="outer").sort_index()


def _upsert_rows(panel: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Overwrite / extend ``panel`` with ``new``'s values where it has them — for a recent window."""
    if new.empty:
        return panel
    extra = [c for c in new.columns if c not in panel.columns]
    panel = panel.reindex(index=panel.index.union(new.index), columns=[*panel.columns, *extra])
    panel.update(new)
    return panel


def _replace_events(events: pd.DataFrame, tickers: list[str], new: pd.DataFrame) -> pd.DataFrame:
    """``tickers``' event history as ``new`` has it; everyone else's untouched.

    Replaced wholesale rather than merged: a full-history fetch is the only
    thing that calls this, and Yahoo restating an event is a correction.
    """
    keep = events[~events["ticker"].isin(tickers)]
    add = new[new["ticker"].isin(tickers)]
    if add.empty:
        return keep.reset_index(drop=True)
    if keep.empty:
        return add.reset_index(drop=True)
    return pd.concat([keep, add], ignore_index=True)


def _meta_ok(ticker: str, series: pd.Series, now: pd.Timestamp, dividends: int | None) -> dict:
    """``dividends`` is how many dividend rows the cache holds for the ticker,
    or ``None`` where nothing has ever counted them.

    Zero is a real answer (most tickers never paid one); what matters is that
    the column is *set*, which is how :func:`fetch_prices` tells a ticker whose
    dividends were fetched from one cached before the table existed. So a
    refresh that splices bars onto a ticker whose dividends were never fetched
    — an interrupted backfill, say — passes ``None`` rather than zero, leaving
    it for the backfill to finish.
    """
    s = series.dropna()
    return {
        "ticker": ticker,
        "status": "ok",
        "rows": int(len(s)),
        "dividends": int(dividends) if pd.notna(dividends) else None,
        "first_date": s.index.min(),
        "last_date": s.index.max(),
        "last_fetch": now,
    }


def _meta_missing(ticker: str, status: str, now: pd.Timestamp) -> dict:
    return {
        "ticker": ticker, "status": status, "rows": 0, "dividends": 0,
        "first_date": None, "last_date": None, "last_fetch": now,
    }


# --- fetching -------------------------------------------------------------


def _melt_events(col: pd.DataFrame, value: str) -> pd.DataFrame:
    """A ``Stock Splits`` / ``Dividends`` column block as ``ticker, date, <value>`` rows.

    Yahoo reports a zero on every day without an event, so only positives are
    kept (a NaN compares false and drops out with them).
    """
    if col.empty or not len(col.columns):
        return _empty_events(value)
    long = col.reset_index().melt(id_vars="date", var_name="ticker", value_name=value)
    long = long[long[value] > 0]
    if long.empty:
        return _empty_events(value)
    return pd.DataFrame(
        {
            "ticker": long["ticker"].astype("string").to_numpy(),
            "date": pd.to_datetime(long["date"]).to_numpy(),
            value: long[value].astype("float64").to_numpy(),
        }
    )


def _parse_download(data: pd.DataFrame | None, tickers: list[str]) -> PriceData:
    """Split a ``yf.download(auto_adjust=False, actions=True)`` frame into the
    adjusted-close panel, the split-adjusted close panel, and the split and
    dividend events.

    Yahoo states both the close and the dividend amounts on today's share
    basis (a dividend paid before a 4:1 split comes back quartered), so the
    dividends line up with ``close`` without any restating of their own.
    """
    if data is None or data.empty:
        return PriceData(pd.DataFrame(), pd.DataFrame(), empty_splits(), empty_dividends())

    if isinstance(data.columns, pd.MultiIndex):
        fields = set(data.columns.get_level_values(0))

        def field(name: str) -> pd.DataFrame:
            return data[name].copy() if name in fields else pd.DataFrame(index=data.index)
    else:  # a single ticker without multi-level columns

        def field(name: str) -> pd.DataFrame:
            if name not in data.columns:
                return pd.DataFrame(index=data.index)
            return data[[name]].set_axis([tickers[0]], axis=1)

    adj, close = field("Adj Close"), field("Close")
    split_col, div_col = field("Stock Splits"), field("Dividends")
    for frame in (adj, close, split_col, div_col):
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        frame.columns = [str(c) for c in frame.columns]

    return PriceData(
        adj.dropna(how="all"),
        close.dropna(how="all"),
        _melt_events(split_col, "ratio"),
        _melt_events(div_col, "amount"),
    )


def _download(tickers: list[str], start: str, end: str | None) -> PriceData:
    import yfinance as yf

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,  # keep Close (split-adjusted) alongside Adj Close ...
        actions=True,  # ... and the split events that relate Close to the 10-Ks
        progress=False,
        group_by="column",
        threads=True,
    )
    return _parse_download(data, tickers)


def _fetched(bars: PriceData, tickers: list[str]) -> list[str]:
    """The tickers ``bars`` actually holds data for, in both panels."""
    return [
        t
        for t in tickers
        if t in bars.adj.columns and t in bars.close.columns and bars.adj[t].notna().any()
    ]


def fetch_prices(
    tickers: list[str],
    start: str = _DEFAULT_START,
    end: str | None = None,
    batch_size: int = 40,
    force: bool = False,
    pause: float = 1.0,
) -> PriceData:
    """Fetch full history for ``tickers`` not already cached; upsert all four artifacts.

    A ticker counts as cached once it has a split-adjusted close series and a
    recorded dividend count (or was found to have no data at all). A cache
    built before either existed therefore re-fetches every ticker once: that
    backfill is what lets the screens value a company on the same share basis
    as its price, and rank it on what it actually pays out.
    """
    tickers = sorted({t.upper().strip() for t in tickers if t and isinstance(t, str)})
    paths = config.get_paths()
    adj = _load_panel(paths.adj_close_parquet)
    close = _load_panel(paths.close_parquet)
    splits = load_splits()
    dividends = load_dividends()
    meta = _load_meta()

    done: set[str] = set()
    if not force and not meta.empty:
        # no dividend count means the row predates dividends.parquet: a ticker
        # that pays nothing and one never asked look identical in the table
        # itself, so the meta column is what says its dividends were fetched
        counted = meta["dividends"].notna() if "dividends" in meta.columns else pd.Series(False, index=meta.index)
        ok = set(meta.loc[(meta["status"] == "ok") & counted, "ticker"]) & set(close.columns)
        done = ok | set(meta.loc[(meta["status"] == "no_data") & counted, "ticker"])
    todo = [t for t in tickers if t not in done]
    LOGGER.info("prices: %d requested, %d already cached, %d to fetch", len(tickers), len(tickers) - len(todo), len(todo))

    meta_rows = {r["ticker"]: dict(r) for _, r in meta.iterrows()}

    bar = _make_bar(total=len(todo), desc="prices")
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        now = pd.Timestamp.utcnow()
        try:
            bars = _download(batch, start, end)
        except Exception as exc:  # noqa: BLE001 - resumable, record and move on
            LOGGER.warning("prices: batch failed (%s); marking error", exc)
            for t in batch:
                meta_rows[t] = _meta_missing(t, "error", now)
            bar.update(len(batch))
            continue

        got = _fetched(bars, batch)
        for t in batch:
            if t not in got:
                # a ticker that used to have data is more likely a transient
                # failure than gone for good: keep it retryable
                was_ok = meta_rows.get(t, {}).get("status") == "ok"
                meta_rows[t] = _meta_missing(t, "error" if was_ok else "no_data", now)
        adj = _upsert_columns(adj, bars.adj[got])
        close = _upsert_columns(close, bars.close[got])
        splits = _replace_events(splits, got, bars.splits)
        dividends = _replace_events(dividends, got, bars.dividends)
        n_divs = bars.dividends["ticker"].value_counts()
        for t in got:
            meta_rows[t] = _meta_ok(t, bars.adj[t], now, int(n_divs.get(t, 0)))

        _save_all(adj, close, splits, dividends, meta_rows)
        n_ok = sum(1 for r in meta_rows.values() if r.get("status") == "ok")
        bar.set_postfix_str(f"{n_ok} ok")
        bar.update(len(batch))
        if pause and i + batch_size < len(todo):
            time.sleep(pause)

    bar.close()
    return PriceData(adj, close, splits, dividends)


def _rebased(cached: pd.Series, fresh: pd.Series, rtol: float = _REBASE_RTOL) -> bool:
    """Whether ``fresh`` fails to continue ``cached``.

    Compares the bars both hold, excluding the cached series' last bar (it may
    have been captured intraday). No such bar to check counts as a failure too:
    continuity that can't be verified isn't assumed.
    """
    cached, fresh = cached.dropna(), fresh.dropna()
    if cached.empty:
        return True
    overlap = cached.index[cached.index < cached.index[-1]].intersection(fresh.index)
    if overlap.empty:
        return True
    return not np.allclose(
        cached.loc[overlap].to_numpy(dtype="float64"),
        fresh.loc[overlap].to_numpy(dtype="float64"),
        rtol=rtol,
        atol=0.0,
    )


def refresh_prices(
    tickers: list[str] | None = None,
    lookback_days: int = 7,
    batch_size: int = 40,
    pause: float = 1.0,
) -> PriceData:
    """Top up cached tickers with recent bars; re-fetch any whose history Yahoo re-based.

    The cheap daily counterpart to :func:`fetch_prices` — it never adds new
    tickers (run ``fetch-prices`` after the universe grows). Each download
    starts a few days before the cache ends, so the overlap can be checked
    against what's stored. Yahoo rewrites a ticker's whole history on a split
    (both panels) or a dividend (the adjusted close), and splicing a fresh
    window onto the old history would put a fake jump at the seam — a 10:1
    split reads as a 90% crash. So a ticker whose overlap disagrees, that has
    no overlap to check, or that reports a split or a dividend in the window is
    re-fetched in full instead of spliced. A dividend is named explicitly
    because the re-basing it causes lands on the bars *before* its ex-date: one
    falling on the first bar of the window leaves the overlap looking
    untouched, and its event row would go unrecorded.

    ``tickers`` restricts the refresh to a subset; ``None`` refreshes every
    ticker currently marked ``ok`` that has both price series.
    """
    paths = config.get_paths()
    adj = _load_panel(paths.adj_close_parquet)
    close = _load_panel(paths.close_parquet)
    splits = load_splits()
    dividends = load_dividends()
    meta = _load_meta()
    if adj.empty or close.empty or meta.empty:
        LOGGER.info("prices: nothing to refresh; run `lti fetch-prices` first")
        return PriceData(adj, close, splits, dividends)

    cached = meta.loc[meta["status"] == "ok", "ticker"].tolist()
    if tickers is not None:
        want = {t.upper().strip() for t in tickers if isinstance(t, str)}
        cached = [t for t in cached if t in want]
    cached = sorted(set(cached) & set(adj.columns) & set(close.columns))
    if not cached:
        LOGGER.info("prices: no cached tickers to refresh")
        return PriceData(adj, close, splits, dividends)

    today = pd.Timestamp.today().normalize()
    start = min(today - pd.Timedelta(days=lookback_days), adj.index.max() - pd.Timedelta(days=5))
    LOGGER.info("prices: refreshing %d tickers from %s", len(cached), start.date())
    meta_rows = {r["ticker"]: dict(r) for _, r in meta.iterrows()}

    win_adj: list[pd.DataFrame] = []
    win_close: list[pd.DataFrame] = []
    rebased: list[str] = []
    bar = _make_bar(total=len(cached), desc="refresh")
    for i in range(0, len(cached), batch_size):
        batch = cached[i : i + batch_size]
        try:
            bars = _download(batch, start.strftime("%Y-%m-%d"), None)
        except Exception as exc:  # noqa: BLE001 - resumable, skip and move on
            LOGGER.warning("prices: refresh batch failed (%s); skipping", exc)
            bar.update(len(batch))
            continue

        acted = set(bars.splits["ticker"].astype(str)) | set(bars.dividends["ticker"].astype(str))
        spliced = []
        for t in _fetched(bars, batch):
            if t in acted or _rebased(adj[t], bars.adj[t]) or _rebased(close[t], bars.close[t]):
                rebased.append(t)
            else:
                spliced.append(t)
        win_adj.append(bars.adj[spliced])
        win_close.append(bars.close[spliced])
        bar.set_postfix_str(f"{len(rebased)} re-based")
        bar.update(len(batch))
        if pause and i + batch_size < len(cached):
            time.sleep(pause)
    bar.close()

    now = pd.Timestamp.utcnow()
    if win_adj:
        fresh_adj = pd.concat(win_adj, axis=1)
        adj = _upsert_rows(adj, fresh_adj)
        close = _upsert_rows(close, pd.concat(win_close, axis=1))
        for t in fresh_adj.columns:
            # spliced means no event in the window, so its dividends are unchanged
            meta_rows[t] = _meta_ok(t, adj[t], now, meta_rows.get(t, {}).get("dividends"))

    if rebased:
        LOGGER.info("prices: re-fetching %d re-based tickers in full", len(rebased))
        full_start = adj.index.min().strftime("%Y-%m-%d")
        for i in range(0, len(rebased), batch_size):
            batch = rebased[i : i + batch_size]
            try:
                bars = _download(batch, full_start, None)
            except Exception as exc:  # noqa: BLE001 - the stale history stays until next run
                LOGGER.warning("prices: full re-fetch failed (%s); leaving %d tickers as they were", exc, len(batch))
                continue
            got = _fetched(bars, batch)
            adj = _upsert_columns(adj, bars.adj[got])
            close = _upsert_columns(close, bars.close[got])
            splits = _replace_events(splits, got, bars.splits)
            dividends = _replace_events(dividends, got, bars.dividends)
            n_divs = bars.dividends["ticker"].value_counts()
            for t in got:
                meta_rows[t] = _meta_ok(t, bars.adj[t], now, int(n_divs.get(t, 0)))

    _save_all(adj, close, splits, dividends, meta_rows)
    LOGGER.info(
        "prices: spliced %d tickers, re-fetched %d re-based ones",
        sum(len(w.columns) for w in win_adj), len(rebased),
    )
    return PriceData(adj, close, splits, dividends)


# --- reads --------------------------------------------------------------


def load_adj_close(tickers: list[str] | None = None) -> pd.DataFrame:
    panel = _load_panel(config.get_paths().adj_close_parquet)
    if tickers is not None:
        cols = [t.upper() for t in tickers if t.upper() in panel.columns]
        panel = panel[cols]
    return panel


def load_close() -> pd.DataFrame:
    return _load_panel(config.get_paths().close_parquet)


def load_price_data() -> PriceData:
    return PriceData(
        adj=load_adj_close(), close=load_close(), splits=load_splits(), dividends=load_dividends()
    )


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
    window_days: int | None = 7,
) -> pd.Series:
    """:func:`price_on_or_before` for many tickers at once.

    Same answer, one vectorised pass instead of a Python loop per ticker — which
    is the difference between a screener page that renders and one that appears
    to hang, since every page prices the whole universe on load.

    ``tickers`` may be a Series (the result keeps its index, so it aligns with a
    snapshot indexed by cik) or a plain list (the result is indexed by ticker).
    ``window_days=None`` takes the last price however old it is.
    """
    want = tickers if isinstance(tickers, pd.Series) else pd.Series(list(tickers), index=list(tickers))
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
    fresh = any_valid
    if window_days is not None:
        index_ns = sub.index.to_numpy(dtype="datetime64[ns]")
        age_days = (np.datetime64(pd.Timestamp(date), "ns") - index_ns[last_row]) / np.timedelta64(1, "D")
        fresh = any_valid & (age_days <= window_days)

    by_ticker = pd.Series(np.where(fresh, last_price, np.nan), index=sub.columns)
    symbols = want.astype("string").str.upper()
    return pd.Series(by_ticker.reindex(symbols).to_numpy(), index=want.index, dtype="float64")


def trailing_dividends(
    dividends: pd.DataFrame | None,
    tickers: "pd.Series | list[str]",
    asof,
    months: int = 12,
) -> pd.Series:
    """Cash paid per share over the ``months`` before ``asof``, per ticker.

    Point-in-time by construction: only ex-dates in ``(asof - months, asof]``
    count, so a screen run at a past date can't see a dividend declared after
    it. The window is half-open at the bottom, so a payer on a fixed schedule
    contributes exactly its year of payments rather than thirteen months'.

    A ticker with no dividend rows sums to **0.0**, not NaN: the cache holds
    every fetched ticker's whole history, so silence there means the company
    paid nothing, which is a fact worth ranking on. Callers that can't tell a
    non-payer from an unfetched ticker should mask on the price instead.

    ``tickers`` may be a Series (the result keeps its index, so it aligns with
    a snapshot indexed by cik) or a plain list (indexed by ticker).
    """
    want = tickers if isinstance(tickers, pd.Series) else pd.Series(list(tickers), index=list(tickers))
    symbols = want.astype("string").str.upper()
    if dividends is None or dividends.empty:
        return pd.Series(0.0, index=want.index, dtype="float64")

    asof = pd.Timestamp(asof)
    lo = asof - pd.DateOffset(months=months)
    win = dividends[(dividends["date"] > lo) & (dividends["date"] <= asof)]
    paid = win.groupby(win["ticker"].astype("string").str.upper())["amount"].sum()
    return pd.Series(
        paid.reindex(symbols).fillna(0.0).to_numpy(dtype="float64"), index=want.index, dtype="float64"
    )


def dividend_growth(
    dividends: pd.DataFrame | None,
    tickers: "pd.Series | list[str]",
    asof,
    years: int = 5,
) -> pd.Series:
    """CAGR of the trailing-twelve-month dividend over the last ``years``.

    Both ends are a full year of payments, so a shifted payment date doesn't
    read as growth. NaN unless the company paid in both windows: a starter has
    no rate to speak of, and a cutter that went to zero has no finite one.
    """
    asof = pd.Timestamp(asof)
    now = trailing_dividends(dividends, tickers, asof)
    then = trailing_dividends(dividends, tickers, asof - pd.DateOffset(years=years))
    ratio = now.where(now > 0) / then.where(then > 0)
    return ratio ** (1.0 / years) - 1.0


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


def forward_returns(
    panel: pd.DataFrame,
    tickers: "pd.Series | list[str]",
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """:func:`forward_return`'s return for many tickers at once (indexed like ``tickers``).

    Entry is the price on or just before ``start_date``; exit is the last price
    on or before ``end_date``, however stale — a position in a company that
    stopped trading is held at its last price. NaN where there is no entry price.
    """
    entry = prices_asof(panel, tickers, pd.Timestamp(start_date))
    exit_ = prices_asof(panel, tickers, pd.Timestamp(end_date), window_days=None)
    return exit_ / entry.where(entry > 0) - 1.0


def missing_report(tickers: list[str]) -> pd.DataFrame:
    """Which of ``tickers`` have no / short price history in the cache."""
    meta = _load_meta()
    wanted = pd.DataFrame({"ticker": sorted({t.upper() for t in tickers if isinstance(t, str)})})
    merged = wanted.merge(meta, on="ticker", how="left")
    merged["status"] = merged["status"].fillna("not_fetched")
    return merged
