"""Tests for the price cache: download parsing, the backfill, and re-base-aware refresh.

No network: ``prices._download`` is swapped for a fake Yahoo that serves
history the way the real one does — every split re-bases all earlier bars.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
import lti.config as config
from lti import prices


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the whole price cache at a temp dir."""
    d = tmp_path / "prices"
    fake = dataclasses.replace(
        config.get_paths(),
        prices_dir=d,
        adj_close_parquet=d / "adj_close.parquet",
        close_parquet=d / "close.parquet",
        splits_parquet=d / "splits.parquet",
        prices_meta_parquet=d / "_prices_meta.parquet",
    )
    monkeypatch.setattr(config, "get_paths", lambda: fake)
    return fake


class FakeYahoo:
    """Actual traded prices + split events in; Yahoo-style adjusted history out."""

    def __init__(self):
        self.actual: dict[str, pd.Series] = {}
        self.splits: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        self.dividend_factor: dict[str, float] = {}  # applied to adj before the last bar
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def split(self, t: str, date: str, ratio: float) -> None:
        """The traded price drops by ``ratio`` from ``date`` on."""
        d = pd.Timestamp(date)
        px = self.actual[t]
        self.actual[t] = px.where(px.index < d, px / ratio)
        self.splits.setdefault(t, []).append((d, ratio))

    def close(self, t: str) -> pd.Series:
        px = self.actual[t].copy()
        for d, ratio in self.splits.get(t, []):
            px[px.index < d] /= ratio
        return px

    def adj(self, t: str) -> pd.Series:
        px = self.close(t)
        f = self.dividend_factor.get(t)
        if f:
            px.iloc[:-1] *= f
        return px

    def download(self, tickers, start, end):
        self.calls.append((tuple(tickers), start))
        lo = pd.Timestamp(start)
        adj, close, events = {}, {}, []
        for t in tickers:
            if t not in self.actual:
                continue
            adj[t] = self.adj(t)[lambda s: s.index >= lo]
            close[t] = self.close(t)[lambda s: s.index >= lo]
            events += [(t, d, r) for d, r in self.splits.get(t, []) if d >= lo]
        ev = prices.empty_splits()
        if events:
            ev = pd.DataFrame(events, columns=["ticker", "date", "ratio"]).astype({"ticker": "string"})
        return prices.PriceData(pd.DataFrame(adj), pd.DataFrame(close), ev)


@pytest.fixture
def yahoo(monkeypatch):
    fake = FakeYahoo()
    days = pd.bdate_range("2020-01-01", "2020-06-30")
    fake.actual["AAA"] = pd.Series(np.linspace(100, 120, len(days)), index=days)
    fake.actual["BBB"] = pd.Series(np.linspace(50, 40, len(days)), index=days)
    fake.actual["SPY"] = pd.Series(np.linspace(300, 310, len(days)), index=days)
    monkeypatch.setattr(prices, "_download", fake.download)
    return fake


def _extend(fake: FakeYahoo, until: str) -> None:
    """Let time pass: each series continues at its last price."""
    for t, s in fake.actual.items():
        more = pd.bdate_range(s.index[-1] + pd.Timedelta(days=1), until)
        fake.actual[t] = pd.concat([s, pd.Series(float(s.iloc[-1]), index=more)])


# --- parsing ------------------------------------------------------------


def test_parse_download_splits_the_three_series():
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=4), name="Date")
    fields = ["Adj Close", "Close", "Dividends", "Stock Splits", "Volume"]
    cols = pd.MultiIndex.from_product([fields, ["AAA", "BBB"]], names=["Price", "Ticker"])
    data = pd.DataFrame(1.0, index=idx, columns=cols)
    data[("Close", "AAA")] = [10.0, 11.0, 12.0, 13.0]
    data[("Adj Close", "AAA")] = [9.0, 10.0, 11.0, 13.0]
    data[("Stock Splits", "AAA")] = [0.0, 0.0, 2.0, 0.0]
    data[("Stock Splits", "BBB")] = [0.0, np.nan, 0.0, 0.0]

    out = prices._parse_download(data, ["AAA", "BBB"])
    assert list(out.close["AAA"]) == [10.0, 11.0, 12.0, 13.0]
    assert list(out.adj["AAA"]) == [9.0, 10.0, 11.0, 13.0]
    assert out.splits.to_dict("records") == [
        {"ticker": "AAA", "date": pd.Timestamp("2021-01-06"), "ratio": 2.0}
    ]


def test_parse_download_single_ticker_flat_columns():
    idx = pd.bdate_range("2021-01-04", periods=3)
    data = pd.DataFrame({"Adj Close": [1.0, 2.0, 3.0], "Close": [1.5, 2.5, 3.5], "Stock Splits": 0.0}, index=idx)
    out = prices._parse_download(data, ["AAA"])
    assert list(out.adj.columns) == ["AAA"] and list(out.close["AAA"]) == [1.5, 2.5, 3.5]
    assert out.splits.empty


def test_parse_download_empty():
    out = prices._parse_download(pd.DataFrame(), ["AAA"])
    assert out.adj.empty and out.close.empty and out.splits.empty


# --- continuity check ---------------------------------------------------


def test_rebased_detects_a_rescaled_overlap():
    idx = pd.bdate_range("2021-01-04", periods=6)
    cached = pd.Series([10.0, 11, 12, 13, 14, 15], index=idx)
    assert not prices._rebased(cached, cached.iloc[2:] * (1 + 1e-6))  # float noise is fine
    assert prices._rebased(cached, cached.iloc[2:] / 4)  # a 4:1 split re-based it
    assert prices._rebased(cached, cached.iloc[2:] * 0.99)  # a dividend re-based it


def test_rebased_ignores_the_possibly_intraday_last_bar():
    idx = pd.bdate_range("2021-01-04", periods=4)
    cached = pd.Series([10.0, 11, 12, 13], index=idx)
    fresh = pd.Series([11.0, 12, 13.4, 14], index=pd.bdate_range("2021-01-05", periods=4))
    assert not prices._rebased(cached, fresh)  # only the last cached bar moved


def test_rebased_when_nothing_overlaps():
    cached = pd.Series([1.0, 2.0], index=pd.bdate_range("2021-01-04", periods=2))
    fresh = pd.Series([3.0], index=pd.bdate_range("2021-03-01", periods=1))
    assert prices._rebased(cached, fresh)


# --- fetch --------------------------------------------------------------


def test_fetch_writes_all_three_artifacts(cache, yahoo):
    yahoo.split("AAA", "2020-03-02", 2.0)
    px = prices.fetch_prices(["AAA", "BBB", "SPY", "NOPE"], pause=0)

    assert cache.adj_close_parquet.exists() and cache.close_parquet.exists() and cache.splits_parquet.exists()
    loaded = prices.load_price_data()
    assert set(loaded.close.columns) == {"AAA", "BBB", "SPY"}
    pd.testing.assert_series_equal(loaded.close["AAA"], yahoo.close("AAA"), check_names=False, check_freq=False)
    assert loaded.splits[["ticker", "ratio"]].values.tolist() == [["AAA", 2.0]]
    meta = prices._load_meta().set_index("ticker")
    assert meta.loc["NOPE", "status"] == "no_data" and meta.loc["AAA", "status"] == "ok"
    assert px.close.shape == loaded.close.shape


def test_fetch_backfills_an_old_cache_without_the_close_panel(cache, yahoo):
    # a cache from before close.parquet existed: adj panel + "ok" meta only
    old = pd.DataFrame({t: yahoo.adj(t) for t in ["AAA", "BBB"]})
    prices._save_panel(old, cache.adj_close_parquet)
    prices._save_meta(pd.DataFrame({"ticker": ["AAA", "BBB"], "status": ["ok", "ok"]}))

    prices.fetch_prices(["AAA", "BBB"], pause=0)
    assert {t for call in yahoo.calls for t in call[0]} == {"AAA", "BBB"}
    assert set(prices.load_close().columns) == {"AAA", "BBB"}
    assert set(prices.load_adj_close().columns) == {"AAA", "BBB"}  # replaced, not duplicated

    yahoo.calls.clear()
    prices.fetch_prices(["AAA", "BBB"], pause=0)  # now fully cached
    assert yahoo.calls == []


def test_fetch_keeps_a_previously_ok_ticker_retryable(cache, yahoo):
    prices.fetch_prices(["AAA", "BBB"], pause=0)
    del yahoo.actual["BBB"]  # a transient Yahoo miss
    prices.fetch_prices(["BBB"], force=True, pause=0)
    assert prices._load_meta().set_index("ticker").loc["BBB", "status"] == "error"


# --- refresh ------------------------------------------------------------


def test_refresh_splices_a_continuous_window(cache, yahoo):
    prices.fetch_prices(["AAA", "BBB", "SPY"], pause=0)
    _extend(yahoo, "2020-07-31")
    yahoo.calls.clear()

    px = prices.refresh_prices(pause=0)
    assert px.close.index.max() == pd.Timestamp("2020-07-31")
    assert all(start != "2008-01-01" for _, start in yahoo.calls)  # no full re-fetch
    pd.testing.assert_series_equal(px.close["AAA"].dropna(), yahoo.close("AAA"), check_names=False, check_freq=False)


def test_refresh_refetches_a_ticker_that_split_instead_of_splicing(cache, yahoo):
    prices.fetch_prices(["AAA", "BBB", "SPY"], pause=0)
    _extend(yahoo, "2020-07-31")
    yahoo.split("AAA", "2020-07-15", 4.0)  # re-bases all earlier AAA bars
    yahoo.calls.clear()

    px = prices.refresh_prices(pause=0)
    full = [tickers for tickers, start in yahoo.calls if start == "2020-01-01"]
    assert full == [("AAA",)]
    # no fake crash at the seam: the stored history is Yahoo's re-based one
    pd.testing.assert_series_equal(px.close["AAA"].dropna(), yahoo.close("AAA"), check_names=False, check_freq=False)
    daily = px.close["AAA"].pct_change().dropna()
    assert daily.abs().max() < 0.05
    assert prices.load_splits()[["ticker", "ratio"]].values.tolist() == [["AAA", 4.0]]


def test_refresh_refetches_a_ticker_whose_adjusted_history_moved(cache, yahoo):
    prices.fetch_prices(["AAA", "BBB", "SPY"], pause=0)
    _extend(yahoo, "2020-07-31")
    yahoo.dividend_factor["BBB"] = 0.98  # a dividend re-scales every earlier adjusted bar
    yahoo.calls.clear()

    px = prices.refresh_prices(pause=0)
    assert [tickers for tickers, start in yahoo.calls if start == "2020-01-01"] == [("BBB",)]
    pd.testing.assert_series_equal(px.adj["BBB"].dropna(), yahoo.adj("BBB"), check_names=False, check_freq=False)


# --- vectorised returns -------------------------------------------------


def test_forward_returns_matches_the_scalar_version():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    panel = pd.DataFrame(
        {
            "UP": np.linspace(10, 20, len(idx)),
            "GONE": np.where(idx < "2020-06-01", 5.0, np.nan),  # stops trading mid-year
            "LATE": np.where(idx >= "2020-03-01", 7.0, np.nan),  # no price at entry
        },
        index=idx,
    )
    start, end = pd.Timestamp("2020-01-15"), pd.Timestamp("2020-12-15")
    vec = prices.forward_returns(panel, ["UP", "GONE", "LATE", "MISSING"], start, end)
    for i, t in enumerate(["UP", "GONE"]):
        assert vec.iloc[i] == pytest.approx(prices.forward_return(panel, t, start, end)[0])
    assert np.isnan(vec.iloc[2]) and np.isnan(vec.iloc[3])
