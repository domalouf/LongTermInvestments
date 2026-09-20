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
        dividends_parquet=d / "dividends.parquet",
        prices_meta_parquet=d / "_prices_meta.parquet",
    )
    monkeypatch.setattr(config, "get_paths", lambda: fake)
    return fake


class FakeYahoo:
    """Actual traded prices + split events in; Yahoo-style adjusted history out."""

    def __init__(self):
        self.actual: dict[str, pd.Series] = {}
        self.splits: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        self.dividends: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        self.dividend_factor: dict[str, float] = {}  # applied to adj before the last bar
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def split(self, t: str, date: str, ratio: float) -> None:
        """The traded price drops by ``ratio`` from ``date`` on."""
        d = pd.Timestamp(date)
        px = self.actual[t]
        self.actual[t] = px.where(px.index < d, px / ratio)
        self.splits.setdefault(t, []).append((d, ratio))

    def dividend(self, t: str, date: str, amount: float) -> None:
        """A cash dividend of ``amount`` per share, ex on ``date``."""
        self.dividends.setdefault(t, []).append((pd.Timestamp(date), amount))

    def close(self, t: str) -> pd.Series:
        px = self.actual[t].copy()
        for d, ratio in self.splits.get(t, []):
            px[px.index < d] /= ratio
        return px

    def adj(self, t: str) -> pd.Series:
        px = self.close(t)
        for d, amount in self.dividends.get(t, []):
            on = px[px.index >= d]
            if len(on):  # Yahoo scales every bar before the ex-date, none after
                px[px.index < d] *= 1 - amount / float(on.iloc[0])
        f = self.dividend_factor.get(t)
        if f:
            px.iloc[:-1] *= f
        return px

    def download(self, tickers, start, end):
        self.calls.append((tuple(tickers), start))
        lo = pd.Timestamp(start)
        adj, close, events, paid = {}, {}, [], []
        for t in tickers:
            if t not in self.actual:
                continue
            adj[t] = self.adj(t)[lambda s: s.index >= lo]
            close[t] = self.close(t)[lambda s: s.index >= lo]
            events += [(t, d, r) for d, r in self.splits.get(t, []) if d >= lo]
            paid += [(t, d, a) for d, a in self.dividends.get(t, []) if d >= lo]
        ev = prices.empty_splits()
        if events:
            ev = pd.DataFrame(events, columns=["ticker", "date", "ratio"]).astype({"ticker": "string"})
        dv = prices.empty_dividends()
        if paid:
            dv = pd.DataFrame(paid, columns=["ticker", "date", "amount"]).astype({"ticker": "string"})
        return prices.PriceData(pd.DataFrame(adj), pd.DataFrame(close), ev, dv)


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


def test_parse_download_splits_the_four_series():
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=4), name="Date")
    fields = ["Adj Close", "Close", "Dividends", "Stock Splits", "Volume"]
    cols = pd.MultiIndex.from_product([fields, ["AAA", "BBB"]], names=["Price", "Ticker"])
    data = pd.DataFrame(1.0, index=idx, columns=cols)
    data[("Close", "AAA")] = [10.0, 11.0, 12.0, 13.0]
    data[("Adj Close", "AAA")] = [9.0, 10.0, 11.0, 13.0]
    data[("Stock Splits", "AAA")] = [0.0, 0.0, 2.0, 0.0]
    data[("Stock Splits", "BBB")] = [0.0, np.nan, 0.0, 0.0]
    data[("Dividends", "AAA")] = [0.0, 0.25, 0.0, 0.0]
    data[("Dividends", "BBB")] = [0.0, 0.0, np.nan, 0.0]

    out = prices._parse_download(data, ["AAA", "BBB"])
    assert list(out.close["AAA"]) == [10.0, 11.0, 12.0, 13.0]
    assert list(out.adj["AAA"]) == [9.0, 10.0, 11.0, 13.0]
    assert out.splits.to_dict("records") == [
        {"ticker": "AAA", "date": pd.Timestamp("2021-01-06"), "ratio": 2.0}
    ]
    assert out.dividends.to_dict("records") == [
        {"ticker": "AAA", "date": pd.Timestamp("2021-01-05"), "amount": 0.25}
    ]


def test_parse_download_single_ticker_flat_columns():
    idx = pd.bdate_range("2021-01-04", periods=3)
    data = pd.DataFrame({"Adj Close": [1.0, 2.0, 3.0], "Close": [1.5, 2.5, 3.5], "Stock Splits": 0.0}, index=idx)
    out = prices._parse_download(data, ["AAA"])
    assert list(out.adj.columns) == ["AAA"] and list(out.close["AAA"]) == [1.5, 2.5, 3.5]
    assert out.splits.empty and out.dividends.empty  # no Dividends column at all


def test_parse_download_empty():
    out = prices._parse_download(pd.DataFrame(), ["AAA"])
    assert out.adj.empty and out.close.empty and out.splits.empty and out.dividends.empty


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


def test_fetch_writes_all_four_artifacts(cache, yahoo):
    yahoo.split("AAA", "2020-03-02", 2.0)
    yahoo.dividend("BBB", "2020-02-14", 0.30)
    yahoo.dividend("BBB", "2020-05-15", 0.30)
    px = prices.fetch_prices(["AAA", "BBB", "SPY", "NOPE"], pause=0)

    for path in (cache.adj_close_parquet, cache.close_parquet, cache.splits_parquet, cache.dividends_parquet):
        assert path.exists()
    loaded = prices.load_price_data()
    assert set(loaded.close.columns) == {"AAA", "BBB", "SPY"}
    pd.testing.assert_series_equal(loaded.close["AAA"], yahoo.close("AAA"), check_names=False, check_freq=False)
    assert loaded.splits[["ticker", "ratio"]].values.tolist() == [["AAA", 2.0]]
    assert loaded.dividends[["ticker", "amount"]].values.tolist() == [["BBB", 0.30], ["BBB", 0.30]]
    meta = prices._load_meta().set_index("ticker")
    assert meta.loc["NOPE", "status"] == "no_data" and meta.loc["AAA", "status"] == "ok"
    # a payer and a non-payer are both *counted*, which is what marks them fetched
    assert meta.loc["BBB", "dividends"] == 2 and meta.loc["AAA", "dividends"] == 0
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


def test_fetch_backfills_a_cache_without_the_dividend_table(cache, yahoo):
    # a cache from before dividends.parquet: both panels and "ok" meta, no counts
    for t in ["AAA", "BBB"]:
        yahoo.dividend(t, "2020-03-16", 0.50)
    prices._save_panel(pd.DataFrame({t: yahoo.adj(t) for t in ["AAA", "BBB"]}), cache.adj_close_parquet)
    prices._save_panel(pd.DataFrame({t: yahoo.close(t) for t in ["AAA", "BBB"]}), cache.close_parquet)
    prices._save_meta(pd.DataFrame({"ticker": ["AAA", "BBB"], "status": ["ok", "ok"], "rows": [120, 120]}))

    prices.fetch_prices(["AAA", "BBB"], pause=0)
    assert {t for call in yahoo.calls for t in call[0]} == {"AAA", "BBB"}
    assert sorted(prices.load_dividends()["ticker"]) == ["AAA", "BBB"]

    yahoo.calls.clear()
    prices.fetch_prices(["AAA", "BBB"], pause=0)  # counted now, so nothing to do
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


def test_refresh_records_a_dividend_paid_since_the_last_fetch(cache, yahoo):
    prices.fetch_prices(["AAA", "BBB", "SPY"], pause=0)
    assert prices.load_dividends().empty
    _extend(yahoo, "2020-07-31")
    yahoo.dividend("BBB", "2020-07-20", 0.40)
    yahoo.calls.clear()

    px = prices.refresh_prices(pause=0)
    assert [tickers for tickers, start in yahoo.calls if start == "2020-01-01"] == [("BBB",)]
    assert px.dividends[["ticker", "amount"]].values.tolist() == [["BBB", 0.40]]
    assert prices._load_meta().set_index("ticker").loc["BBB", "dividends"] == 1
    # the re-based adjusted history came back whole, not spliced onto the old one
    pd.testing.assert_series_equal(px.adj["BBB"].dropna(), yahoo.adj("BBB"), check_names=False, check_freq=False)


def test_refresh_catches_a_dividend_on_the_first_bar_of_the_window(cache, yahoo):
    """Yahoo re-bases the bars *before* an ex-date, so one at the window's edge
    leaves the overlap untouched: without the event check it would be missed."""
    prices.fetch_prices(["AAA", "BBB", "SPY"], pause=0)
    _extend(yahoo, "2020-07-31")
    prices.refresh_prices(pause=0)  # cache now ends 2020-07-31
    yahoo.dividend("BBB", "2020-07-27", 0.40)  # inside the next refresh's lookback
    yahoo.calls.clear()

    px = prices.refresh_prices(lookback_days=7, pause=0)
    assert px.dividends[["ticker", "amount"]].values.tolist() == [["BBB", 0.40]]


def test_refresh_leaves_an_uncounted_ticker_for_the_backfill(cache, yahoo):
    """An interrupted backfill leaves tickers with no dividend count. A refresh
    that only splices new bars must not claim they were counted."""
    yahoo.dividend("BBB", "2020-03-16", 0.50)
    prices._save_panel(pd.DataFrame({t: yahoo.adj(t) for t in ["AAA", "BBB"]}), cache.adj_close_parquet)
    prices._save_panel(pd.DataFrame({t: yahoo.close(t) for t in ["AAA", "BBB"]}), cache.close_parquet)
    prices._save_meta(pd.DataFrame({"ticker": ["AAA", "BBB"], "status": ["ok", "ok"], "rows": [120, 120]}))
    _extend(yahoo, "2020-07-31")

    prices.refresh_prices(pause=0)
    assert prices._load_meta().set_index("ticker")["dividends"].isna().all()

    yahoo.calls.clear()
    prices.fetch_prices(["AAA", "BBB"], pause=0)  # still due a backfill, and it lands
    assert {t for call in yahoo.calls for t in call[0]} == {"AAA", "BBB"}
    assert prices.load_dividends()["ticker"].tolist() == ["BBB"]


# --- dividends ----------------------------------------------------------


def _paid(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [(t, pd.Timestamp(d), a) for t, d, a in rows], columns=["ticker", "date", "amount"]
    ).astype({"ticker": "string"})


def test_trailing_dividends_sums_the_last_year_only():
    paid = _paid(
        ("PAY", "2019-03-01", 0.25), ("PAY", "2019-06-01", 0.25),  # the year before
        ("PAY", "2020-03-02", 0.30), ("PAY", "2020-06-01", 0.30), ("PAY", "2020-09-01", 0.30),
        ("PAY", "2020-12-15", 0.30),  # declared after the as-of date below
    )
    dps = prices.trailing_dividends(paid, ["PAY", "NONE"], "2020-10-01")
    assert dps["PAY"] == pytest.approx(0.90)  # three payments, not the fourth
    assert dps["NONE"] == 0.0  # no rows means it paid nothing, not "unknown"


def test_trailing_dividends_window_is_half_open_at_the_bottom():
    paid = _paid(("PAY", "2019-10-01", 1.0), ("PAY", "2020-04-01", 2.0))
    # a payment exactly twelve months back belongs to the previous year
    assert prices.trailing_dividends(paid, ["PAY"], "2020-10-01")["PAY"] == pytest.approx(2.0)
    assert prices.trailing_dividends(paid, ["PAY"], "2020-09-30")["PAY"] == pytest.approx(3.0)


def test_trailing_dividends_keeps_the_callers_index():
    paid = _paid(("PAY", "2020-06-01", 0.50))
    want = pd.Series(["PAY", "NONE"], index=pd.Index([101, 202], name="cik"))
    dps = prices.trailing_dividends(paid, want, "2020-10-01")
    assert list(dps.index) == [101, 202] and dps.loc[101] == pytest.approx(0.50)


def test_trailing_dividends_without_a_table():
    assert prices.trailing_dividends(None, ["AAA"], "2020-10-01")["AAA"] == 0.0
    assert prices.trailing_dividends(prices.empty_dividends(), ["AAA"], "2020-10-01")["AAA"] == 0.0


def test_dividend_growth_needs_a_payer_at_both_ends():
    paid = _paid(
        ("GROW", "2015-06-01", 1.0), ("GROW", "2020-06-01", 2.0),
        ("NEW", "2020-06-01", 1.0),   # started paying inside the window
        ("CUT", "2015-06-01", 1.0),   # stopped
    )
    g = prices.dividend_growth(paid, ["GROW", "NEW", "CUT", "NONE"], "2020-10-01", years=5)
    assert g["GROW"] == pytest.approx(2 ** 0.2 - 1)
    assert np.isnan(g["NEW"]) and np.isnan(g["CUT"]) and np.isnan(g["NONE"])


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
