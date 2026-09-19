"""The investable universe: split-correct per-share figures, the operating-company
filter, and the backtest's same-universe benchmark.

Pure logic — no SEC data or network needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import pit, sectors
from lti.backtest import BacktestConfig, rebalance_month_spread, run_backtest
from lti.fundamentals import _add_prior_year
from lti.prices import PriceData, empty_splits
from lti.ranking import ScreenSpec


def _splits(*rows: tuple[str, str, float]) -> pd.DataFrame:
    if not rows:
        return empty_splits()
    df = pd.DataFrame(rows, columns=["ticker", "date", "ratio"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# --- split factor -----------------------------------------------------------


def test_split_factor_is_the_product_of_later_splits_only():
    splits = _splits(("AAA", "2015-06-01", 2.0), ("AAA", "2020-06-01", 3.0), ("BBB", "2018-01-02", 0.1))
    tickers = pd.Series(["AAA", "AAA", "AAA", "AAA", "aaa", "BBB", "CCC", "AAA"])
    dates = pd.Series(pd.to_datetime(
        ["2014-01-01", "2016-01-01", "2021-01-01", "2015-06-01", "2014-01-01", "2017-01-01", "2014-01-01", None]
    ))
    f = pit.split_factor_after(tickers, dates, splits)
    assert f.tolist() == [
        6.0,   # both splits still to come
        3.0,   # only the 2020 one
        1.0,   # none left
        3.0,   # a split on the filing day is already in the filing
        6.0,   # ticker case doesn't matter
        0.1,   # a 1-for-10 reverse split
        1.0,   # never split
        1.0,   # no date, no restatement
    ]


def test_split_factor_without_splits_is_one():
    f = pit.split_factor_after(pd.Series(["AAA"], index=[7]), pd.Series(pd.to_datetime(["2015-01-01"])), empty_splits())
    assert f.tolist() == [1.0] and list(f.index) == [7]


def test_restate_per_share_moves_eps_and_shares_onto_todays_basis():
    snap = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "filed": pd.to_datetime(["2016-03-01", "2016-03-01"]),
            "filed_prev": pd.to_datetime(["2015-03-01", "2015-03-01"]),
            "eps": [10.0, 2.0],
            "eps_prev": [8.0, 1.0],
            "shares_outstanding": [100.0, 50.0],
            "net_income": [1000.0, 100.0],
        },
        index=pd.Index([1, 2], name="cik"),
    )
    out = pit.restate_per_share(snap, _splits(("AAA", "2018-06-01", 10.0)))
    assert out.loc[1, ["eps", "eps_prev", "shares_outstanding"]].tolist() == [1.0, 0.8, 1000.0]
    assert out.loc[2, ["eps", "eps_prev", "shares_outstanding"]].tolist() == [2.0, 1.0, 50.0]
    assert out.loc[1, "net_income"] == 1000.0  # totals are basis-free
    assert out["split_factor"].tolist() == [10.0, 1.0]


def test_restate_per_share_refuses_eps_prev_without_its_filing_date():
    snap = pd.DataFrame({"ticker": ["AAA"], "filed": pd.to_datetime(["2016-03-01"]), "eps": [1.0], "eps_prev": [1.0]})
    with pytest.raises(KeyError, match="filed_prev"):
        pit.restate_per_share(snap, _splits(("AAA", "2018-06-01", 2.0)))


def test_prior_year_records_which_filing_it_came_from():
    df = pd.DataFrame(
        {
            "cik": [1, 1, 1],
            "period_end": pd.to_datetime(["2015-12-31", "2016-12-31", "2016-12-31"]),
            "filed": pd.to_datetime(["2016-03-01", "2017-03-01", "2017-05-01"]),  # + a 10-K/A
            "eps": [1.0, 2.0, 2.1],
        }
    )
    out = _add_prior_year(df)
    assert out["filed_prev"].isna().iloc[0]
    assert (out["filed_prev"].iloc[1:] == pd.Timestamp("2016-03-01")).all()
    assert out["eps_prev"].iloc[1:].tolist() == [1.0, 1.0]


# --- the look-ahead regression -------------------------------------------------


@pytest.fixture
def twins():
    """AAA and ZZZ are the same business at the same price; ZZZ split 10:1 in mid-2018.

    ZZZ's 10-Ks before the split report a tenth of the shares and ten times the
    EPS — at ten times AAA's traded price. Yahoo back-adjusts that away, so on
    the price panels the two are identical.
    """
    split_day = pd.Timestamp("2018-06-01")
    rows = []
    for cik, ticker in [(1, "AAA"), (2, "ZZZ")]:
        for fy in range(2010, 2021):
            filed = pd.Timestamp(fy + 1, 3, 1)
            pre = ticker == "ZZZ" and filed < split_day
            rows.append(
                dict(
                    cik=cik, ticker=ticker, company=f"{ticker} Corp", sic=2000, form="10-K",
                    fiscal_year=fy, period_end=pd.Timestamp(fy, 12, 31), filed=filed,
                    revenues=1000.0, net_income=100.0, equity=800.0, liabilities=400.0,
                    assets_current=300.0, liabilities_current=150.0,
                    shares_outstanding=10.0 if pre else 100.0,
                    eps=10.0 if pre else 1.0,
                    cfo=120.0, capex=20.0,
                )
            )
    fund = _add_prior_year(pd.DataFrame(rows))
    fund["free_cash_flow"] = fund["cfo"] - fund["capex"]

    idx = pd.bdate_range("2009-01-01", "2021-06-30")
    grow = 20 * np.cumprod(np.full(len(idx), 1.0003))
    panel = pd.DataFrame({"AAA": grow, "ZZZ": grow, "SPY": 100 * np.cumprod(np.full(len(idx), 1.0002))}, index=idx)
    fixed = PriceData(adj=panel, close=panel, splits=_splits(("ZZZ", "2018-06-01", 10.0)))
    naive = PriceData(adj=panel, close=panel, splits=empty_splits())
    return fund, fixed, naive


def test_a_future_split_does_not_make_a_company_look_cheap(twins):
    fund, fixed, naive = twins
    snap = pit.priced_snapshot(fund, "2016-04-01", fixed).set_index("ticker")
    assert snap.loc["ZZZ", "pe"] == pytest.approx(snap.loc["AAA", "pe"])
    assert snap.loc["ZZZ", "market_cap"] == pytest.approx(snap.loc["AAA", "market_cap"])
    assert snap.loc["ZZZ", "pb"] == pytest.approx(snap.loc["AAA", "pb"])

    # without the split history, the old bug: ZZZ at a tenth of AAA's multiple
    biased = pit.priced_snapshot(fund, "2016-04-01", naive).set_index("ticker")
    assert biased.loc["ZZZ", "pe"] == pytest.approx(biased.loc["AAA", "pe"] / 10)


def test_eps_growth_across_a_split_is_not_a_collapse(twins):
    fund, fixed, _ = twins
    # the first post-split 10-K compares against a pre-split one
    snap = pit.priced_snapshot(fund, "2019-04-01", fixed).set_index("ticker")
    assert snap.loc["ZZZ", "eps_growth_1y"] == pytest.approx(0.0)


def test_backtest_does_not_pick_the_future_splitter(twins):
    fund, fixed, naive = twins
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["pe"], top_n=1), start="2011-01-01", end="2021-01-01", market_cap_min=0.0
    )
    # identical businesses, identical multiples: the tiebreak (ticker) decides
    assert set(run_backtest(cfg, fund, fixed).holdings["ticker"]) == {"AAA"}
    # the old bug picked ZZZ every year until it had split
    before = run_backtest(cfg, fund, naive).holdings
    assert set(before.loc[before["rebalance_date"] < "2018-06-01", "ticker"]) == {"ZZZ"}


def test_valuation_uses_the_split_adjusted_close_not_the_dividend_adjusted_one(twins):
    fund, fixed, _ = twins
    px = PriceData(adj=fixed.adj * 0.5, close=fixed.close, splits=fixed.splits)
    snap = pit.priced_snapshot(fund, "2016-04-01", px).set_index("ticker")
    close_then = fixed.close.loc[:"2016-04-01", "AAA"].iloc[-1]
    assert snap.loc["AAA", "price"] == pytest.approx(close_then)


# --- operating companies ---------------------------------------------------------


def test_operating_mask_drops_non_businesses():
    snap = pd.DataFrame(
        {
            "revenues": [100.0, np.nan, 0.0, -5.0, 50.0, 50.0, 50.0],
            "sic": [2000, 2000, 6770, 2000, 6221, 6221, 6221],
            "company": ["ACME INC", "SHELL CO", "BLANK CHECK CORP", "ODD INC",
                        "INVESCO DB COMMODITY INDEX TRACKING FUND", "SEABOARD CORP /DE/", "ISHARES BITCOIN TRUST ETF"],
        }
    )
    assert pit.operating_mask(snap).tolist() == [True, False, False, False, False, True, False]


def test_commodity_pool_needs_both_the_sic_and_a_fund_like_name():
    sic = pd.Series([6221, 6221, 6221, 1040, 6221])
    names = pd.Series(["SPDR GOLD TRUST", "UNITED STATES OIL FUND, LP", "INTL FCSTONE INC.",
                       "GOLD RESOURCE CORP TRUST", "ML WINTON FUTURESACCESS LLC"])
    assert sectors.is_commodity_pool(sic, names).tolist() == [True, True, False, False, True]


def test_priced_snapshot_can_keep_non_operating_filers(twins):
    fund, fixed, _ = twins
    fund = fund.copy()
    fund.loc[fund["ticker"] == "ZZZ", "revenues"] = np.nan
    assert set(pit.priced_snapshot(fund, "2016-04-01", fixed)["ticker"]) == {"AAA"}
    assert set(pit.priced_snapshot(fund, "2016-04-01", fixed, operating_only=False)["ticker"]) == {"AAA", "ZZZ"}


# --- same-universe benchmark -----------------------------------------------------


@pytest.fixture
def world():
    """Six companies; lower cik = less leverage and faster compounding."""
    rows = []
    for cik in range(1, 7):
        for fy in range(2010, 2021):
            rows.append(
                dict(
                    cik=cik, ticker=f"T{cik}", company=f"Co{cik}", form="10-K", fiscal_year=fy,
                    period_end=pd.Timestamp(fy, 12, 31), filed=pd.Timestamp(fy + 1, 3, 1),
                    revenues=1000.0, net_income=80.0 + 5 * cik, equity=500.0 + 10 * cik,
                    liabilities=200.0 + 30 * cik, shares_outstanding=100.0, eps=(80.0 + 5 * cik) / 100,
                )
            )
    fund = _add_prior_year(pd.DataFrame(rows))
    idx = pd.bdate_range("2009-01-01", "2021-06-30")
    panel = pd.DataFrame(
        {f"T{c}": 100 * np.cumprod(np.full(len(idx), 1.0 + 0.0003 * (7 - c))) for c in range(1, 7)}, index=idx
    )
    panel["SPY"] = 100 * np.cumprod(np.full(len(idx), 1.0002))
    return fund, PriceData(adj=panel, close=panel, splits=empty_splits())


def test_universe_benchmark_is_every_ranked_stock_equal_weighted(world):
    fund, px = world
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=2), start="2012-01-01", end="2020-06-01", market_cap_min=0.0
    )
    r = run_backtest(cfg, fund, px)
    ps = r.period_summary
    assert (ps["n_universe"] == 6).all() and (ps["n_selected"] == 2).all()

    rd, nrd = ps.loc[0, "rebalance_date"], ps.loc[0, "exit_date"]
    one_each = [px.adj.loc[:nrd, t].iloc[-1] / px.adj.loc[:rd, t].iloc[-1] - 1 for t in px.adj.columns if t != "SPY"]
    assert ps.loc[0, "univ_return"] == pytest.approx(np.mean(one_each))

    # the least-levered pair compounds fastest in this world
    assert r.stats["port_cagr"] > r.stats["univ_cagr"] > 0
    assert r.stats["excess_cagr_vs_univ"] == pytest.approx(r.stats["port_cagr"] - r.stats["univ_cagr"])
    assert r.stats["periods_beat_univ"] == 1.0
    assert r.universe_curve.index.equals(r.equity_curve.index)
    assert {"company", "market_cap", "debt_to_equity", "universe_period_return"} <= set(r.holdings.columns)


def test_backtest_defaults_to_an_april_rebalance(world):
    fund, px = world
    r = run_backtest(BacktestConfig(screen=ScreenSpec(metrics=["roe"], top_n=2), market_cap_min=0.0), fund, px)
    assert (pd.to_datetime(r.period_summary["rebalance_date"]).dt.month == 4).all()


def test_backtest_leaves_the_callers_screen_untouched(world):
    fund, px = world
    spec = ScreenSpec(metrics=["roe"], top_n=2, filters={"exclude_financials": False})
    run_backtest(BacktestConfig(screen=spec, market_cap_min=0.0), fund, px)
    assert spec.filters == {"exclude_financials": False}


def test_rebalance_month_spread_runs_each_month(world):
    fund, px = world
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=2), start="2012-01-01", end="2020-06-01", market_cap_min=0.0
    )
    spread = rebalance_month_spread(cfg, fund, px, months=[1, 4, 7])
    assert spread["rebalance_month"].tolist() == [1, 4, 7]
    assert pd.to_datetime(spread["first_rebalance"]).dt.month.tolist() == [1, 4, 7]
    assert spread[["port_cagr", "univ_cagr", "excess_vs_univ"]].notna().all().all()
