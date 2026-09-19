"""Normalized earnings: the point-in-time history, the valuation built on it, and
the Undervalued list's consistency rules.

Pure logic — no SEC data or network needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import history, pit, ranking, tickers
from lti.backtest import BacktestConfig, run_backtest
from lti.fundamentals import _add_prior_year
from lti.prices import PriceData, empty_splits
from lti.ranking import ScreenSpec
from lti.valuation import MAX_UPSIDE, add_fair_value_metrics, add_valuation_models, rank_undervalued

YEARS = range(2014, 2025)
ASOF = "2025-06-02"


def _row(cik, ticker, fy, *, eps, op=None, ni=None, revenue=1000.0, shares=100.0, fcf=None, filed=None):
    ni = eps * shares if ni is None else ni
    op = ni * 1.3 if op is None else op
    fcf = ni * 0.9 if fcf is None else fcf
    return dict(
        cik=cik, ticker=ticker, company=f"{ticker} Inc", sic=2000, form="10-K", fiscal_year=fy,
        period_end=pd.Timestamp(fy, 12, 31), filed=filed or pd.Timestamp(fy + 1, 3, 1),
        revenues=revenue, net_income=ni, eps=eps, operating_income=op, operating_income_reported=op,
        equity=1000.0, liabilities=500.0, assets_current=400.0, liabilities_current=200.0, cash=50.0,
        debt_current=0.0, total_debt=100.0, ppe_net=600.0, shares_outstanding=shares,
        cfo=fcf + 50.0, capex=50.0, free_cash_flow=fcf, dividends_paid=-40.0,
    )


@pytest.fixture
def world():
    rows = []
    for fy in YEARS:
        # a steady earner, cheap on any basis
        rows.append(_row(1, "STDY", fy, eps=2.0, revenue=1000 * 1.05 ** (fy - 2014)))
        # a normal $1.00 earner whose latest year is a $5.00 peak
        rows.append(_row(2, "PEAK", fy, eps=5.0 if fy == 2024 else 1.0))
        # losses, then a one-off gain on an operating loss
        rows.append(_row(3, "ONCE", fy, eps=5.0 if fy == 2024 else -1.0, op=-50.0))
        # profitable until the latest year
        rows.append(_row(4, "SLIP", fy, eps=-0.5 if fy == 2024 else 1.5))
    fund = _add_prior_year(pd.DataFrame(rows))

    # each at its level on the as-of date the list tests use, trending at its own rate
    idx = pd.bdate_range("2013-01-01", "2025-12-31")
    years = (idx - pd.Timestamp(ASOF)).days / 365.25
    level = {"STDY": (10.0, 0.10), "PEAK": (15.0, 0.02), "ONCE": (10.0, -0.05), "SLIP": (12.0, 0.05), "SPY": (100.0, 0.07)}
    panel = pd.DataFrame({t: p * np.exp(mu * years) for t, (p, mu) in level.items()}, index=idx)
    return fund, PriceData(adj=panel, close=panel, splits=empty_splits())


# --- the history ---------------------------------------------------------------


def test_history_is_point_in_time():
    rows = [_row(1, "A", 2020, eps=1.0), _row(1, "A", 2021, eps=2.0), _row(1, "A", 2022, eps=3.0),
            _row(1, "A", 2021, eps=9.0, filed=pd.Timestamp(2024, 6, 1))]  # a 10-K/A filed later
    fund = pd.DataFrame(rows)
    h = history.history_asof(fund, "2023-06-01", years=2)
    assert list(h["fiscal_year"]) == [2021, 2022]
    assert list(h["eps"]) == [2.0, 3.0]  # the restatement isn't known yet
    assert list(history.history_asof(fund, "2025-01-01", years=2)["eps"]) == [9.0, 3.0]


def test_normalized_eps_is_the_median_on_todays_share_basis():
    rows = [_row(1, "A", fy, eps=e) for fy, e in zip(range(2019, 2024), [10.0, 12.0, 11.0, 2.4, 2.6])]
    splits = pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2022-06-01")], "ratio": [5.0]})
    s = history.summarize_history(pd.DataFrame(rows), splits).iloc[0]
    # 10, 12, 11 were filed before a 5:1 split -> 2.0, 2.4, 2.2 on today's basis
    assert s["eps_norm"] == pytest.approx(2.4)
    assert s["history_years"] == 5 and s["profit_years"] == 5


def test_a_profitable_year_needs_every_reported_figure_positive():
    rows = [
        _row(1, "A", 2020, eps=1.0),
        _row(1, "A", 2021, eps=5.0, op=-50.0),  # a one-off gain on an operating loss
        _row(1, "A", 2022, eps=1.0, ni=-10.0),  # EPS sign disagrees with net income
        _row(1, "A", 2023, eps=np.nan, ni=80.0, op=np.nan),  # only net income reported
    ]
    s = history.summarize_history(pd.DataFrame(rows)).iloc[0]
    assert (s["profit_years"], s["history_years"]) == (2, 4)


def test_too_few_years_give_no_norm():
    rows = [_row(1, "A", 2022, eps=1.0), _row(1, "A", 2023, eps=2.0)]
    s = history.summarize_history(pd.DataFrame(rows)).iloc[0]
    assert np.isnan(s["eps_norm"]) and np.isnan(s["fcf_conversion"])


def test_growth_conversion_and_the_latest_year_against_the_norm():
    rows = [_row(1, "A", fy, eps=e, revenue=r, fcf=f)
            for fy, e, r, f in zip(range(2019, 2024), [1, 1, 1, 1, 3.0], [100, 110, 121, 133.1, 146.41],
                                   [90.0, 90.0, 90.0, 90.0, 90.0])]
    s = history.summarize_history(pd.DataFrame(rows)).iloc[0]
    assert s["revenue_cagr"] == pytest.approx(0.10, abs=0.002)
    assert s["fcf_conversion"] == pytest.approx(450 / 700)
    assert s["eps_vs_norm"] == pytest.approx(3.0)


# --- valuing on it ---------------------------------------------------------------


def test_a_peak_year_no_longer_inflates_fair_value(world):
    fund, px = world
    snap = pit.priced_snapshot(fund, ASOF, px, with_history=True).set_index("ticker")
    peak = snap.loc["PEAK"]
    assert peak["eps"] == 5.0 and peak["eps_norm"] == 1.0
    norm = add_valuation_models(snap, snap["price"], basis="normalized").loc["PEAK", "fair_value_est"]
    latest = add_valuation_models(snap, snap["price"], basis="latest").loc["PEAK", "fair_value_est"]
    assert latest > 3 * norm


def test_normalized_valuation_needs_the_history():
    df = pd.DataFrame({"eps": [1.0], "shares_outstanding": [10.0]})
    with pytest.raises(KeyError, match="history"):
        add_valuation_models(df, pd.Series([10.0]))
    with pytest.raises(ValueError, match="basis"):
        add_valuation_models(df, pd.Series([10.0]), basis="yearly")


def test_fair_value_metrics_drop_what_the_list_would_drop(world):
    fund, px = world
    snap = pit.priced_snapshot(fund, ASOF, px, with_history=True).set_index("ticker")
    assert {"fair_value_upside", "fair_value_upside_1y"} <= set(snap.columns)
    assert (snap["fair_value_upside"].dropna() <= MAX_UPSIDE).all()
    cheap = snap.assign(price=snap["price"] / 1000)  # absurd upside is a data error, not a bargain
    out = add_fair_value_metrics(cheap)
    assert out["fair_value_upside"].isna().all()


def test_the_list_keeps_steady_earners_only(world):
    fund, px = world
    r = rank_undervalued(fund, px, ASOF, market_cap_min=0.0, top_n=None)
    # ONCE: no profitable year; SLIP: losing money now; PEAK: cheap only on its peak year
    assert list(r["ticker"]) == ["STDY"]
    on_latest = rank_undervalued(fund, px, ASOF, market_cap_min=0.0, top_n=None, basis="latest")
    assert "PEAK" in set(on_latest["ticker"])
    loose = rank_undervalued(fund, px, ASOF, market_cap_min=0.0, top_n=None,
                             require_positive_eps=False, min_profit_years=None)
    assert "SLIP" in set(loose["ticker"])


def test_the_list_leaves_out_bdcs_with_financials(world):
    fund, px = world
    fund = fund.copy()
    fund.loc[fund["ticker"] == "STDY", "sic"] = np.nan  # BDCs carry no SIC
    r = rank_undervalued(fund, px, ASOF, market_cap_min=0.0, top_n=None)
    assert "STDY" not in set(r["ticker"])
    kept = rank_undervalued(fund, px, ASOF, market_cap_min=0.0, top_n=None, exclude_financials=False)
    assert "STDY" in set(kept["ticker"])


# --- ranking and testing on it -----------------------------------------------------


def test_min_profit_years_needs_the_history(world):
    fund, px = world
    snap = pit.priced_snapshot(fund, ASOF, px)
    spec = ScreenSpec(metrics=["pe"], top_n=5, filters={"min_profit_years": 4})
    with pytest.raises(KeyError, match="history"):
        ranking.select(snap, spec)
    with_hist = pit.priced_snapshot(fund, ASOF, px, with_history=True)
    assert "ONCE" not in ranking.select(with_hist, spec)


def test_backtest_can_rank_on_fair_value_upside(world):
    fund, px = world
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["fair_value_upside"], top_n=1, filters={"min_profit_years": 4}),
        start="2019-01-01", end="2025-06-30", market_cap_min=0.0,
    )
    r = run_backtest(cfg, fund, px)
    assert set(r.holdings["ticker"]) == {"STDY"}
    assert "fair_value_upside" in r.holdings.columns


def test_factor_analysis_measures_the_fair_value_signal(world):
    from lti.factor import ICConfig, compute_ic

    cfg = ICConfig(metrics=["fair_value_upside", "fair_value_upside_1y", "pe_norm", "profit_years"],
                   start="2019-04-01", end="2025-06-30", market_cap_min=0.0, min_names=2, quantiles=2)
    result = compute_ic(cfg, fund=world[0], px=world[1])
    assert {"fair_value_upside", "fair_value_upside_1y"} <= set(result.ic_by_period.columns)


# --- the ticker map ------------------------------------------------------------------


def test_the_first_listed_ticker_is_the_primary():
    rows = pd.DataFrame(
        {
            "cik": [1137774, 19617, 1137774, 19617, 1067983, 1067983],
            "ticker": ["PRU", "JPM", "PFH", "JPM-PC", "BRK-B", "BRK-A"],
            "company": ["PRUDENTIAL FINANCIAL INC", "JPMORGAN CHASE & CO"] * 2 + ["BERKSHIRE HATHAWAY INC"] * 2,
        }
    )
    m = tickers.collapse_ticker_list(rows).set_index("cik")
    assert m.loc[1137774, "ticker"] == "PRU"  # not the baby bond PFH, which sorts first
    assert m.loc[19617, "ticker"] == "JPM"
    assert m.loc[1067983, "ticker"] == "BRK-B"
    assert m.loc[1137774, "tickers_all"] == "PFH,PRU"
