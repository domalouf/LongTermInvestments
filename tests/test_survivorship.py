"""Measuring what the backtests can't see: the census, and the bounds on what it hides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import survivorship as sv
from lti.prices import PriceData, empty_splits, rebase_jumps


def _k(cik, year, *, ticker=None, net_income=100.0, equity=1000.0, revenues=1e9, eps=1.0):
    return dict(cik=cik, ticker=ticker, company=f"Co {cik}", form="10-K", sic=2000, fiscal_year=year,
                period_end=pd.Timestamp(year, 12, 31), filed=pd.Timestamp(year + 1, 3, 1), revenues=revenues,
                net_income=net_income, equity=equity, eps=eps, assets=3000.0, liabilities=2000.0,
                operating_income=150.0, shares_outstanding=1e8, cfo=120.0)


@pytest.fixture
def world():
    """Sixty survivors that file through 2023, and five kinds of company a backtest never sees."""
    rows = [_k(c, y, ticker=f"T{c}") for c in range(1, 61) for y in range(2010, 2024)]
    rows += [_k(101, y) for y in range(2010, 2014)]  # acquired: healthy, last 10-K in March 2014
    rows += [_k(102, y, net_income=-50.0 if y == 2013 else 100.0) for y in range(2010, 2014)]  # failed
    rows += [_k(103, y) for y in range(2010, 2018)]  # left later, in 2018
    rows += [_k(104, y, eps=np.nan) for y in range(2010, 2024)]  # a subsidiary with public debt: no shares
    rows += [_k(105, y) for y in range(2010, 2024)]  # still filing, never listed
    rows += [_k(106, y, revenues=1e8) for y in range(2010, 2014)]  # too small
    fund = pd.DataFrame(rows)
    idx = pd.bdate_range("2010-01-01", "2024-12-31")
    panel = pd.DataFrame({f"T{c}": 10.0 * 1.10 ** ((idx - idx[0]).days / 365.25) for c in range(1, 61)}, index=idx)
    return fund, PriceData(adj=panel, close=panel, splits=empty_splits())


def test_the_census_sorts_the_missing_by_when_and_how_they_left(world):
    fund, px = world
    u = sv.classify(fund, px, "2014-04-01")
    status = u["status"]
    assert (status == "priced").sum() == 60
    assert status[101] == "exit_healthy" and status[102] == "exit_distressed" and status[103] == "later_gone"
    assert not {104, 105, 106} & set(u.index)
    assert u.attrs["excluded"]["no_listed_equity"] == 1 and u.attrs["excluded"]["unlisted_still_filing"] == 1
    # a year on, the two that left are no longer anyone's to hold
    later = sv.classify(fund, px, "2015-06-01")
    assert not {101, 102} & set(later.index) and later.attrs["excluded"]["already_gone"] == 2


def test_putting_the_missing_back_moves_the_universe_by_what_the_scenario_says(world):
    fund, px = world
    r = sv.analyze(fund, px, start=2014, end=2014)
    row, census = r.universe.loc[2014], r.census.loc[2014]
    assert census["companies"] == 63 and census["exit_healthy"] == pytest.approx(1 / 63)
    assert row["survivors"] == pytest.approx(0.10, abs=0.002)
    harsh = next(s for s in sv.SCENARIOS if s.name == "harsh")
    expected = (61 * row["survivors"] + harsh.healthy + harsh.distressed) / 63
    assert row["harsh"] == pytest.approx(expected)
    assert row["harsh"] < row["central"] < row["kind"]


def test_a_factor_whose_best_fifth_holds_the_failures_loses_its_edge(world):
    fund, px = world
    fund = fund.copy()
    # the failing company has the highest "profitability" there is; the acquired one the lowest
    fund["operating_income"] = 100.0 + fund["cik"]  # no ties: the best fifth is thirteen companies
    fund.loc[fund["cik"] == 102, "operating_income"] = 2000.0
    fund.loc[fund["cik"] == 101, "operating_income"] = -2000.0
    f = sv.analyze(fund, px, start=2014, end=2014).factors.loc["op_profitability"]
    assert f["exit_distressed"] > 0 and f["survivors"] == pytest.approx(0.0, abs=1e-9)
    assert f["harsh"] < f["central"] < 0  # no edge among the survivors, a loss once the failure counts


def test_scenario_returns_ignore_empty_groups():
    assert np.isnan(sv._scenario_return(pd.Series(dtype=float), 0.1, sv.SCENARIOS[0]))
    only_priced = pd.Series({"priced": 10})
    assert sv._scenario_return(only_priced, 0.1, sv.SCENARIOS[0]) == pytest.approx(0.1)


# --- the seam Yahoo leaves where a bankrupt company's new shares take its ticker ---


def test_a_bankruptcy_seam_is_not_a_return():
    idx = pd.bdate_range("2020-11-10", periods=8)
    close = pd.DataFrame({"OAS": [3, 2, 1, 0.12, 31, 30, 33, 36.0], "OK": [10, 11, 10, 12, 11, 12, 13, 12.0]}, index=idx)
    adj, cut, found = rebase_jumps(close * 0.9, close)
    assert found[["ticker"]].values.tolist() == [["OAS"]] and found["ratio"].iloc[0] == pytest.approx(31 / 0.12)
    assert cut["OAS"].iloc[:4].isna().all() and cut["OAS"].iloc[4:].tolist() == [31, 30, 33, 36.0]
    # held across the seam: the fall before it, nothing for the seam, the new shares after
    assert adj["OAS"].iloc[4] / adj["OAS"].iloc[3] == pytest.approx(1.0)
    assert adj["OAS"].iloc[3] / adj["OAS"].iloc[0] == pytest.approx(0.12 / 3)
    pd.testing.assert_series_equal(adj["OK"], close["OK"] * 0.9)
    pd.testing.assert_series_equal(cut["OK"], close["OK"])


def test_ordinary_moves_and_empty_panels_are_left_alone():
    idx = pd.bdate_range("2021-01-25", periods=4)
    close = pd.DataFrame({"GME": [20.0, 77.0, 148.0, 347.0]}, index=idx)  # a real squeeze: under 5x a day
    adj, cut, found = rebase_jumps(close, close)
    assert found.empty and cut.equals(close) and adj.equals(close)
    assert rebase_jumps(pd.DataFrame(), pd.DataFrame())[2].empty
