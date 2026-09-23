"""Factor attribution (lti.attribution): Ken French's files, the regression, and a backtest's four legs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import attribution

# the shape of the data library's CSVs: description, a header row opening with a
# comma, the monthly block in percent, then the same factors year by year
FIVE = """This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202607 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RMW,CMA,RF
201301,   5.57,   0.36,   0.93,  -1.24,   1.36,   0.00
201302,   1.29,  -0.46,   0.02,  -0.25,   0.30,   0.00
201303,   4.03,   0.80,  -0.29,  -0.44,   0.41,   0.00

  Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
  2013,  35.20,   7.29,   1.38,  -3.43,   5.49,   0.02
"""
MOMENTUM = """This file was created by CMPT_ME_PRIOR_RETS using the 202607 CRSP database.
Missing data are indicated by -99.99 or -999.

,Mom
201301,   -1.87
201302,  -99.99
201303,    0.41

Annual Factors:

,Mom
  2013,   6.29
"""


def test_the_monthly_block_of_a_french_file_parses_to_fractions():
    five = attribution.parse_french_csv(FIVE)
    assert list(five.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert list(five.index) == list(pd.to_datetime(["2013-01-31", "2013-02-28", "2013-03-31"]))  # annual block ignored
    assert five.loc["2013-01-31", "mkt_rf"] == pytest.approx(0.0557)

    mom = attribution.parse_french_csv(MOMENTUM)
    assert list(mom.columns) == ["mom"]
    assert np.isnan(mom.loc["2013-02-28", "mom"])  # -99.99 is missing, not a 99.99% loss
    with pytest.raises(ValueError):
        attribution.parse_french_csv("no factors in here\n")


def _factors(months: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.normal(0, 0.03, (len(months), 6)), index=months,
                     columns=["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
    f["mkt_rf"] += 0.006
    f["rf"] = 0.001
    return f


def test_the_regression_recovers_known_loadings_and_alpha():
    months = pd.date_range("2010-01-31", periods=180, freq="ME")
    f = _factors(months)
    noise = np.random.default_rng(1).normal(0, 0.002, len(months))
    y = 0.003 + 1.2 * f["mkt_rf"] + 0.4 * f["smb"] - 0.3 * f["hml"] + noise
    fit = attribution.regress(y, f[["mkt_rf", "smb", "hml"]])
    assert fit["coef"]["alpha"] == pytest.approx(0.003, abs=5e-4)
    assert fit["coef"][["mkt_rf", "smb", "hml"]].tolist() == pytest.approx([1.2, 0.4, -0.3], abs=0.02)
    assert fit["t"]["alpha"] > 5 and fit["r2"] > 0.95 and fit["n"] == 180
    assert attribution.regress(y.iloc[:10], f[["mkt_rf", "smb", "hml"]].iloc[:10]) is None  # too short


def test_monthly_returns_keep_whole_months_only():
    # a backtest curve: the rebalance day, month ends, and the next rebalance day
    idx = pd.DatetimeIndex(["2013-04-01", "2013-04-30", "2013-05-31", "2013-06-28", "2013-07-01"])
    curve = pd.Series([100.0, 102.0, 101.0, 105.0, 104.0], index=idx)
    r = attribution.monthly_returns(curve)
    assert list(r.index) == list(pd.to_datetime(["2013-04-30", "2013-05-31", "2013-06-30"]))  # July's 1 day dropped
    assert r.tolist() == pytest.approx([0.02, 101 / 102 - 1, 105 / 101 - 1])


def _curve(returns: pd.Series, start: str) -> pd.Series:
    """An equity curve from monthly returns, opening on a first-of-month rebalance day."""
    values = 100 * (1 + returns).cumprod()
    return pd.concat([pd.Series({pd.Timestamp(start): 100.0}), values])


def test_a_backtests_edge_splits_into_tilts_and_alpha():
    months = pd.date_range("2012-04-30", periods=120, freq="ME")
    f = _factors(months, seed=2)
    wobble = np.random.default_rng(3).normal(0, 0.001, (3, len(months)))
    spy = f["rf"] + f["mkt_rf"] + wobble[0]
    universe = f["rf"] + f["mkt_rf"] + 0.3 * f["smb"] + wobble[1]
    # the strategy adds a value tilt to its universe, and 2.4% a year the tilts don't explain
    strategy = universe + 0.5 * f["hml"] + 0.002 + wobble[2]
    result = SimpleNamespace(
        equity_curve=_curve(strategy, "2012-04-02"),
        universe_curve=_curve(universe, "2012-04-02"),
        benchmark_curve=_curve(spy, "2012-04-02"),
    )
    a = attribution.attribute(result, f)
    edge = a.coef["Strategy − universe"]
    assert edge["hml"] == pytest.approx(0.5, abs=0.03)
    assert edge["alpha"] == pytest.approx(0.024, abs=0.004)  # a year
    assert abs(edge["mkt_rf"]) < 0.03 and abs(edge["smb"]) < 0.03
    assert a.t.loc["alpha", "Strategy − universe"] > 3
    # the universe owns the size tilt; SPY is the market and nothing else
    assert a.coef.loc["smb", "Universe"] == pytest.approx(0.3, abs=0.03)
    assert a.coef.loc["mkt_rf", "SPY"] == pytest.approx(1.0, abs=0.03)
    assert abs(a.coef.loc["alpha", "SPY"]) < 0.005
    assert (a.n == 120).all() and a.months == (months[0], months[-1])

    capm = attribution.attribute(result, f, model="capm")
    assert list(capm.coef.index) == ["alpha", "mkt_rf"]
    with pytest.raises(ValueError):
        attribution.attribute(result, pd.DataFrame())  # nothing fetched
    no_mom = attribution.attribute(result, f.drop(columns="mom"))
    assert "mom" not in no_mom.coef.index and any("mom" in w for w in no_mom.warnings)
