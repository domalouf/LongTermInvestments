"""Unit tests for the pure logic (no SEC data / network needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import metrics, pit, ranking
from lti.backtest import BacktestConfig, run_backtest
from lti.performance import cagr, max_drawdown, sharpe
from lti.ranking import ScreenSpec


@pytest.fixture
def fund() -> pd.DataFrame:
    rows = []
    for cik in range(1, 7):
        for fy in range(2010, 2021):
            rows.append(
                dict(
                    cik=cik,
                    ticker=f"T{cik}",
                    tickers_all=f"T{cik}",
                    company=f"Co{cik}",
                    adsh=f"{cik}-{fy}",
                    form="10-K",
                    fiscal_year=fy,
                    period_end=pd.Timestamp(fy, 12, 31),
                    filed=pd.Timestamp(fy + 1, 3, 1),
                    revenues=1000 + 100 * cik,
                    net_income=80 + cik * 5,
                    equity=500 + cik * 10,
                    liabilities=200 + cik * 30,
                    assets_current=300.0,
                    liabilities_current=150.0,
                    gross_profit=400.0,
                    shares_outstanding=100.0,
                    eps=(80 + cik * 5) / 100.0,
                    cfo=90.0,
                    capex=20.0,
                    revenues_prev=950 + 100 * cik,
                    net_income_prev=75 + cik * 5,
                    eps_prev=(75 + cik * 5) / 100.0,
                    equity_prev=490 + cik * 10,
                )
            )
    df = pd.DataFrame(rows)
    df["free_cash_flow"] = df.cfo - df.capex.abs()
    return df


@pytest.fixture
def panel() -> pd.DataFrame:
    idx = pd.bdate_range("2009-01-01", "2021-06-30")
    p = pd.DataFrame(index=idx)
    for cik in range(1, 7):
        drift = 1.0 + 0.0003 * (7 - cik)  # cheaper (low cik) compounds faster
        p[f"T{cik}"] = 100 * np.cumprod(np.full(len(idx), drift))
    p["SPY"] = 100 * np.cumprod(np.full(len(idx), 1.0002))
    return p


def test_snapshot_no_lookahead(fund):
    for asof in pd.to_datetime(["2013-05-01", "2016-06-01", "2019-11-30"]):
        snap = pit.snapshot_asof(fund, asof)
        assert snap["filed"].max() <= asof
    snap = pit.snapshot_asof(fund, pd.Timestamp("2016-06-01"))
    assert (snap["fiscal_year"] == 2015).all()


def test_fundamental_metrics(fund):
    snap = pit.snapshot_asof(fund, pd.Timestamp("2018-06-01"))
    snap = metrics.add_fundamental_metrics(snap)
    assert (snap["debt_to_equity"] > 0).all()
    assert (snap["current_ratio"] == 2.0).all()
    assert snap["roe"].notna().all()


def test_ranking_prefers_cheap(fund):
    snap = pit.snapshot_asof(fund, pd.Timestamp("2018-06-01"))
    snap = metrics.add_fundamental_metrics(snap)
    picks = ranking.select(snap, ScreenSpec(metrics=["debt_to_equity"], top_n=3))
    assert picks == ["T1", "T2", "T3"]


def test_backtest_runs_and_is_deterministic(fund, panel):
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["pe", "debt_to_equity"], top_n=3),
        start="2012-01-01",
        end="2021-01-01",
        market_cap_min=0.0,
    )
    r1 = run_backtest(cfg, fund=fund, price_panel=panel)
    r2 = run_backtest(cfg, fund=fund, price_panel=panel)
    assert r1.holdings.equals(r2.holdings)
    assert len(r1.equity_curve) > 5
    assert r1.equity_curve.iloc[0] == cfg.initial_capital
    # cheap names outperform the flat benchmark in this toy world
    assert r1.stats["port_cagr"] > r1.stats["bench_cagr"]
    assert (r1.period_summary["n_selected"] == 3).all()


def test_factor_ic_detects_monotonic_signal(fund, panel):
    from lti.factor import ICConfig, compute_ic

    cfg = ICConfig(
        metrics=["debt_to_equity"],
        start="2012-01-01",
        end="2020-06-30",
        horizon_months=12,
        step_months=12,
        market_cap_min=0.0,
        min_names=5,
        quantiles=3,
    )
    result = compute_ic(cfg, fund=fund, price_panel=panel)

    row = result.summary.loc["debt_to_equity"]
    # low debt/equity (low cik) compounds fastest in the toy panel -> strongly negative IC
    assert row["mean_ic"] < -0.9
    assert row["hit_rate"] == 1.0
    assert row["n_periods"] >= 5
    buckets = result.bucket_returns.loc["debt_to_equity"].dropna()
    assert buckets.iloc[0] > buckets.iloc[-1]  # Q1 (cheap debt) beats Q3


def test_factor_ic_is_deterministic_and_scoped(fund, panel):
    from lti.factor import ICConfig, compute_ic

    cfg = ICConfig(
        metrics=["pe", "roe", "debt_to_equity", "not_a_metric"],
        start="2013-01-01",
        end="2020-06-30",
        market_cap_min=0.0,
        min_names=5,
        quantiles=3,
    )
    r1 = compute_ic(cfg, fund=fund, price_panel=panel)
    r2 = compute_ic(cfg, fund=fund, price_panel=panel)
    assert r1.ic_by_period.equals(r2.ic_by_period)
    assert set(r1.summary.index) <= {"pe", "roe", "debt_to_equity"}
    assert any("not_a_metric" in w for w in r1.warnings)


def test_performance_helpers():
    curve = pd.Series(
        [100, 110, 90, 120, 130],
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]),
    )
    assert cagr(curve) > 0
    mdd, peak, trough = max_drawdown(curve)
    assert mdd == pytest.approx(-90 / 110 + 1 - 1, rel=1e-6) or mdd < 0
    assert peak < trough
    assert np.isfinite(sharpe(curve))
