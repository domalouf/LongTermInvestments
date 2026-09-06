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


def test_rolling_backtest_spans_all_windows(fund, panel):
    from lti.rolling import RollingConfig, run_rolling_backtest

    cfg = RollingConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=3),
        window_years=[3, 5],
        step_months=12,
        start="2012-01-01",
        end="2020-01-01",
        market_cap_min=0.0,
    )
    result = run_rolling_backtest(cfg, fund=fund, price_panel=panel)

    assert set(result.windows["window_years"]) == {3, 5}
    assert set(result.summary["window_years"]) == {3, 5}
    # more 3y windows fit in the same span than 5y windows
    counts = result.summary.set_index("window_years")["n_windows"]
    assert counts[3] > counts[5]
    # every window's end date is exactly window_years after its start
    deltas = (result.windows["end"] - result.windows["start"]).dt.days / 365.25
    assert np.allclose(deltas, result.windows["window_years"], atol=0.05)
    # cheap-debt names beat the flat benchmark in this toy world, in (almost) every window
    assert (result.summary["win_rate"] > 0.9).all()


def test_rolling_backtest_rejects_window_longer_than_history(fund, panel):
    from lti.rolling import RollingConfig, run_rolling_backtest

    cfg = RollingConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=3),
        window_years=[50],
        start="2012-01-01",
        end="2020-01-01",
        market_cap_min=0.0,
    )
    with pytest.raises(RuntimeError):
        run_rolling_backtest(cfg, fund=fund, price_panel=panel)


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


def test_stock_resolve_and_annual(fund):
    from lti import stock

    cik, sym = stock.resolve(fund, "t3")  # case-insensitive
    assert (cik, sym) == (3, "T3")
    assert stock.resolve(fund, "NOPE") == (None, "NOPE")

    annual = stock.annual_fundamentals(fund, cik)
    assert list(annual["fiscal_year"]) == sorted(annual["fiscal_year"])
    assert annual["period_end"].is_unique
    assert annual["roe"].notna().all()
    assert "T3" in stock.list_tickers(fund)


def test_stock_valuation_history_split_adjust(fund, panel):
    from lti import stock

    annual = stock.annual_fundamentals(fund, 1)
    val = stock.valuation_history(annual, panel, "T1", freq="ME")
    assert not val.empty
    assert (val["price"] > 0).all()
    assert val["pe"].notna().any()
    assert (val["pe"].dropna() > 0).all()
    assert stock.valuation_history(annual, panel, "ZZZ").empty

    # a 2:1 split after every early filing halves pre-split EPS -> doubles P/E
    splits = pd.Series([2.0], index=pd.to_datetime(["2099-01-01"]))
    val_adj = stock.valuation_history(annual, panel, "T1", freq="ME", splits=splits)
    pe_ratio = (val_adj["pe"].dropna() / val["pe"].dropna()).dropna()
    assert np.allclose(pe_ratio, 2.0)


def test_peg_metric(fund):
    snap = pit.snapshot_asof(fund, pd.Timestamp("2018-06-01"))
    snap = metrics.add_fundamental_metrics(snap)
    snap = metrics.add_price_metrics(snap, price=pd.Series(100.0, index=snap.index))
    assert "peg" in snap.columns
    assert (snap["peg"].dropna() > 0).all()


def test_valuation_models_math():
    from lti import valuation as val

    df = pd.DataFrame(
        {
            "eps": [5.0, -1.0, 4.0],
            "book_value_per_share": [20.0, 10.0, 0.0],
            "free_cash_flow": [1000.0, 500.0, 800.0],
            "shares_outstanding": [100.0, 100.0, 100.0],
            "dividends_paid": [-200.0, 0.0, -100.0],
            "eps_growth_1y": [0.10, 0.10, 0.50],
        },
        index=pd.Index([1, 2, 3], name="cik"),
    )
    price = pd.Series({1: 50.0, 2: 8.0, 3: 40.0})
    a = val.ValuationAssumptions(discount_rate=0.10, terminal_growth=0.02, dcf_years=10, growth_cap=0.15)
    out = val.add_valuation_models(df, price, assumptions=a)

    assert out.loc[1, "graham_number"] == pytest.approx((22.5 * 5 * 20) ** 0.5)
    assert np.isnan(out.loc[2, "graham_number"])  # negative eps
    assert np.isnan(out.loc[3, "graham_number"])  # zero bvps

    assert out.loc[1, "epv_value"] == pytest.approx(5.0 / 0.10)
    assert np.isnan(out.loc[2, "epv_value"])

    # Lynch: eps * (g% + div_yield%) = 5 * (10 + 200/100/50*100)
    assert out.loc[1, "lynch_fair_value"] == pytest.approx(5.0 * (10.0 + 4.0))

    assert out.loc[1, "dcf_value"] > out.loc[1, "epv_value"]  # growth adds value
    assert out.loc[1, "fair_value_est_upside"] == pytest.approx(out.loc[1, "fair_value_est"] / 50.0 - 1)
    assert np.isnan(out.loc[2, "ddm_value"])  # no dividend


def test_historical_cagr(fund):
    from lti import stock
    from lti.valuation import historical_cagr

    annual = stock.annual_fundamentals(fund, 1)
    assert historical_cagr(annual, "revenues", 5) == pytest.approx(0.0)  # flat in the fixture
    assert np.isnan(historical_cagr(annual, "not_a_column"))


def test_rank_undervalued(fund, panel):
    from lti.valuation import ValuationAssumptions, rank_undervalued

    # price the toy names cheap so the models flag big upside
    cheap = panel.copy()
    for c in [c for c in cheap.columns if c != "SPY"]:
        cheap[c] = cheap[c] * 0.02

    ranked = rank_undervalued(
        fund, cheap, "2020-06-01",
        assumptions=ValuationAssumptions(discount_rate=0.10),
        market_cap_min=0.0, min_models=2, max_upside=None, top_n=5,
    )
    assert not ranked.empty
    assert list(ranked["rank"]) == sorted(ranked["rank"])
    # sorted by upside, descending
    up = ranked["fair_value_est_upside"].to_numpy()
    assert (up[:-1] >= up[1:]).all()
    assert (ranked["n_models"] >= 2).all()
    assert (ranked["fair_value_est_upside"] > 0).all()


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
