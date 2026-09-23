"""Unit tests for the pure logic (no SEC data / network needed)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import metrics, pit, ranking
from lti.backtest import BacktestConfig, run_backtest
from lti.prices import PriceData, empty_splits
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


@pytest.fixture
def px(panel) -> PriceData:
    """No dividends and no splits in the toy world: both price series coincide."""
    return PriceData(adj=panel, close=panel, splits=empty_splits())


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


def test_backtest_runs_and_is_deterministic(fund, px):
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["pe", "debt_to_equity"], top_n=3),
        start="2012-01-01",
        end="2021-01-01",
        market_cap_min=0.0,
    )
    r1 = run_backtest(cfg, fund=fund, px=px)
    r2 = run_backtest(cfg, fund=fund, px=px)
    assert r1.holdings.equals(r2.holdings)
    assert len(r1.equity_curve) > 5
    assert r1.equity_curve.iloc[0] == cfg.initial_capital
    # cheap names outperform the flat benchmark in this toy world
    assert r1.stats["port_cagr"] > r1.stats["bench_cagr"]
    assert (r1.period_summary["n_selected"] == 3).all()


def test_rolling_backtest_spans_all_windows(fund, px):
    from lti.rolling import RollingConfig, run_rolling_backtest

    cfg = RollingConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=3),
        window_years=[3, 5],
        step_months=12,
        start="2012-01-01",
        end="2020-01-01",
        market_cap_min=0.0,
    )
    result = run_rolling_backtest(cfg, fund=fund, px=px)

    assert set(result.windows["window_years"]) == {3, 5}
    assert set(result.summary["window_years"]) == {3, 5}
    # more 3y windows fit in the same span than 5y windows
    counts = result.summary.set_index("window_years")["n_windows"]
    assert counts[3] > counts[5]
    # every window's end date is exactly window_years after its start
    deltas = (result.windows["end"] - result.windows["start"]).dt.days / 365.25
    assert np.allclose(deltas, result.windows["window_years"], atol=0.05)
    # cheap-debt names beat the flat benchmark in this toy world, in (almost) every window
    assert (result.summary["win_rate_vs_bench"] > 0.9).all()
    # ...and each window also measures the universe the screen picked from
    assert result.windows["univ_cagr"].notna().all()
    assert (result.summary["win_rate_vs_univ"] > 0.9).all()


def test_rolling_backtest_rejects_window_longer_than_history(fund, px):
    from lti.rolling import RollingConfig, run_rolling_backtest

    cfg = RollingConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=3),
        window_years=[50],
        start="2012-01-01",
        end="2020-01-01",
        market_cap_min=0.0,
    )
    with pytest.raises(RuntimeError):
        run_rolling_backtest(cfg, fund=fund, px=px)


def test_factor_ic_detects_monotonic_signal(fund, px):
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
    result = compute_ic(cfg, fund=fund, px=px)

    row = result.summary.loc["debt_to_equity"]
    # low debt/equity (low cik) compounds fastest in the toy panel -> strongly negative IC
    assert row["mean_ic"] < -0.9
    assert row["hit_rate"] == 1.0
    assert row["n_periods"] >= 5
    buckets = result.bucket_returns.loc["debt_to_equity"].dropna()
    assert buckets.iloc[0] > buckets.iloc[-1]  # Q1 (cheap debt) beats Q3


def test_factor_ic_is_deterministic_and_scoped(fund, px):
    from lti.factor import ICConfig, compute_ic

    cfg = ICConfig(
        metrics=["pe", "roe", "debt_to_equity", "not_a_metric"],
        start="2013-01-01",
        end="2020-06-30",
        market_cap_min=0.0,
        min_names=5,
        quantiles=3,
    )
    r1 = compute_ic(cfg, fund=fund, px=px)
    r2 = compute_ic(cfg, fund=fund, px=px)
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
    out = val.add_valuation_models(df, price, assumptions=a, basis="latest")

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


def test_every_model_is_documented_and_explained():
    """MODEL_DOCS covers MODELS, and explain() prints the inputs each model ran on."""
    from lti import valuation as val

    assert list(val.MODEL_DOCS) == val.MODELS  # documented, in display order
    for doc in val.MODEL_DOCS.values():
        assert all(getattr(doc, f) for f in ("label", "formula", "idea", "inputs",
                                             "at_defaults", "silent", "misleads"))

    df = pd.DataFrame(
        {
            "eps": [3.05],
            "book_value_per_share": [12.40],
            "free_cash_flow": [210.0],
            "shares_outstanding": [100.0],
            "dps_ttm": [0.64],
            "eps_growth_1y": [0.043],
        },
        index=pd.Index([1], name="cik"),
    )
    out = val.add_valuation_models(df, pd.Series({1: 30.0}), basis="latest")
    row = out.loc[1]

    # the inputs are recorded as fed, so a fair value can be checked against them
    assert row["eps_used"] == pytest.approx(3.05)
    assert row["bvps_used"] == pytest.approx(12.40)
    assert row["fcf_ps_used"] == pytest.approx(2.10)
    assert row["dps_used"] == pytest.approx(0.64)

    a = val.ValuationAssumptions()
    for m in val.MODELS:
        line = val.explain(m, row, a)
        assert line.endswith(f"= ${row[m]:,.2f}")  # the equation ends at the value it produced
    assert val.explain("graham_number", row, a).startswith("√(22.5 × EPS $3.05 × BVPS $12.40)")
    assert val.explain("epv_value", row, a) == "EPS $3.05 / 9.0% = $33.89"

    # a model that can't answer says why, in its own documented words
    loss = val.add_valuation_models(
        df.assign(eps=-1.0, free_cash_flow=-50.0, dps_ttm=0.0), pd.Series({1: 30.0}), basis="latest"
    ).loc[1]
    for m in val.MODELS:
        assert val.MODEL_DOCS[m].silent in val.explain(m, loss, a)

    with pytest.raises(KeyError):
        val.explain("not_a_model", row, a)


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
        fund, PriceData(cheap, cheap, empty_splits()), "2020-06-01",
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


def _frictions_cfg(**kw) -> BacktestConfig:
    return BacktestConfig(
        screen=ScreenSpec(metrics=["pe", "debt_to_equity"], top_n=3),
        start="2012-01-01", end="2021-01-01", market_cap_min=0.0, **kw,
    )


def test_a_frictionless_backtest_is_the_gross_one(fund, px):
    r = run_backtest(_frictions_cfg(cost_bps=0.0), fund=fund, px=px)
    assert r.equity_curve.equals(r.equity_curve_gross)
    for leg in ("port", "univ", "bench"):
        assert r.stats[f"{leg}_cagr"] == r.stats[f"{leg}_cagr_gross"] == r.stats[f"{leg}_cagr_liquidated"]
        assert r.stats[f"{leg}_costs_pa"] == r.stats[f"{leg}_taxes_pa"] == 0.0
    assert (r.period_summary["port_return"] == r.period_summary["port_return_gross"]).all()


def test_costs_and_taxes_come_out_of_all_three(fund, panel):
    from lti.frictions import TaxRates

    # a 2%-a-year dividend: the price rises more slowly than the total return
    close = panel.mul(np.exp(-0.02 / 252 * np.arange(len(panel))), axis=0)
    px = PriceData(adj=panel, close=close, splits=empty_splits())
    costs = run_backtest(_frictions_cfg(cost_bps=20.0), fund=fund, px=px)
    taxed = run_backtest(_frictions_cfg(cost_bps=20.0, tax=TaxRates()), fund=fund, px=px)

    for leg in ("port", "univ", "bench"):
        assert costs.stats[f"{leg}_cagr"] < costs.stats[f"{leg}_cagr_gross"]
        assert costs.stats[f"{leg}_costs_pa"] > 0 and costs.stats[f"{leg}_taxes_pa"] == 0.0
        assert taxed.stats[f"{leg}_cagr"] < costs.stats[f"{leg}_cagr"]
        assert taxed.stats[f"{leg}_taxes_pa"] > 0
        # selling at the end taxes the gains still unrealized
        assert taxed.stats[f"{leg}_cagr_liquidated"] < taxed.stats[f"{leg}_cagr"]
    assert taxed.stats["port_cagr_gross"] == pytest.approx(costs.stats["port_cagr_gross"])
    assert taxed.equity_curve.iloc[0] == taxed.equity_curve_gross.iloc[0] == 100_000.0
    # each period's net return is what the curve did between rebalances
    ps = taxed.period_summary
    curve = taxed.equity_curve
    assert ps.loc[1, "port_return"] == pytest.approx(curve[ps.loc[1, "exit_date"]] / curve[ps.loc[1, "rebalance_date"]] - 1)
    assert (ps["costs"] > 0).all() and (ps["taxes"] > 0).all()
    assert ps.loc[0, "turnover"] == pytest.approx(0.5, abs=0.01)  # the first date only buys


def test_holding_past_a_year_makes_every_gain_long_term(fund, px):
    from lti.frictions import TaxRates

    calendar = run_backtest(_frictions_cfg(tax=TaxRates()), fund=fund, px=px)
    patient = run_backtest(_frictions_cfg(tax=TaxRates(), hold_past_one_year=True), fund=fund, px=px)

    # the first trading day of April falls a year to the day, or less, after the last one in some years
    assert calendar.stats["port_short_term_share"] > 0
    assert patient.stats["port_short_term_share"] == 0.0
    dates = pd.to_datetime(patient.period_summary["rebalance_date"])
    assert all(b > a + pd.DateOffset(years=1) for a, b in zip(dates[:-1], dates[1:]))
    assert (dates.dt.month == 4).all()  # it drifts a few days a year, not out of the month


def test_a_sell_buffer_keeps_holdings_until_they_leave_it():
    ranked = pd.DataFrame({"ticker": ["A", "B", "C", "D", "E", "F"], "composite_score": np.linspace(0.1, 0.9, 6)})

    def pick(held, sell_rank, top_n=2):
        return ranking.select_holdings(ranked, top_n, held=held, sell_rank=sell_rank)

    # no buffer to speak of: exactly the top N
    assert pick(["E", "F"], 2) == ranking.top_picks(ranked, 2) == ["A", "B"]
    # E still ranks inside the top 5, so it stays; the one free place goes to the best name not held
    assert pick(["E", "F"], 5) == ["A", "E"]
    # both inside the buffer: nothing is traded
    assert pick(["D", "E"], 5) == ["D", "E"]
    # a holding that left the universe altogether is sold
    assert pick(["Z", "C"], 4) == ["A", "C"]
    with pytest.raises(ValueError):
        pick([], 2, top_n=3)


def test_an_industry_cap_passes_over_names_whose_industry_is_full():
    ranked = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "industry": ["Shops", "Shops", "Shops", "Telecom", None, "Health"],
        }
    )

    def pick(top_n, cap, **kw):
        return ranking.select_holdings(ranked, top_n, group="industry", max_per_group=cap, **kw)

    assert pick(4, 2) == ["A", "B", "D", "E"]  # C would be a third Shops name
    assert pick(4, 1) == ["A", "D", "E", "F"]
    assert pick(3, None) == ranking.top_picks(ranked, 3)  # no cap: the plain top N
    # a name with no industry is never capped
    assert pick(6, 1) == ["A", "D", "E", "F"]
    # kept holdings count toward their industry, but aren't sold to make room
    assert pick(3, 1, held=["B", "C"], sell_rank=6) == ["B", "C", "D"]


def test_industries_follow_fama_french():
    from lti.sectors import OTHER_INDUSTRY, industry

    codes = pd.Series([5311, 5731, 4813, 2834, 3674, 7372, 3711, 1311, 4911, 4953, 2080, None], dtype="Int64")
    assert industry(codes).tolist() == [
        "Shops", "Shops", "Telecom", "Health", "Business Equipment", "Business Equipment",
        "Consumer Durables", "Energy", "Utilities", OTHER_INDUSTRY, "Consumer Non-Durables", pd.NA,
    ]


def test_an_industry_cap_spreads_a_one_industry_screen(fund, px):
    # the three least-levered companies are all retailers
    world = fund.assign(sic=np.where(fund["cik"] <= 3, 5311, 2834 + fund["cik"]))
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=4), start="2012-01-01", end="2021-01-01",
        market_cap_min=0.0, cost_bps=0.0,
    )
    plain = run_backtest(cfg, fund=world, px=px)
    capped = run_backtest(dataclasses.replace(cfg, industry_cap=0.5), fund=world, px=px)

    assert (plain.period_summary["top_industry"] == "Shops").all()
    assert (plain.period_summary["top_industry_share"] == 0.75).all()
    assert (capped.period_summary["top_industry_share"] == 0.5).all()
    assert capped.stats["port_top_industry_share"] == 0.5
    assert (capped.period_summary["n_selected"] == 4).all()  # the place goes to the next name down
    assert set(capped.holdings["ticker"]) == {"T1", "T2", "T4", "T5"}
    assert capped.holdings.groupby(["rebalance_date", "industry"]).size().max() == 2

    with pytest.raises(ValueError):
        run_backtest(dataclasses.replace(cfg, industry_cap=1.5), fund=world, px=px)
    with pytest.raises(KeyError):  # no SIC codes to cap by
        run_backtest(dataclasses.replace(cfg, industry_cap=0.5), fund=fund, px=px)


def test_a_sell_buffer_trades_less_when_the_ranking_churns(fund, px):
    # debt/equity rotates a place a year, so the cheapest two change every year
    churn = fund.assign(liabilities=200 + ((fund["cik"] + fund["fiscal_year"]) % 6) * 30)
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["debt_to_equity"], top_n=2), start="2012-01-01", end="2021-01-01", market_cap_min=0.0
    )
    plain = run_backtest(cfg, fund=churn, px=px)
    buffered = run_backtest(dataclasses.replace(cfg, sell_rank=4), fund=churn, px=px)

    assert run_backtest(dataclasses.replace(cfg, sell_rank=2), fund=churn, px=px).holdings.equals(plain.holdings)
    later = slice(1, None)  # the first rebalance only buys
    assert buffered.period_summary["turnover"][later].mean() < plain.period_summary["turnover"][later].mean()
    assert buffered.period_summary["n_held_over"][later].sum() > plain.period_summary["n_held_over"][later].sum()
    assert buffered.stats["port_costs_pa"] < plain.stats["port_costs_pa"]
    assert (buffered.period_summary["n_selected"] == 2).all()
    # a kept name can rank below the top N, never below the buffer
    assert buffered.holdings["rank"].max() <= 4 and plain.holdings["rank"].max() <= 2
    assert buffered.holdings.loc[buffered.holdings["rank"] > 2, "held_over"].all()

    with pytest.raises(ValueError):
        run_backtest(dataclasses.replace(cfg, sell_rank=1), fund=churn, px=px)


def test_dated_valuations_discount_at_the_rates_of_their_date(fund, panel):
    from lti.valuation import market_assumptions, rank_undervalued

    cheap = panel.copy()
    for c in [c for c in cheap.columns if c != "SPY"]:
        cheap[c] = cheap[c] * 0.02
    days = pd.bdate_range("2019-01-01", "2020-12-31")
    low = pd.DataFrame({"treasury_10y": 0.01, "aaa": 0.025}, index=pd.DatetimeIndex(days, name="date"))
    high = low.assign(treasury_10y=0.06, aaa=0.07)

    def upside(rates):
        px = PriceData(cheap, cheap, empty_splits(), rates=rates)
        ranked = rank_undervalued(fund, px, "2020-06-01", market_cap_min=0.0, min_models=2, max_upside=None, top_n=None)
        snap = pit.priced_snapshot(fund, "2020-06-01", px, with_history=True)
        return ranked.set_index("ticker")["fair_value_est_upside"], snap.set_index("ticker")["fair_value_upside"]

    listed_low, metric_low = upside(low)
    listed_high, metric_high = upside(high)
    listed_fixed, metric_fixed = upside(PriceData(panel, panel, empty_splits()).rates)  # none cached

    def higher(a, b):  # on the names both lists hold — the upside floor moves with the rates
        both = a.dropna().index.intersection(b.dropna().index)
        return len(both) > 0 and bool((a[both] > b[both]).all())

    # cheaper money, higher fair values — in the published list and in the backtest metric alike
    assert higher(listed_low, listed_fixed) and higher(listed_fixed, listed_high)
    assert higher(metric_low, metric_high)
    assert len(listed_low) >= len(listed_high)  # and more names clear the positive-upside floor
    # nothing cached is the fixed 9% of before
    assert market_assumptions("2020-06-01", PriceData(panel, panel, empty_splits()).rates).discount_rate == 0.09


def test_a_backtest_can_be_attributed_to_factors(fund, px):
    from lti import attribution

    r = run_backtest(_frictions_cfg(cost_bps=0.0), fund=fund, px=px)
    months = pd.date_range("2011-01-31", "2021-06-30", freq="ME")
    rng = np.random.default_rng(0)
    factors = pd.DataFrame(rng.normal(0, 0.03, (len(months), 6)), index=months,
                           columns=["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]).assign(rf=0.0)
    a = attribution.attribute(r, factors)
    # every month of the backtest between its first and last rebalance, whole months only
    assert a.n["Strategy"] == len(attribution.monthly_returns(r.equity_curve)) > 90
    assert a.coef.notna().all().all()
    assert a.months[0] == pd.Timestamp("2012-04-30")
