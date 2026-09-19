"""The published factors, the composite and Newey-West machinery, and the study's protocol.

Pure logic — no SEC data or network needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import history, metrics, pit, ranking, study
from lti.factor import composite_score, newey_west_t
from lti.prices import PriceData, empty_splits
from lti.ranking import ScreenSpec


# --- single-filing factors ---------------------------------------------------


def test_profitability_and_accruals_scale_by_assets():
    df = pd.DataFrame({"assets": [1000.0], "operating_income": [120.0], "revenues": [900.0],
                       "cfo": [150.0], "net_income": [100.0]})
    out = metrics.add_quality_metrics(df).iloc[0]
    assert out["op_profitability"] == pytest.approx(0.12)
    assert out["cash_profitability"] == pytest.approx(0.15)
    assert out["accruals"] == pytest.approx(-0.05)  # 50 of earnings more cash than reported


def test_yields_and_the_altman_score():
    df = pd.DataFrame(
        {
            "market_cap": [1000.0, 1000.0], "enterprise_value": [1200.0, 1200.0],
            "free_cash_flow": [60.0, 60.0], "ebit": [100.0, 100.0], "dep_amort": [20.0, 20.0],
            "revenues": [600.0, 600.0], "equity": [400.0, -5.0], "cfo": [90.0, np.nan],
            "dividends_paid": [-30.0, -30.0], "buybacks": [np.nan, 10.0],
            "assets": [800.0, 800.0], "assets_current": [300.0, 300.0], "liabilities_current": [100.0, 100.0],
            "retained_earnings": [200.0, 200.0], "liabilities": [400.0, 400.0],
        }
    )
    out = metrics.add_yield_metrics(df)
    a = out.iloc[0]
    assert a["fcf_yield"] == pytest.approx(0.06)
    assert a["ebitda_ev"] == pytest.approx(0.10)
    assert a["sales_ev"] == pytest.approx(0.5)
    assert a["book_to_market"] == pytest.approx(0.4)
    assert a["shareholder_yield"] == pytest.approx(0.03)  # untagged buybacks count as none
    z = 1.2 * 200 / 800 + 1.4 * 200 / 800 + 3.3 * 100 / 800 + 0.6 * 1000 / 400 + 1.0 * 600 / 800
    assert a["altman_z"] == pytest.approx(z)
    b = out.iloc[1]
    assert np.isnan(b["book_to_market"])  # negative equity
    assert np.isnan(b["shareholder_yield"])  # no cash-flow statement, so no telling


def test_momentum_skips_the_last_month():
    idx = pd.bdate_range("2019-01-01", "2020-06-30")
    px_series = pd.Series(100.0, index=idx)
    px_series[idx >= "2019-05-01"] = 120.0  # a year ago → a month ago: +20%
    px_series[idx >= "2020-03-15"] = 60.0  # the last month's crash is skipped
    panel = pd.DataFrame({"AAA": px_series})
    fund = pd.DataFrame(
        {"cik": [1], "ticker": ["AAA"], "company": ["A Inc"], "form": ["10-K"], "fiscal_year": [2019],
         "period_end": [pd.Timestamp("2019-12-31")], "filed": [pd.Timestamp("2020-02-15")],
         "revenues": [100.0], "net_income": [10.0], "shares_outstanding": [10.0], "eps": [1.0]}
    )
    snap = pit.priced_snapshot(fund, "2020-04-01", PriceData(panel, panel, empty_splits()))
    assert snap["momentum_12_1"].iloc[0] == pytest.approx(0.2)


# --- year-over-year factors --------------------------------------------------------


def _year(fy, **kw):
    base = dict(cik=1, ticker="A", period_end=pd.Timestamp(fy, 12, 31), filed=pd.Timestamp(fy + 1, 3, 1),
                eps=1.0, net_income=100.0, operating_income_reported=150.0, revenues=1000.0,
                assets=2000.0, cfo=160.0, liabilities_noncurrent=500.0, assets_current=600.0,
                liabilities_current=300.0, shares_outstanding=100.0)
    base.update(kw)
    return base


def test_share_growth_counts_issuance_not_splits():
    rows = [_year(2022), _year(2023, shares_outstanding=210.0)]  # a 2:1 split, plus 5% issuance
    splits = pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2023-06-01")], "ratio": [2.0]})
    s = history.summarize_history(pd.DataFrame(rows), splits).iloc[0]
    assert s["share_growth"] == pytest.approx(0.05)
    assert s["asset_growth"] == pytest.approx(0.0)


def test_a_gap_year_leaves_the_changes_unset():
    rows = [_year(2020), _year(2023, assets=3000.0)]
    s = history.summarize_history(pd.DataFrame(rows)).iloc[0]
    assert np.isnan(s["asset_growth"]) and np.isnan(s["f_score"])


def test_piotroski_scores_each_signal():
    strong = [_year(2022), _year(2023, net_income=130.0, operating_income_reported=210.0, revenues=1100.0,
                                  cfo=180.0, liabilities_noncurrent=450.0, assets_current=700.0, shares_outstanding=98.0)]
    weak = [_year(2022), _year(2023, net_income=-20.0, eps=-0.2, operating_income_reported=-10.0, revenues=900.0,
                                cfo=-30.0, liabilities_noncurrent=700.0, assets_current=500.0, liabilities_current=400.0,
                                assets=2100.0, shares_outstanding=120.0)]
    assert history.summarize_history(pd.DataFrame(strong)).iloc[0]["f_score"] == 9
    assert history.summarize_history(pd.DataFrame(weak)).iloc[0]["f_score"] == 0


# --- composites and statistics ------------------------------------------------------


def test_composite_orients_each_part_and_needs_half_of_them():
    snap = pd.DataFrame({"ebit_ev": [0.10, 0.05, np.nan], "accruals": [-0.05, 0.05, np.nan],
                         "fcf_yield": [np.nan, 0.02, 0.04]}, index=[1, 2, 3])
    score = composite_score(snap, ["ebit_ev", "accruals", "fcf_yield"])
    assert score[1] > score[2]  # cheaper on EBIT/EV and lower accruals
    assert np.isnan(score[3])  # one of three parts isn't half


def test_newey_west_widens_the_error_for_overlapping_windows():
    rng = np.random.default_rng(0)
    iid = pd.Series(rng.normal(0.02, 0.05, 200))
    e = iid - iid.mean()
    assert newey_west_t(iid, 0) == pytest.approx(iid.mean() / np.sqrt((e @ e) / len(e) / len(e)))
    smooth = iid.rolling(12).mean().dropna()  # like ICs from 12-month windows a month apart
    assert abs(newey_west_t(smooth, 11)) < abs(newey_west_t(smooth, 0)) / 2


def test_ranking_on_partial_coverage_averages_what_each_company_has():
    snap = pd.DataFrame({"ticker": ["A", "B", "C"], "ebit_ev": [0.10, 0.05, np.nan],
                         "fcf_yield": [0.01, 0.02, 0.09]}, index=[1, 2, 3])
    full = ranking.rank(snap, ScreenSpec(metrics=["ebit_ev", "fcf_yield"]))
    assert list(full["ticker"]) == ["A", "B"]  # every metric needed by default
    half = ranking.rank(snap, ScreenSpec(metrics=["ebit_ev", "fcf_yield"], min_coverage=0.5))
    assert list(half["ticker"]) == ["C", "A", "B"]  # C ranks on its FCF yield alone


# --- the study's protocol ---------------------------------------------------------


def test_the_first_half_alone_chooses_the_screen(monkeypatch):
    """Only factors with the expected sign and t >= 1 in the first half join the
    screen, and the second half is scored with that screen as a composite."""

    def summary(t_by_metric):
        return pd.DataFrame({"mean_ic": {m: 0.01 * t for m, t in t_by_metric.items()},
                             "t_stat_nw": t_by_metric})

    first_t = {h.metric: 0.0 for h in study.FACTORS}
    first_t.update({"op_profitability": 2.0, "accruals": -1.5, "momentum_12_1": -3.0, "ebit_ev": 0.9})
    second_t = {h.metric: 0.5 for h in study.FACTORS}
    second_t.update({"final_screen": 2.2, "op_profitability": 1.8})
    calls = []

    class Result:
        def __init__(self, s):
            self.summary, self.warnings = s, []

    def fake_ic(cfg, fund, px):
        calls.append(cfg)
        return Result(summary(first_t if len(calls) == 1 else second_t))

    class BT:
        stats = {"port_cagr": 0.1, "univ_cagr": 0.08, "bench_cagr": 0.12, "excess_cagr_vs_univ": 0.02}

    monkeypatch.setattr(study, "compute_ic", fake_ic)
    monkeypatch.setattr(study, "run_backtest", lambda *a, **k: BT())
    r = study.run_study(fund=pd.DataFrame(), px=PriceData(pd.DataFrame(), pd.DataFrame(), empty_splits()),
                        month_spread=False)
    # accruals: lower is better, so a negative t is the expected direction
    assert r.selected == ["op_profitability", "accruals"]
    assert calls[1].composites["final_screen"] == ["op_profitability", "accruals"]
    assert calls[1].start == study.SECOND_HALF[0]
    assert r.table.loc["momentum_12_1", "first_t"] == -3.0  # reported signed, not selected
    assert r.table.loc["op_profitability", "verdict"] == "held up"
    assert r.screen["t_stat_nw"] == 2.2


def test_verdicts():
    assert study.verdict(2.0) == "held up"
    assert study.verdict(0.3) == "right way, weak"
    assert study.verdict(-0.3) == "wrong way"
    assert study.verdict(float("nan")) == "no data"
