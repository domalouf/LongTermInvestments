"""Trailing-twelve-month rows from 10-Qs (lti.quarterly), and how snapshots use them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import pit, quarterly

# one calendar-year company. Each 10-Q: (adsh, period, filed, {qtrs: (revenue, net income)}, (cfo, capex) YTD)
QS = [
    ("q2021-1", 20210331, 20210503, {1: (220, 20)}, (30, 9)),
    ("q2022-1", 20220331, 20220502, {1: (240, 22)}, (35, 10)),
    ("q2022-2", 20220630, 20220801, {1: (250, 25), 2: (490, 47)}, (72, 22)),
    ("q2023-1", 20230331, 20230501, {1: (300, 30)}, (45, 12)),
    ("q2023-2", 20230630, 20230801, {1: (310, 32), 2: (610, 62)}, (95, 25)),
]
# its 10-Ks: (adsh, period, filed, revenue, net income, cfo, capex)
KS = [
    ("k2021", 20211231, 20220215, 900, 90, 140, 45),
    ("k2022", 20221231, 20230215, 1000, 100, 150, 50),
]


def _standardized():
    is_rows, cf_rows, bs_rows, idx = [], [], [], []
    for adsh, period, filed, by_qtrs, (cfo, capex) in QS:
        for qtrs, (rev, ni) in by_qtrs.items():
            is_rows.append({"adsh": adsh, "coreg": None, "report": 2, "ddate": period, "qtrs": qtrs,
                            "Revenues": rev, "NetIncomeLoss": ni, "OperatingIncomeLoss": ni * 1.3,
                            "OutstandingShares": 100.0 if qtrs == 1 else np.nan,
                            "EarningsPerShare": ni / 100 if qtrs == 1 else np.nan})
        n = max(by_qtrs)
        cf_rows.append({"adsh": adsh, "coreg": None, "report": 3, "ddate": period, "qtrs": n,
                        "NetCashProvidedByUsedInOperatingActivities": cfo,
                        "PaymentsToAcquirePropertyPlantAndEquipment": capex})
        bs_rows.append({"adsh": adsh, "coreg": None, "report": 1, "ddate": period, "qtrs": 0,
                        "Equity": 500.0 + period % 10_000, "Assets": 1_200.0, "Liabilities": 700.0})
        idx.append({"adsh": adsh, "cik": 1, "name": "Co", "form": "10-Q", "filed": filed, "period": period})
    annual = pd.DataFrame(
        [{"cik": 1, "ticker": "CO", "company": "Co", "adsh": a, "form": "10-K",
          "fiscal_year": p // 10_000, "period_end": pd.to_datetime(str(p)), "filed": pd.to_datetime(str(f)),
          "revenues": rev, "net_income": ni, "cfo": cfo, "capex": capex, "operating_income": ni * 1.3,
          "operating_income_reported": ni * 1.3, "equity": 480.0, "eps": ni / 100, "eps_reported": ni / 100, "shares_outstanding": 100.0,
          "free_cash_flow": cfo - capex, "free_cash_flow_reported": cfo - capex, "revenues_prev": np.nan,
          "net_income_prev": np.nan, "eps_prev": np.nan, "equity_prev": np.nan, "filed_prev": pd.NaT}
         for a, p, f, rev, ni, cfo, capex in KS]
    )
    return pd.DataFrame(bs_rows), pd.DataFrame(is_rows), pd.DataFrame(cf_rows), pd.DataFrame(idx), annual


@pytest.fixture
def built():
    bs, is_, cf, idx, annual = _standardized()
    return quarterly.build_quarterly(bs, is_, cf, idx, annual), annual


def test_ttm_is_the_last_year_plus_this_year_to_date_less_last_years(built):
    q, _ = built
    rows = q.set_index("adsh")
    # Q1 2023: FY2022 + Q1 2023 − Q1 2022
    assert rows.loc["q2023-1", "revenues"] == pytest.approx(1000 + 300 - 240)
    assert rows.loc["q2023-1", "net_income"] == pytest.approx(100 + 30 - 22)
    assert rows.loc["q2023-1", "cfo"] == pytest.approx(150 + 45 - 35)
    # Q2 2023: FY2022 + H1 2023 − H1 2022, with the six-month figures, not the quarter's
    assert rows.loc["q2023-2", "revenues"] == pytest.approx(1000 + 610 - 490)
    assert rows.loc["q2023-2", "free_cash_flow"] == pytest.approx((150 + 95 - 72) - (50 + 25 - 22))
    assert rows.loc["q2023-2", "ttm_quarters"] == 2
    assert (q["form"] == "10-Q").all() and (q["basis"] == "ttm").all()


def test_a_10q_without_a_year_earlier_one_is_left_out(built):
    q, _ = built
    # Q1 2021 has no FY2020 10-K; Q2 2022 has no Q2 2021 to subtract
    assert set(q["adsh"]) == {"q2022-1", "q2023-1", "q2023-2"}


def test_eps_is_ttm_income_over_the_quarters_share_count_and_growth_is_like_for_like(built):
    q, _ = built
    row = q.set_index("adsh").loc["q2023-1"]
    assert row["shares_outstanding"] == pytest.approx(100.0)
    assert row["eps"] == pytest.approx((100 + 30 - 22) / 100)
    assert np.isnan(row["eps_reported"])  # the quarter's own EPS is no year's
    # a year ago is the TTM row of Q1 2022: FY2021 + Q1 2022 − Q1 2021
    assert row["revenues_prev"] == pytest.approx(900 + 240 - 220)
    assert row["filed_prev"] == pd.Timestamp("2022-05-02")
    assert np.isnan(q.set_index("adsh").loc["q2023-2", "revenues_prev"])  # Q2 2022 made no row


def test_operating_income_counts_as_reported_only_if_the_10k_reported_it():
    bs, is_, cf, idx, annual = _standardized()
    q = quarterly.build_quarterly(bs, is_, cf, idx, annual).set_index("adsh")
    assert q.loc["q2023-1", "operating_income_reported"] == pytest.approx(1.3 * (100 + 30 - 22))
    untagged = quarterly.build_quarterly(bs, is_, cf, idx, annual.assign(operating_income_reported=np.nan))
    assert untagged["operating_income_reported"].isna().all()


def test_a_10q_whose_quarters_do_not_line_up_with_the_last_10k_is_left_out():
    bs, is_, cf, idx, annual = _standardized()
    # 10-Ks ending in September: a March 10-Q is two quarters on, but its year to date covers one
    shifted = annual.assign(period_end=annual["period_end"] + pd.DateOffset(months=-3))
    q = quarterly.build_quarterly(bs, is_, cf, idx, shifted)
    assert q.empty


def test_snapshots_take_the_freshest_filing_and_annual_strips_the_quarters(built):
    q, annual = built
    fund = pd.concat([annual, q], ignore_index=True)
    before = pit.snapshot_asof(fund, pd.Timestamp("2023-04-15"))
    assert before.loc[1, "form"] == "10-K" and before.loc[1, "revenues"] == 1000
    after = pit.snapshot_asof(fund, pd.Timestamp("2023-05-15"))
    assert after.loc[1, "form"] == "10-Q" and after.loc[1, "revenues"] == pytest.approx(1060)
    assert list(pit.annual(fund)["adsh"]) == ["k2021", "k2022"]
    assert pit.annual(annual) is annual  # nothing to strip


def test_the_loader_appends_the_quarters_and_can_leave_them_out(built, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lti import fundamentals

    q, annual = built
    annual.to_parquet(tmp_path / "fundamentals.parquet", index=False)
    q.to_parquet(tmp_path / "quarterly.parquet", index=False)
    monkeypatch.setattr(
        fundamentals.config, "get_paths",
        lambda: SimpleNamespace(fundamentals_parquet=tmp_path / "fundamentals.parquet", derived_dir=tmp_path,
                                quarterly_parquet=tmp_path / "quarterly.parquet"),
    )
    both = fundamentals.load_fundamentals()
    assert (both["form"] == "10-Q").sum() == len(q) and (both["form"] == "10-K").sum() == 2
    assert (fundamentals.load_fundamentals(quarterly=False)["form"] == "10-K").all()


def test_the_five_year_history_and_the_registered_study_count_years_not_quarters(built):
    from lti import history, study

    q, annual = built
    fund = pd.concat([annual, q], ignore_index=True)
    hist = history.history_asof(fund, pd.Timestamp("2023-12-31"))
    assert list(hist["form"]) == ["10-K", "10-K"]
    assert (study.as_registered(fund)["form"] == "10-K").all()
