"""Free cash flow nets out stock-based pay (lti.fundamentals.add_free_cash_flow)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import fundamentals, metrics, study


def _filings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": [1, 2, 3],
            "filed": pd.to_datetime(["2020-03-01"] * 3),
            "cfo": [500.0, 300.0, 200.0],
            "capex": [-100.0, 50.0, np.nan],  # filers sign capex either way
            "stock_comp": [150.0, np.nan, 10.0],
        }
    )


def test_stock_based_pay_comes_out_of_free_cash_flow():
    df = fundamentals.add_free_cash_flow(_filings())

    assert df["free_cash_flow_reported"].tolist()[:2] == [400.0, 250.0]
    assert df.loc[0, "free_cash_flow"] == 250.0  # 500 − 100 − 150
    # no stock-comp line on the cash-flow statement: taken as none, FCF as reported
    assert df.loc[1, "free_cash_flow"] == 250.0
    # no capex: no free cash flow, stock comp or not
    assert np.isnan(df.loc[2, "free_cash_flow"]) and np.isnan(df.loc[2, "free_cash_flow_reported"])


def test_a_sign_flipped_stock_comp_still_counts_as_a_cost():
    df = fundamentals.add_free_cash_flow(_filings().assign(stock_comp=[-150.0, 0.0, 0.0]))
    assert df.loc[0, "free_cash_flow"] == 250.0


def test_it_flows_into_the_cash_yield_metrics():
    df = fundamentals.add_free_cash_flow(_filings()).assign(revenues=1000.0)
    df = metrics.add_fundamental_metrics(df)
    assert df.loc[0, "fcf_margin"] == pytest.approx(0.25)


def test_a_table_built_before_the_change_is_upgraded_on_load(tmp_path, monkeypatch):
    old = _filings()
    old["free_cash_flow"] = old["cfo"] - old["capex"].abs()  # the old definition
    old["filed_prev"] = pd.NaT  # recent enough to carry the prior-year filing date
    path = tmp_path / "fundamentals.parquet"
    old.to_parquet(path, index=False)
    monkeypatch.setattr(
        fundamentals.config, "get_paths",
        lambda: SimpleNamespace(
            fundamentals_parquet=path, derived_dir=tmp_path, quarterly_parquet=tmp_path / "quarterly.parquet"
        ),
    )

    df = fundamentals.load_fundamentals()
    assert df.loc[0, "free_cash_flow"] == 250.0
    assert df.loc[0, "free_cash_flow_reported"] == 400.0


def test_the_factor_study_keeps_the_definition_it_was_registered_on():
    df = fundamentals.add_free_cash_flow(_filings())
    registered = study.as_registered(df)

    assert registered.loc[0, "free_cash_flow"] == 400.0
    assert df.loc[0, "free_cash_flow"] == 250.0  # a copy: the caller's table is untouched
    # a table without the column has nothing to restore
    bare = _filings()
    assert study.as_registered(bare) is bare
