"""Unit tests for the Greenblatt-style capital metrics and their inputs.

Pure logic — no SEC data or network needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import metrics, rawtags, ranking, sectors
from lti.ranking import ScreenSpec


# --- return on capital ------------------------------------------------------


def test_roic_uses_working_capital_plus_net_fixed_assets():
    df = pd.DataFrame(
        {
            "operating_income": [100.0],
            "assets_current": [300.0],
            "liabilities_current": [150.0],
            "cash": [50.0],
            "debt_current": [30.0],
            "ppe_net": [70.0],
        }
    )
    out = metrics.add_capital_metrics(df)
    # working capital = (300 - 50) - (150 - 30) = 130; + 70 of PP&E = 200
    assert out["invested_capital"].iloc[0] == pytest.approx(200.0)
    assert out["roic"].iloc[0] == pytest.approx(0.5)


def test_roic_floors_negative_working_capital_at_zero():
    """Customer float is a financing benefit, not negative invested capital."""
    df = pd.DataFrame(
        {
            "operating_income": [100.0],
            "assets_current": [100.0],
            "liabilities_current": [300.0],
            "cash": [20.0],
            "debt_current": [0.0],
            "ppe_net": [50.0],
        }
    )
    out = metrics.add_capital_metrics(df)
    assert out["invested_capital"].iloc[0] == pytest.approx(50.0)
    assert out["roic"].iloc[0] == pytest.approx(2.0)


def test_roic_is_nan_when_invested_capital_is_not_positive():
    df = pd.DataFrame(
        {
            "operating_income": [100.0],
            "assets_current": [100.0],
            "liabilities_current": [300.0],
            "cash": [0.0],
            "debt_current": [0.0],
            "ppe_net": [0.0],
        }
    )
    out = metrics.add_capital_metrics(df)
    assert np.isnan(out["invested_capital"].iloc[0])
    assert np.isnan(out["roic"].iloc[0])


def test_roic_is_skipped_when_ppe_is_absent():
    df = pd.DataFrame(
        {"operating_income": [100.0], "assets_current": [1.0], "liabilities_current": [1.0]}
    )
    assert "roic" not in metrics.add_capital_metrics(df).columns


# --- enterprise value / EBIT-EV --------------------------------------------


def test_ebit_ev_adds_debt_and_subtracts_cash():
    df = pd.DataFrame(
        {
            "market_cap": [1000.0],
            "total_debt": [500.0],
            "cash": [100.0],
            "operating_income": [140.0],
        }
    )
    out = metrics.add_enterprise_value(df)
    assert out["enterprise_value"].iloc[0] == pytest.approx(1400.0)
    assert out["ebit_ev"].iloc[0] == pytest.approx(0.1)


def test_ebit_ev_is_nan_when_debt_is_unknown():
    """Unknown debt must not be silently treated as zero."""
    df = pd.DataFrame(
        {
            "market_cap": [1000.0],
            "total_debt": [np.nan],
            "cash": [100.0],
            "operating_income": [140.0],
        }
    )
    out = metrics.add_enterprise_value(df)
    assert np.isnan(out["enterprise_value"].iloc[0])
    assert np.isnan(out["ebit_ev"].iloc[0])


def test_ebit_ev_is_nan_when_net_cash_exceeds_market_cap():
    df = pd.DataFrame(
        {
            "market_cap": [100.0],
            "total_debt": [0.0],
            "cash": [500.0],
            "operating_income": [50.0],
        }
    )
    out = metrics.add_enterprise_value(df)
    assert np.isnan(out["enterprise_value"].iloc[0])
    assert np.isnan(out["ebit_ev"].iloc[0])


def test_ebit_ev_differs_from_eps_over_price_when_leverage_differs():
    """The whole point of EBIT/EV: two identical operations, different balance sheets."""
    df = pd.DataFrame(
        {
            "operating_income": [100.0, 100.0],
            "total_debt": [0.0, 900.0],
            "cash": [0.0, 0.0],
            "eps": [1.0, 1.0],
            "shares_outstanding": [100.0, 100.0],
        },
        index=[1, 2],
    )
    price = pd.Series([10.0, 10.0], index=[1, 2])
    out = metrics.add_price_metrics(df, price=price, market_cap=price * df["shares_outstanding"])

    # identical on the equity-only view
    assert out["earnings_yield"].iloc[0] == pytest.approx(out["earnings_yield"].iloc[1])
    # but the levered company is markedly more expensive to buy outright:
    # EV of 1000 vs 1900, so the same 100 of EBIT yields far less
    assert out["ebit_ev"].iloc[0] == pytest.approx(0.1)
    assert out["ebit_ev"].iloc[1] == pytest.approx(100 / 1900)


# --- debt derivation --------------------------------------------------------


def test_derive_debt_prefers_rollup_over_instrument_tags():
    df = pd.DataFrame(
        {
            "LongTermDebtNoncurrent": [800.0],
            "ConvertibleDebtNoncurrent": [111.0],  # ignored: a roll-up is present
            "SecuredLongTermDebt": [222.0],
            "DebtCurrent": [200.0],
        }
    )
    out = rawtags._derive_debt(df)
    assert out["debt_noncurrent"].iloc[0] == pytest.approx(800.0)
    assert out["total_debt"].iloc[0] == pytest.approx(1000.0)
    assert not out["total_debt_partial"].iloc[0]


def test_derive_debt_sums_instrument_tags_when_no_rollup():
    df = pd.DataFrame(
        {
            "ConvertibleDebtNoncurrent": [300.0],
            "SecuredLongTermDebt": [200.0],
            "LongTermDebtCurrent": [50.0],
        }
    )
    out = rawtags._derive_debt(df)
    assert out["debt_noncurrent"].iloc[0] == pytest.approx(500.0)
    assert out["debt_current"].iloc[0] == pytest.approx(50.0)
    assert out["total_debt"].iloc[0] == pytest.approx(550.0)


def test_derive_debt_falls_back_to_total_tag_when_only_one_half_known():
    df = pd.DataFrame({"LongTermDebtNoncurrent": [800.0], "LongTermDebt": [950.0]})
    out = rawtags._derive_debt(df)
    assert out["total_debt"].iloc[0] == pytest.approx(950.0)
    assert not out["total_debt_partial"].iloc[0]


def test_derive_debt_flags_a_partial_total():
    df = pd.DataFrame({"LongTermDebtNoncurrent": [800.0]})
    out = rawtags._derive_debt(df)
    assert out["total_debt"].iloc[0] == pytest.approx(800.0)
    assert out["total_debt_partial"].iloc[0]


def test_derive_debt_is_nan_when_nothing_is_reported():
    df = pd.DataFrame({"Goodwill": [10.0]})
    out = rawtags._derive_debt(df)
    assert np.isnan(out["total_debt"].iloc[0])


# --- debt provenance --------------------------------------------------------


def test_debt_provenance_assumes_zero_for_a_clean_balance_sheet():
    df = pd.DataFrame(
        {
            "total_debt": [np.nan],
            "liabilities": [100.0],
            "liabilities_current": [99.0],
            "liabilities_noncurrent": [1.0],
            "assets": [1000.0],
        }
    )
    out = rawtags.add_debt_provenance(df)
    assert out["total_debt"].iloc[0] == 0.0
    assert out["debt_source"].iloc[0] == "assumed_zero"


def test_debt_provenance_stays_unknown_with_material_noncurrent_liabilities():
    df = pd.DataFrame(
        {
            "total_debt": [np.nan],
            "liabilities": [600.0],
            "liabilities_current": [200.0],
            "liabilities_noncurrent": [400.0],
            "assets": [1000.0],
        }
    )
    out = rawtags.add_debt_provenance(df)
    assert np.isnan(out["total_debt"].iloc[0])
    assert out["debt_source"].iloc[0] == "unknown"


def test_debt_provenance_marks_reported_values():
    df = pd.DataFrame(
        {
            "total_debt": [500.0],
            "liabilities": [600.0],
            "liabilities_current": [200.0],
            "liabilities_noncurrent": [400.0],
            "assets": [1000.0],
        }
    )
    out = rawtags.add_debt_provenance(df)
    assert out["debt_source"].iloc[0] == "reported"


def test_debt_provenance_refuses_to_infer_on_an_unclassified_balance_sheet():
    """A homebuilder files every liability as current — that is not 'no debt'."""
    df = pd.DataFrame(
        {
            "total_debt": [np.nan],
            "liabilities": [1868.0],
            "liabilities_current": [1868.0],
            "liabilities_noncurrent": [0.0],
            "assets": [4460.0],
        }
    )
    out = rawtags.add_debt_provenance(df)
    assert np.isnan(out["total_debt"].iloc[0])
    assert out["debt_source"].iloc[0] == "unknown"


def test_debt_provenance_still_infers_on_a_classified_balance_sheet():
    df = pd.DataFrame(
        {
            "total_debt": [np.nan],
            "liabilities": [500.0],
            "liabilities_current": [480.0],
            "liabilities_noncurrent": [20.0],
            "assets": [4000.0],
        }
    )
    out = rawtags.add_debt_provenance(df)
    assert out["total_debt"].iloc[0] == 0.0
    assert out["debt_source"].iloc[0] == "assumed_zero"


# --- EBIT sanity guard ------------------------------------------------------


def test_ebit_prefers_the_as_reported_tag_over_the_derived_one():
    """KB Home's shape: the standardizer derives a ~100% operating margin."""
    df = pd.DataFrame(
        {
            "operating_income": [6214.0],  # derived, wrong
            "operating_income_reported": [520.0],  # as tagged by the filer
            "revenues": [6236.0],
        }
    )
    assert metrics.ebit(df).iloc[0] == pytest.approx(520.0)


def test_ebit_is_nan_when_the_filer_never_tagged_it():
    """No as-reported value means no EBIT — the derived one is not trusted."""
    df = pd.DataFrame(
        {
            "operating_income": [6214.0],
            "operating_income_reported": [np.nan],
            "revenues": [6236.0],
        }
    )
    assert np.isnan(metrics.ebit(df).iloc[0])


def test_ebit_rejects_a_reported_value_above_revenue():
    df = pd.DataFrame(
        {"operating_income_reported": [4622.0], "revenues": [4117.0]}
    )
    assert np.isnan(metrics.ebit(df).iloc[0])


def test_ebit_keeps_a_genuinely_high_margin_business():
    df = pd.DataFrame({"operating_income_reported": [600.0], "revenues": [1000.0]})
    assert metrics.ebit(df).iloc[0] == pytest.approx(600.0)


def test_ebit_falls_back_to_the_standardized_column_on_an_older_build():
    df = pd.DataFrame({"operating_income": [120.0], "revenues": [1000.0]})
    assert metrics.ebit(df).iloc[0] == pytest.approx(120.0)


def test_untrusted_ebit_propagates_to_both_magic_formula_metrics():
    df = pd.DataFrame(
        {
            "operating_income": [4622.0],
            "operating_income_reported": [np.nan],
            "revenues": [4117.0],
            "assets_current": [4460.0],
            "liabilities_current": [1868.0],
            "cash": [109.0],
            "debt_current": [0.0],
            "ppe_net": [69.0],
            "market_cap": [1930.0],
            "total_debt": [0.0],
        }
    )
    out = metrics.add_enterprise_value(metrics.add_capital_metrics(df))
    assert np.isnan(out["roic"].iloc[0])
    assert np.isnan(out["ebit_ev"].iloc[0])


# --- sectors ----------------------------------------------------------------


def test_sic_classification():
    sic = pd.Series([6021, 4911, 3674, 5812, np.nan])
    assert list(sectors.is_financial(sic)) == [True, False, False, False, False]
    assert list(sectors.is_utility(sic)) == [False, True, False, False, False]
    divisions = sectors.sic_division(sic)
    assert divisions.iloc[0] == "Finance, Insurance & Real Estate"
    assert divisions.iloc[2] == "Manufacturing"
    assert pd.isna(divisions.iloc[4])


def test_reit_counts_as_financial():
    """SIC 6798 — REITs sit in the financials division and Greenblatt drops them."""
    assert bool(sectors.is_financial(pd.Series([6798])).iloc[0])


def test_missing_sic_is_unclassified_with_nullable_dtypes():
    """A missing code must not inherit the last division tested."""
    sic = pd.Series([pd.NA, 3674], dtype="Int32")
    divisions = sectors.sic_division(sic)
    assert pd.isna(divisions.iloc[0])
    assert divisions.iloc[1] == "Manufacturing"
    assert not sectors.is_financial(sic).iloc[0]
    assert not sectors.is_utility(sic).iloc[0]


# --- the exclusions actually bite ------------------------------------------


def _screen_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BANK", "UTIL", "MFG"],
            "sic": [6021, 4911, 3674],
            "is_financial": [True, False, False],
            "is_utility": [False, True, False],
            "ebit_ev": [0.30, 0.20, 0.10],
            "roic": [0.90, 0.80, 0.70],
        },
        index=[1, 2, 3],
    )


def test_exclude_financials_and_utilities():
    spec = ScreenSpec(
        metrics=["ebit_ev", "roic"],
        top_n=10,
        filters={"exclude_financials": True, "exclude_utilities": True},
    )
    assert ranking.select(_screen_frame(), spec) == ["MFG"]


def test_excluding_financials_also_drops_investment_companies():
    """BDCs (Ares Capital, FS KKR) have no SIC at all, so the SIC test alone misses them."""
    df = pd.concat([_screen_frame(), pd.DataFrame(
        {"ticker": ["ARCC"], "sic": [np.nan], "is_financial": [False], "is_utility": [False],
         "ebit_ev": [0.50], "roic": [0.95]}, index=[4])])
    spec = ScreenSpec(metrics=["ebit_ev", "roic"], top_n=10, filters={"exclude_financials": True})
    assert ranking.select(df, spec) == ["UTIL", "MFG"]
    assert list(sectors.is_investment_company(df["sic"])) == [False, False, False, True]


def test_exclusions_fall_back_to_raw_sic():
    df = _screen_frame().drop(columns=["is_financial", "is_utility"])
    spec = ScreenSpec(
        metrics=["ebit_ev", "roic"], top_n=10, filters={"exclude_financials": True}
    )
    assert ranking.select(df, spec) == ["UTIL", "MFG"]


def test_exclusion_raises_rather_than_silently_passing_everything_through():
    """The old filter no-opped when the column was missing — that must not recur."""
    df = _screen_frame().drop(columns=["is_financial", "is_utility", "sic"])
    spec = ScreenSpec(
        metrics=["ebit_ev", "roic"], top_n=10, filters={"exclude_financials": True}
    )
    with pytest.raises(KeyError, match="is_financial"):
        ranking.select(df, spec)
