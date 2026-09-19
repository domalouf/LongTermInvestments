"""Share counts from the raw SEC facts, and the cross-check that picks one per filing.

The cases are the real ones that shaped the rules (see ``rawtags.reconcile_shares``).
Pure logic — no SEC data or network needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import rawtags


def _facts(*rows: tuple[str, str, str | None, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["adsh", "tag", "segments", "value"])


def _summary(*rows) -> pd.Series:
    out = rawtags._summarize_shares(_facts(*rows)).set_index("adsh")
    return out.iloc[0]


# --- reading the balance-sheet count -------------------------------------------


def test_a_tagged_outstanding_count_is_used_as_is():
    s = _summary(
        ("a", "CommonStockSharesOutstanding", None, 15.1e9),
        ("a", "CommonStockSharesIssued", None, 15.1e9),
        ("a", "WeightedAverageNumberOfSharesOutstandingBasic", None, 15.3e9),
        ("a", "WeightedAverageNumberOfDilutedSharesOutstanding", None, 15.4e9),
    )
    assert s["shares_bs"] == 15.1e9 and s["shares_bs_direct"]
    assert s["shares_wavg"] == 15.3e9  # basic before diluted
    assert not s["shares_multi_class"]


def test_the_equity_statements_common_stock_column_counts_as_the_total():
    # Procter & Gamble tags its count only on the equity statement
    s = _summary(("pg", "CommonStockSharesOutstanding", "EquityComponents=CommonStock;", 2.357e9))
    assert s["shares_bs"] == 2.357e9 and s["shares_bs_direct"]


def test_a_dimensionless_value_beats_the_equity_statements():
    s = _summary(
        ("a", "CommonStockSharesOutstanding", "EquityComponents=CommonStock;", 9.0e8),
        ("a", "CommonStockSharesOutstanding", None, 1.0e9),
    )
    assert s["shares_bs"] == 1.0e9


def test_issued_less_treasury_when_outstanding_is_untagged():
    # Merck: 3,577M issued, 1,049M in treasury
    s = _summary(
        ("mrk", "CommonStockSharesIssued", None, 3.577e9),
        ("mrk", "TreasuryStockCommonShares", None, 1.049e9),
    )
    assert s["shares_bs"] == pytest.approx(2.528e9)
    assert not s["shares_bs_direct"]


def test_a_placeholder_zero_is_not_a_count():
    s = _summary(
        ("a", "CommonStockSharesOutstanding", None, 0.0),
        ("a", "CommonStockSharesIssued", None, 5.0e7),
    )
    assert s["shares_bs"] == 5.0e7 and not s["shares_bs_direct"]


def test_share_classes_are_counted_not_summed():
    # UPS lists class A and B, each twice (with and without the equity-component axis)
    s = _summary(
        ("ups", "CommonStockSharesIssued", "ClassOfStock=CommonClassA;EquityComponents=CommonStock;", 1.21e8),
        ("ups", "CommonStockSharesIssued", "ClassOfStock=CommonClassA;", 1.21e8),
        ("ups", "CommonStockSharesIssued", "ClassOfStock=CommonClassB;", 7.33e8),
        ("ups", "ShareRepurchaseProgram", "ShareRepurchaseProgram=Plan;", 1.0),
    )
    assert np.isnan(s["shares_bs"])
    assert s["shares_multi_class"]


def test_a_single_listed_class_is_the_total():
    # Baker Hughes tags its count only under its one class, Class A
    s = _summary(
        ("bkr", "CommonStockSharesOutstanding", "ClassOfStock=CommonClassA;", 9.868e8),
        ("bkr", "CommonStockSharesIssued", "ClassOfStock=CommonClassA;", 9.87e8),
    )
    assert s["shares_bs"] == 9.868e8 and s["shares_bs_direct"]
    assert not s["shares_multi_class"]


def test_partnership_units_count_like_shares():
    # Enterprise Products: dimensionless; Energy Transfer: the common-units column,
    # alongside its subsidiaries' units, which must not be added in
    epd = _summary(("epd", "LimitedPartnersCapitalAccountUnitsOutstanding", None, 2.16e9))
    assert epd["shares_bs"] == 2.16e9
    et = _summary(
        ("et", "LimitedPartnersCapitalAccountUnitsOutstanding", "EquityComponents=CommonUnits;", 3.44e9),
        ("et", "LimitedPartnersCapitalAccountUnitsOutstanding", "LegalEntity=USAC;", 1.27e8),
    )
    assert et["shares_bs"] == 3.44e9
    paa = _summary(
        ("paa", "LimitedPartnersCapitalAccountUnitsOutstanding", "LimitedPartnersCapitalAccountByClass=CommonUnits;", 7.055e8),
        ("paa", "LimitedPartnersCapitalAccountUnitsOutstanding",
         "LimitedPartnersCapitalAccountByClass=SeriesAPreferredUnits;", 5.84e7),
    )
    assert paa["shares_bs"] == 7.055e8


# --- picking one count ------------------------------------------------------------


def _reconcile(**cols) -> pd.DataFrame:
    n = max(len(v) for v in cols.values())
    base = {
        "shares_outstanding": [np.nan] * n,
        "shares_wavg": [np.nan] * n,
        "shares_bs": [np.nan] * n,
        "shares_bs_direct": [True] * n,
        "shares_multi_class": [False] * n,
        "eps": [np.nan] * n,
        "net_income": [np.nan] * n,
    }
    base.update(cols)
    return rawtags.reconcile_shares(pd.DataFrame(base))


def _one(**cols) -> pd.Series:
    return _reconcile(**{k: [v] for k, v in cols.items()}).iloc[0]


def test_a_footnoted_weighted_average_is_filled_from_the_balance_sheet():
    r = _one(shares_bs=2.342e9, eps=6.67, net_income=15.97e9)  # Procter & Gamble
    assert (r["shares_outstanding"], r["shares_source"]) == (2.342e9, "balance_sheet")
    assert np.isnan(r["shares_reported"])


def test_the_raw_weighted_average_fills_in_before_the_balance_sheet():
    r = _one(shares_wavg=1.0e8, shares_bs=1.02e8)
    assert (r["shares_outstanding"], r["shares_source"]) == (1.0e8, "reported")


def test_an_issued_count_defers_to_net_income_over_eps():
    # Boeing: 1.01B issued, treasury untagged; ~0.9B by net income / EPS
    r = _one(shares_bs=1.012e9, shares_bs_direct=False, eps=2.49, net_income=2.235e9)
    assert r["shares_source"] == "implied"
    assert r["shares_outstanding"] == pytest.approx(2.235e9 / 2.49)


def test_several_share_classes_use_net_income_over_eps():
    r = _one(shares_multi_class=True, eps=6.56, net_income=5.572e9)  # UPS, class counts only
    assert r["shares_source"] == "implied"
    assert r["shares_outstanding"] == pytest.approx(5.572e9 / 6.56)
    r = _one(shares_multi_class=True, shares_bs=1.0e9, eps=1.0, net_income=3.0e9)
    assert r["shares_source"] == "implied"  # a raw total across classes of unequal weight loses


def test_a_mis_scaled_reported_count_is_replaced_when_the_others_agree():
    # Bruker tagged its weighted average in millions: 146.4
    r = _one(shares_outstanding=146.4, shares_bs=1.452e8, eps=2.92, net_income=4.27e8)
    assert (r["shares_outstanding"], r["shares_source"]) == (1.452e8, "balance_sheet")
    assert r["shares_reported"] == 146.4


def test_a_mis_scaled_balance_sheet_count_is_ignored():
    # Waters' balance-sheet count came through in thousands: 59,388
    r = _one(shares_outstanding=5.933e7, shares_bs=59_388.0, eps=10.75, net_income=6.38e8)
    assert (r["shares_outstanding"], r["shares_source"]) == (5.933e7, "reported")


def test_two_counts_a_scale_error_apart_with_no_tiebreak_give_nothing():
    # a small bank with 23.8 *billion* reported shares against 23.7M on the balance sheet
    r = _one(shares_outstanding=2.38e10, shares_bs=2.37e7)
    assert np.isnan(r["shares_outstanding"]) and r["shares_source"] == "conflict"


def test_a_moderate_gap_keeps_the_reported_count():
    # an IPO year: the weighted average is half the year-end count
    r = _one(shares_outstanding=5.0e7, shares_bs=1.0e8)
    assert (r["shares_outstanding"], r["shares_source"]) == (5.0e7, "reported")


def test_net_income_over_eps_confirms_but_never_overrules():
    # large preferred dividends: net income -1.7M against a -51M loss to common
    r = _one(shares_bs=6.66e7, eps=-0.77, net_income=-1.69e6)
    assert (r["shares_outstanding"], r["shares_source"]) == (6.66e7, "balance_sheet")
    r = _one(shares_outstanding=6.65e7, shares_bs=6.66e7, eps=-0.77, net_income=-1.69e6)
    assert (r["shares_outstanding"], r["eps"], r["eps_source"]) == (6.65e7, -0.77, "reported")


@pytest.mark.parametrize(
    "case",
    [
        dict(shares_outstanding=8.99e8, shares_bs=8.89e8, eps=2.93e6, net_income=2.638e9),  # EPS in the wrong unit
        dict(shares_outstanding=21_012.0, shares_bs=22_830.0, eps=-1.74, net_income=-3.666e7),  # both counts in thousands
    ],
)
def test_a_unit_error_between_eps_shares_and_net_income_leaves_both_unset(case):
    r = _one(**case)
    assert np.isnan(r["shares_outstanding"]) and np.isnan(r["eps"])
    assert (r["shares_source"], r["eps_source"]) == ("conflict", "conflict")
    assert r["eps_reported"] == case["eps"]


def test_nothing_to_go_on_is_missing():
    r = _one(eps=0.05, net_income=1.0e6)  # too small an EPS to divide by
    assert np.isnan(r["shares_outstanding"]) and r["shares_source"] == "missing"
    assert r["eps_source"] == "reported"
