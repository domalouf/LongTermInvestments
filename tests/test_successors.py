"""Joining a company's filings across a change of SEC registrant (Google -> Alphabet)."""

from __future__ import annotations

import pandas as pd

import lti  # noqa: F401  (configures secfsdstools)
from lti import pit, successors


def _k(cik, year, revenue, *, ticker=None, sic=7370, filed_month=2):
    return dict(cik=cik, ticker=ticker, tickers_all=ticker, company=f"Co {cik}", form="10-K", sic=sic,
                fiscal_year=year, period_end=pd.Timestamp(year, 12, 31),
                filed=pd.Timestamp(year + 1, filed_month, 10), revenues=revenue, eps=1.0)


def _world():
    rows = [_k(100, y, 50e9 + 5e9 * (y - 2010)) for y in range(2010, 2015)]  # old Google: no ticker today
    rows += [_k(200, y, 75e9 + 10e9 * (y - 2015), ticker="GOOGL") for y in range(2015, 2019)]  # Alphabet
    fund = pd.DataFrame(rows)
    cik_map = pd.DataFrame({"cik": [200], "ticker": ["GOOGL"], "tickers_all": ["GOOG,GOOGL"]})
    prefixes = pd.DataFrame({"cik": [100, 200], "prefix": ["GOOG", "GOOG"], "n": [20, 16]})
    return fund, cik_map, prefixes


def test_prefix_reads_only_the_conventional_instance_name():
    names = pd.Series(["goog-20131231.xml", "aapl-20200926_htm.xml", "form10k_htm.xml", "d123456d10k.htm", None])
    assert successors.prefix_of(names).tolist()[:2] == ["GOOG", "AAPL"]
    assert successors.prefix_of(names).iloc[2:].isna().all()


def test_a_reorganized_company_is_linked_to_its_successor():
    fund, cik_map, prefixes = _world()
    found = successors.find_successors(fund, cik_map, prefixes)
    assert found[["cik", "successor_cik", "ticker"]].values.tolist() == [[100, 200, "GOOGL"]]

    linked = successors.apply_successors(fund, found)
    assert set(linked["cik"]) == {200} and linked["ticker"].eq("GOOGL").all()
    assert set(linked.loc[linked["fiscal_year"] < 2015, "cik_reported"]) == {100}
    # one unbroken history: a 2014 snapshot now finds the company, under its ticker
    snap = pit.snapshot_asof(linked, pd.Timestamp("2014-04-01"))
    assert snap.loc[200, "ticker"] == "GOOGL" and snap.loc[200, "fiscal_year"] == 2013


def test_a_reused_ticker_is_not_a_successor():
    fund, cik_map, prefixes = _world()
    late = fund.copy()  # the symbol reissued four years after the old company's last filing
    late.loc[late["cik"] == 200, "filed"] += pd.DateOffset(years=4)
    assert successors.find_successors(late, cik_map, prefixes).empty

    small = fund.copy()  # ... or to a company a hundredth the size
    small.loc[small["cik"] == 200, "revenues"] /= 100
    assert successors.find_successors(small, cik_map, prefixes).empty

    other = fund.copy()  # ... or in another industry
    other.loc[other["cik"] == 200, "sic"] = 2834
    assert successors.find_successors(other, cik_map, prefixes).empty

    both = fund.copy()  # two registrants filing side by side are two companies
    both.loc[both["cik"] == 200, "filed"] -= pd.DateOffset(years=2)
    assert successors.find_successors(both, cik_map, prefixes).empty


def test_listed_companies_and_unlinked_ones_are_left_alone():
    fund, cik_map, prefixes = _world()
    found = successors.find_successors(fund, cik_map, prefixes)
    extra = pd.DataFrame([_k(300, 2012, 1e9), _k(400, 2012, 2e9, ticker="XYZ")])
    out = successors.apply_successors(pd.concat([fund, extra], ignore_index=True), found)
    assert out.loc[out["cik"] == 300, "ticker"].isna().all()  # really gone: stays unpriced
    assert out.loc[out["cik"] == 400, "ticker"].eq("XYZ").all()
    assert successors.apply_successors(fund, found.iloc[:0]) is fund


def test_where_both_registrants_reported_a_year_the_successors_row_stays():
    fund, cik_map, prefixes = _world()
    overlap = pd.concat([fund, pd.DataFrame([_k(100, 2015, 74e9)])], ignore_index=True)
    out = successors.apply_successors(overlap, successors.find_successors(fund, cik_map, prefixes))
    y2015 = out[out["fiscal_year"] == 2015]
    assert len(y2015) == 1 and y2015["cik_reported"].iloc[0] == 200
