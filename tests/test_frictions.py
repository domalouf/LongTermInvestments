"""Trading costs and taxes (lti.frictions): the arithmetic of one portfolio's book."""

from __future__ import annotations

import pandas as pd
import pytest

from lti.frictions import Book, TaxRates

RATES = TaxRates(short_term=0.30, long_term=0.15, dividends=0.10)


def _lot(ticker: str, acquired: str, basis: float, value: float) -> dict:
    return {"ticker": ticker, "acquired": pd.Timestamp(acquired), "basis": basis, "value": value}


def _book(lots: list[dict], cost_bps: float = 0.0, tax: TaxRates | None = RATES) -> Book:
    book = Book(0.0, cost_bps, tax)
    book.lots = pd.DataFrame(lots).sort_values(["ticker", "acquired"], ignore_index=True)
    return book


def test_buying_pays_the_cost_once_and_it_goes_into_the_basis():
    book = Book(100.0, cost_bps=10)
    trade = book.rebalance("2013-04-01", pd.Series({"A": 1.0, "B": 1.0}))

    assert book.value == pytest.approx(100 / 1.001)
    assert book.cash == pytest.approx(0.0, abs=1e-9)
    assert trade.cost == pytest.approx(100 - 100 / 1.001)
    assert book.lots["basis"].sum() == pytest.approx(100.0)  # every dollar spent is cost basis
    assert book.positions().to_dict() == pytest.approx({"A": 50 / 1.001, "B": 50 / 1.001})
    assert trade.turnover == pytest.approx(0.5 / 1.001)  # bought the lot, sold nothing


@pytest.mark.parametrize(
    "sold, term",
    [("2014-04-01", "short"), ("2014-04-02", "long")],  # a year to the day is still short-term
)
def test_a_gain_is_long_term_only_after_more_than_a_year(sold, term):
    book = _book([_lot("A", "2013-04-01", 100.0, 150.0)])
    trade = book.rebalance(sold, pd.Series({"B": 1.0}))

    rate = RATES.short_term if term == "short" else RATES.long_term
    assert trade.short_term_gain == pytest.approx(50.0 if term == "short" else 0.0)
    assert trade.long_term_gain == pytest.approx(50.0 if term == "long" else 0.0)
    assert trade.tax == pytest.approx(50.0 * rate)
    assert book.value == pytest.approx(150.0 - 50.0 * rate)


def test_a_rebalance_costs_exactly_its_costs_and_taxes():
    book = _book(
        [_lot("A", "2012-01-03", 40.0, 90.0), _lot("B", "2013-06-03", 60.0, 45.0), _lot("C", "2013-01-02", 10.0, 30.0)],
        cost_bps=25,
    )
    before = book.value
    trade = book.rebalance("2013-12-02", pd.Series({"A": 1.0, "C": 1.0, "D": 1.0}))

    assert book.value == pytest.approx(before - trade.cost - trade.tax)
    assert abs(book.cash) < 1e-6
    held = book.positions()
    assert set(held.index) == {"A", "C", "D"}
    assert held.to_numpy() == pytest.approx([held.mean()] * 3)  # equal weights of what's left


def test_sales_take_the_oldest_lot_first():
    # tax-free, so the sale is exactly half of A — a tax bill would sell a little more to pay it
    book = _book([_lot("A", "2010-01-04", 50.0, 100.0), _lot("A", "2014-01-02", 90.0, 100.0)], tax=None)
    trade = book.rebalance("2014-06-02", pd.Series({"A": 1.0, "B": 1.0}))

    assert trade.long_term_gain == pytest.approx(50.0)  # the 2010 lot, all of it
    assert trade.short_term_gain == pytest.approx(0.0)
    left = book.lots[book.lots["ticker"] == "A"]
    assert len(left) == 1 and left["acquired"].iloc[0] == pd.Timestamp("2014-01-02")


def test_losses_offset_gains_and_what_is_left_carries_forward():
    book = _book([_lot("A", "2013-04-01", 100.0, 40.0), _lot("B", "2013-04-01", 100.0, 120.0)])
    trade = book.rebalance("2013-10-01", pd.Series({"C": 1.0}))  # both short-term: −60 + 20
    assert trade.short_term_gain == pytest.approx(-40.0)
    assert trade.tax == 0.0
    assert book.carry_short == pytest.approx(40.0)

    book.lots["value"] = book.lots["value"] * 1.5  # C gains 80 over the next year and a half
    trade = book.rebalance("2015-04-06", pd.Series({"D": 1.0}))
    assert trade.long_term_gain == pytest.approx(80.0)
    # the short-term loss carried forward offsets the long-term gain
    assert trade.tax == pytest.approx(40.0 * RATES.long_term)
    assert book.carry_short == book.carry_long == 0.0


def test_dividends_are_taxed_as_they_come_and_the_rest_reinvested():
    book = _book([_lot("A", "2013-04-01", 100.0, 100.0)])
    tax = book.grow(pd.Series({"A": 1.10}), pd.Series({"A": 1.06}))  # a 4% yield

    assert tax == pytest.approx(4.0 * RATES.dividends)
    assert book.value == pytest.approx(110.0 - tax)
    assert book.lots["basis"].iloc[0] == pytest.approx(100.0 + 4.0 - tax)

    # no split-adjusted price for a ticker: no dividend is inferred
    assert book.grow(pd.Series({"A": 1.10}), pd.Series(dtype=float)) == 0.0


def test_a_tax_free_account_pays_no_tax():
    book = _book([_lot("A", "2013-04-01", 100.0, 150.0)], tax=None)
    assert book.rebalance("2013-05-01", pd.Series({"B": 1.0})).tax == 0.0
    assert book.grow(pd.Series({"B": 1.10}), pd.Series({"B": 1.00})) == 0.0
    assert book.value == pytest.approx(165.0)


def test_liquidation_pays_the_cost_of_selling_and_the_tax_on_every_gain():
    book = _book([_lot("A", "2013-04-01", 100.0, 150.0), _lot("B", "2015-01-02", 100.0, 120.0)], cost_bps=10)
    cash = book.liquidation_value("2015-04-01")

    proceeds = 270.0 * 0.999
    gains_long, gains_short = 150.0 * 0.999 - 100.0, 120.0 * 0.999 - 100.0
    assert cash == pytest.approx(proceeds - gains_long * RATES.long_term - gains_short * RATES.short_term)
    assert book.value == 270.0  # a what-if: the book itself is untouched
