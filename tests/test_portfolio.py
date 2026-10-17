"""Your own portfolio (lti.portfolio): the ledger, its replay, and the comparison with SPY."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import portfolio
from lti.prices import PriceData, empty_dividends, empty_splits

DAYS = pd.bdate_range("2020-01-01", "2022-12-30")


def _px(splits=None, dividends=None, **extra) -> PriceData:
    """SPY and a stock that tracks it exactly (AAA), a stock that doubles (BBB) and a fund."""
    growth = np.cumprod(np.full(len(DAYS), 1.0003))
    panel = pd.DataFrame(
        {
            "SPY": 100 * growth,
            "AAA": 100 * growth,
            "BBB": np.linspace(50, 100, len(DAYS)),
            "VTI": 200 * growth,
            **extra,
        },
        index=DAYS,
    )
    return PriceData(
        adj=panel, close=panel,
        splits=splits if splits is not None else empty_splits(),
        dividends=dividends if dividends is not None else empty_dividends(),
    )


def _ledger(*rows) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"id": str(i), "fees": 0.0, "kind": None, "note": "", "logged_utc": "", **r} for i, r in enumerate(rows)]
    ).reindex(columns=portfolio.COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _at(px: PriceData, ticker: str, day: str) -> float:
    return float(px.close[ticker].loc[:day].iloc[-1])


def test_xirr_is_the_annual_rate():
    assert portfolio.xirr(["2020-01-01", "2021-01-01"], [-100, 110]) == pytest.approx(0.10, abs=1e-3)
    assert portfolio.xirr(["2020-01-01", "2022-01-01"], [-100, 121]) == pytest.approx(0.10, abs=1e-3)
    assert np.isnan(portfolio.xirr(["2020-01-01", "2021-01-01"], [-100, -5]))


def test_an_account_invested_exactly_like_spy_matches_spy():
    px = _px()
    day = "2020-01-02"
    price = _at(px, "AAA", day)
    acct = portfolio.replay(
        _ledger(
            {"date": day, "action": "deposit", "amount": 10 * price},
            {"date": day, "action": "buy", "ticker": "AAA", "shares": 10.0, "price": price},
        ),
        px,
    )
    s = acct.summary
    assert s["value"] == pytest.approx(10 * px.close["AAA"].iloc[-1])
    assert s["value"] == pytest.approx(s["spy_same_flows"])
    assert s["vs_spy"] == pytest.approx(0.0, abs=1e-6)
    assert s["money_weighted"] == pytest.approx(s["spy_money_weighted"], abs=1e-6)
    assert s["time_weighted"] == pytest.approx(s["spy_return"], rel=1e-6)
    assert acct.holdings.set_index("ticker").loc["AAA", "vs_spy"] == pytest.approx(0.0, abs=1e-6)


def test_a_buy_beyond_the_cash_is_new_money_and_same_day_order_does_not_matter():
    px = _px()
    day = "2020-03-02"
    price = _at(px, "BBB", day)
    # logged buy-first; the deposit still settles before it
    acct = portfolio.replay(
        _ledger(
            {"date": day, "action": "buy", "ticker": "BBB", "shares": 10.0, "price": price},
            {"date": day, "action": "deposit", "amount": 4 * price},
        ),
        px,
    )
    flows = acct.flows
    assert flows["amount"].sum() == pytest.approx(10 * price)  # 4 deposited, 6 of new money
    assert flows["implicit"].sum() == 1
    assert acct.summary["cash"] == pytest.approx(0.0, abs=1e-9)
    # BBB doubled while SPY did less: the picks beat the same money in the index
    assert acct.summary["vs_spy"] > 0
    assert acct.summary["money_weighted"] > acct.summary["spy_money_weighted"]


def test_a_split_does_not_read_as_a_loss_and_sales_use_the_days_basis():
    # AAA split 2-for-1 in mid-2021: the cache's closes are on today's basis, half the traded price before it
    split_day = pd.Timestamp("2021-06-01")
    px = _px(splits=pd.DataFrame({"ticker": ["AAA"], "date": [split_day], "ratio": [2.0]}))
    traded_before = 2 * _at(px, "AAA", "2020-01-02")  # what the broker showed, pre-split
    acct = portfolio.replay(
        _ledger(
            {"date": "2020-01-02", "action": "buy", "ticker": "AAA", "shares": 10.0, "price": traded_before},
            {"date": "2022-01-03", "action": "sell", "ticker": "AAA", "shares": 5.0, "price": _at(px, "AAA", "2022-01-03")},
        ),
        px,
    )
    row = acct.holdings.set_index("ticker").loc["AAA"]
    assert row["shares"] == pytest.approx(15.0)  # 20 after the split, 5 sold
    assert row["avg_cost"] == pytest.approx(traded_before / 2)
    assert row["unrealized"] > 0 and row["realized"] > 0  # it only ever rose
    assert acct.daily["value"].pct_change().min() > -0.01  # no fake crash at the split


def test_dividends_are_credited_for_the_shares_held_the_day_before():
    divs = pd.DataFrame({"ticker": ["AAA", "AAA"], "date": pd.to_datetime(["2020-06-01", "2021-06-01"]), "amount": [1.0, 1.0]})
    px = _px(dividends=divs)
    acct = portfolio.replay(
        _ledger(
            {"date": "2020-01-02", "action": "buy", "ticker": "AAA", "shares": 10.0, "price": _at(px, "AAA", "2020-01-02")},
            # bought on the second ex-date itself: too late for that dividend
            {"date": "2021-06-01", "action": "buy", "ticker": "AAA", "shares": 5.0, "price": _at(px, "AAA", "2021-06-01")},
        ),
        px,
    )
    assert acct.holdings.set_index("ticker").loc["AAA", "income"] == pytest.approx(20.0)
    # the second buy spends the $20 of dividends first; only the rest is new money
    second = acct.flows.set_index("date")["amount"].loc["2021-06-01"]
    assert second == pytest.approx(5 * _at(px, "AAA", "2021-06-01") - 20.0)
    assert acct.summary["cash"] == pytest.approx(0.0, abs=1e-9)


def test_stocks_and_funds_are_compared_as_sleeves():
    px = _px()
    day = "2020-01-02"
    acct = portfolio.replay(
        _ledger(
            {"date": day, "action": "deposit", "amount": 10_000.0},
            {"date": day, "action": "buy", "ticker": "BBB", "shares": 20.0, "price": _at(px, "BBB", day)},
            {"date": day, "action": "buy", "ticker": "VTI", "shares": 20.0, "price": _at(px, "VTI", day)},
            # a foreign stock the 10-K filers don't include, marked by hand
            {"date": day, "action": "buy", "ticker": "AAA", "shares": 1.0, "price": _at(px, "AAA", day), "kind": "stock"},
        ),
        px,
        stocks={"BBB"},
    )
    kinds = acct.holdings.set_index("ticker")["kind"].to_dict()
    assert kinds == {"AAA": "stock", "BBB": "stock", "VTI": "fund"}
    sleeves = acct.sleeves.set_index("sleeve")
    assert set(sleeves.index) == {"Stocks you picked", "Funds", "Cash"}
    assert sleeves["weight"].sum() == pytest.approx(1.0)
    # VTI tracks SPY, so the fund sleeve is level with it; BBB beat it
    assert sleeves.loc["Funds", "vs_spy"] == pytest.approx(0.0, abs=1e-6)
    assert sleeves.loc["Stocks you picked", "vs_spy"] > 0


def test_cash_alone_is_an_account_too():
    acct = portfolio.replay(_ledger({"date": "2020-01-02", "action": "deposit", "amount": 1_000.0}), _px())
    assert acct.holdings.empty and list(acct.holdings.columns) == portfolio.HOLDING_COLUMNS
    assert acct.summary["value"] == pytest.approx(1_000.0)
    assert acct.summary["vs_spy"] < 0  # cash sat still while SPY rose
    assert list(acct.sleeves["sleeve"]) == ["Cash"]


def test_an_uncached_ticker_is_valued_at_its_last_traded_price():
    px = _px()
    acct = portfolio.replay(
        _ledger({"date": "2020-01-02", "action": "buy", "ticker": "ZZZ", "shares": 3.0, "price": 40.0}), px
    )
    assert acct.holdings.set_index("ticker").loc["ZZZ", "value"] == pytest.approx(120.0)
    assert any("ZZZ" in w for w in acct.warnings)


def test_the_ledger_appends_and_refuses_what_cannot_be(tmp_path, monkeypatch):
    monkeypatch.setattr(
        portfolio.config, "get_paths", lambda: SimpleNamespace(portfolio_jsonl=tmp_path / "portfolio.jsonl")
    )
    px = _px()
    portfolio.add_transaction("deposit", amount=5_000, date="2020-01-02")
    e = portfolio.add_transaction("buy", ticker="aaa", shares=10, date="2020-01-02", px=px)  # at that day's close
    assert e["ticker"] == "AAA" and e["price"] == pytest.approx(_at(px, "AAA", "2020-01-02"))
    portfolio.add_transaction("income", amount=-15.0, note="account fee", date="2020-02-03")

    with pytest.raises(ValueError, match="only 10 are held"):
        portfolio.add_transaction("sell", ticker="AAA", shares=11, date="2020-03-02", px=px)
    with pytest.raises(ValueError):
        portfolio.add_transaction("deposit", amount=0)
    with pytest.raises(ValueError):
        portfolio.add_transaction("buy", ticker="AAA", shares=1, px=px, kind="crypto")

    ledger = portfolio.load_ledger()
    assert list(ledger["action"]) == ["deposit", "buy", "income"]
    assert list(ledger["id"]) == ["20200102-1", "20200102-2", "20200203-1"]
    assert portfolio.ledger_tickers(ledger) == ["AAA"]
    assert portfolio.replay(ledger, px).summary["net_deposits"] == pytest.approx(5_000)


def test_stock_tickers_are_the_operating_filers():
    fund = pd.DataFrame(
        {
            "ticker": ["AAPL", "GLD", None],
            "tickers_all": ["AAPL", "GLD", "XYZ"],
            "sic": [3571, 6221, 1000],
            "company": ["Apple Inc", "SPDR GOLD TRUST", "Nobody"],
        }
    )
    assert portfolio.stock_tickers(fund) == {"AAPL"}  # the gold trust counts as a fund
