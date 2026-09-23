"""Interest rates at a date (lti.rates) and the valuation assumptions built on them."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import rates
from lti.valuation import EQUITY_PREMIUM, ValuationAssumptions, market_assumptions

# FRED has shipped both of these: an older DATE header with "." for a missing
# day, and the current observation_date header with a blank
OLD_CSV = "DATE,DGS10\n2013-03-28,1.87\n2013-03-29,.\n2013-04-01,1.86\n"
NEW_CSV = "observation_date,DAAA\n2013-03-28,3.93\n2013-03-29,\n2013-04-01,3.90\n"


def _rates() -> pd.DataFrame:
    table = pd.DataFrame({"treasury_10y": rates.parse_fred_csv(OLD_CSV), "aaa": rates.parse_fred_csv(NEW_CSV)})
    table.index.name = "date"
    return table


def test_a_fred_csv_parses_to_fractions_by_date_in_either_format():
    for text, first in ((OLD_CSV, 0.0187), (NEW_CSV, 0.0393)):
        s = rates.parse_fred_csv(text)
        assert list(s.index) == [pd.Timestamp("2013-03-28"), pd.Timestamp("2013-04-01")]  # the gap dropped
        assert s.iloc[0] == pytest.approx(first)
    with pytest.raises(ValueError):  # a date column and nothing else
        rates.parse_fred_csv("DATE\n2013-04-01\n")


def test_the_rate_on_a_date_is_the_last_reading_before_it_if_recent():
    table = _rates()
    easter = rates.rates_asof(table, "2013-03-31")  # a Sunday: Thursday's reading
    assert easter["treasury_10y"] == pytest.approx(0.0187) and easter["aaa"] == pytest.approx(0.0393)
    assert np.isnan(rates.rates_asof(table, "2013-03-27")).all()  # before the series
    assert np.isnan(rates.rates_asof(table, "2013-06-01")).all()  # two months stale
    assert np.isnan(rates.rates_asof(rates.empty_rates(), "2013-04-01")).all()


def test_market_assumptions_discount_at_the_treasury_plus_the_premium():
    a = market_assumptions("2013-04-01", _rates())
    assert a.discount_rate == pytest.approx(0.0186 + EQUITY_PREMIUM)
    assert a.bond_yield == pytest.approx(0.0390)

    base = ValuationAssumptions(growth_cap=0.10)
    assert market_assumptions("2013-04-01", _rates(), base, equity_premium=0.03).discount_rate == pytest.approx(0.0486)
    assert market_assumptions("2013-04-01", _rates(), base).growth_cap == 0.10  # the rest carried over
    # nothing cached: the fixed rates, untouched
    assert market_assumptions("2013-04-01", rates.empty_rates(), base) is base
    # a rate so low the terminal growth has to give way to keep the perpetuity finite
    low = market_assumptions("2013-04-01", _rates(), equity_premium=0.01)
    assert low.terminal_growth == pytest.approx(low.discount_rate - 0.01)


def _fake_fred(monkeypatch, responses: dict):
    import requests

    def get(url, **_):
        body = next((v for k, v in responses.items() if url.endswith(f"id={k}")), None)
        if isinstance(body, Exception) or body is None:
            raise body or requests.ConnectionError("no route")
        return SimpleNamespace(text=body, raise_for_status=lambda: None)

    monkeypatch.setattr(requests, "get", get)


def test_fetching_caches_both_series_and_a_failure_keeps_the_old_one(tmp_path, monkeypatch):
    monkeypatch.setattr(rates.config, "get_paths", lambda: SimpleNamespace(rates_parquet=tmp_path / "rates.parquet"))

    _fake_fred(monkeypatch, {"DGS10": OLD_CSV, "DAAA": NEW_CSV})
    table = rates.fetch_rates()
    assert list(table.columns) == ["treasury_10y", "aaa"]
    pd.testing.assert_frame_equal(rates.load_rates(), table, check_freq=False)

    # DAAA down: the Treasury refreshes, the cached AAA survives
    _fake_fred(monkeypatch, {"DGS10": OLD_CSV + "2013-04-02,1.84\n", "DAAA": None})
    table = rates.fetch_rates()
    assert table["treasury_10y"].dropna().iloc[-1] == pytest.approx(0.0184)
    assert table["aaa"].dropna().iloc[-1] == pytest.approx(0.0390)

    # everything down: an error, and the cache untouched
    _fake_fred(monkeypatch, {})
    with pytest.raises(RuntimeError):
        rates.fetch_rates()
    assert rates.load_rates()["treasury_10y"].dropna().iloc[-1] == pytest.approx(0.0184)
