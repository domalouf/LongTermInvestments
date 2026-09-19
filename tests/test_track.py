"""The forward track record and the decision journal.

Pure logic — no SEC data or network needed; the data dir points at a temp dir.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
import lti.config as config
from lti import journal, track
from lti.fundamentals import _add_prior_year
from lti.prices import PriceData, empty_splits

RECORD_DAY = "2025-06-02"  # a Monday
ENTRY_DAY = "2025-06-03"  # the list is made after the close, so it's bought at the next one


@pytest.fixture
def paths(tmp_path, monkeypatch):
    d = tmp_path / "track"
    fake = dataclasses.replace(config.get_paths(), track_dir=d, track_records_dir=d / "records",
                               journal_jsonl=d / "journal.jsonl")
    monkeypatch.setattr(config, "get_paths", lambda: fake)
    return fake


def _row(cik, fy, *, shares, eps=2.0, payout=-50.0, assets=4000.0):
    ni = eps * shares
    return dict(
        cik=cik, ticker=f"T{cik:02d}", company=f"Co {cik}", sic=2000, form="10-K", fiscal_year=fy,
        period_end=pd.Timestamp(fy, 12, 31), filed=pd.Timestamp(fy + 1, 3, 1), revenues=5000.0 + 10 * cik,
        net_income=ni, eps=eps, operating_income=ni * 1.3, operating_income_reported=ni * 1.3,
        equity=2000.0, liabilities=2000.0, assets=assets, assets_current=1500.0, liabilities_current=700.0,
        liabilities_noncurrent=1300.0, retained_earnings=900.0, cash=200.0, debt_current=0.0, total_debt=800.0,
        ppe_net=1200.0, shares_outstanding=shares, cfo=ni * 1.2, capex=100.0, free_cash_flow=ni * 1.2 - 100.0,
        dividends_paid=payout, buybacks=np.nan, dep_amort=150.0,
    )


@pytest.fixture
def world():
    """Ten steady companies; T10 doubles its share count in the latest year, T01 pays out the most."""
    rows = []
    for cik in range(1, 11):
        for fy in range(2016, 2025):
            shares = 2e8 * (2 if cik == 10 and fy == 2024 else 1)
            rows.append(_row(cik, fy, shares=shares, payout=-500.0 if cik == 1 else -50.0))
    fund = _add_prior_year(pd.DataFrame(rows))
    idx = pd.bdate_range("2015-01-01", "2026-06-30")
    years = (idx - pd.Timestamp(RECORD_DAY)).days / 365.25
    panel = pd.DataFrame({f"T{c:02d}": 15.0 * np.exp(0.02 * c * years) for c in range(1, 11)}, index=idx)
    panel["SPY"] = 100.0 * np.exp(0.08 * years)
    return fund, PriceData(adj=panel, close=panel, splits=empty_splits())


# --- recording -------------------------------------------------------------------


def _as_of(px: PriceData, day: str) -> PriceData:
    """The price cache as it stood at the close on ``day``."""
    return PriceData(px.adj.loc[:day], px.close.loc[:day], px.splits)


def test_record_writes_every_strategy_once(paths, world):
    fund, px = world
    px = _as_of(px, RECORD_DAY)
    rows = track.record(None, fund, px)
    day = paths.track_records_dir / f"{RECORD_DAY}.parquet"
    assert day.exists() and day.with_suffix(".json").exists()
    meta = json.loads(day.with_suffix(".json").read_text())
    assert meta["final_screen"] == track.FINAL_SCREEN and "git_commit" in meta

    held = rows.groupby("strategy")["ticker"].apply(set)
    assert held["universe"] == {f"T{c:02d}" for c in range(1, 11)}
    assert "T10" not in held["no_diluters"]  # doubled its share count
    assert "T01" in held["cash_returners"]  # the biggest payout
    assert set(held.index) >= {"universe", "no_diluters", "cash_returners", "quality_value", "final_screen"}
    assert rows.groupby("strategy")["weight"].sum().round(12).eq(1.0).all()

    assert track.record(RECORD_DAY, fund, px) is None  # already on record: left alone
    assert len(track.load_records()) == len(rows)


def test_a_record_is_made_on_the_day_not_after(paths, world):
    fund, px = world
    now = _as_of(px, "2025-06-10")
    with pytest.raises(ValueError, match="not after the fact"):
        track.record("2025-05-30", fund, now)  # 11 days back: a backfill
    with pytest.raises(ValueError, match="yet"):
        track.record("2025-06-11", fund, now)
    rows = track.record("2025-06-08", fund, now)  # a Sunday: the Friday close
    assert rows["record_date"].iloc[0] == pd.Timestamp("2025-06-06")
    assert (paths.track_records_dir / "2025-06-06.parquet").exists()


# --- scoring -------------------------------------------------------------------------


def _records(day: str, baskets: dict[str, list[str]]) -> pd.DataFrame:
    frames = []
    for strategy, tickers in baskets.items():
        frames.append(pd.DataFrame({"record_date": pd.Timestamp(day), "strategy": strategy,
                                    "rank": range(1, len(tickers) + 1), "ticker": tickers,
                                    "cik": [int(t[1:]) for t in tickers], "company": "",
                                    "weight": 1.0 / len(tickers), "close": np.nan, "score": np.nan}))
    return pd.concat(frames, ignore_index=True)


def test_performance_against_the_same_day_universe(world):
    _, px = world
    rec = _records(RECORD_DAY, {"universe": [f"T{c:02d}" for c in range(1, 11)], "cash_returners": ["T09", "T10"]})
    perf = track.performance(rec, px, ticker_now=pd.Series(dtype=object))
    assert set(perf["horizon"]) == {1, 3, 6, 12}  # the panel runs to mid-2026
    row = perf[(perf["strategy"] == "cash_returners") & (perf["horizon"] == 12)].iloc[0]
    end = pd.Timestamp(ENTRY_DAY) + pd.DateOffset(months=12)

    def ret(t):
        return px.adj.loc[:end, t].iloc[-1] / px.adj.loc[ENTRY_DAY, t] - 1

    assert row["ret"] == pytest.approx(np.mean([ret("T09"), ret("T10")]))
    assert row["universe"] == pytest.approx(np.mean([ret(f"T{c:02d}") for c in range(1, 11)]))
    assert row["vs_universe"] > 0  # the fastest compounders
    assert row["spy"] == pytest.approx(ret("SPY"))


def test_holdings_are_bought_at_the_next_close(world):
    _, px = world
    adj = px.adj.copy()
    adj.loc[adj.index >= ENTRY_DAY, "T05"] *= 1.2  # jumps the day after the list, before anyone could buy
    jumped = PriceData(adj, adj, empty_splits())
    perf = track.performance(_records(RECORD_DAY, {"universe": ["T05"]}), jumped, horizons=(1,),
                             ticker_now=pd.Series(dtype=object))
    end = pd.Timestamp(ENTRY_DAY) + pd.DateOffset(months=1)
    assert perf["ret"].iloc[0] == pytest.approx(adj.loc[:end, "T05"].iloc[-1] / adj.loc[ENTRY_DAY, "T05"] - 1)
    assert perf["ret"].iloc[0] < 0.05  # the jump isn't in it


def test_horizons_not_yet_due_are_left_out(world):
    _, px = world
    none = pd.Series(dtype=object)
    perf = track.performance(_records("2026-03-02", {"universe": ["T01", "T02"]}), px, ticker_now=none)
    assert set(perf["horizon"]) == {1, 3}
    last = _records("2026-06-30", {"universe": ["T01"]})  # made on the last close in the cache: not bought yet
    assert track.performance(last, px, ticker_now=none).empty
    assert track.paper_curves(last, px, ticker_now=none).empty


def test_a_delisted_name_stays_at_its_last_price(world):
    _, px = world
    adj = px.adj.copy()
    adj.loc[adj.index > "2025-09-30", "T05"] = np.nan  # acquired and gone
    gone = PriceData(adj, adj, empty_splits())
    rec = _records(RECORD_DAY, {"universe": ["T05"]})
    perf = track.performance(rec, gone, horizons=(12,), ticker_now=pd.Series(dtype=object))
    frozen = adj.loc[:"2025-09-30", "T05"].iloc[-1] / adj.loc[ENTRY_DAY, "T05"] - 1
    assert perf["ret"].iloc[0] == pytest.approx(frozen)


def test_a_renamed_ticker_is_found_by_its_cik(world):
    _, px = world
    rec = _records(RECORD_DAY, {"universe": ["T03"]}).assign(ticker="OLD")
    perf = track.performance(rec, px, horizons=(1,), ticker_now=pd.Series({3: "T03"}))
    assert perf["names"].iloc[0] == 1 and np.isfinite(perf["ret"].iloc[0])


def test_summary_and_paper_curves(world):
    _, px = world
    rec = pd.concat([_records("2025-06-02", {"universe": ["T01", "T10"], "cash_returners": ["T10"]}),
                     _records("2025-07-01", {"universe": ["T01", "T10"], "cash_returners": ["T01"]})])
    perf = track.performance(rec, px, ticker_now=pd.Series(dtype=object))
    s = track.summary(perf).set_index(["strategy", "horizon"])
    assert s.loc[("cash_returners", 1), "records"] == 2
    assert np.isnan(s.loc[("universe", 1), "beat_universe"])  # not scored against itself
    curve = track.paper_curves(rec, px, ticker_now=pd.Series(dtype=object))
    assert list(curve.columns) == ["cash_returners", "universe", "SPY"]
    assert curve.index[0] == pd.Timestamp(ENTRY_DAY) and curve.iloc[0].eq(1.0).all()
    june = px.adj.loc["2025-07-02", "T10"] / px.adj.loc[ENTRY_DAY, "T10"]  # to the next close after 07-01
    assert curve.loc[pd.Timestamp("2025-07-02"), "cash_returners"] == pytest.approx(june)


# --- the journal -----------------------------------------------------------------------


def test_journal_entries_are_checked_and_scored(paths, world):
    _, px = world
    buy = journal.add_entry("t10", "buy", "fastest compounder in the group", conviction=4, date="2025-06-02", px=px)
    assert buy["ticker"] == "T10" and buy["review_by"] == "2026-06-02"
    journal.add_entry("T01", "pass", "slowest grower", date="2025-06-02", px=px)
    journal.add_entry("T10", "review", "on track", refers_to=buy["id"], date="2026-06-03", px=px)
    with pytest.raises(ValueError, match="action"):
        journal.add_entry("T01", "yolo", "because", px=px)
    with pytest.raises(ValueError, match="price cache"):
        journal.add_entry("NOPE", "buy", "typo", px=px)
    with pytest.raises(ValueError, match="why"):
        journal.add_entry("T01", "buy", "   ", px=px)

    out = journal.outcomes(px=px).set_index("action")
    assert out.loc["buy", "right_so_far"] == 1.0  # beat SPY since
    assert out.loc["pass", "right_so_far"] == 1.0  # T01 then lagged SPY, so passing was right
    assert np.isnan(out.loc["review", "right_so_far"])
    assert not out.loc["buy", "review_due"]  # reviewed
    assert out.loc["pass", "review_due"]  # due 2026-06-02 and never reviewed
