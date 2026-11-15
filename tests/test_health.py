"""`lti health`: each check reads staleness off the files, not off a job's exit code."""

from __future__ import annotations

import dataclasses
import json
import os

import pandas as pd
import pytest

import lti.config as config
from lti import health

NOW = pd.Timestamp("2026-11-16 08:00")  # a Monday, just after the nightly run


@pytest.fixture
def paths(tmp_path, monkeypatch):
    d = tmp_path
    fake = dataclasses.replace(
        config.get_paths(),
        close_parquet=d / "close.parquet", fundamentals_parquet=d / "fundamentals.parquet",
        quarterly_parquet=d / "quarterly.parquet", rates_parquet=d / "rates.parquet",
        track_dir=d / "track", track_records_dir=d / "track" / "records",
    )
    fake.track_records_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "get_paths", lambda: fake)
    return fake


def _write(paths, *, close="2026-11-13", record="2026-11-13", built="2026-11-14", filed="2026-09-30",
           rate="2026-11-12", quarterly=True):
    idx = pd.bdate_range("2026-10-01", close)
    pd.DataFrame({"AAA": 1.0, "SPY": 2.0}, index=idx).to_parquet(paths.close_parquet)
    for old in paths.track_records_dir.iterdir():
        old.unlink()
    (paths.track_records_dir / f"{record}.parquet").write_bytes(b"")
    pd.DataFrame({"filed": pd.to_datetime(["2025-03-01", filed])}).to_parquet(paths.fundamentals_parquet)
    stamp = pd.Timestamp(built).timestamp()
    os.utime(paths.fundamentals_parquet, (stamp, stamp))
    if quarterly:
        pd.DataFrame({"filed": pd.to_datetime([filed])}).to_parquet(paths.quarterly_parquet)
    pd.DataFrame({"treasury_10y": [0.04], "aaa": [0.05]}, index=pd.to_datetime([rate])).to_parquet(paths.rates_parquet)


def _failed(checks):
    return {c.name for c in checks if not c.ok}


def test_a_monday_after_a_working_week_is_healthy(paths):
    _write(paths)
    checks = health.run(NOW)
    assert not _failed(checks), health.render(checks)
    assert json.loads(health.to_json(checks))["ok"] is True


def test_each_stale_input_fails_its_own_check(paths):
    _write(paths, close="2026-11-06")  # a week of missed price refreshes
    assert _failed(health.run(NOW)) == {"prices"}
    _write(paths, record="2026-11-06")
    assert _failed(health.run(NOW)) == {"track record"}
    _write(paths, built="2026-11-01")  # the weekly rebuild skipped a Saturday
    assert _failed(health.run(NOW)) == {"fundamentals rebuild"}
    _write(paths, filed="2026-06-30")  # rebuilding, but on the same old quarter
    assert _failed(health.run(NOW)) == {"filings"}
    _write(paths, rate="2026-10-30")
    assert _failed(health.run(NOW)) == {"rates"}


def test_a_long_weekend_is_not_a_missed_night(paths):
    _write(paths, close="2026-11-25", record="2026-11-25", built="2026-11-28", rate="2026-11-25")
    # Thanksgiving Thursday and the weekend: Monday's run still sees Wednesday's close
    assert not _failed(health.run(pd.Timestamp("2026-11-30 08:00")))


def test_missing_files_fail_rather_than_crash(paths):
    failed = _failed(health.run(NOW))
    assert {"prices", "track record", "fundamentals", "rates"} <= failed
    _write(paths, quarterly=False)
    paths.quarterly_parquet.unlink(missing_ok=True)
    assert _failed(health.run(NOW)) == {"10-Q rows"}


def test_the_published_list_is_checked_by_its_own_timestamp(paths, tmp_path):
    _write(paths)
    page = tmp_path / "undervalued.json"
    page.write_text(json.dumps({"generated_utc": "2026-11-16T07:44:01+00:00"}))
    assert not _failed(health.run(NOW, published=page))
    page.write_text(json.dumps({"generated_utc": "2026-11-13T07:44:01+00:00"}))
    assert _failed(health.run(NOW, published=page)) == {"published list"}
    assert _failed(health.run(NOW, published=tmp_path / "gone.json")) == {"published list"}


def test_the_cli_exits_nonzero_when_something_is_stale(paths, capsys):
    from lti import cli

    _write(paths, close="2026-10-20")
    with pytest.raises(SystemExit) as exc:
        cli.main(["health"])
    assert exc.value.code == 1 and "FAIL  prices" in capsys.readouterr().out
