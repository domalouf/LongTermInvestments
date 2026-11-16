"""Is the data the nightly job runs on still current?

Every step of that job can fail quietly: Yahoo stops answering and the price
cache stays where it was, the weekly rebuild stops and the list goes on ranking
old filings, a record isn't written and nobody looks for weeks. Each check here
is one of those, read off the files themselves rather than off whether a job
exited cleanly. ``lti health`` prints them and exits 1 if any fails;
``deploy/check-health.sh`` turns that into an alert.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import lti.config as config

# Trading days a daily series may trail today: a weekend is none, a holiday one,
# and the job runs before the next close — beyond three, a night was missed.
MAX_TRADING_DAYS_BEHIND = 3
# The weekly rebuild rewrites the table even when the SEC published nothing new.
MAX_FUNDAMENTALS_AGE_DAYS = 9
# The SEC publishes a quarter at a time, a few weeks after it ends: filings trail
# today by one to about three and a half months when the rebuild is working.
MAX_FILINGS_LAG_DAYS = 125
# FRED posts a day or two late; the valuations fall back to fixed rates at 14.
MAX_RATES_AGE_DAYS = 10
MAX_PUBLISHED_AGE_HOURS = 36


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _trading_days_behind(latest, now: pd.Timestamp) -> int:
    return int(np.busday_count(pd.Timestamp(latest).date(), now.date()))


def _daily(name: str, latest, now: pd.Timestamp, what: str) -> Check:
    if latest is None or pd.isna(latest):
        return Check(name, False, f"no {what} at all")
    behind = _trading_days_behind(latest, now)
    return Check(
        name, behind <= MAX_TRADING_DAYS_BEHIND,
        f"latest {what} {pd.Timestamp(latest).date()}, {behind} trading day{'s' if behind != 1 else ''} behind",
    )


def check_prices(now: pd.Timestamp) -> Check:
    path = config.get_paths().close_parquet
    if not path.exists():
        return Check("prices", False, f"{path.name} not found")
    import pyarrow.parquet as pq

    # one ticker's column, for the dates in its index: the whole panel is thousands
    index_cols = set(pq.read_schema(path).pandas_metadata["index_columns"]) if pq.read_schema(path).pandas_metadata else set()
    first = next(n for n in pq.read_schema(path).names if n not in index_cols)
    dates = pd.read_parquet(path, columns=[first]).index
    return _daily("prices", pd.to_datetime(dates).max(), now, "close")


def check_track(now: pd.Timestamp) -> Check:
    days = sorted(p.stem for p in config.get_paths().track_records_dir.glob("*.parquet"))
    return _daily("track record", pd.Timestamp(days[-1]) if days else None, now, "record")


def check_fundamentals(now: pd.Timestamp) -> list[Check]:
    paths = config.get_paths()
    path = paths.fundamentals_parquet
    if not path.exists():
        return [Check("fundamentals", False, f"{path.name} not found")]
    built = pd.Timestamp(path.stat().st_mtime, unit="s")
    age = (now - built).days
    filed = [pd.read_parquet(p, columns=["filed"])["filed"] for p in (path, paths.quarterly_parquet) if p.exists()]
    through = pd.to_datetime(pd.concat(filed), errors="coerce").max()
    lag = (now - through).days if pd.notna(through) else None
    return [
        Check("fundamentals rebuild", age <= MAX_FUNDAMENTALS_AGE_DAYS,
              f"last built {built.date()}, {age} days ago (weekly)"),
        Check("filings", lag is not None and lag <= MAX_FILINGS_LAG_DAYS,
              "no filing dates" if lag is None else f"through {through.date()}, {lag} days ago"),
        Check("10-Q rows", paths.quarterly_parquet.exists(),
              "quarterly.parquet " + ("present" if paths.quarterly_parquet.exists() else "not built")),
    ]


def check_rates(now: pd.Timestamp) -> Check:
    from lti import rates

    table = rates.load_rates()
    if table.empty:
        return Check("rates", False, "no rates cached — valuing at the fixed 9%")
    latest = pd.to_datetime(table.dropna(how="all").index).max()
    age = (now - latest).days
    return Check("rates", age <= MAX_RATES_AGE_DAYS, f"latest reading {latest.date()}, {age} days ago")


def check_published(path: str | Path, now: pd.Timestamp) -> Check:
    """The public snapshot's own ``generated_utc`` — the nightly job's last step."""
    path = Path(path)
    try:
        generated = pd.Timestamp(json.loads(path.read_text(encoding="utf-8"))["generated_utc"])
    except (OSError, ValueError, KeyError) as exc:
        return Check("published list", False, f"{path}: {type(exc).__name__}")
    generated = generated.tz_convert("UTC").tz_localize(None) if generated.tzinfo else generated
    hours = (now - generated).total_seconds() / 3600
    return Check("published list", hours <= MAX_PUBLISHED_AGE_HOURS,
                 f"generated {generated:%Y-%m-%d %H:%M} UTC, {hours:.0f} hours ago")


def run(now=None, published: str | Path | None = None) -> list[Check]:
    """Every check, as of ``now`` (UTC; default the present). A check that itself
    breaks is a failed check, not a crash: this is what reports the breakage."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    steps = [check_prices, check_track, check_fundamentals, check_rates]
    if published:
        steps.append(lambda n: check_published(published, n))
    out: list[Check] = []
    for step in steps:
        try:
            got = step(now)
        except Exception as exc:  # noqa: BLE001
            name = getattr(step, "__name__", "check_published").removeprefix("check_")
            got = Check(name, False, f"check failed: {type(exc).__name__}: {exc}")
        out.extend(got if isinstance(got, list) else [got])
    return out


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    return "\n".join(f"{'ok  ' if c.ok else 'FAIL'}  {c.name:<{width}}  {c.detail}" for c in checks)


def to_json(checks: list[Check]) -> str:
    return json.dumps({"ok": all(c.ok for c in checks), "checks": [asdict(c) for c in checks]}, indent=2)
