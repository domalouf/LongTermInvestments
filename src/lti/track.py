"""A forward track record: what each strategy picks each day, and how it did since.

Every backtest here runs on a survivor-only universe, and every idea in this
project was chosen after looking at the same fifteen years. The one test free of
both is the future. :func:`record` writes down — once, never to be rewritten —
what each strategy in :data:`STRATEGIES` would hold today; :func:`performance`
later measures what those holdings returned, against SPY and against the whole
universe recorded the same day. The list is made after the market closes, so
returns run from the next close — the first price anyone following it could get. A company acquired or delisted after a record
stays in it at its last price, so unlike the backtests this record has no
survivorship bias.

Records are one parquet file per day under ``data/track/records/``, with a JSON
sidecar holding the parameters and the git commit that made it. Returns are
always computed from the current total-return prices — never from a price
stored at the time, since Yahoo re-bases that history with every dividend.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd

import lti.config as config
from lti import factor, pit, prices as prices_mod, ranking, sectors, study, valuation
from lti.ranking import ScreenSpec

LOGGER = logging.getLogger(__name__)

UNIVERSE_CAP_MIN = 1e9
TOP_N = 30
HORIZONS = (1, 3, 6, 12)  # months
BENCHMARK = "SPY"
# how far behind the latest close a record may be made: a missed long weekend,
# not a backfill — an old day would be a backtest, with its survivorship bias
MAX_RECORD_LAG_DAYS = 7

# what `lti factor-study` chose from its first half on 2026-09-19 — fixed here so
# the record keeps testing that exact screen, whatever the study says later
FINAL_SCREEN = ["share_growth", "asset_growth", "shareholder_yield", "f_score"]


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    why: str


STRATEGIES: list[Strategy] = [
    Strategy("undervalued_list", "Undervalued list (top 30)",
             "What the landing page and domalouf.com/invest publish."),
    Strategy("final_screen", "Final screen (top 30)",
             "The factor study's screen — share issuance, asset growth, shareholder yield, "
             "F-score. It trailed its universe in 2019-25; does that go on?"),
    Strategy("no_diluters", "No heavy diluters",
             "The universe without its top tenth by share issuance: the study's strongest, "
             "most consistent signal, used as a filter rather than a screen."),
    Strategy("cash_returners", "Cash returners (top fifth)",
             "The top fifth by shareholder yield, the other signal that held up in both halves."),
    Strategy("quality_value", "Quality + value (top fifth)",
             "The top fifth of the quality-plus-value composite, which held up in 2019-25."),
    Strategy("universe", "Universe (all)",
             f"Every company in the universe — ${UNIVERSE_CAP_MIN / 1e9:.0f}B+, operating, "
             "financials and BDCs out — equal-weighted: the yardstick."),
]
RECORD_COLUMNS = ["record_date", "strategy", "rank", "ticker", "cik", "company", "weight", "close", "score"]


# --- recording ---------------------------------------------------------------


def universe(snap: pd.DataFrame, cap_min: float = UNIVERSE_CAP_MIN) -> pd.DataFrame:
    """The tracked universe, with the study's composites scored within it."""
    u = snap[snap["price"].notna() & (snap["price"] > 0) & (snap["market_cap"] >= cap_min)]
    if "is_financial" in u.columns:
        u = u[~u["is_financial"].fillna(False).astype(bool)]
    if "sic" in u.columns:
        u = u[~sectors.is_investment_company(u["sic"])]
    u = u.copy()
    for name, parts in study.COMPOSITES.items():
        u[name] = factor.composite_score(u, parts)
    return u


def _top_fraction(u: pd.DataFrame, col: str, fraction: float) -> pd.DataFrame:
    """The top ``fraction`` of ``u`` by ``col``, best first."""
    cut = u[col].quantile(1.0 - fraction)
    return u[u[col] >= cut].sort_values(col, ascending=False)


def holdings(
    fund: pd.DataFrame, px: prices_mod.PriceData, asof, *, cap_min: float = UNIVERSE_CAP_MIN, top_n: int = TOP_N
) -> pd.DataFrame:
    """What each strategy holds on ``asof`` — :data:`RECORD_COLUMNS`, equal weights."""
    asof = pd.Timestamp(asof)
    u = universe(pit.priced_snapshot(fund, asof, px, with_history=True), cap_min)

    listed = valuation.rank_undervalued(fund, px, asof, market_cap_min=cap_min, top_n=top_n)
    screen = ranking.rank(u, ScreenSpec(metrics=FINAL_SCREEN, top_n=top_n, min_coverage=0.5)).head(top_n)
    diluters = u["share_growth"] > u["share_growth"].quantile(0.9)
    picks = {
        "undervalued_list": (listed, "fair_value_est_upside"),
        "final_screen": (screen, "composite_score"),
        "no_diluters": (u[~diluters], "share_growth"),
        "cash_returners": (_top_fraction(u, "shareholder_yield", 0.2), "shareholder_yield"),
        "quality_value": (_top_fraction(u, "quality_value", 0.2), "quality_value"),
        "universe": (u, "market_cap"),
    }

    frames = []
    for name, (df, score_col) in picks.items():
        df = df[df["ticker"].notna()].drop_duplicates("ticker")
        if df.empty:
            LOGGER.warning("track: %s holds nothing on %s", name, asof.date())
            continue
        frames.append(
            pd.DataFrame(
                {
                    "record_date": asof,
                    "strategy": name,
                    "rank": np.arange(1, len(df) + 1),
                    "ticker": df["ticker"].astype(str).to_numpy(),
                    "cik": df.index.to_numpy(dtype="int64"),
                    "company": df["company"].astype(str).to_numpy() if "company" in df.columns else "",
                    "weight": 1.0 / len(df),
                    "close": df["price"].to_numpy(dtype="float64"),
                    "score": df[score_col].to_numpy(dtype="float64") if score_col in df.columns else np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RECORD_COLUMNS)


def _git_state() -> tuple[str | None, bool | None]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except Exception:  # noqa: BLE001 - not a checkout, or no git: the record still stands
        return None, None


def record(
    asof=None,
    fund: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame | None:
    """Write today's holdings for every strategy — once.

    ``asof`` defaults to the latest close in the price cache, and may be at most
    :data:`MAX_RECORD_LAG_DAYS` before it: the price cache only holds companies
    still trading, so a record made long after its date would quietly leave out
    the ones that failed in between. A day already on record is left alone
    (returns None): nothing in the record is revised after the fact. ``force``
    exists only to redo a record made earlier the same day by mistake.
    """
    if px is None:
        px = prices_mod.load_price_data()
    latest = px.close.index.max()
    asof = latest if asof is None else pd.Timestamp(asof).normalize()
    if asof > latest:
        raise ValueError(f"no close for {asof.date()} in the price cache yet (latest {latest.date()})")
    if (latest - asof).days > MAX_RECORD_LAG_DAYS:
        raise ValueError(
            f"{asof.date()} is {(latest - asof).days} days before the latest close ({latest.date()}): "
            "the record is made on the day, not after the fact — for the past, run a backtest"
        )
    asof = px.close.index[px.close.index <= asof].max()  # a weekend or holiday means the close before
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()

    out_dir = config.get_paths().track_records_dir
    path = out_dir / f"{asof.date()}.parquet"
    if path.exists() and not force:
        LOGGER.info("track: %s is already on record", asof.date())
        return None

    rows = holdings(fund, px, asof)
    if rows.empty:
        raise RuntimeError(f"no strategy held anything on {asof.date()} — is the price cache current?")
    commit, dirty = _git_state()
    meta = {
        "record_date": str(asof.date()),
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "universe_cap_min": UNIVERSE_CAP_MIN,
        "top_n": TOP_N,
        "final_screen": FINAL_SCREEN,
        "holdings": rows.groupby("strategy").size().to_dict(),
    }
    prices_mod._write_parquet(rows, path, index=False)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("track: recorded %s (%s)", asof.date(), ", ".join(f"{k} {v}" for k, v in meta["holdings"].items()))
    return rows


def load_records() -> pd.DataFrame:
    files = sorted(config.get_paths().track_records_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    records = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    records["record_date"] = pd.to_datetime(records["record_date"])
    return records


# --- scoring -------------------------------------------------------------------


def _next_close(index: pd.DatetimeIndex, record_date) -> pd.Timestamp | None:
    """The first close after ``record_date``, where a record's holdings are bought."""
    pos = index.searchsorted(pd.Timestamp(record_date), side="right")
    return index[pos] if pos < len(index) else None


def _basket_return(
    px: prices_mod.PriceData, basket: pd.DataFrame, start, end, ticker_now: pd.Series | None = None
) -> tuple[float, int]:
    """Weighted total return of ``basket`` from ``start`` to ``end``, and how many names priced.

    A name that stopped trading is held at its last price. When a recorded ticker
    has no price at all (renamed since), the company's current ticker is tried.
    """
    tickers = pd.Series(basket["ticker"].astype(str).to_numpy(), index=basket.index)
    r = prices_mod.forward_returns(px.adj, tickers, start, end)
    if ticker_now is not None and r.isna().any():
        alt = pd.Series(basket["cik"].map(ticker_now).to_numpy(), index=basket.index)
        retry = r.isna() & alt.notna() & (alt != tickers)
        if retry.any():
            r[retry] = prices_mod.forward_returns(px.adj, alt[retry].astype(str), start, end)
    w = basket["weight"].where(r.notna())
    if w.sum() <= 0:
        return float("nan"), 0
    return float((r.fillna(0.0) * w).sum() / w.sum()), int(r.notna().sum())


def _ticker_now() -> pd.Series | None:
    try:
        from lti import tickers

        m = tickers.get_cik_ticker_map()
        return pd.Series(m["ticker"].to_numpy(), index=m["cik"].to_numpy())
    except Exception:  # noqa: BLE001 - the fallback is a nicety
        return None


def performance(
    records: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
    *,
    horizons=HORIZONS,
    ticker_now: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per record date, strategy and horizon that has come due: the
    strategy's return from the next close, the same-day universe's, SPY's, and
    the gaps."""
    records = load_records() if records is None else records
    px = prices_mod.load_price_data() if px is None else px
    cols = ["record_date", "strategy", "horizon", "ret", "universe", "spy", "vs_universe", "vs_spy", "names"]
    if records.empty:
        return pd.DataFrame(columns=cols)
    ticker_now = _ticker_now() if ticker_now is None else ticker_now
    latest = px.adj.index.max()

    rows = []
    for date, day in records.groupby("record_date"):
        date = pd.Timestamp(date)
        start = _next_close(px.adj.index, date)
        if start is None:
            continue
        for h in horizons:
            end = start + pd.DateOffset(months=h)
            if end > latest:
                continue
            spy = (
                float(prices_mod.forward_returns(px.adj, [BENCHMARK], start, end).iloc[0])
                if BENCHMARK in px.adj.columns
                else float("nan")
            )
            rets = {s: _basket_return(px, g, start, end, ticker_now) for s, g in day.groupby("strategy")}
            univ = rets.get("universe", (float("nan"), 0))[0]
            for s, (r, n) in rets.items():
                rows.append(
                    {"record_date": date, "strategy": s, "horizon": h, "ret": r, "universe": univ, "spy": spy,
                     "vs_universe": r - univ, "vs_spy": r - spy, "names": n}
                )
    return pd.DataFrame(rows, columns=cols)


def summary(perf: pd.DataFrame) -> pd.DataFrame:
    """Per strategy and horizon: records come due, average return and gaps, how
    often it beat the universe — and a Newey-West t on one record a month
    (records a day apart share almost all their returns)."""
    cols = ["strategy", "horizon", "records", "months", "ret", "vs_universe", "vs_spy", "beat_universe", "t_vs_universe"]
    if perf.empty:
        return pd.DataFrame(columns=cols)
    month = perf["record_date"].dt.to_period("M")
    first_in_month = perf["record_date"] == perf.groupby([month, "strategy", "horizon"])["record_date"].transform("min")
    rows = []
    for (s, h), g in perf.groupby(["strategy", "horizon"]):
        m = perf[first_in_month & (perf["strategy"] == s) & (perf["horizon"] == h)].sort_values("record_date")
        yardstick = s == "universe"  # measured against itself: nothing to test
        rows.append(
            {
                "strategy": s, "horizon": h, "records": len(g), "months": len(m),
                "ret": g["ret"].mean(), "vs_universe": g["vs_universe"].mean(), "vs_spy": g["vs_spy"].mean(),
                "beat_universe": np.nan if yardstick else float((g["vs_universe"] > 0).mean()),
                "t_vs_universe": np.nan if yardstick else factor.newey_west_t(m["vs_universe"], lags=max(0, h - 1)),
            }
        )
    order = {s.name: i for i, s in enumerate(STRATEGIES)}
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["horizon", "strategy"], key=lambda c: c.map(order) if c.name == "strategy" else c)


def paper_curves(
    records: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
    *,
    ticker_now: pd.Series | None = None,
) -> pd.DataFrame:
    """Growth of $1 in each strategy (and SPY) from the first record, moving into
    each month's first record at the next close — the portfolio you'd hold
    rebalancing monthly. Indexed by those closes."""
    records = load_records() if records is None else records
    px = prices_mod.load_price_data() if px is None else px
    if records.empty:
        return pd.DataFrame()
    ticker_now = _ticker_now() if ticker_now is None else ticker_now
    dates = pd.Series(sorted(pd.to_datetime(records["record_date"].unique())))
    firsts = dates.groupby(dates.dt.to_period("M")).min()
    legs = [(d, e) for d in firsts if (e := _next_close(px.adj.index, d)) is not None]
    if not legs:
        return pd.DataFrame()
    stops = [e for _, e in legs[1:]] + [px.adj.index.max()]

    names = [s.name for s in STRATEGIES if s.name in set(records["strategy"])] + [BENCHMARK]
    value = dict.fromkeys(names, 1.0)
    points = {legs[0][1]: dict(value)}
    for (d0, e0), e1 in zip(legs, stops):
        if e1 <= e0:
            continue
        day = records[records["record_date"] == d0]
        for s, g in day.groupby("strategy"):
            r, _ = _basket_return(px, g, e0, e1, ticker_now)
            if not np.isnan(r):
                value[s] *= 1.0 + r
        if BENCHMARK in px.adj.columns:
            spy = float(prices_mod.forward_returns(px.adj, [BENCHMARK], e0, e1).iloc[0])
            if not np.isnan(spy):
                value[BENCHMARK] *= 1.0 + spy
        points[e1] = dict(value)
    return pd.DataFrame(points).T[names]
