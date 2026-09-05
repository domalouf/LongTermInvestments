"""Simple annual-rebalance backtest engine.

Each rebalance date: take a point-in-time snapshot, rank by the screen's metrics,
buy the top-N equal-weighted, hold until the next rebalance. Benchmark is a single
buy-and-hold position in ``config`` ticker (default SPY).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import lti.config as config
from lti import metrics, pit, prices as prices_mod, ranking
from lti.performance import summarize
from lti.ranking import ScreenSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    screen: ScreenSpec
    start: str = "2011-01-01"
    end: str | None = None
    rebalance_month: int = 1
    benchmark: str = "SPY"
    rf_annual: float = 0.0
    initial_capital: float = 100_000.0
    market_cap_min: float = 500_000_000.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    holdings: pd.DataFrame
    period_summary: pd.DataFrame
    stats: dict
    warnings: list[str] = field(default_factory=list)


def _rebalance_dates(trading_days: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, month: int) -> list[pd.Timestamp]:
    dates = []
    for year in range(start.year, end.year + 1):
        target = pd.Timestamp(year=year, month=month, day=1)
        pos = trading_days.searchsorted(target)
        if pos < len(trading_days):
            d = trading_days[pos]
            if start <= d <= end:
                dates.append(d)
    return sorted(set(dates))


def _monthly_grid(rd: pd.Timestamp, nrd: pd.Timestamp) -> pd.DatetimeIndex:
    months = pd.date_range(rd, nrd, freq="ME")
    return months.append(pd.DatetimeIndex([nrd])).unique()


def _relative_path(panel: pd.DataFrame, ticker: str, entry: float, grid: pd.DatetimeIndex) -> pd.Series:
    """price(t)/entry on the monthly grid, frozen at last known price after delisting."""
    if ticker not in panel.columns or entry is None or entry <= 0:
        return pd.Series(1.0, index=grid)
    series = panel[ticker].dropna()
    reindexed = series.reindex(series.index.union(grid)).ffill().reindex(grid)
    return (reindexed / entry).ffill().fillna(1.0)


def run_backtest(
    cfg: BacktestConfig,
    fund: pd.DataFrame | None = None,
    price_panel: pd.DataFrame | None = None,
) -> BacktestResult:
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if price_panel is None:
        price_panel = prices_mod.load_adj_close()

    warnings: list[str] = []
    bench = cfg.benchmark.upper()
    if bench not in price_panel.columns:
        raise RuntimeError(f"benchmark {bench} not in price cache — run `lti fetch-prices`")

    trading_days = price_panel[bench].dropna().index
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end) if cfg.end else trading_days[-1]
    end = min(end, trading_days[-1])

    rebal_dates = _rebalance_dates(trading_days, start, end, cfg.rebalance_month)
    if len(rebal_dates) < 2:
        raise RuntimeError("need at least two rebalance dates in the date range")

    spec = cfg.screen
    spec.filters = {**spec.filters, "market_cap_min": cfg.market_cap_min, "require_price": True}

    equity_segments: list[pd.Series] = []
    bench_segments: list[pd.Series] = []
    holdings_rows: list[dict] = []
    period_rows: list[dict] = []

    port_value = cfg.initial_capital
    bench_value = cfg.initial_capital
    equity_segments.append(pd.Series({rebal_dates[0]: port_value}))
    bench_segments.append(pd.Series({rebal_dates[0]: bench_value}))

    for rd, nrd in zip(rebal_dates[:-1], rebal_dates[1:]):
        snap = pit.snapshot_asof(fund, rd)
        snap = snap[snap["ticker"].notna()]
        if snap.empty:
            warnings.append(f"{rd.date()}: no companies with fundamentals+ticker known")
            continue

        snap = metrics.add_fundamental_metrics(snap)
        entry_prices = pd.Series(
            {cik: prices_mod.price_on_or_before(price_panel, t, rd) for cik, t in snap["ticker"].items()}
        )
        mcap = entry_prices * snap.get("shares_outstanding", pd.Series(np.nan, index=snap.index))
        snap = metrics.add_price_metrics(snap, price=entry_prices, market_cap=mcap)

        picks = ranking.select(snap, spec)
        if not picks:
            warnings.append(f"{rd.date()}: screen produced no picks")
            continue
        if len(picks) < spec.top_n:
            warnings.append(f"{rd.date()}: only {len(picks)}/{spec.top_n} names qualified")

        weight = 1.0 / len(picks)
        pick_rows = snap[snap["ticker"].isin(picks)].drop_duplicates("ticker")
        entry_by_ticker = dict(zip(pick_rows["ticker"], entry_prices.reindex(pick_rows.index)))

        bench_ret, _ = prices_mod.forward_return(price_panel, bench, rd, nrd)
        n_delisted = 0
        holding_returns = []
        for t in picks:
            entry = entry_by_ticker.get(t)
            r, delisted = prices_mod.forward_return(price_panel, t, rd, nrd)
            if r is None:
                r = 0.0
                warnings.append(f"{rd.date()}: no forward price for {t}; treated as cash")
            n_delisted += int(bool(delisted))
            holding_returns.append(r)
            holdings_rows.append(
                {
                    "rebalance_date": rd,
                    "exit_date": nrd,
                    "ticker": t,
                    "weight": weight,
                    "entry_price": entry,
                    "period_return": r,
                    "benchmark_period_return": bench_ret,
                    "delisted": bool(delisted),
                }
            )

        port_period_return = float(np.mean(holding_returns))

        grid = _monthly_grid(rd, nrd)
        rel = pd.DataFrame(
            {t: _relative_path(price_panel, t, entry_by_ticker.get(t), grid) for t in picks}
        )
        port_path = port_value * rel.mul(weight).sum(axis=1)
        bench_rel = _relative_path(price_panel, bench, prices_mod.price_on_or_before(price_panel, bench, rd), grid)
        bench_path = bench_value * bench_rel

        equity_segments.append(port_path)
        bench_segments.append(bench_path)
        port_value = float(port_path.iloc[-1])
        bench_value = float(bench_path.iloc[-1])

        period_rows.append(
            {
                "rebalance_date": rd,
                "exit_date": nrd,
                "n_selected": len(picks),
                "n_delisted": n_delisted,
                "port_return": port_period_return,
                "bench_return": bench_ret,
                "excess_return": port_period_return - (bench_ret or 0.0),
            }
        )

    equity_curve = pd.concat(equity_segments).sort_index()
    equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]
    benchmark_curve = pd.concat(bench_segments).sort_index()
    benchmark_curve = benchmark_curve[~benchmark_curve.index.duplicated(keep="last")]

    holdings = pd.DataFrame(holdings_rows)
    period_summary = pd.DataFrame(period_rows)
    stats = summarize(equity_curve, benchmark_curve, cfg.rf_annual, holdings)

    # de-duplicate warnings, keep order
    warnings = list(dict.fromkeys(warnings))
    return BacktestResult(equity_curve, benchmark_curve, holdings, period_summary, stats, warnings)
