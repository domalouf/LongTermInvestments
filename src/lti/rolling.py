"""Rolling-window backtests.

A single start/end backtest answers "how did this strategy do over one
particular stretch of history?" That's one draw from a small sample and easy
to curve-fit to. This module instead reruns ``backtest.run_backtest`` over
every overlapping N-year window spanning all available data (e.g. every 3-year
window and every 5-year window from the earliest cached price to the latest),
so a strategy's edge can be judged by how consistently it holds up rather than
by one lucky (or unlucky) period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lti.backtest import BacktestConfig, run_backtest
from lti.ranking import ScreenSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class RollingConfig:
    screen: ScreenSpec
    window_years: list[int] = field(default_factory=lambda: [3, 5])
    step_months: int = 12
    start: str | None = None  # default: earliest available price date
    end: str | None = None  # default: latest available price date
    rebalance_month: int = 1
    benchmark: str = "SPY"
    rf_annual: float = 0.0
    initial_capital: float = 100_000.0
    market_cap_min: float = 500_000_000.0


@dataclass
class RollingResult:
    windows: pd.DataFrame  # one row per (window length, window start)
    summary: pd.DataFrame  # one row per window length, aggregated across its windows
    warnings: list[str] = field(default_factory=list)


def _window_starts(first: pd.Timestamp, last: pd.Timestamp, years: int, step_months: int) -> list[pd.Timestamp]:
    starts = []
    cur = first
    while cur + pd.DateOffset(years=years) <= last:
        starts.append(cur)
        cur = cur + pd.DateOffset(months=step_months)
    return starts


def run_rolling_backtest(
    cfg: RollingConfig,
    fund: pd.DataFrame | None = None,
    price_panel: pd.DataFrame | None = None,
) -> RollingResult:
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if price_panel is None:
        from lti import prices as prices_mod

        price_panel = prices_mod.load_adj_close()

    bench = cfg.benchmark.upper()
    if bench not in price_panel.columns:
        raise RuntimeError(f"benchmark {bench} not in price cache — run `lti fetch-prices`")

    trading_days = price_panel[bench].dropna().index
    if trading_days.empty:
        raise RuntimeError(f"no price history for benchmark {bench}")

    first = pd.Timestamp(cfg.start) if cfg.start else trading_days[0]
    last = pd.Timestamp(cfg.end) if cfg.end else trading_days[-1]
    first = max(first, trading_days[0])
    last = min(last, trading_days[-1])

    warnings: list[str] = []
    rows: list[dict] = []

    for years in cfg.window_years:
        starts = _window_starts(first, last, years, cfg.step_months)
        if not starts:
            warnings.append(
                f"{years}y: available range ({first.date()}–{last.date()}) is shorter than the window; skipped"
            )
            continue
        for w_start in starts:
            w_end = w_start + pd.DateOffset(years=years)
            bt_cfg = BacktestConfig(
                screen=cfg.screen,
                start=w_start.strftime("%Y-%m-%d"),
                end=w_end.strftime("%Y-%m-%d"),
                rebalance_month=cfg.rebalance_month,
                benchmark=cfg.benchmark,
                rf_annual=cfg.rf_annual,
                initial_capital=cfg.initial_capital,
                market_cap_min=cfg.market_cap_min,
            )
            try:
                result = run_backtest(bt_cfg, fund=fund, price_panel=price_panel)
            except RuntimeError as exc:
                warnings.append(f"{years}y window {w_start.date()}–{w_end.date()}: {exc}")
                continue

            port_cagr = result.stats.get("port_cagr", np.nan)
            bench_cagr = result.stats.get("bench_cagr", np.nan)
            beat_benchmark = np.nan if (np.isnan(port_cagr) or np.isnan(bench_cagr)) else float(port_cagr > bench_cagr)

            rows.append(
                {
                    "window_years": years,
                    "start": w_start,
                    "end": w_end,
                    "port_cagr": port_cagr,
                    "bench_cagr": bench_cagr,
                    "excess_cagr": result.stats.get("excess_cagr"),
                    "port_max_drawdown": result.stats.get("port_max_drawdown"),
                    "port_sharpe": result.stats.get("port_sharpe"),
                    "hit_rate": result.stats.get("hit_rate"),
                    "beat_benchmark": beat_benchmark,
                    "n_warnings": len(result.warnings),
                }
            )

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise RuntimeError("no rolling windows could be run — check the date range vs the chosen window lengths")

    summary = (
        windows.groupby("window_years")
        .agg(
            n_windows=("port_cagr", "size"),
            mean_cagr=("port_cagr", "mean"),
            median_cagr=("port_cagr", "median"),
            worst_cagr=("port_cagr", "min"),
            best_cagr=("port_cagr", "max"),
            mean_excess_cagr=("excess_cagr", "mean"),
            win_rate=("beat_benchmark", "mean"),
            mean_max_drawdown=("port_max_drawdown", "mean"),
            worst_max_drawdown=("port_max_drawdown", "min"),
            mean_sharpe=("port_sharpe", "mean"),
        )
        .reset_index()
    )

    return RollingResult(windows, summary, list(dict.fromkeys(warnings)))
