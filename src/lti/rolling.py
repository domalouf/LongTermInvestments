"""Rolling-window backtests.

A single start/end backtest answers "how did this strategy do over one
particular stretch of history?" That's one draw from a small sample and easy
to curve-fit to. This module instead reruns ``backtest.run_backtest`` over
every overlapping N-year window spanning all available data (e.g. every 3-year
window and every 5-year window from the earliest cached price to the latest),
so a strategy's edge can be judged by how consistently it holds up rather than
by one lucky (or unlucky) period.

Each window is judged as the single backtest is: against the benchmark, and
against the equal-weighted universe the screen ranked. The universe carries the
same survivorship bias as the strategy, so beating it is the test that counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lti import prices as prices_mod
from lti.backtest import BacktestConfig, run_backtest
from lti.frictions import DEFAULT_COST_BPS, TaxRates
from lti.ranking import ScreenSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class RollingConfig:
    screen: ScreenSpec
    window_years: list[int] = field(default_factory=lambda: [3, 5])
    step_months: int = 12
    start: str | None = None  # default: earliest available price date
    end: str | None = None  # default: latest available price date
    rebalance_month: int = 4  # as BacktestConfig: once calendar-year 10-Ks are in
    benchmark: str = "SPY"
    rf_annual: float = 0.0
    initial_capital: float = 100_000.0
    market_cap_min: float = 500_000_000.0
    # as BacktestConfig: what trading and taxes take
    cost_bps: float = DEFAULT_COST_BPS
    tax: TaxRates | None = None
    hold_past_one_year: bool = False
    sell_rank: int | None = None  # as BacktestConfig: the buffer against turnover
    industry_cap: float | None = None  # and the most in any one industry


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


def _beat(a: float, b: float) -> float:
    return np.nan if (np.isnan(a) or np.isnan(b)) else float(a > b)


def run_rolling_backtest(
    cfg: RollingConfig,
    fund: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
) -> RollingResult:
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if px is None:
        px = prices_mod.load_price_data()

    bench = cfg.benchmark.upper()
    if bench not in px.adj.columns:
        raise RuntimeError(f"benchmark {bench} not in price cache — run `lti fetch-prices`")

    trading_days = px.adj[bench].dropna().index
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
                cost_bps=cfg.cost_bps,
                tax=cfg.tax,
                hold_past_one_year=cfg.hold_past_one_year,
                sell_rank=cfg.sell_rank,
                industry_cap=cfg.industry_cap,
            )
            try:
                result = run_backtest(bt_cfg, fund=fund, px=px)
            except RuntimeError as exc:
                warnings.append(f"{years}y window {w_start.date()}–{w_end.date()}: {exc}")
                continue

            stats = result.stats
            port_cagr = stats.get("port_cagr", np.nan)
            bench_cagr = stats.get("bench_cagr", np.nan)
            univ_cagr = stats.get("univ_cagr", np.nan)
            rows.append(
                {
                    "window_years": years,
                    "start": w_start,
                    "end": w_end,
                    "port_cagr": port_cagr,
                    "port_cagr_gross": stats.get("port_cagr_gross", np.nan),
                    "univ_cagr": univ_cagr,
                    "bench_cagr": bench_cagr,
                    "excess_cagr_vs_univ": port_cagr - univ_cagr,
                    "excess_cagr": port_cagr - bench_cagr,
                    "beat_universe": _beat(port_cagr, univ_cagr),
                    "beat_benchmark": _beat(port_cagr, bench_cagr),
                    "port_max_drawdown": stats.get("port_max_drawdown"),
                    "port_sharpe": stats.get("port_sharpe"),
                    "hit_rate": stats.get("hit_rate"),
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
            median_cagr=("port_cagr", "median"),
            worst_cagr=("port_cagr", "min"),
            best_cagr=("port_cagr", "max"),
            mean_excess_vs_univ=("excess_cagr_vs_univ", "mean"),
            win_rate_vs_univ=("beat_universe", "mean"),
            mean_excess_vs_bench=("excess_cagr", "mean"),
            win_rate_vs_bench=("beat_benchmark", "mean"),
            mean_max_drawdown=("port_max_drawdown", "mean"),
            worst_max_drawdown=("port_max_drawdown", "min"),
            mean_sharpe=("port_sharpe", "mean"),
        )
        .reset_index()
    )

    return RollingResult(windows, summary, list(dict.fromkeys(warnings)))
