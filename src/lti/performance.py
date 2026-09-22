"""Portfolio performance statistics on a periodic (monthly) equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

_PPY = 12  # periods per year (monthly curves)


def _returns(curve: pd.Series) -> pd.Series:
    return curve.pct_change().dropna()


def cagr(curve: pd.Series) -> float:
    if len(curve) < 2 or curve.iloc[0] <= 0:
        return float("nan")
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1


def total_return(curve: pd.Series) -> float:
    if len(curve) < 2 or curve.iloc[0] <= 0:
        return float("nan")
    return curve.iloc[-1] / curve.iloc[0] - 1


def annual_vol(curve: pd.Series, periods_per_year: int = _PPY) -> float:
    r = _returns(curve)
    return float(r.std(ddof=0) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")


def sharpe(curve: pd.Series, rf_annual: float = 0.0, periods_per_year: int = _PPY) -> float:
    r = _returns(curve)
    if len(r) < 2:
        return float("nan")
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = r - rf_period
    denom = excess.std(ddof=0)
    if denom == 0:
        return float("nan")
    return float(excess.mean() / denom * np.sqrt(periods_per_year))


def max_drawdown(curve: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if curve.empty:
        return float("nan"), None, None
    running_max = curve.cummax()
    dd = curve / running_max - 1
    trough = dd.idxmin()
    peak = curve.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def hit_rate(holdings: pd.DataFrame) -> float:
    if holdings.empty or "period_return" not in holdings or "benchmark_period_return" not in holdings:
        return float("nan")
    valid = holdings.dropna(subset=["period_return", "benchmark_period_return"])
    if valid.empty:
        return float("nan")
    return float((valid["period_return"] > valid["benchmark_period_return"]).mean())


def turnover(holdings: pd.DataFrame) -> float:
    if holdings.empty or "rebalance_date" not in holdings:
        return float("nan")
    sets = [set(g["ticker"]) for _, g in holdings.groupby("rebalance_date")]
    if len(sets) < 2:
        return float("nan")
    changes = [len(a ^ b) / (2 * max(len(a), 1)) for a, b in zip(sets[:-1], sets[1:])]
    return float(np.mean(changes))


def summarize(
    port: pd.Series,
    bench: pd.Series,
    rf_annual: float = 0.0,
    holdings: pd.DataFrame | None = None,
    universe: pd.Series | None = None,
) -> dict:
    p_mdd, p_peak, p_trough = max_drawdown(port)
    b_mdd, _, _ = max_drawdown(bench)
    p_cagr, b_cagr = cagr(port), cagr(bench)
    stats = {
        "port_cagr": p_cagr,
        "bench_cagr": b_cagr,
        "excess_cagr": p_cagr - b_cagr,
        "port_total_return": total_return(port),
        "bench_total_return": total_return(bench),
        "port_vol": annual_vol(port),
        "bench_vol": annual_vol(bench),
        "port_sharpe": sharpe(port, rf_annual),
        "bench_sharpe": sharpe(bench, rf_annual),
        "port_max_drawdown": p_mdd,
        "bench_max_drawdown": b_mdd,
        "port_max_dd_peak": p_peak,
        "port_max_dd_trough": p_trough,
    }
    # emitted whenever a universe was measured at all, NaN where the curve is too
    # short to say anything: a caller reading stats["univ_cagr"] shouldn't have to
    # know whether this particular run got as far as trading
    if universe is not None:
        stats["univ_cagr"] = cagr(universe)
        stats["excess_cagr_vs_univ"] = stats["port_cagr"] - stats["univ_cagr"]
        stats["univ_total_return"] = total_return(universe)
        stats["univ_vol"] = annual_vol(universe)
        stats["univ_sharpe"] = sharpe(universe, rf_annual)
        stats["univ_max_drawdown"] = max_drawdown(universe)[0]
    if holdings is not None:
        stats["hit_rate"] = hit_rate(holdings)
        stats["avg_turnover"] = turnover(holdings)
    return stats
