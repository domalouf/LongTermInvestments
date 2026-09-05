"""Factor analysis — which metrics rank stocks by forward return.

For a grid of historical as-of dates we take a point-in-time snapshot (the same
no-look-ahead path the backtest uses), compute every metric and each stock's
forward return over a fixed horizon, then measure the **cross-sectional**
relationship between the metric and the forward return on that date — the
Information Coefficient (IC). Averaging the per-date ICs separates
stock-selection signal from market direction.

Sign convention: the IC is reported *signed*. A negative mean IC means lower
values of the metric went with higher forward returns (expected for ``pe``,
``pb``, ``debt_to_equity``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lti import metrics as metrics_mod, pit, prices as prices_mod
from lti.metrics import FUNDAMENTAL_METRICS, PRICE_METRICS

LOGGER = logging.getLogger(__name__)

ALL_METRICS: list[str] = FUNDAMENTAL_METRICS + PRICE_METRICS


@dataclass
class ICConfig:
    metrics: list[str] = field(default_factory=lambda: list(ALL_METRICS))
    start: str = "2011-01-01"
    end: str | None = None
    horizon_months: int = 12   # forward-return window
    step_months: int = 12      # spacing of as-of dates (== horizon ⇒ non-overlapping)
    market_cap_min: float = 500_000_000.0
    require_positive_eps: bool = False
    quantiles: int = 5
    method: str = "spearman"   # "spearman" (rank IC) | "pearson"
    winsorize: float = 0.01    # per-period tail clip on forward returns (and on the
                               # metric too, for a pearson IC); 0 disables
    min_names: int = 20        # skip a date with a thinner cross-section


@dataclass
class ICResult:
    summary: pd.DataFrame        # index = metric
    ic_by_period: pd.DataFrame   # index = as-of date, columns = metrics
    n_by_period: pd.DataFrame    # index = as-of date, columns = metrics (cross-section size)
    bucket_returns: pd.DataFrame  # index = metric, columns = Q1..Qn (mean forward return)
    warnings: list[str] = field(default_factory=list)


# --- helpers -------------------------------------------------------------


def _asof_grid(start: pd.Timestamp, end: pd.Timestamp, horizon_months: int, step_months: int) -> list[pd.Timestamp]:
    """As-of dates from ``start``, spaced ``step_months`` apart, leaving a full
    ``horizon_months`` of price history after the last one."""
    last_usable = end - pd.DateOffset(months=horizon_months)
    dates: list[pd.Timestamp] = []
    cur = start
    while cur <= last_usable:
        dates.append(cur)
        cur = cur + pd.DateOffset(months=step_months)
    return dates


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """Pearson, or Spearman as Pearson on ranks (no scipy dependency)."""
    if method == "spearman":
        x, y = x.rank(), y.rank()
    return float(x.corr(y))  # pandas' default pearson needs no scipy


def _winsorize(s: pd.Series, frac: float) -> pd.Series:
    lo, hi = s.quantile(frac), s.quantile(1.0 - frac)
    return s.clip(lo, hi)


def _bucket_means(metric: pd.Series, fwd: pd.Series, q: int) -> pd.Series | None:
    """Mean forward return per metric quantile (Q1 = lowest metric value)."""
    ranks = metric.rank(method="first")
    try:
        buckets = pd.qcut(ranks, q, labels=False)
    except ValueError:  # not enough distinct values to form q buckets
        return None
    out = fwd.groupby(buckets).mean().reindex(range(q))
    out.index = pd.RangeIndex(1, q + 1)
    return out


def _prepare_snapshot(fund: pd.DataFrame, panel: pd.DataFrame, asof: pd.Timestamp, cfg: ICConfig) -> pd.DataFrame:
    snap = pit.snapshot_asof(fund, asof)
    snap = snap[snap["ticker"].notna()]
    if snap.empty:
        return snap

    snap = metrics_mod.add_fundamental_metrics(snap)
    entry = pd.Series(
        {cik: prices_mod.price_on_or_before(panel, t, asof) for cik, t in snap["ticker"].items()}
    )
    shares = snap["shares_outstanding"] if "shares_outstanding" in snap.columns else None
    mcap = entry * shares if shares is not None else None
    snap = metrics_mod.add_price_metrics(snap, price=entry, market_cap=mcap)

    if cfg.market_cap_min and "market_cap" in snap.columns:
        snap = snap[snap["market_cap"] >= cfg.market_cap_min]
    if cfg.require_positive_eps and "eps" in snap.columns:
        snap = snap[snap["eps"] > 0]
    return snap


def _summarize(ic_by_period: pd.DataFrame, n_by_period: pd.DataFrame, bucket_returns: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for m in ic_by_period.columns:
        ic = ic_by_period[m].dropna()
        if ic.empty:
            continue
        mean_ic = float(ic.mean())
        std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
        n = int(len(ic))
        row = {
            "mean_ic": mean_ic,
            "ic_std": std_ic,
            "ic_ir": mean_ic / std_ic if std_ic else np.nan,
            "t_stat": mean_ic / (std_ic / np.sqrt(n)) if std_ic else np.nan,
            "hit_rate": float(np.mean(np.sign(ic) == np.sign(mean_ic))) if mean_ic else np.nan,
            "n_periods": n,
            "avg_n_stocks": float(n_by_period[m].dropna().mean()) if m in n_by_period else np.nan,
        }
        if m in bucket_returns.index:
            br = bucket_returns.loc[m].dropna()
            if len(br) >= 2:
                row["q_spread"] = float(br.iloc[-1] - br.iloc[0])
                row["monotonicity"] = _corr(
                    pd.Series(np.arange(len(br))), pd.Series(br.to_numpy()), "spearman"
                )
        rows[m] = row

    out = pd.DataFrame(rows).T
    if out.empty:
        return out
    out = out.sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)
    out["n_periods"] = out["n_periods"].astype(int)
    return out


# --- public API --------------------------------------------------------


def compute_ic(
    cfg: ICConfig,
    fund: pd.DataFrame | None = None,
    price_panel: pd.DataFrame | None = None,
) -> ICResult:
    """Cross-sectional IC of each metric in ``cfg.metrics`` vs forward return."""
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if price_panel is None:
        price_panel = prices_mod.load_adj_close()
    if price_panel.empty:
        raise RuntimeError("price cache is empty — run `lti fetch-prices`")

    warnings: list[str] = []
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end) if cfg.end else price_panel.index.max()
    grid = _asof_grid(start, end, cfg.horizon_months, cfg.step_months)
    if len(grid) < 2:
        raise RuntimeError("date range is too short for the chosen horizon / step")
    if cfg.step_months < cfg.horizon_months:
        warnings.append(
            f"as-of step ({cfg.step_months}m) is shorter than the return horizon "
            f"({cfg.horizon_months}m): forward-return windows overlap, so t-stats are optimistic."
        )

    metrics_wanted = [m for m in cfg.metrics if m in ALL_METRICS]
    unknown = [m for m in cfg.metrics if m not in ALL_METRICS]
    if unknown:
        warnings.append(f"ignored unknown metrics: {unknown}")
    if not metrics_wanted:
        raise RuntimeError("no known metrics requested")

    ic_rows: dict[pd.Timestamp, dict] = {}
    n_rows: dict[pd.Timestamp, dict] = {}
    bucket_acc: dict[str, list[pd.Series]] = {m: [] for m in metrics_wanted}

    for asof in grid:
        exit_date = asof + pd.DateOffset(months=cfg.horizon_months)
        snap = _prepare_snapshot(fund, price_panel, asof, cfg)
        if len(snap) < cfg.min_names:
            warnings.append(f"{asof.date()}: only {len(snap)} names after filters; skipped")
            continue

        fwd = pd.Series(
            {
                cik: prices_mod.forward_return(price_panel, t, asof, exit_date)[0]
                for cik, t in snap["ticker"].items()
            }
        )
        if cfg.winsorize > 0 and fwd.notna().sum() > 2:
            fwd = _winsorize(fwd, cfg.winsorize)
        snap = snap.assign(_fwd=fwd)

        ic_rows[asof], n_rows[asof] = {}, {}
        for m in metrics_wanted:
            if m not in snap.columns:
                continue
            pair = snap[[m, "_fwd"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < cfg.min_names:
                continue

            x, y = pair[m], pair["_fwd"]
            if cfg.method == "pearson" and cfg.winsorize > 0:
                x = _winsorize(x, cfg.winsorize)
            ic = _corr(x, y, cfg.method)

            ic_rows[asof][m] = ic
            n_rows[asof][m] = len(pair)

            buckets = _bucket_means(pair[m], pair["_fwd"], cfg.quantiles)
            if buckets is not None:
                bucket_acc[m].append(buckets)

    ic_by_period = pd.DataFrame(ic_rows).T.reindex(columns=metrics_wanted).sort_index()
    n_by_period = pd.DataFrame(n_rows).T.reindex(columns=metrics_wanted).sort_index()
    if ic_by_period.dropna(how="all").empty:
        raise RuntimeError("no usable periods — check the filters, date range and price coverage")

    bucket_returns = pd.DataFrame(
        {m: pd.concat(v, axis=1).mean(axis=1) for m, v in bucket_acc.items() if v}
    ).T
    if not bucket_returns.empty:
        bucket_returns.columns = [f"Q{c}" for c in bucket_returns.columns]

    summary = _summarize(ic_by_period, n_by_period, bucket_returns)
    return ICResult(summary, ic_by_period, n_by_period, bucket_returns, list(dict.fromkeys(warnings)))
