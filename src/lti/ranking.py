"""Rank a point-in-time snapshot by one or more metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lti import sectors
from lti.metrics import LOWER_IS_BETTER


@dataclass
class ScreenSpec:
    metrics: list[str]
    ascending: list[bool] | None = None  # True = lower is better; defaults per-metric
    top_n: int = 10
    weights: list[float] | None = None
    filters: dict = field(default_factory=dict)
    # share of the metrics a company must have to be ranked; its composite then
    # averages over the ones it has. 1.0 (the default) needs every one.
    min_coverage: float = 1.0

    def directions(self) -> list[bool]:
        if self.ascending is not None:
            return self.ascending
        return [m in LOWER_IS_BETTER for m in self.metrics]

    def metric_weights(self) -> np.ndarray:
        if self.weights is None:
            return np.ones(len(self.metrics)) / len(self.metrics)
        w = np.asarray(self.weights, dtype="float64")
        return w / w.sum()


def _apply_filters(snapshot: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df = snapshot
    if filters.get("market_cap_min") is not None and "market_cap" in df.columns:
        df = df[df["market_cap"] >= filters["market_cap_min"]]
    if filters.get("require_positive_eps") and "eps" in df.columns:
        df = df[df["eps"] > 0]
    if filters.get("require_price") and "price" in df.columns:
        df = df[df["price"].notna()]
    if filters.get("min_profit_years") is not None:
        if "profit_years" not in df.columns:
            raise KeyError(
                "cannot apply min_profit_years: the snapshot has no multi-year history "
                "— build it with pit.priced_snapshot(..., with_history=True)"
            )
        df = df[df["profit_years"] >= filters["min_profit_years"]]
    if filters.get("exclude_financials"):
        df = _drop_sector(df, "is_financial", sectors.is_financial)
        if "sic" in df.columns:  # BDCs and other investment companies carry no SIC
            df = df[~sectors.is_investment_company(df["sic"])]
    if filters.get("exclude_utilities"):
        df = _drop_sector(df, "is_utility", sectors.is_utility)
    return df


def _drop_sector(df: pd.DataFrame, flag_col: str, classify) -> pd.DataFrame:
    """Drop a sector, using the precomputed flag if present and ``sic`` otherwise.

    Raises when neither is available: silently returning an unfiltered universe
    would quietly answer a different question than the one that was asked.
    """
    if flag_col in df.columns:
        return df[~df[flag_col].fillna(False).astype(bool)]
    if "sic" in df.columns:
        return df[~classify(df["sic"])]
    raise KeyError(
        f"cannot apply the {flag_col} filter: the snapshot has neither "
        f"'{flag_col}' nor 'sic' — rebuild with `lti build-fundamentals`"
    )


def rank(snapshot: pd.DataFrame, spec: ScreenSpec) -> pd.DataFrame:
    df = _apply_filters(snapshot, spec.filters).copy()

    missing = [m for m in spec.metrics if m not in df.columns]
    if missing:
        raise KeyError(f"metrics not in snapshot: {missing}")
    need = max(1, math.ceil(spec.min_coverage * len(spec.metrics)))
    df = df[df[spec.metrics].notna().sum(axis=1) >= need]
    if df.empty:
        df["composite_score"] = []
        df["rank"] = []
        return df

    weights = spec.metric_weights()
    total = np.zeros(len(df))
    weight_seen = np.zeros(len(df))
    for metric, ascending, weight in zip(spec.metrics, spec.directions(), weights):
        pct = df[metric].rank(pct=True, ascending=ascending).to_numpy()  # lower pct = better
        have = ~np.isnan(pct)
        total += np.where(have, weight * pct, 0.0)
        weight_seen += np.where(have, weight, 0.0)

    df["composite_score"] = total / weight_seen
    tiebreak = df["ticker"] if "ticker" in df.columns else df.index.to_series().astype("string")
    df = df.assign(_tb=tiebreak.values).sort_values(["composite_score", "_tb"]).drop(columns="_tb")
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def top_picks(ranked: pd.DataFrame, top_n: int) -> list[str]:
    """The first ``top_n`` tickers of a :func:`rank` result, de-duplicated."""
    if ranked.empty or "ticker" not in ranked.columns:
        return []
    picks = ranked.loc[ranked["ticker"].notna(), "ticker"].head(top_n)
    return list(dict.fromkeys(picks.tolist()))


def buffered_picks(ranked: pd.DataFrame, top_n: int, held: list[str], sell_rank: int) -> list[str]:
    """:func:`top_picks` with a buffer against turnover, in rank order.

    A name already ``held`` stays while it still ranks in the top ``sell_rank``;
    only the slots its departures free up go to the best-ranked names not held.
    Rank noise near the cut-off — a holding slipping from 28th to 33rd — no
    longer forces a sale, and the costs and taxes that come with one. With
    ``sell_rank == top_n`` it picks exactly what :func:`top_picks` does.
    """
    if sell_rank < top_n:
        raise ValueError(f"sell_rank ({sell_rank}) can't be inside the top_n ({top_n}) the screen buys")
    order = top_picks(ranked, len(ranked))
    keep = set(held) & set(order[:sell_rank])
    fill = [t for t in order if t not in keep][: max(top_n - len(keep), 0)]
    chosen = keep | set(fill)
    return [t for t in order if t in chosen]


def select(snapshot: pd.DataFrame, spec: ScreenSpec) -> list[str]:
    return top_picks(rank(snapshot, spec), spec.top_n)
