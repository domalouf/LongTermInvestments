"""Rank a point-in-time snapshot by one or more metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lti.metrics import LOWER_IS_BETTER


@dataclass
class ScreenSpec:
    metrics: list[str]
    ascending: list[bool] | None = None  # True = lower is better; defaults per-metric
    top_n: int = 10
    weights: list[float] | None = None
    filters: dict = field(default_factory=dict)

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
    if filters.get("exclude_financials") and "sic" in df.columns:
        df = df[~df["sic"].astype("string").str.startswith("6", na=False)]
    return df


def rank(snapshot: pd.DataFrame, spec: ScreenSpec) -> pd.DataFrame:
    df = _apply_filters(snapshot, spec.filters).copy()
    df = df.dropna(subset=[m for m in spec.metrics if m in df.columns])

    missing = [m for m in spec.metrics if m not in df.columns]
    if missing:
        raise KeyError(f"metrics not in snapshot: {missing}")
    if df.empty:
        df["composite_score"] = []
        df["rank"] = []
        return df

    weights = spec.metric_weights()
    composite = np.zeros(len(df))
    for metric, ascending, weight in zip(spec.metrics, spec.directions(), weights):
        pct = df[metric].rank(pct=True, ascending=ascending)  # lower pct = better
        composite += weight * pct.to_numpy()

    df["composite_score"] = composite
    tiebreak = df["ticker"] if "ticker" in df.columns else df.index.to_series().astype("string")
    df = df.assign(_tb=tiebreak.values).sort_values(["composite_score", "_tb"]).drop(columns="_tb")
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def select(snapshot: pd.DataFrame, spec: ScreenSpec) -> list[str]:
    ranked = rank(snapshot, spec)
    if ranked.empty or "ticker" not in ranked.columns:
        return []
    picks = ranked.loc[ranked["ticker"].notna(), "ticker"].head(spec.top_n)
    return list(dict.fromkeys(picks.tolist()))
