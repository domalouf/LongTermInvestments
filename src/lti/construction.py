"""Does it matter *how* a signal is held? A pre-registered construction study.

The factor study (:mod:`lti.study`) left a puzzle. Its final screen — share
issuance, asset growth, shareholder yield, F-score — ranks stocks: an
out-of-sample rank IC of about +0.10. Yet the thirty names at the top of it
trailed the universe by 3.6% a year in the same period, because thirty names
ranked on those four things turn out to be one bet: shrinking retailers and
telecoms buying back stock. The signal is spread across fifteen hundred
companies; a concentrated screen collects one corner of it and a lot of
industry risk.

So the hypotheses here are about **construction**, not about new signals. The
score is the same one, on the same universe; each rule (:data:`RULES`) is a
different way of holding it, and each is expected to beat the equal-weighted
universe:

* hold everything, tilted toward the better-ranked (``screen_tilt``);
* leave out the worst fifth and hold the rest (``avoid_worst``);
* take the best fifth *of each industry*, so no industry is over- or
  under-weight (``sector_neutral``) — against the best fifth outright
  (``top_fifth``), the control that still carries the industry bets.

**What registration can and can't claim.** These rules were written down on
:data:`REGISTERED`, *after* the factor study's second half had been seen — they
exist because of what it showed. So their 2019– history is not out of sample
the way the factor study's was, and :func:`evaluate` reports it as background,
not as a test. The test is the forward track record (:mod:`lti.track`), which
records every rule here daily from the registration date on; nothing recorded
after that date could have shaped them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lti import factor, prices as prices_mod, sectors

REGISTERED = "2026-11-15"

# the factor study's final screen, as its first half chose it on 2026-09-19
SCREEN = ["share_growth", "asset_growth", "shareholder_yield", "f_score"]
FRACTION = 0.2  # "a fifth", wherever a rule cuts one


def screen_score(u: pd.DataFrame) -> pd.Series:
    """The screen's composite within ``u``: mean percentile of its parts, higher is better."""
    return factor.composite_score(u, SCREEN)


def _equal(index: pd.Index, members: pd.Series) -> pd.Series:
    """Equal weights over the rows of ``index`` where ``members`` is True; zero elsewhere."""
    w = members.reindex(index).fillna(False).astype("float64")
    return w / w.sum() if w.sum() > 0 else w


def tilt_weights(u: pd.DataFrame) -> pd.Series:
    """Every company, weighted by where it ranks: the best about twice the
    average weight, the worst about none. A company the screen can't score gets
    the average."""
    pct = screen_score(u).rank(pct=True).fillna(0.5)
    return pct / pct.sum()


def avoid_worst_weights(u: pd.DataFrame) -> pd.Series:
    """Equal weights on everything but the worst fifth by score. Unscored companies stay."""
    score = screen_score(u)
    return _equal(u.index, ~(score <= score.quantile(FRACTION)))


def top_fifth_weights(u: pd.DataFrame) -> pd.Series:
    """Equal weights on the best fifth by score, wherever they are."""
    score = screen_score(u)
    return _equal(u.index, score >= score.quantile(1.0 - FRACTION))


def sector_neutral_weights(u: pd.DataFrame) -> pd.Series:
    """Equal weights on the best fifth by score *within each industry*.

    Industries are Fama and French's twelve (:func:`lti.sectors.industry`); each
    contributes a fifth of its scored names, rounded up, so the portfolio holds
    every industry in about the proportion the universe does. Companies with no
    SIC code form a group of their own.
    """
    score = screen_score(u)
    industry = sectors.industry(u["sic"]).fillna("(none)") if "sic" in u.columns else pd.Series("(none)", index=u.index)
    chosen = pd.Series(False, index=u.index)
    for _, s in score.dropna().groupby(industry.reindex(score.dropna().index)):
        chosen[s.nlargest(math.ceil(len(s) * FRACTION)).index] = True
    return _equal(u.index, chosen)


@dataclass(frozen=True)
class Rule:
    name: str
    label: str
    why: str
    weights: Callable[[pd.DataFrame], pd.Series]


RULES: list[Rule] = [
    Rule("screen_tilt", "Screen, as a tilt (all)",
         "Every company in the universe, weighted by its rank on the final screen: the "
         "signal held as broadly as it is spread.", tilt_weights),
    Rule("avoid_worst", "Avoid the worst fifth",
         "The universe without the bottom fifth of the final screen: the signal used to "
         "leave things out rather than to pick.", avoid_worst_weights),
    Rule("sector_neutral", "Screen, best fifth of each industry",
         "The best fifth of every industry on the final screen, so the portfolio has the "
         "universe's industry mix and only the ranking differs.", sector_neutral_weights),
    Rule("top_fifth", "Screen, best fifth outright",
         "The best fifth on the final screen wherever they are — the control for the row "
         "above: the same breadth, with the industry bets left in.", top_fifth_weights),
]
BY_NAME = {r.name: r for r in RULES}


# --- the history, as background ---------------------------------------------------


@dataclass
class ConstructionResult:
    summary: pd.DataFrame  # one row per rule and period
    excess: pd.DataFrame  # index = as-of date, columns = rules: 12-month return less the universe's
    universe: pd.Series  # the universe's own 12-month return at each as-of date
    names: pd.DataFrame  # how many companies each rule held
    warnings: list[str]


def _portfolio_return(weights: pd.Series, fwd: pd.Series) -> float:
    """Weighted mean of ``fwd`` over the names that have a return."""
    w = weights.where(fwd.notna(), 0.0)
    return float((w * fwd.fillna(0.0)).sum() / w.sum()) if w.sum() > 0 else float("nan")


def evaluate(
    fund: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
    *,
    start: str = "2011-04-01",
    end: str | None = None,
    horizon_months: int = 12,
    step_months: int = 1,
    cap_min: float | None = None,
) -> ConstructionResult:
    """Each rule's 12-month return less the equal-weighted universe's, from monthly
    as-of dates — the factor study's protocol (10-Ks only, its free-cash-flow
    definition, gross of costs), on the track record's universe.

    The summary splits at the factor study's halves. Read the module docstring
    before reading the second one as a test.
    """
    from lti import pit, study, track

    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if px is None:
        px = prices_mod.load_price_data()
    fund = study.as_registered(fund)
    cap_min = track.UNIVERSE_CAP_MIN if cap_min is None else cap_min

    last = pd.Timestamp(end) if end else px.adj.index.max()
    grid = factor._asof_grid(pd.Timestamp(start), last, horizon_months, step_months)
    excess: dict[pd.Timestamp, dict] = {}
    names: dict[pd.Timestamp, dict] = {}
    universe: dict[pd.Timestamp, float] = {}
    warnings: list[str] = []
    for asof in grid:
        u = track.universe(pit.priced_snapshot(fund, asof, px), cap_min)
        u = u[u["ticker"].notna()].drop_duplicates("ticker")
        if len(u) < 50:
            warnings.append(f"{asof.date()}: only {len(u)} companies in the universe; skipped")
            continue
        fwd = prices_mod.forward_returns(px.adj, u["ticker"], asof, asof + pd.DateOffset(months=horizon_months))
        base = float(fwd.mean())
        universe[asof] = base
        excess[asof], names[asof] = {}, {}
        for rule in RULES:
            w = rule.weights(u)
            excess[asof][rule.name] = _portfolio_return(w, fwd) - base
            names[asof][rule.name] = int((w > 0).sum())

    cols = [r.name for r in RULES]
    excess_df = pd.DataFrame(excess).T.reindex(columns=cols).sort_index()
    names_df = pd.DataFrame(names).T.reindex(columns=cols).sort_index()
    lags = max(0, math.ceil(horizon_months / step_months) - 1)
    periods = {
        "2011-18 (chose the screen)": study.FIRST_HALF,
        "2019- (seen before registration)": study.SECOND_HALF,
        "whole": (start, None),
    }
    rows = []
    for label, (lo, hi) in periods.items():
        # like the factor study: as-of dates whose return window ends inside the period
        hi_asof = pd.Timestamp(hi) - pd.DateOffset(months=horizon_months) if hi else None
        part = excess_df.loc[lo:hi_asof] if hi_asof is not None else excess_df.loc[lo:]
        for name in cols:
            x = part[name].dropna()
            if x.empty:
                continue
            rows.append(
                {
                    "rule": name, "period": label, "months": len(x),
                    "excess": float(x.mean()), "t_nw": factor.newey_west_t(x, lags),
                    "beat_universe": float((x > 0).mean()),
                    "names": float(names_df.loc[x.index, name].mean()),
                }
            )
    summary = pd.DataFrame(rows, columns=["rule", "period", "months", "excess", "t_nw", "beat_universe", "names"])
    return ConstructionResult(summary, excess_df, pd.Series(universe).sort_index(), names_df, warnings)
