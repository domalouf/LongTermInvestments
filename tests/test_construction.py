"""The construction study's rules: the same score, held four ways."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import construction as c, track


@pytest.fixture
def u():
    """Forty companies in four industries; a higher index is better on every part of the screen."""
    n = 40
    return pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(n)],
            "sic": np.repeat([2000, 2834, 3674, 5311], n // 4),  # food, drugs, chips, shops
            "share_growth": np.linspace(0.20, -0.05, n),
            "asset_growth": np.linspace(0.30, -0.05, n),
            "shareholder_yield": np.linspace(0.0, 0.08, n),
            "f_score": np.linspace(2, 9, n),
        },
        index=pd.RangeIndex(1, n + 1, name="cik"),
    )


def test_the_rules_are_the_registered_ones_on_the_track_records_screen():
    assert c.SCREEN == track.FINAL_SCREEN
    assert [r.name for r in c.RULES] == ["screen_tilt", "avoid_worst", "sector_neutral", "top_fifth"]
    tracked = [s.name for s in track.STRATEGIES]
    assert all(r.name in tracked for r in c.RULES) and tracked[-1] == "universe"


def test_every_rule_is_fully_invested(u):
    for rule in c.RULES:
        w = rule.weights(u)
        assert w.index.equals(u.index) and w.sum() == pytest.approx(1.0) and (w >= 0).all(), rule.name


def test_the_tilt_holds_everything_and_leans_toward_the_best(u):
    w = c.tilt_weights(u)
    assert (w > 0).all() and w.is_monotonic_increasing
    assert w.iloc[-1] / w.mean() == pytest.approx(2.0, rel=0.05)
    unscored = u.assign(share_growth=np.nan, asset_growth=np.nan, shareholder_yield=np.nan, f_score=np.nan)
    assert c.tilt_weights(unscored).nunique() == 1  # nothing to rank on: the universe itself


def test_exclusion_drops_the_worst_fifth_and_the_top_fifth_is_its_mirror(u):
    kept = c.avoid_worst_weights(u)
    assert (kept.iloc[:8] == 0).all() and (kept.iloc[8:] > 0).all() and kept.iloc[8:].nunique() == 1
    top = c.top_fifth_weights(u)
    assert (top.iloc[-8:] > 0).all() and (top.iloc[:-8] == 0).all()
    # the best fifth outright sits entirely in the last industry ...
    assert set(u.loc[top > 0, "sic"]) == {5311}


def test_sector_neutral_takes_the_best_of_each_industry(u):
    w = c.sector_neutral_weights(u)
    held = u[w > 0]
    assert held.groupby("sic").size().to_dict() == {2000: 2, 2834: 2, 3674: 2, 5311: 2}
    for _, g in u.groupby("sic"):
        assert set(g.index[-2:]) <= set(held.index)  # each industry's own two best
    no_sic = c.sector_neutral_weights(u.drop(columns="sic"))
    assert (no_sic > 0).sum() == 8  # one group: the best fifth outright


def test_portfolio_return_skips_names_without_a_price():
    w = pd.Series([0.5, 0.3, 0.2])
    assert c._portfolio_return(w, pd.Series([0.10, np.nan, 0.20])) == pytest.approx((0.05 + 0.04) / 0.7)
    assert np.isnan(c._portfolio_return(w, pd.Series([np.nan] * 3)))
