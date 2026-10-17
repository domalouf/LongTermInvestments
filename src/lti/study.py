"""A pre-registered test of the factors with the best published records.

Choosing a strategy and judging it on the same data flatters it: try enough
metrics and one always looks good. So everything here is fixed in advance, in
code, before any result:

* **The hypotheses** (:data:`FACTORS`). Each factor's expected direction — do
  higher values earn more, or less? — comes from the paper that documented it,
  and so do the composites (:data:`COMPOSITES`).
* **The universe.** Operating companies with a market cap of $500M+, financials
  and business development companies out: most of these factors are defined
  for industrial firms.
* **The test.** Monthly as-of dates, 12-month forward returns, rank IC, with
  Newey-West t-stats for the overlapping return windows.
* **The split.** The first half — as-of dates April 2011 to March 2018 —
  chooses; the second — April 2019 on, a year later so no return window
  straddles the two — judges. The one decision made from data: the final screen
  combines the single factors that pointed the expected way in the first half
  (t ≥ 1 that way). Its second-half result is the one out-of-sample number.

``profit_years`` isn't among the hypotheses: step 2 of this project already
looked at it across the whole sample, so it can't be tested out of sample here.

Three things the rest of the project changed after registration are held at
what they were, so the published results stay reproducible: the fundamentals
are 10-Ks only, not the fresher 10-Q rows; free cash flow is measured before
stock-based pay was netted out of it (both :func:`as_registered`); and the
backtests are gross of trading costs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lti import prices as prices_mod
from lti.backtest import BacktestConfig, rebalance_month_spread, run_backtest
from lti.factor import ICConfig, compute_ic
from lti.ranking import ScreenSpec


@dataclass(frozen=True)
class Hypothesis:
    metric: str
    sign: int  # +1: higher values should earn more; -1: lower values should
    source: str


# Composite scores: the mean percentile of the parts, each oriented so higher is
# better, ranked within the universe (lti.factor.composite_score).
COMPOSITES: dict[str, list[str]] = {
    "value_composite": [
        "ebit_ev", "ebitda_ev", "fcf_yield", "sales_ev", "earnings_yield", "book_to_market", "shareholder_yield",
    ],
    "quality_composite": ["op_profitability", "cash_profitability", "accruals", "f_score", "altman_z"],
    "quality_value": ["value_composite", "quality_composite"],
}

FACTORS: list[Hypothesis] = [
    Hypothesis("op_profitability", +1, "Novy-Marx 2013; Fama & French 2015 (RMW)"),
    Hypothesis("cash_profitability", +1, "Ball, Gerakos, Linnainmaa & Nikolaev 2016"),
    Hypothesis("accruals", -1, "Sloan 1996"),
    Hypothesis("share_growth", -1, "Pontiff & Woodgate 2008; Daniel & Titman 2006"),
    Hypothesis("asset_growth", -1, "Cooper, Gulen & Schill 2008"),
    Hypothesis("shareholder_yield", +1, "Boudoukh, Michaely, Richardson & Roberts 2007"),
    Hypothesis("momentum_12_1", +1, "Jegadeesh & Titman 1993"),
    Hypothesis("f_score", +1, "Piotroski 2000"),
    Hypothesis("altman_z", +1, "Campbell, Hilscher & Szilagyi 2008 (distress risk)"),
    Hypothesis("ebit_ev", +1, "Greenblatt 2005; Loughran & Wellman 2011"),
    Hypothesis("fcf_yield", +1, "Lakonishok, Shleifer & Vishny 1994"),
    Hypothesis("book_to_market", +1, "Fama & French 1992"),
    Hypothesis("earnings_yield", +1, "Basu 1977"),
    Hypothesis("value_composite", +1, "Asness, Moskowitz & Pedersen 2013"),
    Hypothesis("quality_composite", +1, "Asness, Frazzini & Pedersen 2019"),
    Hypothesis("quality_value", +1, "Novy-Marx 2013 (profitability + value)"),
]

FIRST_HALF = ("2011-04-01", "2019-03-01")  # as-of dates to 2018-03, returns ending by 2019-03
SECOND_HALF = ("2019-04-01", None)
SELECT_T = 1.0  # first-half t (in the expected direction) a factor needs to join the final screen
HELD_UP_T = 1.65  # second-half t that counts as holding up (one-sided 5%)


@dataclass
class StudyResult:
    table: pd.DataFrame  # one row per hypothesis
    selected: list[str]  # the single factors the first half picked for the final screen
    screen: pd.Series  # the final screen's second-half IC summary
    backtests: dict[str, dict] = field(default_factory=dict)  # half -> backtest stats
    month_spread: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def _signed(summary: pd.DataFrame, metric: str, sign: int, col: str) -> float:
    if metric not in summary.index or pd.isna(summary.at[metric, col]):
        return float("nan")
    return float(summary.at[metric, col]) * sign


def as_registered(fund: pd.DataFrame) -> pd.DataFrame:
    """``fund`` as the study was registered on: 10-Ks only (no trailing-twelve-month
    10-Q rows, :mod:`lti.quarterly`), and free cash flow as operating cash flow
    less capex, with stock-based pay still in it (see
    :func:`lti.fundamentals.add_free_cash_flow`)."""
    from lti import pit

    fund = pit.annual(fund)
    if "free_cash_flow_reported" not in fund.columns:
        return fund
    return fund.assign(free_cash_flow=fund["free_cash_flow_reported"])


def verdict(t_signed: float) -> str:
    if np.isnan(t_signed):
        return "no data"
    if t_signed >= HELD_UP_T:
        return "held up"
    return "right way, weak" if t_signed > 0 else "wrong way"


def run_study(
    fund: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
    *,
    market_cap_min: float = 500e6,
    horizon_months: int = 12,
    step_months: int = 1,
    backtest_cap_min: float = 1e9,
    top_n: int = 30,
    month_spread: bool = True,
) -> StudyResult:
    """Run the protocol in the module docstring."""
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if px is None:
        px = prices_mod.load_price_data()
    fund = as_registered(fund)

    base = ICConfig(
        metrics=[h.metric for h in FACTORS],
        start=FIRST_HALF[0],
        end=FIRST_HALF[1],
        horizon_months=horizon_months,
        step_months=step_months,
        market_cap_min=market_cap_min,
        exclude_financials=True,
        composites=COMPOSITES,
    )
    first = compute_ic(base, fund, px)

    selected = [
        h.metric for h in FACTORS
        if h.metric not in COMPOSITES and _signed(first.summary, h.metric, h.sign, "t_stat_nw") >= SELECT_T
    ]
    screen_parts = selected or [h.metric for h in FACTORS if h.metric not in COMPOSITES]
    second = compute_ic(
        dataclasses.replace(
            base, start=SECOND_HALF[0], end=SECOND_HALF[1],
            metrics=base.metrics + ["final_screen"],
            composites={**COMPOSITES, "final_screen": screen_parts},
        ),
        fund, px,
    )

    rows = []
    for h in FACTORS:
        t2 = _signed(second.summary, h.metric, h.sign, "t_stat_nw")
        rows.append(
            {
                "metric": h.metric,
                "expected": "higher" if h.sign > 0 else "lower",
                "first_ic": _signed(first.summary, h.metric, h.sign, "mean_ic"),
                "first_t": _signed(first.summary, h.metric, h.sign, "t_stat_nw"),
                "second_ic": _signed(second.summary, h.metric, h.sign, "mean_ic"),
                "second_t": t2,
                "verdict": verdict(t2),
                "in_screen": h.metric in selected,
                "source": h.source,
            }
        )
    table = pd.DataFrame(rows).set_index("metric")

    screen = second.summary.loc["final_screen"] if "final_screen" in second.summary.index else pd.Series(dtype=float)
    spec = ScreenSpec(metrics=screen_parts, top_n=top_n, filters={"exclude_financials": True}, min_coverage=0.5)
    backtests = {}
    for label, (start, end) in {"first half": ("2011-04-01", "2018-04-30"), "second half": ("2019-04-01", None)}.items():
        r = run_backtest(
            BacktestConfig(screen=spec, start=start, end=end, market_cap_min=backtest_cap_min, cost_bps=0.0), fund, px
        )
        backtests[label] = r.stats
    spread = (
        rebalance_month_spread(
            BacktestConfig(screen=spec, start=SECOND_HALF[0], market_cap_min=backtest_cap_min, cost_bps=0.0), fund, px
        )
        if month_spread
        else pd.DataFrame()
    )
    return StudyResult(table, selected, screen, backtests, spread, list(dict.fromkeys(first.warnings + second.warnings)))
