"""The six intrinsic-value equations, the blend, and the filters that rank on it.

The multiples asserted here are the ones the model docstrings quote, which
`lti explain-models` and the public page print: if an equation changes, either
the number or the explanation is now wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import valuation as v
from lti.valuation import ValuationAssumptions

A = ValuationAssumptions()


def s(*vals) -> pd.Series:
    return pd.Series(list(vals), dtype="float64")


# --- each model -------------------------------------------------------------------


def test_graham_number_is_the_geometric_mean_of_its_two_ceilings():
    # 10% on book: 15 x EPS and 1.5 x BVPS agree, and so does their geometric mean
    assert v.graham_number(s(2.0), s(20.0))[0] == pytest.approx(30.0)
    out = v.graham_number(s(2.0, -2.0, 2.0, np.nan), s(-20.0, 20.0, 0.0, 20.0))
    assert out.isna().all()  # a loss or negative book has no root, not a small one


def test_graham_intrinsic_matches_grahams_table_and_clips_growth():
    g = v.graham_intrinsic(s(1, 1, 1, 1, 1), s(0.0, 0.05, 0.15, 0.50, -0.10), 0.045)
    assert g.round(1).tolist() == [8.3, 18.1, 37.6, 47.4, 8.3]  # 50% clipped to 20 points, -10% to 0
    assert v.graham_intrinsic(s(1.0), s(np.nan), 0.045)[0] == pytest.approx(8.5 * 4.4 / 4.5)
    # bonds paying twice as much halve the value; a zero yield is floored, not divided by
    assert v.graham_intrinsic(s(1.0), s(0.05), 0.09)[0] == pytest.approx(g[1] / 2)
    assert np.isfinite(v.graham_intrinsic(s(1.0), s(0.05), 0.0)[0])
    assert v.graham_intrinsic(s(-1.0, 0.0), s(0.05, 0.05), 0.045).isna().all()


def test_lynch_is_eps_times_growth_plus_yield_in_points():
    assert v.lynch_fair_value(s(3.0), s(0.12), s(0.02), 0.25)[0] == pytest.approx(42.0)
    assert v.lynch_fair_value(s(3.0), s(0.12), None, 0.25)[0] == pytest.approx(36.0)
    assert v.lynch_fair_value(s(3.0), s(0.60), s(0.0), 0.25)[0] == pytest.approx(75.0)  # capped at 25
    assert v.lynch_fair_value(s(3.0), s(0.0), s(0.0), 0.25)[0] == 0.0  # the rule's own limit
    assert pd.isna(v.lynch_fair_value(s(-3.0), s(0.12), s(0.02), 0.25)[0])


def test_dcf_multiples_and_its_closed_form():
    out = v.two_stage_dcf(s(1.0, 1.0, 2.0, -1.0, 0.0), s(0.05, 0.0, 0.05, 0.05, 0.05), A)
    assert out[0] == pytest.approx(19.04, abs=0.01) and out[1] == pytest.approx(13.08, abs=0.01)
    assert out[2] == pytest.approx(2 * out[0])  # linear in the cash
    assert out[3:].isna().all()  # no cash to discount
    # flat cash: an annuity plus a discounted Gordon perpetuity
    r, gt, n = A.discount_rate, A.terminal_growth, A.dcf_years
    annuity = (1 - (1 + r) ** -n) / r
    assert out[1] == pytest.approx(annuity + (1 + gt) / (r - gt) / (1 + r) ** n)
    capped = v.two_stage_dcf(s(1.0, 1.0), s(A.growth_cap, 0.90), A)
    assert capped[0] == pytest.approx(capped[1])
    # a higher required return is a lower value
    assert v.two_stage_dcf(s(1.0), s(0.05), ValuationAssumptions(discount_rate=0.12))[0] < out[0]


def test_ddm_is_gordon_growth_capped_at_the_terminal_rate():
    out = v.ddm_value(s(1.0, 1.0, 1.0, 0.0, np.nan), s(0.10, A.terminal_growth, 0.0, 0.02, 0.02), A)
    assert out[0] == pytest.approx(15.77, abs=0.01) and out[0] == pytest.approx(out[1])
    assert out[2] == pytest.approx(1 / A.discount_rate)
    assert out[3:].isna().all()  # a non-payer abstains


def test_epv_capitalises_earnings_at_the_discount_rate():
    out = v.epv_value(s(1.0, 1.0, -1.0), 0.09)
    assert out[0] == pytest.approx(11.11, abs=0.01) and pd.isna(out[2])
    assert v.epv_value(s(1.0), 0.07)[0] == pytest.approx(14.29, abs=0.01)


def test_terminal_growth_is_kept_below_the_discount_rate():
    a = ValuationAssumptions(discount_rate=0.03)
    assert a.terminal_growth == pytest.approx(0.02)
    assert np.isfinite(v.two_stage_dcf(s(1.0), s(0.05), a)[0])


# --- the blend ----------------------------------------------------------------------


def frame(**over) -> pd.DataFrame:
    row = dict(eps=4.0, eps_norm=2.0, book_value_per_share=20.0, fcf_ps_norm=1.5, dps_ttm=0.5,
               revenue_cagr=0.05, shares_outstanding=100.0, free_cash_flow=300.0)
    row.update(over)
    return pd.DataFrame([row], index=pd.Index([1], name="cik"))


def test_the_blend_is_the_mean_of_the_models_that_spoke():
    out = v.add_valuation_models(frame(), pd.Series([25.0], index=[1])).iloc[0]
    assert out["n_models"] == 6
    assert out["fair_value_est"] == pytest.approx(np.mean([out[m] for m in v.MODELS]))
    assert out["fair_value_est_upside"] == pytest.approx(out["fair_value_est"] / 25.0 - 1)
    # each model saw the inputs recorded beside it
    assert (out["eps_used"], out["fcf_ps_used"], out["dps_used"], out["est_growth"]) == (2.0, 1.5, 0.5, 0.05)
    assert out["epv_value"] == pytest.approx(2.0 / A.discount_rate)
    assert out["dcf_value"] == pytest.approx(1.5 * v.two_stage_dcf(s(1.0), s(0.05), A)[0])
    assert out["lynch_fair_value"] == pytest.approx(2.0 * (5 + 100 * 0.5 / 25.0))


def test_a_model_without_its_input_abstains_from_the_blend():
    out = v.add_valuation_models(frame(dps_ttm=0.0, fcf_ps_norm=-1.0), pd.Series([25.0], index=[1])).iloc[0]
    assert pd.isna(out["ddm_value"]) and pd.isna(out["dcf_value"]) and out["n_models"] == 4
    assert out["fair_value_est"] == pytest.approx(
        np.mean([out[m] for m in ("lynch_fair_value", "graham_number", "graham_intrinsic", "epv_value")])
    )


def test_basis_picks_normalized_or_latest_earnings():
    px = pd.Series([25.0], index=[1])
    latest = v.add_valuation_models(frame(eps_growth_1y=0.08), px, basis="latest").iloc[0]
    assert latest["eps_used"] == 4.0 and latest["fcf_ps_used"] == 3.0 and latest["est_growth"] == 0.08
    with pytest.raises(KeyError):
        v.add_valuation_models(frame().drop(columns="eps_norm"), px)
    with pytest.raises(ValueError):
        v.add_valuation_models(frame(), px, basis="forward")


def test_growth_is_capped_and_a_shrinking_company_gets_none():
    px = pd.Series([25.0], index=[1])
    assert v.add_valuation_models(frame(revenue_cagr=0.40), px).iloc[0]["est_growth"] == A.growth_cap
    assert v.add_valuation_models(frame(revenue_cagr=-0.10), px).iloc[0]["est_growth"] == 0.0
    fixed = ValuationAssumptions(fixed_growth=0.03)
    assert v.add_valuation_models(frame(), px, assumptions=fixed).iloc[0]["est_growth"] == 0.03


def test_the_dividend_grows_at_its_own_rate_where_known():
    px = pd.Series([25.0], index=[1])
    flat = v.add_valuation_models(frame(dividend_growth_5y=0.0), px).iloc[0]
    assert flat["ddm_growth_used"] == 0.0 and flat["ddm_value"] == pytest.approx(0.5 / A.discount_rate)
    unknown = v.add_valuation_models(frame(dividend_growth_5y=np.nan), px).iloc[0]
    assert unknown["ddm_growth_used"] == A.terminal_growth  # revenue's 5%, capped


def test_rankable_upside_needs_enough_models_and_drops_data_errors():
    snap = pd.concat([frame(), frame(), frame(eps_norm=-1.0)])
    snap.index = pd.Index([1, 2, 3], name="cik")
    snap["price"] = [25.0, 0.5, 25.0]  # cik 2: a price off by 50x
    out = v.add_fair_value_metrics(snap)
    assert pd.notna(out.loc[1, "fair_value_upside"]) and pd.notna(out.loc[1, "fair_value_upside_1y"])
    assert pd.isna(out.loc[2, "fair_value_upside"])  # beyond MAX_UPSIDE
    assert pd.isna(out.loc[3, "fair_value_upside"])  # only the DCF and DDM spoke: two is not a blend
    assert v.MIN_MODELS == 3 and v.MAX_UPSIDE == 5.0


def test_historical_cagr():
    annual = pd.DataFrame({"period_end": pd.date_range("2019-12-31", periods=6, freq="YE"),
                           "eps": [1.0, 1.1, 1.3, 1.2, 1.5, 2.0]})
    assert v.historical_cagr(annual) == pytest.approx(2.0 ** (1 / 5) - 1)
    assert v.historical_cagr(annual, years=2) == pytest.approx((2.0 / 1.2) ** 0.5 - 1)
    assert np.isnan(v.historical_cagr(annual.assign(eps=[-1.0, 1, 1, 1, 1, 2])))
    assert np.isnan(v.historical_cagr(annual.head(1))) and np.isnan(v.historical_cagr(annual, "fcf"))
