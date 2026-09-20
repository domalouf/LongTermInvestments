"""Tests for the static-artifact renderer (no network / SEC data)."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

import lti  # noqa: F401  (configures secfsdstools)
from lti import report


@pytest.fixture
def ranked() -> pd.DataFrame:
    """A rank_undervalued()-shaped frame: cik index, the published columns."""
    return pd.DataFrame(
        {
            "rank": [1, 2],
            "ticker": ["AAA", "BBB"],
            "company": ["Alpha Inc", "Beta & Co <LLC>"],
            "price": [10.0, 20.0],
            "fair_value_est": [25.0, 22.0],
            "fair_value_est_upside": [1.5, 0.1],
            "pe_norm": [8.0, np.nan],
            "dividend_yield": [0.042, 0.0],
            "profit_years": [5.0, 4.0],
            "revenue_cagr": [0.06, -0.02],
            "fcf_conversion": [0.9, np.nan],
            "debt_to_equity": [0.5, 1.2],
        },
        index=pd.Index([101, 202], name="cik"),
    )


@pytest.fixture
def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=report.VIEW_COLUMNS)


def test_build_view_drops_missing_and_resets_index(ranked):
    view = report.build_view(ranked.drop(columns=["fcf_conversion"]))
    assert "fcf_conversion" not in view.columns
    assert list(view.columns) == [c for c in report.VIEW_COLUMNS if c != "fcf_conversion"]
    assert list(view.index) == [0, 1]


def test_dividend_yield_renders_as_a_percent_and_a_non_payer_as_a_dash(ranked):
    html = report.render_html(ranked, asof="2026-09-05", params={})
    assert "<th>Dividend yield</th>" in html
    assert '<td class="num">4.2%</td>' in html
    assert '<td class="num">0.0%</td>' not in html  # BBB pays nothing, so it gets a dash


def test_build_payload_shape_and_null_coercion(ranked):
    payload = report.build_payload(ranked, asof="2026-09-05", params={"top_n": 2})
    assert payload["asof"] == "2026-09-05"
    assert payload["params"] == {"top_n": 2}
    assert payload["count"] == 2
    assert payload["generated_utc"].endswith("+00:00")
    assert payload["rows"][0]["ticker"] == "AAA"
    # NaN must serialize as JSON null, not the string "NaN"
    assert payload["rows"][1]["pe_norm"] is None
    json.dumps(payload)  # must not raise


def test_build_payload_empty(empty):
    payload = report.build_payload(empty, asof="2026-09-05")
    assert payload["count"] == 0
    assert payload["rows"] == []


def test_render_csv_roundtrips(ranked):
    back = pd.read_csv(io.StringIO(report.render_csv(ranked)))
    assert len(back) == 2
    assert list(back["ticker"]) == ["AAA", "BBB"]
    assert pd.isna(back.loc[1, "pe_norm"])


def test_render_html_is_self_contained_and_escaped(ranked):
    doc = report.render_html(ranked, asof="2026-09-05", params={"market_cap_min": 2e9})
    assert "<script" not in doc.lower()
    assert "http://" not in doc and "https://" not in doc  # no external resources
    assert "2026-09-05" in doc
    assert "Alpha Inc" in doc
    assert "Beta &amp; Co &lt;LLC&gt;" in doc  # company name HTML-escaped
    assert "Beta & Co <LLC>" not in doc
    assert "+150%" in doc  # upside formatting
    assert "$25.00" in doc
    assert "Not investment advice." in doc
    assert "market cap &ge; $2,000M" in doc
    assert "+6%" in doc and "90%" in doc  # revenue growth, cash conversion


def test_render_html_empty_has_message(empty):
    doc = report.render_html(empty, asof="2026-09-05")
    assert "No names pass the filters for 2026-09-05" in doc
    assert "<table" not in doc


def test_write_artifacts_writes_all_three(tmp_path, ranked):
    written = report.write_artifacts(ranked, tmp_path / "invest", asof="2026-09-05", params={})
    names = {p.name for p in written}
    assert names == {"index.html", "undervalued.json", "undervalued.csv"}
    assert all(p.exists() for p in written)

    payload = json.loads((tmp_path / "invest" / "undervalued.json").read_text())
    assert payload["count"] == 2
    assert 'href="undervalued.csv"' in (tmp_path / "invest" / "index.html").read_text()


def test_write_artifacts_empty_still_publishes(tmp_path, empty):
    written = report.write_artifacts(empty, tmp_path / "invest", asof="2026-09-05")
    assert all(p.exists() for p in written)
    assert json.loads((written[1]).read_text())["count"] == 0


def test_write_artifacts_uses_shared_generated_timestamp(tmp_path, ranked):
    report.write_artifacts(ranked, tmp_path, asof="2026-09-05", generated="2026-09-05T12:00:00+00:00")
    payload = json.loads((tmp_path / "undervalued.json").read_text())
    assert payload["generated_utc"] == "2026-09-05T12:00:00+00:00"
    assert "2026-09-05 12:00:00 UTC" in (tmp_path / "index.html").read_text()
