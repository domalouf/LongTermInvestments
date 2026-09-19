"""Render the "undervalued today" list to shareable static artifacts.

The Streamlit page ``app/views/undervalued.py`` is the interactive view; this
module is its headless twin. ``lti undervalued --out DIR`` calls
:func:`write_artifacts` to drop a self-contained ``index.html`` plus ``.json`` /
``.csv`` siblings — the daily public snapshot served from domalouf.com.

Nothing here touches the network or SEC data: the input is a finished
:func:`lti.valuation.rank_undervalued` result.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path

import pandas as pd

# Columns lifted from a rank_undervalued() result into the published table, in
# display order. Mirrors cmd_undervalued's console view; missing ones are skipped.
VIEW_COLUMNS: list[str] = [
    "rank",
    "ticker",
    "company",
    "price",
    "fair_value_est",
    "fair_value_est_upside",
    "pe_norm",
    "profit_years",
    "revenue_cagr",
    "fcf_conversion",
    "debt_to_equity",
]

_COLUMN_LABELS: dict[str, str] = {
    "rank": "#",
    "ticker": "Ticker",
    "company": "Company",
    "price": "Price",
    "fair_value_est": "Fair value",
    "fair_value_est_upside": "Upside",
    "pe_norm": "P/E (5y)",
    "profit_years": "Profitable yrs",
    "revenue_cagr": "Revenue growth",
    "fcf_conversion": "Cash conversion",
    "debt_to_equity": "D/E",
}

# Right-aligned, monospaced numeric columns in the HTML table.
_NUMERIC_COLUMNS = {
    "rank",
    "price",
    "fair_value_est",
    "fair_value_est_upside",
    "pe_norm",
    "profit_years",
    "revenue_cagr",
    "fcf_conversion",
    "debt_to_equity",
}

_ARTIFACT_NAMES = ("index.html", "undervalued.json", "undervalued.csv")


# --- shared shaping -------------------------------------------------------


def build_view(ranked: pd.DataFrame) -> pd.DataFrame:
    """The published columns of ``ranked``, index reset, values left raw.

    Formatting happens at render time so the JSON payload stays numeric.
    """
    cols = [c for c in VIEW_COLUMNS if c in ranked.columns]
    return ranked.loc[:, cols].reset_index(drop=True).copy()


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def build_payload(
    ranked: pd.DataFrame,
    *,
    asof: str,
    params: dict | None = None,
    generated: str | None = None,
) -> dict:
    """A JSON-serializable dict: metadata + one record per row (NaN -> null)."""
    view = build_view(ranked)
    rows = json.loads(view.to_json(orient="records")) if not view.empty else []
    return {
        "generated_utc": generated or _now_utc_iso(),
        "asof": str(asof),
        "params": params or {},
        "count": len(rows),
        "columns": list(view.columns),
        "rows": rows,
    }


def render_csv(ranked: pd.DataFrame) -> str:
    return build_view(ranked).to_csv(index=False)


# --- HTML ---------------------------------------------------------------


def _fmt_cell(col: str, val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "&mdash;"
    if col in ("rank", "n_models", "profit_years"):
        return f"{int(val)}"
    if col in ("price", "fair_value_est"):
        return f"${val:,.2f}"
    if col in ("fair_value_est_upside", "revenue_cagr"):
        return f"{val:+.0%}"
    if col in ("roe", "fcf_conversion"):
        return f"{val:.0%}"
    if col in ("pe", "pe_norm", "debt_to_equity"):
        return f"{val:.1f}"
    return html.escape(str(val))


def _upside_bar_pct(val, ceiling: float) -> float:
    if val is None or ceiling <= 0 or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return max(0.0, min(100.0, 100.0 * float(val) / ceiling))


def _params_summary(params: dict) -> str:
    if not params:
        return ""
    bits: list[str] = []
    if params.get("market_cap_min"):
        bits.append(f"market cap &ge; ${params['market_cap_min'] / 1e6:,.0f}M")
    if params.get("min_profit_years"):
        bits.append(f"profitable in &ge; {params['min_profit_years']} of the last 5 years")
    if params.get("require_positive_eps"):
        bits.append("profitable now")
    if params.get("min_roe"):
        bits.append(f"ROE &ge; {params['min_roe']:.0%}")
    if params.get("discount_rate"):
        bits.append(f"discount rate {params['discount_rate']:.1%}")
    return " &middot; ".join(bits)


_CSS = """
:root {
  color-scheme: light dark;
  --bg: #fbfbfa; --panel: #ffffff; --ink: #1c1c1a; --muted: #6b6b66;
  --line: #e6e6e1; --accent: #1f6f5c; --bar: #cdeae0; --pos: #1f6f5c; --neg: #b23b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16171a; --panel: #1e1f23; --ink: #e9e9e6; --muted: #9a9a94;
    --line: #2c2e33; --accent: #66c9ac; --bar: #244b41; --pos: #66c9ac; --neg: #e08585;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 40px 20px 72px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin: 0 0 2px; }
.meta { color: var(--muted); font-size: 0.85rem; margin: 14px 0 22px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
thead th { position: sticky; top: 0; background: var(--panel); font-weight: 600;
  color: var(--muted); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: color-mix(in srgb, var(--accent) 6%, transparent); }
td.num { text-align: right; font-variant-numeric: tabular-nums;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
td.company { white-space: normal; min-width: 180px; }
td.ticker { font-weight: 600; }
td.upside { position: relative; }
td.upside .val { position: relative; z-index: 1; color: var(--pos); font-weight: 600; }
td.upside::before { content: ""; position: absolute; inset: 4px auto 4px 0;
  width: var(--w, 0%); background: var(--bar); border-radius: 3px; }
.foot { color: var(--muted); font-size: 0.82rem; margin-top: 26px; }
.foot ul { padding-left: 18px; margin: 8px 0; }
.foot a, a { color: var(--accent); }
.dl { display: inline-block; margin-top: 18px; font-size: 0.88rem; }
.empty { padding: 40px 16px; text-align: center; color: var(--muted); }
"""

_CAVEATS = """
<ul>
  <li><strong>Universe:</strong> companies whose latest 10-K was filed on or before
      the as-of date, one row each, with a ticker, positive revenue and above the
      market-cap floor; commodity and crypto trusts are left out. Each filing&rsquo;s
      per-share figures are restated for any stock split since it was filed.</li>
  <li><strong>Fair value</strong> is the median of the DCF, Lynch, Graham (&times;2),
      DDM and EPV models that produced a number, run on <em>normalized</em>
      earnings &mdash; the median of the last five years&rsquo; EPS and free cash
      flow, not the latest year alone &mdash; with growth from the five-year revenue
      trend. <em>Upside</em> is fair value &divide; price &minus; 1. Ranked by upside,
      descending; upsides above +500% are dropped as likely data errors.</li>
  <li><strong>Only steady earners:</strong> profitable now and in most of the last
      five years (a year counts only if EPS, net income and operating income were
      all positive); financials and business development companies are left out.
      <em>Cash conversion</em> is five years&rsquo; free cash flow over net income.</li>
  <li><strong>Candidates, not a buy list.</strong> Backtested, the widest gaps have
      trailed the average steady earner: usually the market knows why a stock is
      cheap. Watch for businesses in lasting decline &mdash; normalized earnings
      assume the last five years are a fair guide. The universe is also
      survivorship-biased.</li>
  <li>Not investment advice.</li>
</ul>
"""


def render_html(
    ranked: pd.DataFrame,
    *,
    asof: str,
    params: dict | None = None,
    generated: str | None = None,
) -> str:
    """A self-contained HTML page: no scripts, no external resources."""
    view = build_view(ranked)
    generated = generated or _now_utc_iso()
    params = params or {}

    asof_disp = html.escape(str(asof))
    gen_disp = html.escape(generated.replace("T", " ").replace("+00:00", " UTC"))
    summary = _params_summary(params)

    if view.empty:
        body = f'<div class="card"><div class="empty">No names pass the filters for {asof_disp}.</div></div>'
    else:
        ceiling = 0.0
        if "fair_value_est_upside" in view.columns:
            ups = view["fair_value_est_upside"].dropna()
            ceiling = float(ups.max()) if not ups.empty else 0.0

        head = "".join(
            f"<th>{html.escape(_COLUMN_LABELS.get(c, c))}</th>" for c in view.columns
        )
        body_rows: list[str] = []
        for rec in view.to_dict("records"):
            cells: list[str] = []
            for c in view.columns:
                val = rec[c]
                text = _fmt_cell(c, val)
                if c == "fair_value_est_upside":
                    w = _upside_bar_pct(val, ceiling)
                    cells.append(f'<td class="num upside" style="--w:{w:.1f}%"><span class="val">{text}</span></td>')
                elif c == "company":
                    cells.append(f'<td class="company">{text}</td>')
                elif c == "ticker":
                    cells.append(f'<td class="ticker">{text}</td>')
                elif c in _NUMERIC_COLUMNS:
                    cells.append(f'<td class="num">{text}</td>')
                else:
                    cells.append(f"<td>{text}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        body = (
            '<div class="card"><div class="scroll"><table>'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
            "</table></div></div>"
            '<a class="dl" href="undervalued.csv">Download CSV</a>'
        )

    meta_line = f"As of <strong>{asof_disp}</strong> &middot; generated {gen_disp}"
    if summary:
        meta_line += f"<br>{summary}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="index, follow">
<title>Undervalued today &mdash; {asof_disp}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>&#127919; Undervalued today</h1>
  <p class="sub">The widest gaps between blended intrinsic value and price across the US 10-K universe.</p>
  <p class="meta">{meta_line}</p>
  {body}
  <div class="foot">
    <strong>How this works</strong>
    {_CAVEATS}
  </div>
</div>
</body>
</html>
"""


# --- write all three ---------------------------------------------------


def write_artifacts(
    ranked: pd.DataFrame,
    out_dir: str | Path,
    *,
    asof: str,
    params: dict | None = None,
    generated: str | None = None,
) -> list[Path]:
    """Write ``index.html``, ``undervalued.json`` and ``undervalued.csv`` into
    ``out_dir`` (created if needed). Returns the written paths.

    An empty ``ranked`` still writes all three, so a stale snapshot never lingers.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated = generated or _now_utc_iso()

    html_path = out / "index.html"
    json_path = out / "undervalued.json"
    csv_path = out / "undervalued.csv"

    html_path.write_text(
        render_html(ranked, asof=asof, params=params, generated=generated),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(build_payload(ranked, asof=asof, params=params, generated=generated), indent=2),
        encoding="utf-8",
    )
    csv_path.write_text(render_csv(ranked), encoding="utf-8")

    return [html_path, json_path, csv_path]
