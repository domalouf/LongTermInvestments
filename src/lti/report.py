"""Render the "undervalued today" list to shareable static artifacts.

The Streamlit page ``app/views/undervalued.py`` is the interactive view; this
module is its headless twin. ``lti undervalued --out DIR`` calls
:func:`write_artifacts` to drop a self-contained ``index.html`` plus ``.json`` /
``.csv`` siblings — the daily public snapshot served from domalouf.com.

Nothing here touches the network or SEC data: the input is a finished
:func:`lti.valuation.rank_undervalued` result.

The page carries the six equations as well as the ranked names — rendered from
:data:`lti.valuation.MODEL_DOCS`, so the published explanation and the code that
does the arithmetic can't drift apart. Each row of the list opens in place to
show how that name's fair value was decided: a ``<details>`` whose summary is
the row itself, so the page stays script-free.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path

import pandas as pd

from lti.valuation import (
    MODEL_DOCS,
    MODELS,
    ValuationAssumptions,
    explain,
    explain_blend,
    explain_inputs,
)

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
    "dividend_yield",
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
    "dividend_yield": "Dividend yield",
    "profit_years": "Profitable yrs",
    "revenue_cagr": "Revenue growth",
    "fcf_conversion": "Cash conversion",
    "debt_to_equity": "D/E",
}

# Each column's track in the list's grid, shared by the header and every row so
# they line up like a table. The company takes what's left.
_COLUMN_TRACKS: dict[str, str] = {
    "rank": "3.2em",
    "ticker": "5.6em",
    "company": "minmax(12em, 1fr)",
    "price": "6.6em",
    "fair_value_est": "6.8em",
    "fair_value_est_upside": "6.8em",
    "pe_norm": "5.4em",
    "dividend_yield": "6em",
    "profit_years": "6.2em",
    "revenue_cagr": "6.4em",
    "fcf_conversion": "6.8em",
    "debt_to_equity": "4.6em",
}


def _tracks_width(tracks: list[str]) -> float:
    """The narrowest the grid can be, in em: every track at its minimum. The
    list is held at least this wide, so where the view is narrower the rows
    scroll sideways whole rather than overflowing their own backgrounds."""
    return sum(float(t.removeprefix("minmax(").split(",")[0].removesuffix("em")) for t in tracks)

# Not in the table but in the JSON and CSV: each model's value and the inputs it
# ran on, so a fair value can be checked against the numbers that produced it.
DETAIL_COLUMNS: list[str] = [
    *MODELS,
    "n_models",
    "eps_used",
    "bvps_used",
    "fcf_ps_used",
    "dps_used",
    "est_growth",
    "ddm_growth_used",
]

# Right-aligned, monospaced numeric columns in the HTML table.
_NUMERIC_COLUMNS = {
    "rank",
    "price",
    "fair_value_est",
    "fair_value_est_upside",
    "pe_norm",
    "dividend_yield",
    "profit_years",
    "revenue_cagr",
    "fcf_conversion",
    "debt_to_equity",
}

# --- shared shaping -------------------------------------------------------


def build_view(ranked: pd.DataFrame, *, detail: bool = False) -> pd.DataFrame:
    """The published columns of ``ranked``, index reset, values left raw —
    with :data:`DETAIL_COLUMNS` after them when ``detail``.

    Formatting happens at render time so the JSON payload stays numeric.
    """
    wanted = VIEW_COLUMNS + DETAIL_COLUMNS if detail else VIEW_COLUMNS
    cols = [c for c in wanted if c in ranked.columns]
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
    view = build_view(ranked, detail=True)
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
    return build_view(ranked, detail=True).to_csv(index=False)


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
    if col == "dividend_yield":
        return f"{val:.1%}" if val else "&mdash;"  # a non-payer yields 0, which is a dash here
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
        rate = f"discount rate {params['discount_rate']:.1%}"
        if params.get("treasury_10y") is not None:
            rate += f" (the 10-year Treasury&rsquo;s {params['treasury_10y']:.1%} plus a risk premium)"
        bits.append(rate)
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
/* Wide enough for every column on a 1280px laptop: at 1040px the table scrolled sideways there. */
.wrap { max-width: 1280px; margin: 0 auto; padding: 40px 20px 72px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin: 0 0 2px; }
.meta { color: var(--muted); font-size: 0.85rem; margin: 14px 0 14px; }
.notice { font-size: 0.85rem; margin: 0 0 22px; padding: 10px 14px; border-radius: 8px;
  border: 1px solid var(--line); border-left: 3px solid var(--accent); background: var(--panel); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
/* A container, so an opened row's breakdown can be exactly as wide as what's in view (100cqw)
   while the columns scroll sideways beneath it on a phone. */
.scroll { overflow-x: auto; container-type: inline-size; }
.hint { color: var(--muted); font-size: 0.85rem; margin: 0 0 10px; }
/* The list: a header and one <details> per name, each laid on the same grid tracks so the
   columns line up like a table's. The <summary> is the row, so the whole row opens it. */
.list { font-size: 0.9rem; }
.cells { display: grid; grid-template-columns: var(--tracks); align-items: center; }
.cells > span { padding: 10px 12px; white-space: nowrap; min-width: 0; }
.head { border-bottom: 1px solid var(--line); align-items: end; }
.head > span { font-weight: 600; color: var(--muted); text-transform: uppercase; font-size: 0.72rem;
  letter-spacing: 0.04em; white-space: normal; line-height: 1.3; }
.head > span.num { text-align: right; }
details.row { border-bottom: 1px solid var(--line); scroll-margin-top: 12px; }
details.row:last-child { border-bottom: 0; }
details.row > summary { list-style: none; cursor: pointer; }
details.row > summary::-webkit-details-marker { display: none; }
details.row > summary:hover { background: color-mix(in srgb, var(--accent) 6%, transparent); }
details.row > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
details.row[open] > summary { background: color-mix(in srgb, var(--accent) 10%, transparent); }
.cells > .num { text-align: right; font-variant-numeric: tabular-nums;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
/* the disclosure arrow rides in the rank cell, turning down when the row is open */
summary.cells > .rank { display: flex; justify-content: space-between; gap: 4px; }
summary.cells > .rank::before { content: "\\25B8"; color: var(--muted); font-family: -apple-system, BlinkMacSystemFont,
  "Segoe UI", Roboto, Helvetica, Arial, sans-serif; transition: transform 0.15s; }
details.row[open] .cells > .rank::before { transform: rotate(90deg); color: var(--accent); }
.cells > .company { white-space: normal; line-height: 1.35; }
.cells > .ticker { font-weight: 600; }
.cells > .upside { position: relative; align-self: stretch; display: flex; align-items: center;
  justify-content: flex-end; }
.cells > .upside .val { position: relative; z-index: 1; color: var(--pos); font-weight: 600; }
.cells > .upside::before { content: ""; position: absolute; inset: 6px auto 6px 0;
  width: var(--w, 0%); background: var(--bar); border-radius: 3px; }
/* An opened row's arithmetic. Sticky at the left, as wide as the view, so on a phone it stays
   put and readable however far the columns above it are scrolled. */
.why { position: sticky; left: 0; width: 100cqw; padding: 2px 16px 16px; }
details.models { margin-top: 26px; border: 1px solid var(--line); border-radius: 12px;
  background: var(--panel); padding: 0 16px; }
details.models > summary { cursor: pointer; padding: 14px 0; font-weight: 600; }
details.models[open] > summary { border-bottom: 1px solid var(--line); margin-bottom: 10px; }
details.models .lede { color: var(--muted); font-size: 0.85rem; margin: 0 0 14px; }
.model { padding: 12px 0; border-bottom: 1px solid var(--line); }
.model:last-child { border-bottom: 0; }
.model h3 { font-size: 0.95rem; margin: 0 0 6px; }
.model code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82rem; background: color-mix(in srgb, var(--accent) 8%, transparent);
  padding: 2px 6px; border-radius: 4px; display: inline-block; margin-bottom: 8px; }
.model p { margin: 0 0 8px; font-size: 0.87rem; }
.model dl { margin: 0; font-size: 0.83rem; color: var(--muted); }
.model dt { font-weight: 600; color: var(--ink); float: left; clear: left; margin-right: 6px; }
.model dd { margin: 0 0 4px; }
.blend { font-size: 0.88rem; margin: 12px 0; padding: 8px 12px; border-radius: 6px;
  background: color-mix(in srgb, var(--accent) 8%, transparent); }
.calcs { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }
.calc { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }
.calc h4 { margin: 0 0 6px; font-size: 0.88rem; }
.calc code { display: block; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.76rem; color: var(--muted); margin-bottom: 6px; overflow-wrap: anywhere; }
.calc .arith { margin: 0; font-size: 0.84rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.calc.silent { opacity: 0.55; }
table.vars { width: 100%; border-collapse: collapse; margin: 14px 0 0; font-size: 0.83rem; }
table.vars th { text-align: left; font-weight: 600; color: var(--muted); text-transform: uppercase;
  font-size: 0.7rem; letter-spacing: 0.04em; border-bottom: 1px solid var(--line); }
table.vars td { border-bottom: 1px solid var(--line); }
table.vars tr:last-child td { border-bottom: 0; }
table.vars th, table.vars td { padding: 6px 10px 6px 0; white-space: normal; }
table.vars td.sym { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-weight: 600; }
table.vars td.num { white-space: nowrap; }
table.vars td.src { color: var(--muted); }
.foot { color: var(--muted); font-size: 0.82rem; margin-top: 26px; }
.foot ul { padding-left: 18px; margin: 8px 0; }
.foot a, a { color: var(--accent); }
.dl { display: inline-block; margin-top: 18px; font-size: 0.88rem; }
.empty { padding: 40px 16px; text-align: center; color: var(--muted); }
/* On a phone the numbers still scroll sideways, but the ticker stays put beside them, and company names wrap
   to two short lines rather than three. */
@media (max-width: 700px) {
  .wrap { padding: 28px 14px 56px; }
  .cells > span { padding: 9px 9px; }
  .cells > .ticker { position: sticky; left: 0; z-index: 2; background: var(--panel);
    box-shadow: 1px 0 0 var(--line); align-self: stretch; display: flex; align-items: center; }
  .head > .ticker { align-items: flex-end; }
  details.row[open] .cells > .ticker { background: color-mix(in srgb, var(--accent) 10%, var(--panel)); }
  .cells > .company { font-size: 0.82rem; }
  .why { padding: 2px 12px 14px; }
  .calcs { grid-template-columns: 1fr; }
  table.vars thead { display: none; }
  table.vars td.src { font-size: 0.78rem; }
}
"""


def render_models_html() -> str:
    """The six equations, explained — rendered from :data:`lti.valuation.MODEL_DOCS`.

    A ``<details>`` block so the page still opens on the list; everything inside
    is static HTML, keeping the artifact script-free and self-contained.
    """
    blocks: list[str] = []
    for m in MODELS:
        d = MODEL_DOCS[m]
        blocks.append(
            f"""    <div class="model">
      <h3>{html.escape(d.label)}</h3>
      <code>{html.escape(d.formula)}</code>
      <p>{html.escape(d.idea)} It takes {html.escape(d.inputs)}</p>
      <dl>
        <dt>At the default assumptions:</dt><dd>{html.escape(d.at_defaults)}</dd>
        <dt>No value when:</dt><dd>{html.escape(d.silent)}</dd>
        <dt>Where it misleads:</dt><dd>{html.escape(d.misleads)}</dd>
      </dl>
    </div>"""
        )
    return f"""<details class="models">
  <summary>What each of the {len(MODELS)} equations does &mdash; and what it assumes</summary>
  <p class="lede">Every fair value above is the mean of these six, run on normalized earnings.
    They are not six independent opinions: Lynch, both Grahams and earnings power all multiply
    the same EPS, so they mostly restate that one number at different multiples. The DCF runs on
    cash flow and the dividend discount on cash actually paid out &mdash; those two are the ones
    that can disagree for a reason. The spread between them is as much the output as the average is.</p>
{chr(10).join(blocks)}
</details>"""


def _anchor(ticker) -> str:
    return "why-" + "".join(ch if ch.isalnum() else "-" for ch in str(ticker))


def render_breakdown_html(
    row: pd.Series,
    assumptions: ValuationAssumptions | None = None,
    *,
    treasury_10y: float | None = None,
) -> str:
    """How one name's fair value was decided — what its row opens to.

    The mean that made the fair value, each equation with the company's own
    numbers in place of the symbols (:func:`lti.valuation.explain`), and every
    variable those equations used with where it came from
    (:func:`lti.valuation.explain_inputs`). Script-free, like the rest.
    """
    a = assumptions or ValuationAssumptions()
    esc = html.escape
    models = []
    for m in MODELS:
        if m not in row.index:
            continue
        d = MODEL_DOCS[m]
        silent = pd.isna(row[m])
        models.append(
            f'<div class="calc{" silent" if silent else ""}"><h4>{esc(d.label)}</h4>'
            f"<code>{esc(d.formula)}</code>"
            f'<p class="arith">{esc(explain(m, row, a))}</p></div>'
        )
    variables = "".join(
        f'<tr><td class="sym">{esc(sym)}</td><td>{esc(name)}</td>'
        f'<td class="num">{esc(val)}</td><td class="src">{esc(src)}</td></tr>'
        for sym, name, val, src in explain_inputs(row, a, treasury_10y=treasury_10y)
    )
    return f"""<div class="why">
      <p class="blend"><strong>Fair value:</strong> {esc(explain_blend(row))}</p>
      <div class="calcs">{"".join(models)}</div>
      <table class="vars"><thead><tr><th>Symbol</th><th>Variable</th><th>Value</th><th>Where it comes from</th></tr></thead>
        <tbody>{variables}</tbody></table>
    </div>"""


# Above the table, where it's read: the footer's caveats are a long scroll
# past fifty rows, and a public page of "+180% upside" needs this up front.
_NOTICE = """<p class="notice"><strong>Not investment advice.</strong> This is a
screener's output, for research and learning: the numbers come straight from
SEC filings and simple valuation models, and can be wrong or out of date. Do
your own research before buying or selling anything.</p>"""


_CAVEATS = """
<ul>
  <li><strong>Universe:</strong> companies whose latest 10-K was filed on or before
      the as-of date, one row each, with a ticker, positive revenue and above the
      market-cap floor; commodity and crypto trusts are left out. Each filing&rsquo;s
      per-share figures are restated for any stock split since it was filed.</li>
  <li><strong>Fair value</strong> is the mean of the DCF, Lynch, Graham (&times;2),
      DDM and EPV models that produced a number (each equation is spelled out
      above), run on <em>normalized</em>
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
    assumptions: ValuationAssumptions | None = None,
) -> str:
    """A self-contained HTML page: no scripts, no external resources.

    Every row opens in place to its breakdown (:func:`render_breakdown_html`).
    ``assumptions`` are the ones ``ranked`` was valued at, for those
    breakdowns; without them, the defaults at ``params``' discount rate.
    """
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

        if assumptions is None:
            known = {k: params[k] for k in ("discount_rate", "growth_cap", "bond_yield") if params.get(k)}
            assumptions = ValuationAssumptions(**known)
        treasury_10y = params.get("treasury_10y")

        def cls(c: str) -> str:
            return " ".join(x for x in (c.replace("_", "-"), "num" if c in _NUMERIC_COLUMNS else "") if x)

        tracks = [_COLUMN_TRACKS.get(c, "6em") for c in view.columns]
        head = "".join(
            f'<span class="{cls(c)}">{html.escape(_COLUMN_LABELS.get(c, c))}</span>' for c in view.columns
        )
        rows: list[str] = []
        for i, rec in enumerate(view.to_dict("records")):
            cells: list[str] = []
            for c in view.columns:
                val = rec[c]
                text = _fmt_cell(c, val)
                if c == "fair_value_est_upside":
                    w = _upside_bar_pct(val, ceiling)
                    cells.append(f'<span class="num upside" style="--w:{w:.1f}%"><span class="val">{text}</span></span>')
                else:
                    cells.append(f'<span class="{cls(c)}">{text}</span>')
            why = render_breakdown_html(ranked.iloc[i], assumptions, treasury_10y=treasury_10y)
            anchor = f' id="{_anchor(rec["ticker"])}"' if "ticker" in rec else ""
            rows.append(
                f'<details class="row"{anchor}><summary class="cells">{"".join(cells)}</summary>\n'
                f"    {why}\n  </details>"
            )
        body = (
            f"<style>.list {{ --tracks: {' '.join(tracks)}; min-width: {_tracks_width(tracks):g}em; }}</style>\n"
            '  <p class="hint">Click any row to see how its fair value was decided: each equation with the '
            "company&rsquo;s own numbers, the average that became its fair value, and where every input came from.</p>\n"
            '  <div class="card"><div class="scroll"><div class="list">\n'
            f'  <div class="cells head" aria-hidden="true">{head}</div>\n  '
            + "\n  ".join(rows)
            + "\n  </div></div></div>\n"
            '  <a class="dl" href="undervalued.csv">Download CSV</a>'
        )

    meta_line = f"As of <strong>{asof_disp}</strong> &middot; generated {gen_disp}"
    if summary:
        meta_line += f"<br>{summary}"
    models = render_models_html()

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
  {_NOTICE}
  {body}
  {models}
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
    assumptions: ValuationAssumptions | None = None,
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
        render_html(ranked, asof=asof, params=params, generated=generated, assumptions=assumptions),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(build_payload(ranked, asof=asof, params=params, generated=generated), indent=2),
        encoding="utf-8",
    )
    csv_path.write_text(render_csv(ranked), encoding="utf-8")

    return [html_path, json_path, csv_path]
