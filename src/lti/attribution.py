"""Skill or style: how much of a strategy's return the known factors explain.

A value screen that beats its universe may simply own smaller, cheaper or more
profitable companies than the universe does — tilts anyone can buy in an index
fund, and that the academic factors already price. Regressing the strategy's
monthly returns on Fama and French's factors separates the two. The loadings
are the tilts; the intercept, alpha, is what's left over, the part of the return
the tilts don't explain.

``lti fetch-factors`` caches Ken French's monthly five factors — market, size
(SMB), value (HML), profitability (RMW), investment (CMA), with the T-bill rate —
and his momentum factor in ``data/prices/factors.parquet``, as fractions.

:func:`attribute` runs four regressions on one backtest:

* **Strategy** less the T-bill: its tilts, and its alpha against the factors.
* **Universe** less the T-bill: the tilts the screen inherits from what it picks
  among — $500M+ survivors, often without financials — before ranking anything.
* **Strategy − universe**: the ranking's edge, the gap the backtest page leads
  with, and what's left of it once the tilts it adds are accounted for. A
  long-short difference, so no T-bill comes off.
* **SPY** less the T-bill: a calibration. It should load about 1 on the market
  and next to nothing on the rest, with an alpha near zero.

Standard errors are Newey-West. Two caveats. The backtest can only hold
companies that still exist, while the factors are built from every stock CRSP
has, so the strategy's alpha against the factors carries the survivorship bias;
the strategy − universe regression largely nets it out. And a dozen years of
monthly returns is little evidence: an alpha with a t-stat under 2 is
indistinguishable from none.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

import lti.config as config

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRENCH_FILES = {"five": "F-F_Research_Data_5_Factors_2x3", "momentum": "F-F_Momentum_Factor"}
_COLUMNS = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma", "RF": "rf", "Mom": "mom"}

MODELS: dict[str, list[str]] = {
    "capm": ["mkt_rf"],
    "ff3": ["mkt_rf", "smb", "hml"],
    "ff5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
    "ff5_mom": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"],
}
MODEL_LABELS = {
    "capm": "Market only (CAPM)",
    "ff3": "Fama-French 3 (market, size, value)",
    "ff5": "Fama-French 5 (+ profitability, investment)",
    "ff5_mom": "Fama-French 5 + momentum",
}
TERM_LABELS = {
    "alpha": "Alpha (a year)",
    "mkt_rf": "Market",
    "smb": "Size: small over big",
    "hml": "Value: cheap over dear",
    "rmw": "Profitability: robust over weak",
    "cma": "Investment: conservative over aggressive",
    "mom": "Momentum: winners over losers",
}


# --- the factor data -------------------------------------------------------------


def parse_french_csv(text: str) -> pd.DataFrame:
    """The monthly block of a Ken French data-library CSV, as fractions by month end.

    The files open with a few lines of description, then a header row starting
    with a comma, then ``YYYYMM`` rows in percent, then a blank line and the same
    factors annually — which is where the monthly block stops. ``-99.99`` and
    ``-999`` mark missing values.
    """
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            if line.lstrip().startswith(",") and len(cells) > 1:
                header = [c for c in cells[1:] if c]
            continue
        if re.fullmatch(r"\d{6}", cells[0]) and len(cells) >= len(header) + 1:
            rows.append(cells[: len(header) + 1])
        elif rows:
            break
    if header is None or not rows:
        raise ValueError("not a Ken French factor file: no header row and monthly block found")
    index = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    values = pd.DataFrame(
        [[float(v) for v in r[1:]] for r in rows], index=index, columns=[_COLUMNS.get(h, h.lower()) for h in header]
    )
    return values.mask(values <= -99.99) / 100.0


def fetch_factors() -> pd.DataFrame:
    """Download the five factors and momentum, cache them, return the table.

    Without the five factors there's nothing to regress on, so that failing is an
    error; momentum failing only leaves it out.
    """
    import zipfile

    import requests

    frames = []
    for key, name in FRENCH_FILES.items():
        try:
            resp = requests.get(FRENCH_URL.format(name=name), headers={"User-Agent": config.user_agent()}, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                text = z.read(z.namelist()[0]).decode("latin-1")
            frames.append(parse_french_csv(text))
        except Exception as exc:  # noqa: BLE001 — say which file, and carry on without momentum
            if key == "five":
                raise RuntimeError(f"could not fetch the Fama-French five factors ({name}): {exc}") from exc
            LOGGER.warning("could not fetch the momentum factor (%s): %s — leaving it out", name, exc)
    table = pd.concat(frames, axis=1).sort_index()
    table.index.name = "month"
    path = config.get_paths().factors_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)
    return table


def load_factors() -> pd.DataFrame:
    """The cached factors, or an empty table when ``lti fetch-factors`` hasn't run."""
    path = config.get_paths().factors_parquet
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# --- the regression -----------------------------------------------------------------


def monthly_returns(curve: pd.Series) -> pd.Series:
    """Calendar-month returns of an equity curve, indexed by month end.

    A month the curve only partly covers — a backtest starting on the 15th, or
    ending on the first trading day of a month, as they do — is left out: the
    factors cover whole months.
    """
    curve = curve.dropna()
    if len(curve) < 2:
        return pd.Series(dtype="float64")
    last = curve.resample("ME").last()
    returns = last.pct_change()
    returns.iloc[0] = last.iloc[0] / curve.iloc[0] - 1
    # whole months only: starting near the first of the month, ending near its last
    first_day, last_day = curve.index[0], curve.index[-1]
    if first_day.day > 5:
        returns = returns.iloc[1:]
    if last_day < last_day + pd.offsets.MonthEnd(0) - pd.Timedelta(days=5):
        returns = returns.iloc[:-1]
    return returns


def regress(y: pd.Series, X: pd.DataFrame, lags: int | None = None) -> dict | None:
    """OLS of ``y`` on ``X`` with an intercept, and Newey-West t-stats.

    Returns ``coef`` and ``t`` (Series: ``alpha``, then ``X``'s columns), ``r2``
    and ``n``; ``None`` with fewer than a year of observations beyond the number
    of coefficients. ``lags`` defaults to the Newey-West rule, 4·(n/100)^(2/9).
    """
    df = pd.concat([y.rename("_y"), X], axis=1, join="inner").dropna()
    n, k = len(df), X.shape[1] + 1
    if n < k + 12:
        return None
    Xm = np.column_stack([np.ones(n), df[list(X.columns)].to_numpy(dtype="float64")])
    yv = df["_y"].to_numpy(dtype="float64")
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    e = yv - Xm @ beta
    lags = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
    u = Xm * e[:, None]
    S = u.T @ u
    for lag in range(1, min(lags, n - 1) + 1):
        G = u[lag:].T @ u[:-lag]
        S += (1 - lag / (lags + 1)) * (G + G.T)
    bread = np.linalg.pinv(Xm.T @ Xm)
    se = np.sqrt(np.clip(np.diag(bread @ S @ bread), 0.0, None))
    names = ["alpha", *X.columns]
    total = (yv - yv.mean()) @ (yv - yv.mean())
    return {
        "coef": pd.Series(beta, index=names),
        "t": pd.Series(np.divide(beta, se, out=np.full(k, np.nan), where=se > 0), index=names),
        "r2": float(1 - (e @ e) / total) if total > 0 else float("nan"),
        "n": n,
    }


@dataclass
class Attribution:
    model: str
    coef: pd.DataFrame  # terms × legs; alpha a year, the rest loadings
    t: pd.DataFrame  # the same, Newey-West t-stats
    r2: pd.Series  # by leg
    n: pd.Series  # months, by leg
    months: tuple[pd.Timestamp, pd.Timestamp] | None
    warnings: list[str] = field(default_factory=list)


LEGS = ("Strategy", "Universe", "Strategy − universe", "SPY")


def attribute(result, factors: pd.DataFrame, model: str = "ff5_mom") -> Attribution:
    """The four regressions in the module docstring, on a backtest's curves.

    ``result`` needs ``equity_curve``, ``universe_curve`` and ``benchmark_curve``
    (a :class:`lti.backtest.BacktestResult`). Its curves are used as reported, so
    after costs and taxes when the backtest charged them.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}, not {model!r}")
    if factors.empty or "rf" not in factors.columns:
        raise ValueError("no factors cached — run `lti fetch-factors`")
    warnings: list[str] = []
    cols = [c for c in MODELS[model] if c in factors.columns]
    if len(cols) < len(MODELS[model]):
        missing = sorted(set(MODELS[model]) - set(cols))
        warnings.append(f"{', '.join(missing)} not in the cached factors; the regression leaves it out")

    strat = monthly_returns(result.equity_curve)
    univ = monthly_returns(result.universe_curve)
    bench = monthly_returns(result.benchmark_curve)
    rf = factors["rf"]
    ys = {
        "Strategy": strat - rf.reindex(strat.index),
        "Universe": univ - rf.reindex(univ.index),
        "Strategy − universe": strat - univ,
        "SPY": bench - rf.reindex(bench.index),
    }
    X = factors[cols]
    if not strat.empty and not factors.empty and strat.index.max() > factors.index.max():
        warnings.append(
            f"the factors end in {factors.index.max():%B %Y}; later months are left out — "
            "Ken French updates them every month or so"
        )

    terms = ["alpha", *cols]
    coef = pd.DataFrame(np.nan, index=terms, columns=list(LEGS))
    tstat = coef.copy()
    r2, n = pd.Series(np.nan, index=list(LEGS)), pd.Series(0, index=list(LEGS))
    span = None
    for leg, y in ys.items():
        fit = regress(y, X)
        if fit is None:
            warnings.append(f"{leg}: too few months overlap the factors to regress")
            continue
        coef[leg], tstat[leg] = fit["coef"], fit["t"]
        coef.loc["alpha", leg] *= 12  # monthly intercept -> a year
        r2[leg], n[leg] = fit["r2"], fit["n"]
        used = y.dropna().index.intersection(X.dropna().index)
        span = (used.min(), used.max())
    return Attribution(model, coef, tstat, r2, n, span, warnings)


def as_text(a: Attribution) -> pd.DataFrame:
    """The regressions as a readable table: each coefficient with its t-stat, R², months."""

    def cell(term: str, leg: str) -> str:
        c, t = a.coef.at[term, leg], a.t.at[term, leg]
        if pd.isna(c):
            return "—"
        value = f"{c:+.1%}" if term == "alpha" else f"{c:+.2f}"
        return f"{value} (t {t:.1f})" if pd.notna(t) else value

    rows = {TERM_LABELS.get(term, term): {leg: cell(term, leg) for leg in a.coef.columns} for term in a.coef.index}
    rows["R²"] = {leg: f"{a.r2[leg]:.2f}" if pd.notna(a.r2[leg]) else "—" for leg in a.coef.columns}
    rows["Months"] = {leg: f"{int(a.n[leg])}" for leg in a.coef.columns}
    return pd.DataFrame(rows).T
