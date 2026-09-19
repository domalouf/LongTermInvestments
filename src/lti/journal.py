"""A decision journal: what you decided, why, and what happened next.

Greenblatt found that people running the Magic Formula themselves trailed the
automated version: they overrode picks they didn't like and quit after bad
years. Writing down each decision with its reason — and what would prove it
wrong — then checking it against what happened is the cheapest defence against
that, and the only way to learn whether your own judgement adds anything.

Entries are append-only JSON lines in ``data/track/journal.jsonl``; a later
look at a decision is a new ``review`` entry pointing at it, never an edit.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

import lti.config as config
from lti import prices as prices_mod

ACTIONS = ("buy", "add", "trim", "sell", "watch", "pass", "review")
# a decision to own more is right if the stock then beat SPY; one to own less
# (or none) is right if it then lagged; watching and reviewing aren't scored
_OWN_MORE = {"buy", "add"}
_OWN_LESS = {"trim", "sell", "pass"}
COLUMNS = ["id", "date", "ticker", "action", "thesis", "change_my_mind", "fair_value",
           "conviction", "review_by", "price", "refers_to", "logged_utc"]


def load_journal() -> pd.DataFrame:
    path = config.get_paths().journal_jsonl
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(entries).reindex(columns=COLUMNS)
    for col in ("date", "review_by"):
        df[col] = pd.to_datetime(df[col])
    return df


def add_entry(
    ticker: str,
    action: str,
    thesis: str,
    *,
    change_my_mind: str = "",
    fair_value: float | None = None,
    conviction: int | None = None,
    review_by=None,
    refers_to: str | None = None,
    date=None,
    px: prices_mod.PriceData | None = None,
) -> dict:
    """Append one decision. The price is the split-adjusted close on or before
    ``date`` (default: the latest in the cache), so a typo'd ticker fails here
    rather than turning up later as a decision nobody can score."""
    action = action.strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, not {action!r}")
    if not thesis.strip():
        raise ValueError("write down why — that's the point of the journal")
    if conviction is not None and not 1 <= int(conviction) <= 5:
        raise ValueError("conviction runs from 1 to 5")
    px = prices_mod.load_price_data() if px is None else px
    ticker = ticker.upper().strip()
    if ticker not in px.close.columns:
        raise ValueError(f"{ticker} isn't in the price cache — check the symbol, or `lti fetch-prices` it")

    date = pd.Timestamp(date).normalize() if date is not None else px.close.index.max()
    price = prices_mod.price_on_or_before(px.close, ticker, date)
    if price is None:
        raise ValueError(f"no {ticker} price on or just before {date.date()}")
    review_by = pd.Timestamp(review_by) if review_by is not None else date + pd.DateOffset(years=1)

    existing = load_journal()
    same_day = existing[(existing["date"] == date) & (existing["ticker"] == ticker)] if not existing.empty else existing
    entry = {
        "id": f"{date:%Y%m%d}-{ticker}-{len(same_day) + 1}",
        "date": str(date.date()),
        "ticker": ticker,
        "action": action,
        "thesis": thesis.strip(),
        "change_my_mind": change_my_mind.strip(),
        "fair_value": float(fair_value) if fair_value is not None else None,
        "conviction": int(conviction) if conviction is not None else None,
        "review_by": str(review_by.date()),
        "price": round(float(price), 4),
        "refers_to": refers_to,
        "logged_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    path = config.get_paths().journal_jsonl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def outcomes(journal: pd.DataFrame | None = None, px: prices_mod.PriceData | None = None) -> pd.DataFrame:
    """Each decision with what the stock and SPY have done since (total return),
    whether the decision looks right so far, and whether its review is due."""
    journal = load_journal() if journal is None else journal
    if journal.empty:
        return journal.assign(price_now=[], since=[], spy_since=[], vs_spy=[], right_so_far=[], review_due=[])
    px = prices_mod.load_price_data() if px is None else px
    latest = px.adj.index.max()
    out = journal.copy()
    since = [prices_mod.forward_return(px.adj, t, d, latest)[0] for t, d in zip(out["ticker"], out["date"])]
    out["since"] = pd.Series(since, index=out.index, dtype="float64")
    spy = [prices_mod.forward_return(px.adj, "SPY", d, latest)[0] if "SPY" in px.adj.columns else None for d in out["date"]]
    out["spy_since"] = pd.Series(spy, index=out.index, dtype="float64")
    out["price_now"] = [prices_mod.price_on_or_before(px.close, t, latest, window_days=10_000) for t in out["ticker"]]
    out["vs_spy"] = out["since"] - out["spy_since"]
    out["right_so_far"] = np.select(
        [out["action"].isin(_OWN_MORE), out["action"].isin(_OWN_LESS)],
        [out["vs_spy"] > 0, out["vs_spy"] < 0],
        default=np.nan,
    )
    out["right_so_far"] = out["right_so_far"].where(out["vs_spy"].notna() & out["action"].isin(_OWN_MORE | _OWN_LESS))
    reviewed = set(journal.loc[journal["action"] == "review", "refers_to"].dropna())
    out["review_due"] = (out["review_by"] <= latest) & ~out["id"].isin(reviewed) & (out["action"] != "review")
    return out
