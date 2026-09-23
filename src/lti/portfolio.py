"""Your own portfolio: what you hold, what it cost, and whether picking beat indexing.

The decision journal scores each decision against SPY, but not how much money
rode on it; the backtests and the track record measure strategies, not the
account you actually run. This keeps that account, as an append-only ledger of
transactions (``data/track/portfolio.jsonl``) replayed against the price cache:

========  =========================  =============================================
action    fields                     meaning
========  =========================  =============================================
deposit   amount                     money into the account from outside
withdraw  amount                     money taken out
buy       ticker, shares, price      a purchase; ``fees`` on top of the price
sell      ticker, shares, price      a sale; ``fees`` come out of the proceeds
income    amount (ticker optional)   interest, or a dividend the price cache
                                     doesn't carry; negative for an account fee
========  =========================  =============================================

* **Dividends** on tickers in the price cache are credited on each ex-date for
  the shares held the day before. Log ``income`` only for what the cache lacks.
* **Splits.** Shares are entered as the broker showed them that day and put on
  today's basis — the one the price cache is on — through every split since
  (:func:`lti.pit.split_factor_after`), so a 4-for-1 split doesn't read as a
  75% loss.
* **New money.** A buy costing more than the cash on hand counts the shortfall
  as a deposit that day, so a ledger of trades alone still adds up. Same-day
  transactions settle deposits and income first, then sales, then buys, then
  withdrawals, whatever order they were logged in.
* **Cost basis** is the average cost, for display — not tax lots.

The question it exists for: **what would the same money, moved on the same
days, have made in SPY?** Every deposit buys SPY and every withdrawal sells it,
and the gap between that and the account is what your choices were worth, in
dollars. The money-weighted returns (XIRR) of the two say it per year. The same
comparison runs for each position (its buys, sales and dividends, in SPY
instead) and for the account's stocks against its funds: how much belongs in
picks, and how much in an index, is the decision the numbers here inform.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass, field

import lti.config as config

import numpy as np
import pandas as pd

from lti import pit, prices as prices_mod

ACTIONS = ("deposit", "withdraw", "buy", "sell", "income")
KINDS = ("stock", "fund")
COLUMNS = ["id", "date", "action", "ticker", "shares", "price", "amount", "fees", "kind", "note", "logged_utc"]
BENCHMARK = "SPY"
HOLDING_COLUMNS = [
    "ticker", "kind", "shares", "avg_cost", "price", "value", "weight", "cost_basis", "unrealized",
    "unrealized_pct", "realized", "income", "fees", "first_bought", "money_weighted", "spy_same_flows",
    "vs_spy", "priced",
]
# same-day settlement order: money in before it is spent, sales before the buys they fund
_ORDER = {"dividend": 0, "deposit": 1, "income": 2, "sell": 3, "buy": 4, "withdraw": 5}
_EPS = 1e-9


# --- the ledger ---------------------------------------------------------------


def load_ledger() -> pd.DataFrame:
    path = config.get_paths().portfolio_jsonl
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("shares", "price", "amount", "fees"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _held(ledger: pd.DataFrame, ticker: str, date: pd.Timestamp, splits: pd.DataFrame) -> float:
    """Shares of ``ticker`` held after ``date``'s trades, on today's basis."""
    trades = ledger[(ledger["ticker"] == ticker) & ledger["action"].isin(["buy", "sell"]) & (ledger["date"] <= date)]
    if trades.empty:
        return 0.0
    factor = pit.split_factor_after(trades["ticker"], trades["date"], splits)
    signed = np.where(trades["action"] == "buy", 1.0, -1.0) * trades["shares"].to_numpy() * factor.to_numpy()
    return float(signed.sum())


def add_transaction(
    action: str,
    *,
    ticker: str | None = None,
    shares: float | None = None,
    price: float | None = None,
    amount: float | None = None,
    fees: float = 0.0,
    kind: str | None = None,
    note: str = "",
    date=None,
    px: prices_mod.PriceData | None = None,
) -> dict:
    """Append one transaction to the ledger and return it.

    A buy or sale without a ``price`` takes the split-adjusted close that day,
    restated back onto that day's share basis. A sale can't exceed the shares
    held, and ``kind`` (stock or fund) overrides how a ticker is classified —
    for a foreign stock with no 10-K, say.
    """
    action = action.strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, not {action!r}")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if fees < 0:
        raise ValueError("fees can't be negative")
    date = pd.Timestamp(date).normalize() if date is not None else pd.Timestamp.today().normalize()
    ticker = ticker.upper().strip() if ticker else None
    ledger = load_ledger()

    if action in ("buy", "sell"):
        if not ticker:
            raise ValueError(f"a {action} needs a ticker")
        if not shares or shares <= 0:
            raise ValueError(f"a {action} needs a positive number of shares")
        px = prices_mod.load_price_data() if px is None else px
        if price is None:
            close = prices_mod.price_on_or_before(px.close, ticker, date)
            if close is None:
                raise ValueError(f"no {ticker} price on or just before {date.date()} in the cache — give --price")
            # the close is on today's share basis; the ledger keeps the day's own
            factor = pit.split_factor_after(pd.Series([ticker]), pd.Series([date]), px.splits).iloc[0]
            price = close * factor
        if price <= 0:
            raise ValueError("price must be positive")
        if action == "sell":
            factor = pit.split_factor_after(pd.Series([ticker]), pd.Series([date]), px.splits).iloc[0]
            have = _held(ledger, ticker, date, px.splits)
            if shares * factor > have + _EPS:
                raise ValueError(f"selling {shares:g} {ticker} on {date.date()}, but only {have / factor:g} are held")
        amount = None
    else:
        if amount is None or (action != "income" and amount <= 0) or amount == 0:
            raise ValueError(f"a {action} needs an amount" + ("" if action == "income" else " above zero"))
        shares = price = None

    same_day = ledger[ledger["date"] == date] if not ledger.empty else ledger
    entry = {
        "id": f"{date:%Y%m%d}-{len(same_day) + 1}",
        "date": str(date.date()),
        "action": action,
        "ticker": ticker,
        "shares": float(shares) if shares is not None else None,
        "price": round(float(price), 6) if price is not None else None,
        "amount": float(amount) if amount is not None else None,
        "fees": float(fees),
        "kind": kind,
        "note": note.strip(),
        "logged_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    path = config.get_paths().portfolio_jsonl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def ledger_tickers(ledger: pd.DataFrame | None = None) -> list[str]:
    """Every ticker the ledger has traded — which `lti fetch-prices` caches too."""
    ledger = load_ledger() if ledger is None else ledger
    return sorted(ledger["ticker"].dropna().astype(str).unique().tolist()) if not ledger.empty else []


def stock_tickers(fund: pd.DataFrame) -> set[str]:
    """Tickers of the operating companies that file 10-Ks: individual stocks.

    Anything else in the ledger — an index fund, a bond ETF, a commodity trust —
    counts as a fund unless a transaction says otherwise.
    """
    from lti import sectors

    rows = fund[fund["ticker"].notna()]
    if {"sic", "company"} <= set(rows.columns):
        rows = rows[~sectors.is_commodity_pool(rows["sic"], rows["company"])]
    out = set(rows["ticker"].astype(str).str.upper())
    if "tickers_all" in rows.columns:
        for many in rows["tickers_all"].dropna().astype(str):
            out.update(t.strip().upper() for t in many.split(",") if t.strip())
    return out


# --- returns --------------------------------------------------------------------


def xirr(dates, amounts) -> float:
    """The annual rate at which ``amounts`` on ``dates`` are worth nothing today.

    The money-weighted return: money in is negative, money out (and what's left
    at the end) positive. NaN when the flows never change sign.
    """
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    a = np.asarray(amounts, dtype="float64")
    if len(a) < 2 or not (a > 0).any() or not (a < 0).any():
        return float("nan")
    t = ((d - d.min()).dt.days / 365.25).to_numpy()

    def npv(r: float) -> float:
        return float(np.sum(a / (1.0 + r) ** t))

    lo, hi = -0.9999, 1.0
    f_lo = npv(lo)
    while npv(hi) * f_lo > 0 and hi < 1e6:
        hi *= 10
    if npv(hi) * f_lo > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) * f_lo > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _index_units(adj: pd.Series, flows: list[tuple[pd.Timestamp, float]]) -> float:
    """Units of an index bought (money in) and sold (money out) by ``flows`` —
    investor's sign: money in is negative — at its total-return price that day."""
    units = 0.0
    for day, amt in flows:
        before = adj[adj.index <= day]
        if before.empty:
            return float("nan")
        units -= amt / float(before.iloc[-1])
    return units


@dataclass
class Account:
    daily: pd.DataFrame  # by trading day: holdings, cash, value, net_deposits, spy
    holdings: pd.DataFrame  # one row per ticker ever held
    sleeves: pd.DataFrame  # stocks against funds
    flows: pd.DataFrame  # external flows: date, amount (in = positive), implicit
    summary: dict
    warnings: list[str] = field(default_factory=list)


def replay(
    ledger: pd.DataFrame,
    px: prices_mod.PriceData,
    *,
    stocks: set[str] | None = None,
    asof=None,
    benchmark: str = BENCHMARK,
) -> Account:
    """Run the ledger forward through the price cache; see the module docstring.

    ``stocks`` is the set of tickers that count as individual stocks
    (:func:`stock_tickers`); without it every holding is a fund unless a
    transaction's ``kind`` says otherwise.
    """
    if ledger.empty:
        raise ValueError("the ledger is empty — log a transaction first")
    warnings: list[str] = []
    led = ledger.copy()
    led["date"] = pd.to_datetime(led["date"]).dt.normalize()
    led["fees"] = led["fees"].fillna(0.0)
    led["seq"] = np.arange(len(led))  # logged order, for ties within a day
    start = led["date"].min()
    last_price = px.adj.index.max() if not px.adj.empty else start
    end = pd.Timestamp(asof).normalize() if asof is not None else max(last_price, led["date"].max())
    led = led[led["date"] <= end].reset_index(drop=True)
    trades = led["action"].isin(["buy", "sell"])
    led["factor"] = 1.0
    if trades.any():
        led.loc[trades, "factor"] = pit.split_factor_after(
            led.loc[trades, "ticker"], led.loc[trades, "date"], px.splits
        ).to_numpy()

    days = px.adj.index[(px.adj.index >= start) & (px.adj.index <= end)] if not px.adj.empty else None
    if days is None or len(days) == 0:
        days = pd.bdate_range(start, end)
    days = pd.DatetimeIndex(days).union(pd.DatetimeIndex([end])).unique()
    days = days[days >= start]

    tickers = ledger_tickers(led)
    divs = px.dividends
    divs = divs[divs["ticker"].isin(tickers) & (divs["date"] > start) & (divs["date"] <= end)] if not divs.empty else divs

    events: list[tuple] = [
        (r.date, _ORDER[r.action], r.seq, r.action, r) for r in led.itertuples(index=False)
    ] + [
        (pd.Timestamp(d.date), _ORDER["dividend"], i, "dividend", d) for i, d in enumerate(divs.itertuples(index=False))
    ]
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    cash = 0.0
    shares: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)
    realized: dict[str, float] = defaultdict(float)
    income: dict[str, float] = defaultdict(float)
    fees_paid: dict[str, float] = defaultdict(float)
    first_bought: dict[str, pd.Timestamp] = {}
    pos_flows: dict[str, list] = defaultdict(list)  # investor's sign: money into the position negative
    ext: list[tuple[pd.Timestamp, float, bool]] = []  # money into the account positive
    share_deltas: list[tuple[pd.Timestamp, str, float]] = []
    cash_deltas: list[tuple[pd.Timestamp, float]] = []

    for day, _, _, action, r in events:
        if action == "dividend":
            if shares[r.ticker] <= _EPS:
                continue
            amt = shares[r.ticker] * float(r.amount)  # both on today's share basis
            cash += amt
            income[r.ticker] += amt
            pos_flows[r.ticker].append((day, amt))
            cash_deltas.append((day, amt))
        elif action == "deposit":
            cash += r.amount
            ext.append((day, r.amount, False))
            cash_deltas.append((day, r.amount))
        elif action == "withdraw":
            if r.amount > cash + _EPS:
                warnings.append(f"{day.date()}: withdrew ${r.amount:,.2f} with ${cash:,.2f} in cash")
            cash -= r.amount
            ext.append((day, -r.amount, False))
            cash_deltas.append((day, -r.amount))
        elif action == "income":
            cash += r.amount
            if isinstance(r.ticker, str):
                income[r.ticker] += r.amount
                pos_flows[r.ticker].append((day, r.amount))
            cash_deltas.append((day, r.amount))
        elif action == "buy":
            outlay = r.shares * r.price + r.fees
            if outlay > cash + _EPS:
                new_money = outlay - max(cash, 0.0)
                ext.append((day, new_money, True))
                cash_deltas.append((day, new_money))
                cash += new_money
            cash -= outlay
            held = r.shares * r.factor
            shares[r.ticker] += held
            cost[r.ticker] += outlay
            fees_paid[r.ticker] += r.fees
            first_bought.setdefault(r.ticker, day)
            pos_flows[r.ticker].append((day, -outlay))
            share_deltas.append((day, r.ticker, held))
            cash_deltas.append((day, -outlay))
        elif action == "sell":
            sold = min(r.shares * r.factor, shares[r.ticker])
            if r.shares * r.factor > shares[r.ticker] + 1e-6:
                warnings.append(f"{day.date()}: sold more {r.ticker} than was held; capped at the holding")
            proceeds = (sold / r.factor) * r.price - r.fees
            basis = cost[r.ticker] * sold / shares[r.ticker] if shares[r.ticker] > _EPS else 0.0
            realized[r.ticker] += proceeds - basis
            cost[r.ticker] -= basis
            shares[r.ticker] -= sold
            fees_paid[r.ticker] += r.fees
            cash += proceeds
            pos_flows[r.ticker].append((day, proceeds))
            share_deltas.append((day, r.ticker, -sold))
            cash_deltas.append((day, proceeds))

    def daily_cumsum(deltas: pd.DataFrame) -> pd.DataFrame:
        if deltas.empty:
            return pd.DataFrame(0.0, index=days, columns=deltas.columns)
        at = deltas.groupby(level=0).sum().cumsum()
        return at.reindex(at.index.union(days)).ffill().reindex(days).fillna(0.0)

    held_daily = daily_cumsum(
        pd.DataFrame(share_deltas, columns=["date", "ticker", "d"]).pivot_table(
            index="date", columns="ticker", values="d", aggfunc="sum"
        ).fillna(0.0)
        if share_deltas else pd.DataFrame(columns=tickers)
    )
    cash_daily = daily_cumsum(pd.DataFrame(cash_deltas, columns=["date", "d"]).set_index("date")[["d"]])["d"]

    # a close for every held ticker on every day: the price cache, else the last
    # price it traded at in the ledger (on today's basis)
    held_cols = list(held_daily.columns)
    closes = px.close.reindex(columns=held_cols) if not px.close.empty else pd.DataFrame(columns=held_cols)
    closes = closes.reindex(closes.index.union(days)).ffill().reindex(days)
    traded = led[trades].assign(p=lambda x: x["price"] / x["factor"])
    if traded.empty:
        ledger_px = pd.DataFrame(np.nan, index=days, columns=held_cols)
    else:
        ledger_px = traded.pivot_table(index="date", columns="ticker", values="p", aggfunc="last")
        ledger_px = ledger_px.reindex(columns=held_cols)
        ledger_px = ledger_px.reindex(ledger_px.index.union(days)).ffill().reindex(days)
    unpriced = [t for t in held_cols if t not in px.close.columns]
    if unpriced:
        warnings.append(
            f"no prices cached for {', '.join(unpriced)} — valued at the last price traded; "
            "`lti fetch-prices` picks up every ticker in the ledger"
        )
    marks = closes.fillna(ledger_px)
    holdings_value = (held_daily * marks).sum(axis=1)

    flows = pd.DataFrame(ext, columns=["date", "amount", "implicit"])
    net_deposits = daily_cumsum(flows.set_index("date")[["amount"]])["amount"] if not flows.empty else pd.Series(0.0, index=days)
    daily = pd.DataFrame(
        {"holdings": holdings_value, "cash": cash_daily, "net_deposits": net_deposits}, index=days
    )
    daily["value"] = daily["holdings"] + daily["cash"]

    # the same money, on the same days, in the benchmark
    bench = px.adj[benchmark].dropna() if benchmark in px.adj.columns else pd.Series(dtype="float64")
    investor = [(d, -a) for d, a, _ in ext]
    if not bench.empty and not flows.empty:
        units = pd.Series(
            [-a / float(bench[bench.index <= d].iloc[-1]) if (bench.index <= d).any() else np.nan for d, a in investor],
            index=pd.DatetimeIndex([d for d, _ in investor]),
        )
        units_daily = daily_cumsum(units.to_frame("u"))["u"]
        daily["spy"] = units_daily * bench.reindex(bench.index.union(days)).ffill().reindex(days)
    else:
        daily["spy"] = np.nan
        warnings.append(f"{benchmark} isn't in the price cache, so there's nothing to compare against")

    final_marks = marks.iloc[-1] if len(marks) else pd.Series(dtype="float64")
    kinds = _kinds(led, stocks)
    rows = []
    for t in tickers:
        value = shares[t] * float(final_marks.get(t, np.nan)) if shares[t] > _EPS else 0.0
        value = 0.0 if np.isnan(value) else value
        f = pos_flows[t]
        spy_units = _index_units(bench, f) if not bench.empty else np.nan
        spy_value = spy_units * float(bench.iloc[-1]) if pd.notna(spy_units) else np.nan
        rows.append(
            {
                "ticker": t,
                "kind": kinds.get(t, "fund"),
                "shares": shares[t] if shares[t] > _EPS else 0.0,
                "avg_cost": cost[t] / shares[t] if shares[t] > _EPS else np.nan,
                "price": float(final_marks.get(t, np.nan)),
                "value": value,
                "cost_basis": cost[t] if shares[t] > _EPS else 0.0,
                "unrealized": value - cost[t] if shares[t] > _EPS else 0.0,
                "realized": realized[t],
                "income": income[t],
                "fees": fees_paid[t],
                "first_bought": first_bought.get(t),
                "money_weighted": xirr([d for d, _ in f] + [end], [a for _, a in f] + [value]),
                "spy_same_flows": spy_value,
                "vs_spy": value - spy_value if pd.notna(spy_value) else np.nan,
                "priced": t not in unpriced,
            }
        )
    holdings = pd.DataFrame(rows, columns=[c for c in HOLDING_COLUMNS if c not in ("weight", "unrealized_pct")])
    total = float(daily["value"].iloc[-1])
    holdings["weight"] = holdings["value"] / total if total else np.nan
    holdings["unrealized_pct"] = holdings["unrealized"] / holdings["cost_basis"].where(holdings["cost_basis"] > 0)
    holdings = holdings[HOLDING_COLUMNS]

    sleeves = _sleeves(holdings, pos_flows, bench, end, cash=float(cash_daily.iloc[-1]), total=total)
    summary = _summary(daily, flows, bench, end, total)
    return Account(daily, holdings, sleeves, flows, summary, list(dict.fromkeys(warnings)))


def _kinds(led: pd.DataFrame, stocks: set[str] | None) -> dict[str, str]:
    out = {t: ("stock" if stocks and t in stocks else "fund") for t in ledger_tickers(led)}
    explicit = led.dropna(subset=["ticker", "kind"]) if "kind" in led.columns else led.iloc[0:0]
    for r in explicit.itertuples(index=False):  # the latest word on a ticker wins
        out[r.ticker] = r.kind
    return out


def _sleeves(holdings, pos_flows, bench, end, *, cash: float, total: float) -> pd.DataFrame:
    """Stocks against funds: each sleeve's money-weighted return, and the same flows in SPY."""
    rows = []
    for kind in KINDS:
        members = holdings.loc[holdings["kind"] == kind, "ticker"].tolist() if not holdings.empty else []
        if not members:
            continue
        f = sorted((d, a) for t in members for d, a in pos_flows[t])
        value = float(holdings.loc[holdings["kind"] == kind, "value"].sum())
        spy_units = _index_units(bench, f) if not bench.empty else np.nan
        spy_value = spy_units * float(bench.iloc[-1]) if pd.notna(spy_units) else np.nan
        rows.append(
            {
                "sleeve": {"stock": "Stocks you picked", "fund": "Funds"}[kind],
                "value": value,
                "weight": value / total if total else np.nan,
                "money_in": -sum(a for _, a in f),
                "money_weighted": xirr([d for d, _ in f] + [end], [a for _, a in f] + [value]),
                "spy_same_flows": spy_value,
                "vs_spy": value - spy_value if pd.notna(spy_value) else np.nan,
            }
        )
    rows.append({"sleeve": "Cash", "value": cash, "weight": cash / total if total else np.nan})
    return pd.DataFrame(rows)


def _summary(daily: pd.DataFrame, flows: pd.DataFrame, bench: pd.Series, end: pd.Timestamp, total: float) -> dict:
    first = flows["date"].min() if not flows.empty else daily.index[0]
    years = (end - first).days / 365.25
    investor_dates = list(flows["date"]) + [end]
    investor = list(-flows["amount"])
    spy_now = float(daily["spy"].iloc[-1]) if daily["spy"].notna().any() else np.nan

    # time-weighted: each day's return with that day's deposits and withdrawals taken
    # out — a flow on a weekend lands on the next trading day, when the value shows it
    v = daily["value"]
    f = 0.0
    if not flows.empty:
        pos = np.minimum(daily.index.searchsorted(flows["date"]), len(daily.index) - 1)
        f = flows.groupby(daily.index[pos])["amount"].sum().reindex(daily.index).fillna(0.0)
    prev = v.shift(1)
    r = ((v - f) / prev - 1).where(prev > 0).dropna()
    twr = float((1 + r).prod() - 1) if len(r) else np.nan
    spy_ret = np.nan
    if not bench.empty:
        b = bench[(bench.index >= first) & (bench.index <= end)]
        spy_ret = float(b.iloc[-1] / b.iloc[0] - 1) if len(b) > 1 else np.nan

    def annual(x: float) -> float:
        return (1 + x) ** (1 / years) - 1 if years > 0 and pd.notna(x) and x > -1 else np.nan

    return {
        "asof": end,
        "first_flow": first,
        "years": years,
        "value": total,
        "cash": float(daily["cash"].iloc[-1]),
        "net_deposits": float(flows["amount"].sum()) if not flows.empty else 0.0,
        "implicit_deposits": float(flows.loc[flows["implicit"], "amount"].sum()) if not flows.empty else 0.0,
        "gain": total - (float(flows["amount"].sum()) if not flows.empty else 0.0),
        "spy_same_flows": spy_now,
        "vs_spy": total - spy_now if pd.notna(spy_now) else np.nan,
        "money_weighted": xirr(investor_dates, investor + [total]),
        "spy_money_weighted": xirr(investor_dates, investor + [spy_now]) if pd.notna(spy_now) else np.nan,
        "time_weighted": twr,
        "time_weighted_pa": annual(twr),
        "spy_return": spy_ret,
        "spy_return_pa": annual(spy_ret),
    }
