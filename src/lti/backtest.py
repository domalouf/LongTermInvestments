"""Simple annual-rebalance backtest engine.

Each rebalance date: take a split-correct, priced point-in-time snapshot, rank
by the screen's metrics, buy the top-N equal-weighted, hold until the next
rebalance. Two benchmarks:

* **SPY** (``cfg.benchmark``), bought and held — what the money could have
  done instead.
* **The universe** — every stock the screen ranked, equal-weighted and
  rebalanced on the same dates: what picking at random from the same
  candidates would have returned. It carries the same survivorship bias as the
  picks (both can only hold companies that still exist), so the gap between
  them measures the ranking itself; the gap to SPY has the bias baked in.

All three are run twice: gross, and as a :class:`lti.frictions.Book` that pays
``cost_bps`` on every trade and, with ``tax`` set, the tax on dividends and on
the gains each sale realizes. The reported curves and statistics are the net
ones; the gross CAGRs sit beside them as ``*_cagr_gross``. The universe pays the
same frictions on its own, smaller, turnover, so the gap to it is what the
ranking adds after paying for the trading it takes.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lti import metrics, pit, prices as prices_mod, ranking
from lti.frictions import DEFAULT_COST_BPS, Book, TaxRates, Trade
from lti.performance import cagr, summarize
from lti.ranking import ScreenSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    screen: ScreenSpec
    start: str = "2011-01-01"
    end: str | None = None
    # April: by then nearly every calendar-year company has filed its 10-K
    # (large filers within 60 days of year end, the smallest within 90), so the
    # ranking runs on last year's numbers. In January it runs on the year before's.
    rebalance_month: int = 4
    benchmark: str = "SPY"
    rf_annual: float = 0.0
    initial_capital: float = 100_000.0
    market_cap_min: float = 500_000_000.0
    operating_only: bool = True
    # what trading and taxes take (lti.frictions): every trade pays cost_bps of
    # its value, one way; tax=None is a tax-free account (an IRA, a 401(k))
    cost_bps: float = DEFAULT_COST_BPS
    tax: TaxRates | None = None
    # rebalance no sooner than a year and a day after the last one, so that
    # every sale is a long-term gain — the calendar date lands on or just short
    # of the year in most years, which taxes each year's gains as short-term
    hold_past_one_year: bool = False
    # a buffer against turnover: keep a holding until it drops out of the top
    # sell_rank, rather than the moment it leaves the top N. None: no buffer
    sell_rank: int | None = None

    @property
    def has_frictions(self) -> bool:
        return self.cost_bps > 0 or self.tax is not None


@dataclass
class BacktestResult:
    equity_curve: pd.Series  # after costs and taxes, when the config has them
    benchmark_curve: pd.Series
    universe_curve: pd.Series
    holdings: pd.DataFrame
    period_summary: pd.DataFrame
    stats: dict
    warnings: list[str] = field(default_factory=list)
    equity_curve_gross: pd.Series | None = None  # the strategy before costs and taxes


def _rebalance_dates(
    trading_days: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, month: int, hold_past_one_year: bool = False
) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for year in range(start.year, end.year + 1):
        target = pd.Timestamp(year=year, month=month, day=1)
        if hold_past_one_year and dates:
            # a sale counts as long-term from the day after the purchase's anniversary
            target = max(target, dates[-1] + pd.DateOffset(years=1) + pd.Timedelta(days=1))
        pos = trading_days.searchsorted(target)
        if pos < len(trading_days):
            d = trading_days[pos]
            if start <= d <= end:
                dates.append(d)
    return sorted(set(dates))


def _monthly_grid(rd: pd.Timestamp, nrd: pd.Timestamp) -> pd.DatetimeIndex:
    months = pd.date_range(rd, nrd, freq="ME")
    return months.append(pd.DatetimeIndex([nrd])).unique()


def _growth_paths(panel: pd.DataFrame, tickers: list[str], rd: pd.Timestamp, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Growth of $1 bought at ``rd`` in each ticker, on ``grid``.

    Total return from the adjusted close; held at the last price after a
    delisting; flat at 1.0 (cash) for a ticker with no entry price.
    """
    names = list(dict.fromkeys(tickers))
    out = pd.DataFrame(1.0, index=grid, columns=names)
    cols = [t for t in names if t in panel.columns]
    if not cols:
        return out
    entry = prices_mod.prices_asof(panel, pd.Series(cols, index=cols), rd)
    window = panel.loc[rd - pd.Timedelta(days=10) : grid[-1], cols]
    on_grid = window.reindex(window.index.union(grid)).ffill().reindex(grid)
    out[cols] = on_grid.div(entry.where(entry > 0), axis=1).ffill().fillna(1.0)
    return out


def _price_growth(close: pd.DataFrame, tickers: list[str], rd: pd.Timestamp, nrd: pd.Timestamp) -> pd.Series:
    """Growth of $1 in each ticker's split-adjusted price alone, ``rd`` to ``nrd``:
    the total return less the dividends. A ticker without a split-adjusted close
    is left out, and :meth:`lti.frictions.Book.grow` books it no dividend."""
    cols = [t for t in dict.fromkeys(tickers) if t in close.columns]
    return 1.0 + prices_mod.forward_returns(close, cols, rd, nrd)


def _hold(
    book: Book, paths: pd.DataFrame, weights: pd.Series, price_growth: pd.Series, rd: pd.Timestamp
) -> tuple[pd.Series, Trade, float]:
    """Rebalance ``book`` into ``weights`` on ``rd`` and carry it along ``paths``
    (:func:`_growth_paths`) to the next rebalance.

    Returns its value on the grid — the last point after the tax on the period's
    dividends — the trade, and that dividend tax.
    """
    trade = book.rebalance(rd, weights)
    pos = book.positions()
    values = paths.reindex(columns=pos.index, fill_value=1.0).to_numpy() @ pos.to_numpy() + book.cash
    dividend_tax = book.grow(paths.iloc[-1], price_growth)
    values[-1] = book.value
    return pd.Series(values, index=paths.index), trade, dividend_tax


LEGS = ("port", "univ", "bench")  # the strategy, its universe, the benchmark


def run_backtest(
    cfg: BacktestConfig,
    fund: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
) -> BacktestResult:
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if px is None:
        px = prices_mod.load_price_data()

    warnings: list[str] = []
    bench = cfg.benchmark.upper()
    if bench not in px.adj.columns:
        raise RuntimeError(f"benchmark {bench} not in price cache — run `lti fetch-prices`")
    if px.close.empty:
        raise RuntimeError("no split-adjusted prices cached — run `lti fetch-prices` to backfill them")
    unpriced: set[str] = set()  # companies left out for want of a split-adjusted close

    trading_days = px.adj[bench].dropna().index
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end) if cfg.end else trading_days[-1]
    end = min(end, trading_days[-1])

    rebal_dates = _rebalance_dates(trading_days, start, end, cfg.rebalance_month, cfg.hold_past_one_year)
    if len(rebal_dates) < 2:
        raise RuntimeError("need at least two rebalance dates in the date range")
    if cfg.sell_rank is not None and cfg.sell_rank < cfg.screen.top_n:
        raise ValueError(f"sell_rank ({cfg.sell_rank}) must be at least top_n ({cfg.screen.top_n})")

    spec = dataclasses.replace(
        cfg.screen,
        filters={**cfg.screen.filters, "market_cap_min": cfg.market_cap_min, "require_price": True},
    )
    # several years of history only when a metric or filter needs it — it costs a pass per date
    with_history = metrics.needs_history([*spec.metrics, *(k for k, v in spec.filters.items() if v is not None)])

    port_value = bench_value = univ_value = cfg.initial_capital
    equity_segments = [pd.Series({rebal_dates[0]: port_value})]
    bench_segments = [pd.Series({rebal_dates[0]: bench_value})]
    univ_segments = [pd.Series({rebal_dates[0]: univ_value})]
    holdings_rows: list[dict] = []
    period_rows: list[dict] = []
    # the same three portfolios after costs and taxes
    books = {leg: Book(cfg.initial_capital, cfg.cost_bps, cfg.tax) for leg in LEGS}
    net_segments = {leg: [pd.Series({rebal_dates[0]: cfg.initial_capital})] for leg in LEGS}
    paid: dict[str, list[tuple[float, float]]] = {leg: [] for leg in LEGS}  # (costs, taxes) ÷ value, per period
    realized: list[tuple[float, float]] = []  # the strategy's net gains by term, $, per rebalance
    held: list[str] = []  # the last rebalance's picks, which a buffer may keep
    if cfg.tax is not None and bench not in px.close.columns:
        warnings.append(f"{bench} has no split-adjusted close, so its dividends go untaxed")

    for rd, nrd in zip(rebal_dates[:-1], rebal_dates[1:]):
        snap = pit.priced_snapshot(fund, rd, px, operating_only=cfg.operating_only, with_history=with_history)
        if snap.empty:
            warnings.append(f"{rd.date()}: no companies with fundamentals+ticker known")
            continue
        if "shares_outstanding" in snap.columns:
            lacking = snap["shares_outstanding"].notna() & ~snap["ticker"].isin(px.close.columns)
            unpriced |= set(snap.loc[lacking & snap["ticker"].isin(px.adj.columns), "ticker"])

        ranked = ranking.rank(snap, spec)
        if cfg.sell_rank is None:
            picks = ranking.top_picks(ranked, spec.top_n)
        else:
            picks = ranking.buffered_picks(ranked, spec.top_n, held, cfg.sell_rank)
        if not picks:
            warnings.append(f"{rd.date()}: screen produced no picks")
            continue
        if len(picks) < spec.top_n:
            warnings.append(f"{rd.date()}: only {len(picks)}/{spec.top_n} names qualified")
        universe = list(dict.fromkeys(ranked["ticker"].dropna().tolist()))

        grid = _monthly_grid(rd, nrd)
        paths = _growth_paths(px.adj, universe, rd, grid)
        port_path = port_value * paths[picks].mean(axis=1)
        univ_path = univ_value * paths.mean(axis=1)
        bench_paths = _growth_paths(px.adj, [bench], rd, grid)
        bench_path = bench_value * bench_paths[bench]

        port_ret = float(port_path.iloc[-1] / port_value - 1)
        univ_ret = float(univ_path.iloc[-1] / univ_value - 1)
        bench_ret, _ = prices_mod.forward_return(px.adj, bench, rd, nrd)

        weight = 1.0 / len(picks)
        pick_rows = ranked.drop_duplicates("ticker").set_index("ticker")
        detail = [c for c in ["company", "market_cap", *spec.metrics, "composite_score", "rank"] if c in pick_rows.columns]
        n_delisted = 0
        for t in picks:
            r, delisted = prices_mod.forward_return(px.adj, t, rd, nrd)
            if r is None:
                warnings.append(f"{rd.date()}: no forward price for {t}; treated as cash")
                r = 0.0
            n_delisted += int(bool(delisted))
            holdings_rows.append(
                {
                    "rebalance_date": rd,
                    "exit_date": nrd,
                    "ticker": t,
                    "weight": weight,
                    "entry_price": pick_rows.at[t, "price"],
                    **{c: pick_rows.at[t, c] for c in detail},
                    "period_return": r,
                    "benchmark_period_return": bench_ret,
                    "universe_period_return": univ_ret,
                    "delisted": bool(delisted),
                    "held_over": t in held,  # kept from the last rebalance
                }
            )

        equity_segments.append(port_path)
        bench_segments.append(bench_path)
        univ_segments.append(univ_path)
        port_value = float(port_path.iloc[-1])
        bench_value = float(bench_path.iloc[-1])
        univ_value = float(univ_path.iloc[-1])

        price_growth = _price_growth(px.close, [*universe, bench], rd, nrd)
        legs = {
            "port": (paths, pd.Series(1.0, index=picks)),
            "univ": (paths, pd.Series(1.0, index=universe)),
            "bench": (bench_paths, pd.Series(1.0, index=[bench])),
        }
        net_ret: dict[str, float] = {}
        trades: dict[str, Trade] = {}
        for leg, (leg_paths, weights) in legs.items():
            segment, trade, dividend_tax = _hold(books[leg], leg_paths, weights, price_growth, rd)
            net_segments[leg].append(segment)
            net_ret[leg] = float(segment.iloc[-1] / trade.value_before - 1)
            paid[leg].append((trade.cost / trade.value_before, (trade.tax + dividend_tax) / trade.value_before))
            trades[leg] = trade
        port_ret_gross = port_ret
        if cfg.has_frictions:
            port_ret, univ_ret, bench_ret = net_ret["port"], net_ret["univ"], net_ret["bench"]
        t = trades["port"]
        port_costs, port_taxes = paid["port"][-1]
        realized.append((t.short_term_gain, t.long_term_gain))
        n_held_over = len(set(picks) & set(held))
        held = picks

        period_rows.append(
            {
                "rebalance_date": rd,
                "exit_date": nrd,
                "n_selected": len(picks),
                "n_universe": len(universe),
                "n_delisted": n_delisted,
                "n_held_over": n_held_over,
                "port_return": port_ret,
                "univ_return": univ_ret,
                "bench_return": bench_ret,
                "excess_return": port_ret - (bench_ret or 0.0),
                "excess_vs_univ": port_ret - univ_ret,
                "port_return_gross": port_ret_gross,
                # what the strategy traded on this date and paid over the period, as
                # shares of what it was worth going in; gains are net of losses by term
                "turnover": t.turnover,
                "costs": port_costs,
                "taxes": port_taxes,
                "gains_short_term": t.short_term_gain / t.value_before,
                "gains_long_term": t.long_term_gain / t.value_before,
            }
        )

    def _curve(segments: list[pd.Series]) -> pd.Series:
        curve = pd.concat(segments).sort_index()
        return curve[~curve.index.duplicated(keep="last")]

    gross = {"port": _curve(equity_segments), "univ": _curve(univ_segments), "bench": _curve(bench_segments)}
    # without frictions the books only restate the gross curves, to rounding
    curves = {leg: _curve(net_segments[leg]) for leg in LEGS} if cfg.has_frictions else gross
    equity_curve, universe_curve, benchmark_curve = curves["port"], curves["univ"], curves["bench"]

    holdings = pd.DataFrame(holdings_rows)
    period_summary = pd.DataFrame(period_rows)
    stats = summarize(equity_curve, benchmark_curve, cfg.rf_annual, holdings, universe=universe_curve)
    if not period_summary.empty:
        stats["periods_beat_univ"] = float((period_summary["excess_vs_univ"] > 0).mean())
        stats["periods_beat_bench"] = float((period_summary["excess_return"] > 0).mean())
    stats.update(_friction_stats(cfg, gross, curves, books, paid, realized))

    if unpriced:
        warnings.insert(
            0,
            f"{len(unpriced)} companies were left out for want of a split-adjusted price "
            "— run `lti fetch-prices` to backfill them",
        )
    # de-duplicate warnings, keep order
    warnings = list(dict.fromkeys(warnings))
    return BacktestResult(
        equity_curve, benchmark_curve, universe_curve, holdings, period_summary, stats, warnings, gross["port"]
    )


def _friction_stats(
    cfg: BacktestConfig,
    gross: dict[str, pd.Series],
    curves: dict[str, pd.Series],
    books: dict[str, Book],
    paid: dict[str, list[tuple[float, float]]],
    realized: list[tuple[float, float]],
) -> dict:
    """What trading and taxes took from each leg — emitted for every run, frictionless or not.

    ``*_cagr_gross`` is the CAGR before them; ``*_costs_pa`` / ``*_taxes_pa`` the
    average paid per (roughly yearly) period, as a share of the value going in;
    ``*_cagr_liquidated`` the CAGR had everything been sold on the last day,
    paying the cost and the tax on every gain still unrealized — the fair
    comparison with buy-and-hold SPY, whose gains are otherwise never taxed.
    ``port_short_term_share`` is the share of the strategy's realized gains
    taxed as short-term.
    """
    out: dict = {}
    for leg in LEGS:
        curve = curves[leg]
        out[f"{leg}_cagr_gross"] = cagr(gross[leg])
        costs, taxes = zip(*paid[leg]) if paid[leg] else ((np.nan,), (np.nan,))
        out[f"{leg}_costs_pa"] = float(np.mean(costs))
        out[f"{leg}_taxes_pa"] = float(np.mean(taxes))
        if cfg.has_frictions and len(curve) > 1:
            sold = curve.copy()
            sold.iloc[-1] = books[leg].liquidation_value(curve.index[-1])
            out[f"{leg}_cagr_liquidated"] = cagr(sold)
        else:
            out[f"{leg}_cagr_liquidated"] = cagr(curve)
    short = sum(max(s, 0.0) for s, _ in realized)
    long = sum(max(lg, 0.0) for _, lg in realized)
    out["port_short_term_share"] = float(short / (short + long)) if short + long > 0 else np.nan
    return out


def rebalance_month_spread(
    cfg: BacktestConfig,
    fund: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
    months: range | list[int] = range(1, 13),
) -> pd.DataFrame:
    """The same strategy rebalanced in each month of the year; one row per month.

    With a dozen annual rebalances, *when* in the year the portfolio turns over
    moves the result a lot — the same screen can beat or trail its universe on
    the choice of month alone ("rebalance timing luck"). The spread across
    months is a floor on how much of any single run's edge could be noise.
    """
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if px is None:
        px = prices_mod.load_price_data()

    rows = []
    for m in months:
        try:
            r = run_backtest(dataclasses.replace(cfg, rebalance_month=m), fund, px)
        except RuntimeError as exc:
            LOGGER.warning("rebalance month %d: %s", m, exc)
            continue
        s = r.stats
        rows.append(
            {
                "rebalance_month": m,
                "first_rebalance": r.equity_curve.index[0],
                "periods": len(r.period_summary),
                "port_cagr": s["port_cagr"],
                "univ_cagr": s.get("univ_cagr", np.nan),
                "bench_cagr": s["bench_cagr"],
                "excess_vs_univ": s.get("excess_cagr_vs_univ", np.nan),
                "excess_vs_bench": s["excess_cagr"],
                "periods_beat_univ": s.get("periods_beat_univ", np.nan),
                "port_max_drawdown": s["port_max_drawdown"],
            }
        )
    return pd.DataFrame(rows)
