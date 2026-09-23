"""Command-line entry point: ``python -m lti.cli <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import lti.config as config


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s  %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _print_warnings(warnings: list[str], limit: int = 30) -> None:
    if warnings:
        print(f"\n=== warnings ({len(warnings)}) ===")
        for w in warnings[:limit]:
            print("  ", w)


def cmd_update(args: argparse.Namespace) -> None:
    from lti import sec_update

    sec_update.run_full_update(force=args.force)
    print("latest quarter:", sec_update.latest_quarter())


def cmd_pipeline(args: argparse.Namespace) -> None:
    from lti import sec_update

    sec_update.run_pipeline_only()


def cmd_build_fundamentals(args: argparse.Namespace) -> None:
    from lti import fundamentals

    quarters = args.quarters.split(",") if args.quarters else None
    if quarters:
        quarters = [q if q.endswith(".zip") else f"{q}.zip" for q in quarters]
    path = fundamentals.build_fundamentals(smoke=args.smoke, quarters=quarters)
    print("wrote", path)
    fundamentals.coverage_report(fundamentals.load_fundamentals())


def cmd_refresh_tickers(args: argparse.Namespace) -> None:
    from lti import tickers

    df = tickers.refresh_cik_ticker_map()
    print(f"wrote {len(df)} cik->ticker rows")


def cmd_fetch_prices(args: argparse.Namespace) -> None:
    from lti import fundamentals, prices

    if args.smoke:
        wanted = list(config.SMOKE_TICKERS) + [args.benchmark]
    elif args.universe_file:
        wanted = [line.strip() for line in open(args.universe_file) if line.strip()]
        wanted.append(args.benchmark)
    else:
        wanted = fundamentals.price_universe(fundamentals.load_fundamentals()) + [args.benchmark]

    prices.fetch_prices(wanted, start=args.start, batch_size=args.batch_size, force=args.force)
    report = prices.missing_report(wanted)
    print(report["status"].value_counts().to_string())


def cmd_refresh_prices(args: argparse.Namespace) -> None:
    from lti import prices

    tickers = list(config.SMOKE_TICKERS) + ["SPY"] if args.smoke else None
    prices.refresh_prices(tickers, lookback_days=args.lookback_days, batch_size=args.batch_size)


def cmd_coverage(args: argparse.Namespace) -> None:
    from lti import fundamentals

    fundamentals.coverage_report(fundamentals.load_fundamentals())


def cmd_progress(args: argparse.Namespace) -> None:
    from lti import progress

    print(progress.render())


def _friction_kwargs(args: argparse.Namespace) -> dict:
    """The portfolio-construction, cost and tax flags, as BacktestConfig / RollingConfig fields."""
    from lti.frictions import TaxRates

    if args.sell_rank is not None and args.sell_rank < args.top_n:
        raise SystemExit(f"--sell-rank ({args.sell_rank}) must be at least --top-n ({args.top_n})")
    if args.industry_cap is not None and not 0 < args.industry_cap <= 1:
        raise SystemExit(f"--industry-cap ({args.industry_cap}) is a share of the portfolio, in (0, 1]")
    tax = None
    if args.taxable:
        tax = TaxRates(short_term=args.short_term_tax, long_term=args.long_term_tax, dividends=args.dividend_tax)
    return dict(
        cost_bps=args.cost_bps, tax=tax, hold_past_one_year=args.hold_past_year,
        sell_rank=args.sell_rank, industry_cap=args.industry_cap,
    )


def _add_friction_args(p: argparse.ArgumentParser) -> None:
    from lti.frictions import DEFAULT_COST_BPS, TaxRates

    rates = TaxRates()
    p.add_argument(
        "--sell-rank", type=int, default=None,
        help="keep a holding until it drops out of the top SELL_RANK, not the top N: a buffer "
             "against turnover (e.g. --top-n 30 --sell-rank 60; default: no buffer)",
    )
    p.add_argument(
        "--industry-cap", type=float, default=None,
        help="at most this share of the picks in one Fama-French industry, e.g. 0.2 (default: no cap)",
    )
    p.add_argument(
        "--cost-bps", type=float, default=DEFAULT_COST_BPS,
        help=f"trading cost per dollar traded, one way, in basis points (default {DEFAULT_COST_BPS:g}; 0 = gross)",
    )
    p.add_argument("--taxable", action="store_true", help="a taxable account: tax dividends and realized gains")
    p.add_argument("--short-term-tax", type=float, default=rates.short_term,
                   help=f"rate on gains held a year or less (default {rates.short_term:g})")
    p.add_argument("--long-term-tax", type=float, default=rates.long_term,
                   help=f"rate on gains held longer (default {rates.long_term:g})")
    p.add_argument("--dividend-tax", type=float, default=rates.dividends,
                   help=f"rate on dividends (default {rates.dividends:g})")
    p.add_argument(
        "--hold-past-year", action="store_true",
        help="rebalance no sooner than a year and a day after the last time, so every gain is long-term",
    )


def _screen_from_json(path: str):
    from lti.backtest import BacktestConfig
    from lti.frictions import DEFAULT_COST_BPS, TaxRates
    from lti.ranking import ScreenSpec

    raw = json.load(open(path))
    screen = ScreenSpec(
        metrics=raw["metrics"],
        ascending=raw.get("ascending"),
        top_n=raw.get("top_n", 10),
        weights=raw.get("weights"),
        filters=raw.get("filters", {}),
    )
    return BacktestConfig(
        screen=screen,
        start=raw.get("start", "2011-01-01"),
        end=raw.get("end"),
        rebalance_month=raw.get("rebalance_month", 4),
        benchmark=raw.get("benchmark", "SPY"),
        rf_annual=raw.get("rf_annual", 0.0),
        initial_capital=raw.get("initial_capital", 100_000.0),
        market_cap_min=raw.get("market_cap_min", 50_000_000.0),
        operating_only=raw.get("operating_only", True),
        cost_bps=raw.get("cost_bps", DEFAULT_COST_BPS),
        tax=TaxRates(**raw["tax"]) if raw.get("tax") else None,
        hold_past_one_year=raw.get("hold_past_one_year", False),
        sell_rank=raw.get("sell_rank"),
        industry_cap=raw.get("industry_cap"),
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    from lti.backtest import BacktestConfig, rebalance_month_spread, run_backtest
    from lti.ranking import ScreenSpec

    if args.config:
        cfg = _screen_from_json(args.config)
    else:
        from lti.metrics import MAGIC_FORMULA_METRICS

        metrics = list(MAGIC_FORMULA_METRICS) if args.magic_formula else args.metrics.split(",")
        cfg = BacktestConfig(
            screen=ScreenSpec(
                metrics=metrics,
                top_n=args.top_n,
                filters={
                    "exclude_financials": args.exclude_financials or args.magic_formula,
                    "exclude_utilities": args.exclude_utilities or args.magic_formula,
                },
            ),
            start=args.start,
            end=args.end,
            rebalance_month=args.rebalance_month,
            **_friction_kwargs(args),
        )
    if args.all_months:
        spread = rebalance_month_spread(cfg)
        print("\n=== the same strategy, rebalanced in each month ===")
        print(spread.round(4).to_string(index=False))
        ex = spread["excess_vs_univ"]
        print(
            f"\nexcess CAGR vs the universe: median {ex.median():+.2%}, "
            f"range {ex.min():+.2%} to {ex.max():+.2%} across {len(spread)} months"
        )
        return

    result = run_backtest(cfg)
    print("\n=== stats ===")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")
    print("\n=== period summary ===")
    print(result.period_summary.to_string(index=False))
    _print_warnings(result.warnings)


def cmd_rolling_backtest(args: argparse.Namespace) -> None:
    from lti.ranking import ScreenSpec
    from lti.rolling import RollingConfig, run_rolling_backtest

    cfg = RollingConfig(
        screen=ScreenSpec(metrics=args.metrics.split(","), top_n=args.top_n),
        window_years=[int(y) for y in args.windows.split(",")],
        step_months=args.step_months,
        start=args.start,
        end=args.end,
        rebalance_month=args.rebalance_month,
        **_friction_kwargs(args),
    )
    try:
        result = run_rolling_backtest(cfg)
    except RuntimeError as exc:
        raise SystemExit(f"lti rolling-backtest: {exc}") from None
    print("\n=== rolling-window summary (gaps are strategy minus universe / minus benchmark) ===")
    print(result.summary.round(4).to_string(index=False))
    print(f"\n=== per-window detail ({len(result.windows)} windows) ===")
    print(result.windows.round(4).to_string(index=False))
    _print_warnings(result.warnings)


def cmd_factor_ic(args: argparse.Namespace) -> None:
    from lti.factor import ICConfig, compute_ic

    kwargs = dict(
        start=args.start,
        end=args.end,
        horizon_months=args.horizon,
        step_months=args.step,
        market_cap_min=args.market_cap_min * 1e6,
        require_positive_eps=args.require_positive_eps,
        quantiles=args.quantiles,
        method=args.method,
    )
    if args.metrics:
        kwargs["metrics"] = args.metrics.split(",")
    result = compute_ic(ICConfig(**kwargs))

    print("\n=== IC summary (sorted by |mean IC|) ===")
    print(result.summary.round(4).to_string())
    print("\n=== mean forward return by metric quantile (Q1 = lowest value) ===")
    print(result.bucket_returns.round(4).to_string())
    _print_warnings(result.warnings)


def cmd_factor_study(args: argparse.Namespace) -> None:
    import pandas as pd

    from lti import study

    r = study.run_study(
        market_cap_min=args.market_cap_min * 1e6,
        backtest_cap_min=args.backtest_cap_min * 1e6,
        top_n=args.top_n,
        month_spread=not args.no_month_spread,
    )
    print("\n=== pre-registered factor study: rank IC vs 12-month forward return ===")
    print("monthly as-of dates, Newey-West t; operating companies, financials and BDCs out.")
    print("IC and t are signed: positive means the factor worked the way its paper said.\n")
    t = r.table.copy()
    for c in ("first_ic", "second_ic"):
        t[c] = t[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—")
    for c in ("first_t", "second_t"):
        t[c] = t[c].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—")
    t["in_screen"] = t["in_screen"].map({True: "yes", False: ""})
    t = t.rename(columns={"first_ic": "IC 2011-18", "first_t": "t", "second_ic": "IC 2019-25", "second_t": "t "})
    print(t.to_string())

    print(f"\nfinal screen (chosen on the first half only): {', '.join(r.selected) or 'nothing qualified'}")
    if not r.screen.empty:
        print(f"  second-half IC {r.screen['mean_ic']:+.3f} (t {r.screen['t_stat_nw']:+.1f})")
    for half, s in r.backtests.items():
        tag = " (in-sample)" if half == "first half" else ""
        print(
            f"  backtest, {half}{tag}: {s['port_cagr']:.1%} a year vs {s['univ_cagr']:.1%} for its universe "
            f"({s['excess_cagr_vs_univ']:+.1%}), SPY {s['bench_cagr']:.1%}"
        )
    if not r.month_spread.empty:
        ex = r.month_spread["excess_vs_univ"]
        print(
            f"  second half, by rebalance month vs universe: {ex.min():+.1%} to {ex.max():+.1%} "
            f"(median {ex.median():+.1%}), ahead in {int((ex > 0).sum())} of {len(ex)} months"
        )


def cmd_track_record(args: argparse.Namespace) -> None:
    from lti import track

    try:
        rows = track.record(args.asof, force=args.force)
    except ValueError as exc:
        raise SystemExit(f"lti track-record: {exc}") from None
    if rows is None:
        print("already on record for that day — nothing written")
        return
    counts = rows.groupby("strategy").size()
    print(f"recorded {rows['record_date'].iloc[0].date()}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def cmd_track_report(args: argparse.Namespace) -> None:
    import pandas as pd

    from lti import track

    records = track.load_records()
    if records.empty:
        print("nothing on record yet — run `lti track-record` (the nightly job does)")
        return
    days = records["record_date"].drop_duplicates().sort_values()
    print(f"\n=== forward track record: {len(days)} day(s), {days.iloc[0].date()} to {days.iloc[-1].date()} ===")
    for s in track.STRATEGIES:
        print(f"  {s.label:<30} {s.why}")
    summary = track.summary(track.performance(records))
    if summary.empty:
        print("\nno record has reached its first horizon (1 month) yet — check back later.")
        return
    shown = summary.copy()
    for c in ("ret", "vs_universe", "vs_spy", "beat_universe"):
        fmt = "{:.0%}" if c == "beat_universe" else "{:+.1%}"
        shown[c] = shown[c].map(lambda v, fmt=fmt: fmt.format(v) if pd.notna(v) else "—")
    shown["t_vs_universe"] = shown["t_vs_universe"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—")
    print("\nby horizon (months): average return, gap to the same-day universe and to SPY")
    print(shown.to_string(index=False))
    curves = track.paper_curves(records)
    if len(curves) > 1:
        print("\nfollowing each strategy, rebalancing monthly, $1 grew to:")
        print(curves.iloc[-1].round(3).to_string())


def cmd_journal_add(args: argparse.Namespace) -> None:
    from lti import journal

    try:
        entry = journal.add_entry(
            args.ticker, args.action, args.thesis,
            change_my_mind=args.change_my_mind or "", fair_value=args.fair_value, conviction=args.conviction,
            review_by=args.review_by, refers_to=args.refers_to, date=args.date,
        )
    except ValueError as exc:
        raise SystemExit(f"lti journal-add: {exc}") from None
    print(f"logged {entry['id']}: {entry['action']} {entry['ticker']} at ${entry['price']:,.2f}, review by {entry['review_by']}")


def cmd_journal(args: argparse.Namespace) -> None:
    import pandas as pd

    from lti import journal

    out = journal.outcomes()
    if out.empty:
        print("the journal is empty — `lti journal-add TICKER ACTION --thesis ...`")
        return
    view = out[["id", "date", "ticker", "action", "price", "price_now", "since", "spy_since", "vs_spy",
                "right_so_far", "review_due", "thesis"]].copy()
    view["date"] = view["date"].dt.date
    for c in ("since", "spy_since", "vs_spy"):
        view[c] = view[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "—")
    view["right_so_far"] = view["right_so_far"].map({1.0: "yes", 0.0: "no"}).fillna("")
    view["review_due"] = view["review_due"].map({True: "DUE", False: ""})
    print(view.to_string(index=False, max_colwidth=50))
    scored = out["right_so_far"].dropna()
    if len(scored):
        print(f"\n{int(scored.sum())} of {len(scored)} scored decisions look right so far (against SPY).")


def cmd_undervalued(args: argparse.Namespace) -> None:
    import pandas as pd

    from lti import prices, report
    from lti.fundamentals import load_fundamentals
    from lti.valuation import ValuationAssumptions, rank_undervalued

    asof = args.asof or pd.Timestamp.today().strftime("%Y-%m-%d")
    assumptions = ValuationAssumptions(discount_rate=args.discount_rate, growth_cap=args.growth_cap)
    # the screen's arguments double as the metadata the report publishes
    screen = {
        "asof": asof,
        "top_n": args.top,
        "market_cap_min": args.market_cap_min * 1e6,
        "min_models": args.min_models,
        "min_profit_years": args.min_profit_years,
        "min_roe": args.min_roe,
        "require_positive_eps": not args.allow_negative_eps,
        "exclude_financials": not args.include_financials,
    }
    params = {**screen, "discount_rate": args.discount_rate, "growth_cap": args.growth_cap}
    ranked = rank_undervalued(
        load_fundamentals(), prices.load_price_data(), assumptions=assumptions, **screen
    )

    if args.out:
        written = report.write_artifacts(ranked, args.out, asof=asof, params=params)
        for p in written:
            print("wrote", p)

    if args.format == "json":
        print(json.dumps(report.build_payload(ranked, asof=asof, params=params), indent=2))
    elif args.format == "csv":
        print(report.render_csv(ranked), end="")
    elif args.format == "html":
        print(report.render_html(ranked, asof=asof, params=params), end="")
    elif not args.out:  # text (default) — skip when --out already reported paths
        if ranked.empty:
            print(f"no names pass the filters as of {asof}")
            return
        view = report.build_view(ranked)
        view["fair_value_est_upside"] = (view["fair_value_est_upside"] * 100).round(1)
        if "dividend_yield" in view.columns:
            view["dividend_yield"] = (view["dividend_yield"] * 100).round(1)
        for c in ("price", "fair_value_est", "pe_norm", "revenue_cagr", "fcf_conversion", "debt_to_equity"):
            if c in view.columns:
                view[c] = view[c].round(2)
        print(f"\n=== most undervalued as of {asof} ({len(ranked)} shown) ===")
        print(view.to_string(index=False))


def cmd_explain_models(args: argparse.Namespace) -> None:
    """Print what each intrinsic-value equation does, and what it assumes."""
    import textwrap

    from lti.valuation import MAX_UPSIDE, MIN_MODELS, MODEL_DOCS, MODELS, ValuationAssumptions

    a = ValuationAssumptions()
    width = max(60, min(args.width, 120))

    def para(text: str, label: str = "", indent: str = "  ") -> None:
        body = f"{label} {text}" if label else text
        print(textwrap.fill(body, width=width, initial_indent=indent,
                            subsequent_indent=indent + " " * (len(label) + 1 if label else 0)))

    print(f"\nSix equations. Each fair value is the median of the ones that produced a number "
          f"({MIN_MODELS} minimum).")
    para(f"Defaults: discount rate {a.discount_rate:.1%} · terminal growth {a.terminal_growth:.1%} · "
         f"DCF window {a.dcf_years}y · growth capped at {a.growth_cap:.0%} · AAA yield {a.bond_yield:.1%}",
         indent="")
    for m in MODELS:
        d = MODEL_DOCS[m]
        print(f"\n{d.label}  ({m})")
        print(f"  {d.formula}")
        print()
        para(d.idea + " It takes " + d.inputs)
        para(d.at_defaults, "At the defaults:")
        para(d.silent, "No value when:")
        para(d.misleads, "Where it misleads:")
    print()
    para("Four of the six multiply the same EPS, so six values agreeing is not six independent "
         "opinions — the DCF (cash flow) and the dividend discount (cash actually paid out) are the "
         f"two carrying separate evidence. The screen also drops an upside above {MAX_UPSIDE:+.0%} as "
         "a data error. Rough, assumption-sensitive estimates — not investment advice.", indent="")
    print()


def cmd_smoke(args: argparse.Namespace) -> None:
    """Full smoke chain assuming `lti update` already ran with LTI_SMOKE=1."""
    from lti import fundamentals, prices
    from lti.backtest import BacktestConfig, run_backtest
    from lti.ranking import ScreenSpec

    fundamentals.build_fundamentals(smoke=True)
    from lti import tickers

    tickers.refresh_cik_ticker_map()
    prices.fetch_prices(list(config.SMOKE_TICKERS) + ["SPY"], start="2010-01-01")
    cfg = BacktestConfig(
        screen=ScreenSpec(metrics=["pe", "debt_to_equity"], top_n=5),
        start="2015-01-01",
        end="2023-01-01",
        market_cap_min=0.0,
    )
    result = run_backtest(cfg)
    print("\n=== smoke stats ===")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lti", description="LongTermInvestments CLI")
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("update", help="download SEC data + build index (+ pipeline unless smoke)")
    up.add_argument("--force", action="store_true")
    up.set_defaults(func=cmd_update)

    pl = sub.add_parser("pipeline", help="run only the standardization pipeline")
    pl.set_defaults(func=cmd_pipeline)

    bf = sub.add_parser("build-fundamentals", help="build fundamentals.parquet")
    bf.add_argument("--smoke", action="store_true")
    bf.add_argument("--quarters", help="comma-separated, e.g. 2022q1,2022q2")
    bf.set_defaults(func=cmd_build_fundamentals)

    rt = sub.add_parser("refresh-tickers", help="refresh cik->ticker map")
    rt.set_defaults(func=cmd_refresh_tickers)

    fp = sub.add_parser(
        "fetch-prices",
        help="download/cache yfinance prices (total-return + split-adjusted), splits and dividends",
    )
    fp.add_argument("--smoke", action="store_true")
    fp.add_argument("--universe-file", help="file with one ticker per line")
    fp.add_argument("--benchmark", default="SPY")
    fp.add_argument("--start", default="2008-01-01")
    fp.add_argument("--batch-size", type=int, default=40)
    fp.add_argument("--force", action="store_true")
    fp.set_defaults(func=cmd_fetch_prices)

    rp = sub.add_parser(
        "refresh-prices",
        help="top up cached tickers with recent bars; re-fetch any that split or paid a dividend",
    )
    rp.add_argument("--smoke", action="store_true")
    rp.add_argument("--lookback-days", type=int, default=7, help="recent window to re-download")
    rp.add_argument("--batch-size", type=int, default=40)
    rp.set_defaults(func=cmd_refresh_prices)

    cv = sub.add_parser("coverage", help="print fundamentals coverage report")
    cv.set_defaults(func=cmd_coverage)

    pg = sub.add_parser("progress", help="show a progress bar for each pipeline stage")
    pg.set_defaults(func=cmd_progress)

    bt = sub.add_parser("backtest", help="run a backtest")
    bt.add_argument("--config", help="path to a backtest config JSON")
    bt.add_argument("--metrics", default="pe,debt_to_equity")
    bt.add_argument("--top-n", type=int, default=10)
    bt.add_argument("--start", default="2011-01-01")
    bt.add_argument("--end", default=None)
    bt.add_argument(
        "--rebalance-month", type=int, default=4,
        help="default 4 (April), when calendar-year 10-Ks are in",
    )
    bt.add_argument(
        "--all-months", action="store_true",
        help="run the strategy once per rebalance month and print the spread",
    )
    bt.add_argument(
        "--magic-formula",
        action="store_true",
        help="Greenblatt: rank on ebit_ev + roic, excluding financials and utilities",
    )
    bt.add_argument("--exclude-financials", action="store_true", help="drop SIC 6000-6799")
    bt.add_argument("--exclude-utilities", action="store_true", help="drop SIC 4900-4999")
    _add_friction_args(bt)
    bt.set_defaults(func=cmd_backtest)

    rb = sub.add_parser(
        "rolling-backtest",
        help="backtest a strategy over many rolling N-year windows spanning all available data",
    )
    rb.add_argument("--metrics", default="pe,debt_to_equity")
    rb.add_argument("--top-n", type=int, default=10)
    rb.add_argument("--windows", default="3,5", help="comma-separated window lengths in years")
    rb.add_argument("--step-months", type=int, default=12, help="spacing between window start dates")
    rb.add_argument("--start", default=None, help="default: earliest available price date")
    rb.add_argument("--end", default=None, help="default: latest available price date")
    rb.add_argument("--rebalance-month", type=int, default=4, help="April by default, as for `lti backtest`")
    _add_friction_args(rb)
    rb.set_defaults(func=cmd_rolling_backtest)

    fi = sub.add_parser("factor-ic", help="cross-sectional IC of each metric vs forward return")
    fi.add_argument("--metrics", help="comma-separated; default = all known metrics")
    fi.add_argument("--start", default="2011-04-01")
    fi.add_argument("--end", default=None)
    fi.add_argument("--horizon", type=int, default=12, help="forward-return window (months)")
    fi.add_argument("--step", type=int, default=12, help="spacing of as-of dates (months)")
    fi.add_argument("--market-cap-min", type=float, default=500.0, help="universe floor ($M)")
    fi.add_argument("--require-positive-eps", action="store_true")
    fi.add_argument("--quantiles", type=int, default=5)
    fi.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    fi.set_defaults(func=cmd_factor_ic)

    fs = sub.add_parser(
        "factor-study",
        help="pre-registered test of published factors: chosen on 2011-18, judged on 2019-25",
    )
    fs.add_argument("--market-cap-min", type=float, default=500.0, help="IC universe floor ($M)")
    fs.add_argument("--backtest-cap-min", type=float, default=1000.0, help="backtest universe floor ($M)")
    fs.add_argument("--top-n", type=int, default=30, help="names the final screen's backtest holds")
    fs.add_argument("--no-month-spread", action="store_true", help="skip the 12-month rebalance spread")
    fs.set_defaults(func=cmd_factor_study)

    tr = sub.add_parser("track-record", help="record today's holdings for every tracked strategy (once a day)")
    tr.add_argument("--asof", default=None,
                    help="date (default: the latest close in the price cache; at most a week before it)")
    tr.add_argument("--force", action="store_true", help="redo a record made earlier the same day by mistake")
    tr.set_defaults(func=cmd_track_record)

    trr = sub.add_parser("track-report", help="how the recorded strategies have done since")
    trr.set_defaults(func=cmd_track_report)

    ja = sub.add_parser("journal-add", help="log a decision in the decision journal")
    ja.add_argument("ticker")
    ja.add_argument("action", help="buy, add, trim, sell, watch, pass or review")
    ja.add_argument("--thesis", required=True, help="why — the point of the journal")
    ja.add_argument("--change-my-mind", help="what would prove this wrong")
    ja.add_argument("--fair-value", type=float, help="your estimate, per share")
    ja.add_argument("--conviction", type=int, help="1 (low) to 5 (high)")
    ja.add_argument("--review-by", help="date to look again (default: a year on)")
    ja.add_argument("--refers-to", help="for a review: the id of the decision it looks back on")
    ja.add_argument("--date", help="date of the decision (default: the latest close)")
    ja.set_defaults(func=cmd_journal_add)

    jl = sub.add_parser("journal", help="every logged decision and what the stock did since")
    jl.set_defaults(func=cmd_journal)

    em = sub.add_parser("explain-models", help="what each intrinsic-value equation does and assumes")
    em.add_argument("--width", type=int, default=92, help="wrap width (60-120)")
    em.set_defaults(func=cmd_explain_models)

    uv = sub.add_parser("undervalued", help="most undervalued names by blended intrinsic value")
    uv.add_argument("--asof", default=None, help="date (default today)")
    uv.add_argument("--top", type=int, default=30)
    uv.add_argument("--market-cap-min", type=float, default=1000.0, help="floor ($M)")
    uv.add_argument("--min-models", type=int, default=3, help="valuation models that must produce a value")
    uv.add_argument(
        "--min-profit-years", type=int, default=4,
        help="profitable in at least this many of the last 5 years (default 4)",
    )
    uv.add_argument(
        "--include-financials", action="store_true",
        help="keep banks, insurers, REITs and BDCs (the models assume an operating business)",
    )
    uv.add_argument("--min-roe", type=float, default=None, help="quality floor, e.g. 0.1")
    uv.add_argument("--discount-rate", type=float, default=0.09)
    uv.add_argument("--growth-cap", type=float, default=0.15)
    uv.add_argument("--allow-negative-eps", action="store_true", help="drop the profitable-now requirement")
    uv.add_argument(
        "--format",
        choices=["text", "html", "json", "csv"],
        default="text",
        help="stdout format (default: text)",
    )
    uv.add_argument(
        "--out",
        metavar="DIR",
        help="also write index.html + undervalued.{json,csv} into this directory",
    )
    uv.set_defaults(func=cmd_undervalued)

    sm = sub.add_parser("smoke", help="run the full smoke chain (after `lti update`)")
    sm.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
