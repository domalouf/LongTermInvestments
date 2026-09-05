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
        fund = fundamentals.load_fundamentals()
        mask = fund["ticker"].notna()
        if "shares_outstanding" in fund.columns:
            mask &= fund["shares_outstanding"].notna()
        wanted = sorted(fund.loc[mask, "ticker"].unique().tolist()) + [args.benchmark]

    prices.fetch_prices(wanted, start=args.start, batch_size=args.batch_size, force=args.force)
    report = prices.missing_report(wanted)
    print(report["status"].value_counts().to_string())


def cmd_coverage(args: argparse.Namespace) -> None:
    from lti import fundamentals

    fundamentals.coverage_report(fundamentals.load_fundamentals())


def cmd_progress(args: argparse.Namespace) -> None:
    from lti import progress

    print(progress.render())


def _screen_from_json(path: str):
    from lti.backtest import BacktestConfig
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
        rebalance_month=raw.get("rebalance_month", 1),
        benchmark=raw.get("benchmark", "SPY"),
        rf_annual=raw.get("rf_annual", 0.0),
        initial_capital=raw.get("initial_capital", 100_000.0),
        market_cap_min=raw.get("market_cap_min", 50_000_000.0),
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    from lti.backtest import BacktestConfig, run_backtest
    from lti.ranking import ScreenSpec

    if args.config:
        cfg = _screen_from_json(args.config)
    else:
        cfg = BacktestConfig(
            screen=ScreenSpec(metrics=args.metrics.split(","), top_n=args.top_n),
            start=args.start,
            end=args.end,
            rebalance_month=args.rebalance_month,
        )
    result = run_backtest(cfg)
    print("\n=== stats ===")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")
    print("\n=== period summary ===")
    print(result.period_summary.to_string(index=False))
    if result.warnings:
        print(f"\n=== warnings ({len(result.warnings)}) ===")
        for w in result.warnings[:30]:
            print("  ", w)


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

    fp = sub.add_parser("fetch-prices", help="download/cache yfinance adjusted close")
    fp.add_argument("--smoke", action="store_true")
    fp.add_argument("--universe-file", help="file with one ticker per line")
    fp.add_argument("--benchmark", default="SPY")
    fp.add_argument("--start", default="2008-01-01")
    fp.add_argument("--batch-size", type=int, default=40)
    fp.add_argument("--force", action="store_true")
    fp.set_defaults(func=cmd_fetch_prices)

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
    bt.add_argument("--rebalance-month", type=int, default=1)
    bt.set_defaults(func=cmd_backtest)

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
