#!/usr/bin/env python3
"""
CLI entry point for running backtests.

Usage:
  python -m backtest.run_backtest --start 2025-01-01 --end 2025-12-31
  python -m backtest.run_backtest --months 12
  python -m backtest.run_backtest --download  # download data first

Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars before running.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_spy_benchmark(start: datetime, end: datetime) -> float:
    """Buy-and-hold SPY return over the same period."""
    from backtest.data_loader import DataLoader
    loader = DataLoader.__new__(DataLoader)
    import json
    with open("config.json") as f:
        loader.config = json.load(f)

    df = loader.load_bars("SPY", "1Day")
    if df is None:
        return None
    if df["timestamp"].dt.tz is None:
        start = start.replace(tzinfo=None)
        end = end.replace(tzinfo=None)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if len(df) < 2:
        return None
    return (df["close"].iloc[-1] / df["close"].iloc[0]) - 1


def main():
    parser = argparse.ArgumentParser(description="Run backtest of trading strategy")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=12,
                        help="Months of history (used if --start not given)")
    parser.add_argument("--cash", type=float, default=100000.0, help="Starting capital")
    parser.add_argument("--config", default="config.json", help="Config path")
    parser.add_argument("--download", action="store_true",
                        help="Download data before running")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.download:
        from backtest.data_loader import DataLoader
        loader = DataLoader(args.config)
        loader.download_all(months=args.months)

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc) - timedelta(days=args.months * 30)
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)

    from backtest.engine import Backtester
    from backtest.metrics import calculate_metrics, print_metrics

    bt = Backtester(args.config, initial_cash=args.cash)
    bt.load_data(start, end)

    if not bt.bars_15min:
        logger.error("No 15Min data loaded. Run with --download first.")
        sys.exit(1)

    bt.run(start, end)

    benchmark = compute_spy_benchmark(start, end)
    metrics = calculate_metrics(bt.snapshots, bt.trades, args.cash, benchmark_return=benchmark)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
