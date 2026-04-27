#!/usr/bin/env python3
"""
Download and cache historical bar data from Alpaca for backtesting.
Stores as Parquet files in data/{timeframe}/{symbol}.parquet
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

TF_MAP = {
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Day": TimeFrame.Day,
}


class DataLoader:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        self.stock_client = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
        self.data_feed = self.config["execution"].get("data_feed", "iex")

    def _build_watchlist(self) -> List[str]:
        symbols = []
        for category, syms in self.config["watchlist"].items():
            symbols.extend(syms)
        if "SPY" not in symbols:
            symbols.append("SPY")
        return symbols

    def _fetch_bars(self, symbol: str, timeframe: str,
                    start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        tf = TF_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        try:
            if "/" in symbol:
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                )
                bars = self.crypto_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    feed=self.data_feed,
                )
                bars = self.stock_client.get_stock_bars(request)

            if len(bars.df) == 0:
                return None

            df = bars.df.reset_index()
            df = df[["timestamp", "open", "high", "low", "close", "volume",
                      "trade_count", "vwap"]]
            return df

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
            return None

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "-")
        return DATA_DIR / timeframe / f"{safe_symbol}.parquet"

    def download_symbol(self, symbol: str, timeframe: str,
                        months: int = 12, force: bool = False) -> bool:
        path = self._parquet_path(symbol, timeframe)
        if path.exists() and not force:
            logger.debug(f"Cached: {path}")
            return True

        path.parent.mkdir(parents=True, exist_ok=True)

        end = datetime.now()
        start = end - timedelta(days=months * 30)

        logger.info(f"Downloading {symbol} {timeframe} ({start.date()} to {end.date()})...")
        df = self._fetch_bars(symbol, timeframe, start, end)
        if df is None or len(df) == 0:
            logger.warning(f"No data for {symbol} {timeframe}")
            return False

        df.to_parquet(path, index=False)
        logger.info(f"  Saved {len(df)} bars to {path}")
        return True

    def download_all(self, months: int = 12, force: bool = False):
        symbols = self._build_watchlist()
        total = len(symbols)
        success = 0
        failed = []

        for i, symbol in enumerate(symbols, 1):
            for tf in ["1Day", "15Min"]:
                ok = self.download_symbol(symbol, tf, months=months, force=force)
                if tf == "1Day" and ok:
                    success += 1
                elif tf == "1Day" and not ok:
                    failed.append(symbol)
                # Rate limit: ~200 requests/min for free tier
                time.sleep(0.4)

            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total} symbols")

        logger.info(f"Download complete: {success}/{total} symbols, "
                    f"{len(failed)} failed: {failed}")

    def load_bars(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._parquet_path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Download historical data for backtesting")
    parser.add_argument("--months", type=int, default=12, help="Months of history")
    parser.add_argument("--force", action="store_true", help="Re-download existing data")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--symbol", type=str, help="Download single symbol only")
    args = parser.parse_args()

    loader = DataLoader(args.config)
    if args.symbol:
        for tf in ["1Day", "15Min"]:
            loader.download_symbol(args.symbol, tf, months=args.months, force=args.force)
    else:
        loader.download_all(months=args.months, force=args.force)


if __name__ == "__main__":
    main()
