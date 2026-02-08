#!/usr/bin/env python3
"""
Trading Bot Runner v2.1 - FIXED
===============================
Fixes:
- Added file lock to prevent multiple concurrent cycles
- Tracks last run time to prevent duplicate runs in same interval

Handles:
- 15-minute execution intervals
- Graceful shutdown
- State persistence loading on startup
- Metrics tracking
"""

import os
import sys
import json
import signal
import argparse
import logging
import fcntl
from datetime import datetime, timedelta
from time import sleep
from typing import Optional

from enhanced_trading_bot import EnhancedTradingBot, PositionState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lock file to prevent concurrent runs
LOCK_FILE = "/tmp/trading_bot.lock"
LAST_RUN_FILE = "/tmp/trading_bot_last_run.txt"


class TradingMetrics:
    """Track and persist trading metrics."""

    def __init__(self, filepath: str = "trading_metrics.json"):
        self.filepath = filepath
        self.metrics = self.load()

    def load(self) -> dict:
        """Load metrics from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            'start_date': datetime.now().isoformat(),
            'total_cycles': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'peak_portfolio_value': 0.0,
            'max_drawdown': 0.0,
            'last_updated': datetime.now().isoformat()
        }

    def save(self):
        """Save metrics to disk."""
        self.metrics['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save metrics: {e}")

    def record_cycle(self, portfolio_value: float, trades: list = None):
        """Record a completed cycle and any trades that occurred."""
        self.metrics['total_cycles'] += 1

        if portfolio_value > self.metrics['peak_portfolio_value']:
            self.metrics['peak_portfolio_value'] = portfolio_value

        if self.metrics['peak_portfolio_value'] > 0:
            drawdown = (self.metrics['peak_portfolio_value'] - portfolio_value) / self.metrics['peak_portfolio_value']
            if drawdown > self.metrics['max_drawdown']:
                self.metrics['max_drawdown'] = drawdown

        # Record individual trade metrics
        if trades:
            for trade in trades:
                if trade.get('side') == 'SELL':
                    self.metrics['total_trades'] += 1
                    pnl = trade.get('pnl', 0)
                    self.metrics['total_pnl'] += pnl
                    if pnl > 0:
                        self.metrics['winning_trades'] += 1
                    elif pnl < 0:
                        self.metrics['losing_trades'] += 1

        self.save()

    def display(self):
        """Display current metrics."""
        m = self.metrics
        print("\n" + "=" * 50)
        print(" TRADING METRICS")
        print("=" * 50)
        print(f"  Running since:       {m['start_date'][:10]}")
        print(f"  Total cycles:        {m['total_cycles']}")
        print(f"  Total trades:        {m['total_trades']}")

        if m['total_trades'] > 0:
            win_rate = m['winning_trades'] / m['total_trades'] * 100
            print(f"  Win rate:            {win_rate:.1f}%")

        print(f"  Total PnL:           ${m['total_pnl']:,.2f}")
        print(f"  Peak value:          ${m['peak_portfolio_value']:,.2f}")
        print(f"  Max drawdown:        {m['max_drawdown']*100:.2f}%")
        print(f"  Last updated:        {m['last_updated'][:19]}")
        print("=" * 50 + "\n")


class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM."""

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.info("Shutdown signal received, completing current cycle...")
        self.kill_now = True


def get_current_interval(interval_minutes: int) -> str:
    """Get the current interval identifier (e.g., '2026-01-19T13:15')."""
    now = datetime.now()
    interval_start = now.replace(
        minute=(now.minute // interval_minutes) * interval_minutes,
        second=0,
        microsecond=0
    )
    return interval_start.strftime('%Y-%m-%dT%H:%M')


def already_ran_this_interval(interval_minutes: int) -> bool:
    """Check if we already ran in the current interval."""
    current_interval = get_current_interval(interval_minutes)

    if os.path.exists(LAST_RUN_FILE):
        try:
            with open(LAST_RUN_FILE, 'r') as f:
                last_interval = f.read().strip()

            if last_interval == current_interval:
                return True
        except IOError:
            pass

    return False


def mark_interval_complete(interval_minutes: int):
    """Mark the current interval as complete."""
    current_interval = get_current_interval(interval_minutes)
    try:
        with open(LAST_RUN_FILE, 'w') as f:
            f.write(current_interval)
    except IOError as e:
        logger.warning(f"Could not write last run file: {e}")


def load_and_verify_state(config_path: str = "config.json"):
    """
    Load and verify position state on startup.
    Ensures trailing stop data survives restarts.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    positions_file = config['persistence'].get('positions_file', 'positions.json')

    if os.path.exists(positions_file):
        try:
            with open(positions_file, 'r') as f:
                data = json.load(f)

            # Handle both new format {"positions": {...}, "cooldowns": {...}}
            # and legacy format (flat dict of positions)
            if isinstance(data, dict) and 'positions' in data and isinstance(data['positions'], dict):
                positions = data['positions']
                cooldowns = data.get('cooldowns', {})
                if cooldowns:
                    logger.info(f"  Active cooldowns: {list(cooldowns.keys())}")
            else:
                positions = data

            logger.info(f"Loaded {len(positions)} position states from {positions_file}")

            for symbol, state in positions.items():
                logger.info(f"  {symbol}: entry=${state.get('entry_price', 'N/A')}, "
                           f"peak=${state.get('peak_price', 'N/A')}")

            return positions
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load positions file: {e}")

    return {}


def wait_for_next_interval(interval_minutes: int):
    """Wait until the next interval boundary (e.g., :00, :15, :30, :45 for 15-min)."""
    now = datetime.now()

    # Calculate next interval
    minutes_past = now.minute % interval_minutes
    if minutes_past == 0 and now.second < 5:
        # We're at the boundary, run now
        return

    minutes_to_wait = interval_minutes - minutes_past
    next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_wait)

    wait_seconds = (next_run - now).total_seconds()

    if wait_seconds > 0:
        logger.info(f"Waiting {wait_seconds:.0f} seconds until next interval ({next_run.strftime('%H:%M')})")
        sleep(wait_seconds)


def run_once(config_path: str = "config.json"):
    """Run a single trading cycle."""
    logger.info("Running single cycle...")

    # Verify state is loaded
    load_and_verify_state(config_path)

    bot = EnhancedTradingBot(config_path)
    bot.run_cycle()

    logger.info("Single cycle complete.")


def run_continuous(config_path: str = "config.json", interval: Optional[int] = None):
    """Run continuous trading loop with protection against duplicate runs."""

    # Load config for interval
    with open(config_path, 'r') as f:
        config = json.load(f)

    if interval is None:
        interval = config['scheduler'].get('interval_minutes', 15)

    logger.info("=" * 60)
    logger.info(" STARTING CONTINUOUS TRADING BOT v2.1")
    logger.info("=" * 60)
    logger.info(f"  Interval: {interval} minutes")
    logger.info(f"  Config: {config_path}")
    logger.info("=" * 60)

    # Load and verify existing state
    existing_positions = load_and_verify_state(config_path)
    if existing_positions:
        logger.info(f"Restored {len(existing_positions)} position states from previous session")

    # Initialize
    killer = GracefulKiller()
    metrics = TradingMetrics(config['persistence'].get('metrics_file', 'trading_metrics.json'))

    # Create bot once and reuse across cycles (API clients are expensive to init)
    bot = EnhancedTradingBot(config_path)

    # Wait for next interval boundary
    wait_for_next_interval(interval)

    while not killer.kill_now:
        try:
            # Check if we already ran this interval
            if already_ran_this_interval(interval):
                logger.debug("Already ran this interval, waiting for next...")
                sleep(5)  # Brief sleep before checking again
                wait_for_next_interval(interval)
                continue

            cycle_start = datetime.now()
            logger.info(f"Starting cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")

            # Acquire file lock to prevent concurrent cycle execution
            lock_fd = None
            try:
                lock_fd = open(LOCK_FILE, 'w')
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                logger.warning("Another cycle is already running, skipping this interval")
                if lock_fd:
                    lock_fd.close()
                wait_for_next_interval(interval)
                continue

            try:
                # Reuse bot instance (config hot-reloads inside run_cycle)
                bot.run_cycle()

                # Mark this interval as complete
                mark_interval_complete(interval)

                # Record metrics
                account = bot.get_account()
                metrics.record_cycle(account['portfolio_value'], bot.cycle_trades)
            finally:
                # Release file lock
                if lock_fd:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()

            # Wait for next interval
            if not killer.kill_now:
                wait_for_next_interval(interval)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            # Wait before retrying
            if not killer.kill_now:
                logger.info("Waiting 60 seconds before retry...")
                sleep(60)

    logger.info("Trading bot stopped gracefully.")
    metrics.display()


def show_metrics(config_path: str = "config.json"):
    """Display trading metrics."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    metrics = TradingMetrics(config['persistence'].get('metrics_file', 'trading_metrics.json'))
    metrics.display()


def show_positions(config_path: str = "config.json"):
    """Display current position states."""
    positions = load_and_verify_state(config_path)

    # Also load cooldowns from the file directly
    with open(config_path, 'r') as f:
        config = json.load(f)
    positions_file = config['persistence'].get('positions_file', 'positions.json')
    cooldowns = {}
    if os.path.exists(positions_file):
        try:
            with open(positions_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'cooldowns' in data:
                cooldowns = data['cooldowns']
        except (json.JSONDecodeError, IOError):
            pass

    print("\n" + "=" * 60)
    print(" POSITION STATES (Persistent)")
    print("=" * 60)

    if not positions:
        print("  No positions tracked.")
    else:
        for symbol, state in positions.items():
            entry = state.get('entry_price', 'N/A')
            peak = state.get('peak_price', 'N/A')
            entry_time = state.get('entry_time', 'N/A')
            entry_type = state.get('entry_type', 'unknown')

            if isinstance(entry, (int, float)) and isinstance(peak, (int, float)):
                gain_from_entry = (peak - entry) / entry * 100
                entry_atr = state.get('entry_atr', 0)
                print(f"  {symbol} [{entry_type}]:")
                print(f"    Entry: ${entry:.2f} @ {entry_time[:16] if isinstance(entry_time, str) else 'N/A'}")
                print(f"    Peak:  ${peak:.2f} (+{gain_from_entry:.1f}% from entry)")
                print(f"    Trailing stop: ${peak * 0.88:.2f} (12% base, adaptive in bearish)")
                if entry_atr > 0:
                    print(f"    Initial stop:  ${entry - entry_atr * 2:.2f} (2x ATR=${entry_atr:.2f})")
                if entry_type == 'mr':
                    print(f"    Profit target: BB middle (mean reversion)")
            else:
                print(f"  {symbol}: {state}")

    if cooldowns:
        print("\n  COOLDOWNS (48h post-stop-out):")
        for symbol, timestamp in cooldowns.items():
            print(f"    {symbol}: stopped out at {timestamp[:16]}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Trading Bot Runner')
    parser.add_argument('--once', action='store_true', help='Run single cycle and exit')
    parser.add_argument('--interval', type=int, help='Override interval in minutes')
    parser.add_argument('--metrics', action='store_true', help='Show trading metrics')
    parser.add_argument('--positions', action='store_true', help='Show position states')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')

    args = parser.parse_args()

    # Check for API keys
    if not os.environ.get('ALPACA_API_KEY') or not os.environ.get('ALPACA_SECRET_KEY'):
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        print("\nSet them with:")
        print("  export ALPACA_API_KEY='your-key'")
        print("  export ALPACA_SECRET_KEY='your-secret'")
        sys.exit(1)

    # Check config exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    if args.metrics:
        show_metrics(args.config)
    elif args.positions:
        show_positions(args.config)
    elif args.once:
        run_once(args.config)
    else:
        run_continuous(args.config, args.interval)


if __name__ == '__main__':
    main()
