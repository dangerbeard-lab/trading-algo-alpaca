#!/usr/bin/env python3
"""
Continuous Runner for Enhanced Trading Bot
==========================================
Runs the trading bot on a schedule with proper error handling.

Usage:
    python runner.py                    # Run continuously (hourly)
    python runner.py --once             # Run once and exit
    python runner.py --interval 30      # Run every 30 minutes
    python runner.py --metrics          # Just print metrics summary
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional

from enhanced_trading_bot import EnhancedTradingBot, TradingConfig


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    print("\nShutdown requested, finishing current iteration...")
    shutdown_requested = True


def load_config_from_file(config_path: str = "config.json") -> TradingConfig:
    """Load configuration from JSON file."""
    config = TradingConfig(
        api_key=os.environ.get('ALPACA_API_KEY', ''),
        api_secret=os.environ.get('ALPACA_SECRET_KEY', '')
    )
    
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found, using defaults")
        return config
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # API settings
        if 'api_settings' in data:
            config.paper = data['api_settings'].get('paper', True)
        
        # Instruments
        if 'instruments' in data:
            inst = data['instruments']
            config.crypto_symbols = inst.get('crypto_symbols', config.crypto_symbols)
            config.etf_symbols = inst.get('etf_symbols', config.etf_symbols)
            config.stock_symbols = inst.get('stock_symbols', config.stock_symbols)
        
        # Position sizing
        if 'position_sizing' in data:
            ps = data['position_sizing']
            config.base_position_pct = ps.get('base_position_pct', config.base_position_pct)
            config.max_positions = ps.get('max_positions', config.max_positions)
            config.max_portfolio_exposure = ps.get('max_portfolio_exposure', config.max_portfolio_exposure)
            config.atr_period = ps.get('atr_period', config.atr_period)
            config.atr_target_risk = ps.get('atr_target_risk', config.atr_target_risk)
            config.min_position_pct = ps.get('min_position_pct', config.min_position_pct)
            config.max_position_pct = ps.get('max_position_pct', config.max_position_pct)
        
        # Risk management
        if 'risk_management' in data:
            rm = data['risk_management']
            config.max_drawdown_pct = rm.get('max_drawdown_pct', config.max_drawdown_pct)
            config.take_profit_pct = rm.get('take_profit_pct', config.take_profit_pct)
        
        # Trailing stop
        if 'trailing_stop' in data:
            ts = data['trailing_stop']
            config.trailing_stop_enabled = ts.get('enabled', config.trailing_stop_enabled)
            config.trailing_stop_pct = ts.get('trailing_stop_pct', config.trailing_stop_pct)
            config.trailing_stop_activation_pct = ts.get('activation_pct', config.trailing_stop_activation_pct)
        
        # Technical indicators
        if 'technical_indicators' in data:
            ti = data['technical_indicators']
            config.ema_fast = ti.get('ema_fast', config.ema_fast)
            config.ema_slow = ti.get('ema_slow', config.ema_slow)
            config.rsi_period = ti.get('rsi_period', config.rsi_period)
            config.rsi_oversold = ti.get('rsi_oversold', config.rsi_oversold)
            config.rsi_overbought = ti.get('rsi_overbought', config.rsi_overbought)
            config.bb_period = ti.get('bb_period', config.bb_period)
            config.bb_std = ti.get('bb_std', config.bb_std)
            config.macd_fast = ti.get('macd_fast', config.macd_fast)
            config.macd_slow = ti.get('macd_slow', config.macd_slow)
            config.macd_signal = ti.get('macd_signal', config.macd_signal)
        
        # ADX hysteresis
        if 'adx_hysteresis' in data:
            adx = data['adx_hysteresis']
            config.adx_period = adx.get('adx_period', config.adx_period)
            config.adx_trending_entry = adx.get('trending_entry_threshold', config.adx_trending_entry)
            config.adx_trending_exit = adx.get('trending_exit_threshold', config.adx_trending_exit)
        
        # Volume confirmation
        if 'volume_confirmation' in data:
            vc = data['volume_confirmation']
            config.volume_confirmation_enabled = vc.get('enabled', config.volume_confirmation_enabled)
            config.volume_period = vc.get('period', config.volume_period)
            config.volume_multiplier = vc.get('multiplier', config.volume_multiplier)
        
        # Correlation filter
        if 'correlation_filter' in data:
            cf = data['correlation_filter']
            config.correlation_enabled = cf.get('enabled', config.correlation_enabled)
            config.correlation_lookback = cf.get('lookback_days', config.correlation_lookback)
            config.correlation_threshold = cf.get('threshold', config.correlation_threshold)
            config.max_correlated_positions = cf.get('max_correlated_positions', config.max_correlated_positions)
        
        # Time of day filter
        if 'time_of_day_filter' in data:
            tf = data['time_of_day_filter']
            config.time_filter_enabled = tf.get('enabled', config.time_filter_enabled)
            config.market_open_buffer_minutes = tf.get('market_open_buffer_minutes', config.market_open_buffer_minutes)
            config.market_close_buffer_minutes = tf.get('market_close_buffer_minutes', config.market_close_buffer_minutes)
        
        # Logging
        if 'logging' in data:
            lg = data['logging']
            config.log_file = lg.get('log_file', config.log_file)
            config.metrics_file = lg.get('metrics_file', config.metrics_file)
        
        logging.info(f"Loaded configuration from {config_path}")
        
    except Exception as e:
        logging.error(f"Error loading config: {e}")
    
    return config


def get_next_run_time(interval_minutes: int) -> datetime:
    """Calculate the next aligned run time."""
    now = datetime.now()
    
    # Align to interval boundaries (e.g., run at :00, :30 for 30-min interval)
    minutes_past = now.minute % interval_minutes
    if minutes_past == 0 and now.second < 30:
        # We're at an aligned time, run now
        return now
    
    # Calculate next aligned time
    next_minute = now.minute + (interval_minutes - minutes_past)
    next_run = now.replace(second=0, microsecond=0)
    
    if next_minute >= 60:
        next_run = next_run + timedelta(hours=1)
        next_run = next_run.replace(minute=next_minute - 60)
    else:
        next_run = next_run.replace(minute=next_minute)
    
    return next_run


def run_continuous(config: TradingConfig, interval_minutes: int):
    """Run the bot continuously on a schedule."""
    global shutdown_requested
    
    print(f"\n{'='*60}")
    print("Enhanced Trading Bot - Continuous Mode")
    print(f"{'='*60}")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Press Ctrl+C to stop gracefully")
    print(f"{'='*60}\n")
    
    bot = EnhancedTradingBot(config)
    iteration = 0
    
    while not shutdown_requested:
        iteration += 1
        
        try:
            # Calculate next run time
            next_run = get_next_run_time(interval_minutes)
            wait_seconds = (next_run - datetime.now()).total_seconds()
            
            if wait_seconds > 0:
                print(f"\nNext run at {next_run.strftime('%H:%M:%S')} (waiting {wait_seconds:.0f}s)")
                
                # Wait with periodic checks for shutdown
                while wait_seconds > 0 and not shutdown_requested:
                    sleep_time = min(10, wait_seconds)
                    time.sleep(sleep_time)
                    wait_seconds -= sleep_time
            
            if shutdown_requested:
                break
            
            # Run the bot
            print(f"\n[Iteration {iteration}] Running at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            bot.run_once()
            
        except KeyboardInterrupt:
            shutdown_requested = True
        except Exception as e:
            logging.error(f"Error in iteration {iteration}: {e}")
            # Continue running despite errors
            time.sleep(60)  # Wait a bit before retrying
    
    print("\nShutting down...")
    bot.print_metrics_summary()
    print("Goodbye!")


def run_once(config: TradingConfig):
    """Run the bot once and exit."""
    print("\n" + "="*60)
    print("Enhanced Trading Bot - Single Run")
    print("="*60 + "\n")
    
    bot = EnhancedTradingBot(config)
    bot.run_once()
    bot.print_metrics_summary()


def print_metrics(config: TradingConfig):
    """Print metrics summary without running the bot."""
    from enhanced_trading_bot import MetricsTracker
    
    print("\n" + "="*60)
    print("Trading Metrics Summary")
    print("="*60 + "\n")
    
    metrics = MetricsTracker(config)
    summary = metrics.get_summary()
    
    if 'message' in summary:
        print(summary['message'])
        return
    
    print(f"Total Trades:       {summary['total_trades']}")
    print(f"Win Rate:           {summary['win_rate']:.2%}")
    print(f"Total PnL:          ${summary['total_pnl']:,.2f}")
    print(f"Average R-Multiple: {summary['average_r_multiple']:.2f}")
    print(f"Sharpe Ratio:       {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {summary['max_drawdown']:.2%}")
    print(f"Current Drawdown:   {summary['current_drawdown']:.2%}")
    
    print("\n--- By Regime ---")
    for regime, stats in summary['by_regime'].items():
        if stats['trades'] > 0:
            wr = stats['wins'] / stats['trades']
            print(f"  {regime.capitalize():10} {stats['trades']:3} trades | {wr:6.2%} win rate | ${stats['total_pnl']:>10,.2f} PnL")
    
    print("\n--- By Symbol (sorted by PnL) ---")
    symbol_data = [
        (sym, stats['trades'], stats['wins'], stats['total_pnl'], stats['total_r'])
        for sym, stats in summary['by_symbol'].items()
        if stats['trades'] > 0
    ]
    symbol_data.sort(key=lambda x: x[3], reverse=True)
    
    print(f"  {'Symbol':<10} {'Trades':>6} {'Win%':>7} {'PnL':>12} {'Avg R':>7}")
    print("  " + "-"*45)
    for sym, trades, wins, pnl, total_r in symbol_data:
        wr = wins / trades if trades > 0 else 0
        avg_r = total_r / trades if trades > 0 else 0
        print(f"  {sym:<10} {trades:>6} {wr:>6.1%} ${pnl:>10,.2f} {avg_r:>7.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Trading Bot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python runner.py                    # Run continuously (hourly)
    python runner.py --once             # Run once and exit  
    python runner.py --interval 30      # Run every 30 minutes
    python runner.py --metrics          # Print metrics summary
    python runner.py --config my.json   # Use custom config file
        """
    )
    
    parser.add_argument('--once', action='store_true', 
                       help='Run once and exit')
    parser.add_argument('--interval', type=int, default=60,
                       help='Run interval in minutes (default: 60)')
    parser.add_argument('--metrics', action='store_true',
                       help='Print metrics summary and exit')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_file(args.config)
    
    # Validate API keys
    if not config.api_key or not config.api_secret:
        print("Error: API keys not set!")
        print("\nPlease set environment variables:")
        print("  export ALPACA_API_KEY='your-api-key'")
        print("  export ALPACA_SECRET_KEY='your-secret-key'")
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run appropriate mode
    if args.metrics:
        print_metrics(config)
    elif args.once:
        run_once(config)
    else:
        run_continuous(config, args.interval)


if __name__ == "__main__":
    main()
