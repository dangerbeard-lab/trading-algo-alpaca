#!/usr/bin/env python3
"""
Enhanced Trading Bot v2.0
=========================
Multi-asset hybrid strategy with critical fixes:
- Lookahead bias fix (uses closed candles only)
- State persistence (positions.json)
- Fractional share support
- 12% trailing stop (no fixed TP)
- ATR-based position sizing (4-10%)
- Limit orders for mean reversion
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    GetAssetsRequest, ClosePositionRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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


class PositionState:
    """Manages persistent state for positions (survives restarts)."""

    def __init__(self, filepath: str = "positions.json"):
        self.filepath = filepath
        self.positions: Dict[str, dict] = {}
        self.load()

    def load(self):
        """Load position state from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.positions = json.load(f)
                logger.info(f"Loaded {len(self.positions)} position states from {self.filepath}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load positions file: {e}")
                self.positions = {}
        else:
            logger.info("No existing positions file found, starting fresh")
            self.positions = {}

    def save(self):
        """Save position state to disk immediately."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.positions, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save positions file: {e}")

    def update_position(self, symbol: str, entry_price: float, entry_time: datetime,
                        peak_price: float, current_price: float):
        """Update or create position state and save immediately."""
        new_peak = max(peak_price, current_price)

        self.positions[symbol] = {
            'entry_price': entry_price,
            'entry_time': entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
            'peak_price': new_peak,
            'highest_watermark': new_peak,
            'last_updated': datetime.now().isoformat()
        }
        self.save()  # Persist immediately

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position state if exists."""
        return self.positions.get(symbol)

    def remove_position(self, symbol: str):
        """Remove position state and save."""
        if symbol in self.positions:
            del self.positions[symbol]
            self.save()

    def get_peak_price(self, symbol: str, current_price: float) -> float:
        """Get peak price for trailing stop calculation."""
        pos = self.positions.get(symbol)
        if pos:
            return max(pos.get('peak_price', current_price), current_price)
        return current_price

    def cleanup_stale_positions(self, active_symbols: set, normalize_fn=None):
        """
        Remove position states that no longer exist in Alpaca.
        active_symbols: set of symbols currently held in Alpaca
        normalize_fn: optional function to normalize symbol format
        """
        # Normalize active symbols for comparison
        normalized_active = set()
        for sym in active_symbols:
            if normalize_fn:
                normalized_active.add(normalize_fn(sym))
            else:
                normalized_active.add(sym)

        # Find stale entries
        stale = []
        for tracked_symbol in self.positions.keys():
            if tracked_symbol not in normalized_active:
                stale.append(tracked_symbol)

        # Remove stale entries
        if stale:
            for symbol in stale:
                del self.positions[symbol]
            self.save()
            logger.info(f"Cleaned up {len(stale)} stale position states: {stale}")


class TechnicalIndicators:
    """Calculate technical indicators for signal generation."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx.fillna(20)


class EnhancedTradingBot:
    """
    Multi-asset hybrid trading bot with:
    - Trend following (EMA crossover + MACD + ADX)
    - Mean reversion (Bollinger Bands + RSI)
    - 12% trailing stop (no fixed TP)
    - ATR-based position sizing (4-10%)
    - State persistence for restarts
    """

    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # API credentials from environment
        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        # Initialize clients
        self.trading_client = TradingClient(api_key, secret_key, paper=self.config.get('paper_trading', True))
        self.stock_data_client = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_data_client = CryptoHistoricalDataClient(api_key, secret_key)

        # Position state persistence
        self.position_state = PositionState(
            self.config['persistence'].get('positions_file', 'positions.json')
        )

        # Build watchlist
        self.watchlist = self._build_watchlist()

        # Separate crypto and equity symbols
        self.crypto_symbols = [s for s in self.watchlist if '/' in s]
        self.equity_symbols = [s for s in self.watchlist if '/' not in s]

        logger.info(f"Initialized bot with {len(self.watchlist)} symbols")
        logger.info(f"  Crypto: {len(self.crypto_symbols)}, Equities: {len(self.equity_symbols)}")

    def _build_watchlist(self) -> List[str]:
        """Flatten watchlist from config."""
        watchlist = []
        for category, symbols in self.config['watchlist'].items():
            watchlist.extend(symbols)
        return watchlist

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for consistent tracking.
        Alpaca returns crypto as BTCUSD, but we use BTC/USD in watchlist.
        """
        if '/' not in symbol and symbol.endswith('USD'):
            return f"{symbol[:-3]}/{symbol[-3:]}"  # BTCUSD -> BTC/USD
        return symbol

    def get_account(self) -> dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value)
        }

    def get_positions(self) -> Dict[str, dict]:
        """Get current positions."""
        positions = self.trading_client.get_all_positions()
        return {
            p.symbol: {
                'qty': float(p.qty),
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            }
            for p in positions
        }

    def get_pending_orders(self) -> set:
        """Get symbols with pending (unfilled) orders."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.trading_client.get_orders(request)
            symbols = set()
            for order in orders:
                symbols.add(order.symbol)
            if symbols:
                logger.info(f"Found pending orders for: {symbols}")
            return symbols
        except Exception as e:
            logger.warning(f"Failed to get pending orders: {e}")
            return set()


    def get_bars(self, symbol: str, timeframe: str = "15Min", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical bars for a symbol."""
        try:
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            tf = tf_map.get(timeframe, TimeFrame(15, TimeFrameUnit.Minute))

            end = datetime.now()
            start = end - timedelta(days=10)  # Enough for 100 bars

            if '/' in symbol:  # Crypto
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    limit=limit
                )
                bars = self.crypto_data_client.get_crypto_bars(request)
            else:  # Equity
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    limit=limit,
                    feed=self.config['execution'].get('data_feed', 'iex')  # Explicit IEX feed
                )
                bars = self.stock_data_client.get_stock_bars(request)

            if len(bars.df) == 0:
                return None

            df = bars.df.reset_index()
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']]
            return df

        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe."""
        cfg = self.config['strategy']

        df = df.copy()

        # EMAs
        df['ema_short'] = TechnicalIndicators.ema(df['close'], cfg['ema_short'])
        df['ema_long'] = TechnicalIndicators.ema(df['close'], cfg['ema_long'])

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.macd(
            df['close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal']
        )

        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], cfg['rsi_period'])

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.bollinger_bands(
            df['close'], cfg['bollinger_period'], cfg['bollinger_std']
        )

        # ADX
        df['adx'] = TechnicalIndicators.adx(df['high'], df['low'], df['close'], cfg['adx_period'])

        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], cfg['atr_period'])

        # Volume MA
        df['volume_ma'] = TechnicalIndicators.sma(df['volume'], cfg['volume_ma_period'])
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)

        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Generate trading signal using CLOSED candles only (fix lookahead bias).
        Uses .iloc[-2] for the last CLOSED candle, not .iloc[-1] (forming candle).
        """
        if len(df) < 3:
            return {'signal': 'HOLD', 'reason': 'Insufficient data'}

        cfg = self.config['strategy']

        # CRITICAL FIX: Use .iloc[-2] for closed candle (not -1 which is forming)
        current = df.iloc[-2]  # Last CLOSED candle
        previous = df.iloc[-3]  # Previous closed candle

        signal = {'signal': 'HOLD', 'reason': '', 'order_type': 'market', 'limit_price': None}

        # Check regime
        adx = current['adx']
        is_trending = adx > cfg['adx_trending_threshold']
        is_ranging = adx < cfg['adx_ranging_threshold']

        # Volume confirmation
        volume_confirmed = current['volume_ratio'] > cfg['volume_confirmation_multiplier']

        # ========== TRENDING REGIME: EMA Crossover + MACD ==========
        if is_trending:
            # Bullish: EMA9 crosses above EMA21, MACD confirms
            ema_cross_up = (
                previous['ema_short'] <= previous['ema_long'] and
                current['ema_short'] > current['ema_long']
            )
            macd_bullish = current['macd'] > current['macd_signal']

            if ema_cross_up and macd_bullish and volume_confirmed:
                signal = {
                    'signal': 'BUY',
                    'reason': f'Trending BUY: EMA crossover + MACD (ADX={adx:.1f})',
                    'order_type': 'market',
                    'limit_price': None
                }

            # Bearish: EMA9 crosses below EMA21
            ema_cross_down = (
                previous['ema_short'] >= previous['ema_long'] and
                current['ema_short'] < current['ema_long']
            )
            macd_bearish = current['macd'] < current['macd_signal']

            if ema_cross_down and macd_bearish:
                signal = {
                    'signal': 'SELL',
                    'reason': f'Trending SELL: EMA crossover + MACD (ADX={adx:.1f})',
                    'order_type': 'market',
                    'limit_price': None
                }

        # ========== RANGING REGIME: Mean Reversion with Limit Orders ==========
        elif is_ranging:
            rsi = current['rsi']

            # Oversold at lower Bollinger Band - use LIMIT ORDER
            if rsi < cfg['rsi_oversold'] and current['close'] <= current['bb_lower']:
                signal = {
                    'signal': 'BUY',
                    'reason': f'Mean Reversion BUY: RSI={rsi:.1f}, at BB lower (ADX={adx:.1f})',
                    'order_type': 'limit',
                    'limit_price': current['bb_lower']  # Limit at BB lower
                }

            # Overbought at upper Bollinger Band
            elif rsi > cfg['rsi_overbought'] and current['close'] >= current['bb_upper']:
                signal = {
                    'signal': 'SELL',
                    'reason': f'Mean Reversion SELL: RSI={rsi:.1f}, at BB upper (ADX={adx:.1f})',
                    'order_type': 'market',
                    'limit_price': None
                }

        return signal

    def calculate_position_size(self, symbol: str, df: pd.DataFrame, cash_balance: float) -> float:
        """
        Calculate position size using ATR-based scaling.
        - Low volatility stocks: up to max_position_pct (10%)
        - High volatility stocks: near base_position_pct (4%)

        IMPORTANT: Uses cash_balance (not portfolio_value or buying_power) to avoid
        accidental margin/leverage. Margin accounts have 2:1 buying power by default.
        """
        cfg = self.config['risk_management']

        base_pct = cfg['base_position_pct']  # 4%
        max_pct = cfg['max_position_pct']    # 10%

        # Get ATR from closed candle
        current = df.iloc[-2]
        atr = current['atr']
        price = current['close']

        if pd.isna(atr) or atr <= 0 or price <= 0:
            return cash_balance * base_pct

        # ATR as percentage of price
        atr_pct = atr / price

        # Scale position size inversely with volatility
        # Lower volatility = larger position (up to max_pct)
        # Higher volatility = smaller position (down to base_pct)

        # Typical ATR% ranges: 0.5% (low vol) to 5% (high vol like BTC)
        # Map this to position size
        if atr_pct <= 0.01:  # Very low vol (<1%)
            position_pct = max_pct
        elif atr_pct >= 0.04:  # High vol (>4%)
            position_pct = base_pct
        else:
            # Linear interpolation between base and max
            vol_range = 0.04 - 0.01
            vol_position = (atr_pct - 0.01) / vol_range
            position_pct = max_pct - (vol_position * (max_pct - base_pct))

        position_value = cash_balance * position_pct

        logger.debug(f"{symbol}: ATR%={atr_pct:.2%}, Position={position_pct:.1%} of cash (${position_value:,.0f})")

        return position_value

    def check_trailing_stop(self, symbol: str, current_price: float, position: dict) -> bool:
        """
        Check if trailing stop is hit (12% from peak).
        Updates peak price in persistent state.
        """
        cfg = self.config['risk_management']
        trailing_stop_pct = cfg['trailing_stop_pct']  # 0.12 = 12%

        entry_price = position['avg_entry_price']

        # Get or initialize peak price from persistent state
        peak_price = self.position_state.get_peak_price(symbol, current_price)

        # Update peak if current price is higher
        if current_price > peak_price:
            peak_price = current_price
            self.position_state.update_position(
                symbol=symbol,
                entry_price=entry_price,
                entry_time=datetime.now(),  # Will preserve original if exists
                peak_price=peak_price,
                current_price=current_price
            )

        # Calculate trailing stop level
        stop_level = peak_price * (1 - trailing_stop_pct)

        if current_price <= stop_level:
            drawdown_from_peak = (peak_price - current_price) / peak_price
            logger.info(f"{symbol}: Trailing stop hit! Peak=${peak_price:.2f}, "
                       f"Current=${current_price:.2f}, Stop=${stop_level:.2f} "
                       f"(Down {drawdown_from_peak:.1%} from peak)")
            return True

        return False

    def execute_trade(self, symbol: str, side: str, qty: float,
                      order_type: str = 'market', limit_price: float = None) -> bool:
        """Execute a trade with proper quantity handling (fractional shares)."""
        try:
            # Fractional shares: keep as float, don't round to int
            # Alpaca supports fractional shares for most equities

            is_crypto = '/' in symbol

            if is_crypto:
                # Crypto: use notional or qty with appropriate precision
                qty = round(qty, 8)  # 8 decimal places for crypto
            else:
                # Equities: round to 4 decimal places for fractional shares
                qty = round(qty, 4)

            if qty <= 0:
                logger.warning(f"Invalid quantity {qty} for {symbol}")
                return False

            if order_type == 'limit' and limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol.replace('/', ''),  # Remove slash for crypto
                    qty=qty,
                    side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
                    limit_price=round(limit_price, 2)
                )
                order_type_str = f"LIMIT @ ${limit_price:.2f}"
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol.replace('/', ''),
                    qty=qty,
                    side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY
                )
                order_type_str = "MARKET"

            order = self.trading_client.submit_order(order_request)
            logger.info(f"Executed {side} {order_type_str} order: {qty} {symbol} (Order ID: {order.id})")

            return True

        except Exception as e:
            logger.error(f"Failed to execute {side} order for {symbol}: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Close an entire position."""
        try:
            clean_symbol = symbol.replace('/', '')
            self.trading_client.close_position(clean_symbol)
            self.position_state.remove_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False

    def should_skip_trading(self) -> bool:
        """Check if we should skip trading (market hours, first/last 30 mins)."""
        now = datetime.now()
        cfg = self.config['execution']
        sched = self.config['scheduler']

        # Always allow crypto
        # For equities, check market hours
        market_open = now.replace(
            hour=sched['market_open_hour'],
            minute=sched['market_open_minute'],
            second=0
        )
        market_close = now.replace(
            hour=sched['market_close_hour'],
            minute=sched['market_close_minute'],
            second=0
        )

        # Skip first and last 30 minutes
        skip_open = market_open + timedelta(minutes=cfg['skip_first_minutes'])
        skip_close = market_close - timedelta(minutes=cfg['skip_last_minutes'])

        if now < skip_open or now > skip_close:
            return True

        return False

    def run_cycle(self):
        """Run one complete trading cycle."""
        logger.info("=" * 60)
        logger.info("STARTING TRADING CYCLE")
        logger.info("=" * 60)

        # Get account info
        account = self.get_account()
        portfolio_value = account['portfolio_value']
        cash = account['cash']
        logger.info(f"Portfolio Value: ${portfolio_value:,.2f}, Cash: ${cash:,.2f}")

        # Get current positions
        positions = self.get_positions()
        current_position_count = len(positions)
        logger.info(f"Current Positions: {current_position_count}")

        # Cleanup stale position states (sync with Alpaca)
        self.position_state.cleanup_stale_positions(
            active_symbols=set(positions.keys()),
            normalize_fn=self._normalize_symbol
        )

        # Calculate current exposure
        total_exposure = sum(p['market_value'] for p in positions.values())
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
        logger.info(f"Current Exposure: {exposure_pct:.1%}")

        cfg = self.config['risk_management']

        # ========== CHECK EXISTING POSITIONS FOR EXITS ==========
        for symbol, position in positions.items():
            # Normalize crypto symbols: Alpaca returns BTCUSD, we use BTC/USD
            lookup_symbol = self._normalize_symbol(symbol)

            current_price = position['current_price']

            # Update position state (for trailing stop tracking)
            pos_state = self.position_state.get_position(lookup_symbol)
            if pos_state:
                self.position_state.update_position(
                    symbol=lookup_symbol,
                    entry_price=position['avg_entry_price'],
                    entry_time=pos_state.get('entry_time', datetime.now()),
                    peak_price=pos_state.get('peak_price', current_price),
                    current_price=current_price
                )
            else:
                # New position not in state - initialize
                self.position_state.update_position(
                    symbol=lookup_symbol,
                    entry_price=position['avg_entry_price'],
                    entry_time=datetime.now(),
                    peak_price=current_price,
                    current_price=current_price
                )

            # Check trailing stop (12%)
            if self.check_trailing_stop(lookup_symbol, current_price, position):
                logger.info(f"TRAILING STOP triggered for {symbol}")
                self.close_position(symbol)
                continue

            # Get data for signal check
            df = self.get_bars(lookup_symbol)
            if df is not None and len(df) > 0:
                df = self.calculate_indicators(df)
                signal = self.generate_signal(df, lookup_symbol)

                if signal['signal'] == 'SELL':
                    logger.info(f"SELL signal for {symbol}: {signal['reason']}")
                    self.close_position(symbol)

        # ========== CHECK FOR NEW ENTRIES ==========
        # Refresh positions after exits
        positions = self.get_positions()
        current_position_count = len(positions)

        # Check if we can add more positions
        max_positions = cfg['max_positions']
        max_exposure = cfg['max_portfolio_exposure']

        total_exposure = sum(p['market_value'] for p in positions.values())
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0

        if current_position_count >= max_positions:
            logger.info(f"Max positions reached ({max_positions}), skipping new entries")
            return

        if exposure_pct >= max_exposure:
            logger.info(f"Max exposure reached ({exposure_pct:.1%} >= {max_exposure:.0%}), skipping new entries")
            return

        # Skip equity trading during restricted hours
        skip_equities = self.should_skip_trading()

        # Get symbols with pending orders to avoid duplicates
        pending_symbols = self.get_pending_orders()

        # Scan watchlist for entry opportunities
        for symbol in self.watchlist:
            # Skip if already in position OR has pending order
            clean_symbol = symbol.replace('/', '')
            if clean_symbol in positions or symbol in positions:
                continue
            if clean_symbol in pending_symbols or symbol in pending_symbols:
                logger.debug(f"Skipping {symbol}: pending order exists")
                continue

            # Skip equities during restricted hours
            is_crypto = '/' in symbol
            if not is_crypto and skip_equities:
                continue

            # Re-check limits
            if current_position_count >= max_positions:
                break
            if exposure_pct >= max_exposure:
                break

            # Get data
            df = self.get_bars(symbol)
            if df is None or len(df) < 50:
                continue

            df = self.calculate_indicators(df)
            signal = self.generate_signal(df, symbol)

            if signal['signal'] == 'BUY':
                # Calculate position size using CASH (not portfolio_value) to avoid leverage
                position_value = self.calculate_position_size(symbol, df, cash)

                # Check if this would exceed max exposure
                new_exposure = (total_exposure + position_value) / portfolio_value
                if new_exposure > max_exposure:
                    position_value = (max_exposure * portfolio_value) - total_exposure
                    if position_value <= 0:
                        continue

                # Calculate quantity (fractional shares supported)
                current_price = df.iloc[-2]['close']  # Use closed candle price
                qty = position_value / current_price

                if qty > 0:
                    logger.info(f"BUY signal for {symbol}: {signal['reason']}")

                    success = self.execute_trade(
                        symbol=symbol,
                        side='BUY',
                        qty=qty,
                        order_type=signal['order_type'],
                        limit_price=signal['limit_price']
                    )

                    if success:
                        # Initialize position state
                        self.position_state.update_position(
                            symbol=symbol,
                            entry_price=current_price,
                            entry_time=datetime.now(),
                            peak_price=current_price,
                            current_price=current_price
                        )

                        current_position_count += 1
                        total_exposure += position_value
                        exposure_pct = total_exposure / portfolio_value

        # Log final state
        logger.info("-" * 60)
        logger.info(f"Cycle complete. Positions: {current_position_count}, Exposure: {exposure_pct:.1%}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    bot = EnhancedTradingBot()
    bot.run_cycle()


if __name__ == '__main__':
    main()
