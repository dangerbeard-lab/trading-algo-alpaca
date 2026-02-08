#!/usr/bin/env python3
"""
Enhanced Trading Bot v3.1
=========================
Multi-asset hybrid strategy with down-market optimizations:
- Lookahead bias fix (uses closed candles only)
- State persistence (positions.json)
- Fractional share support
- Two-phase stop: initial 2x ATR stop + adaptive trailing stop
- ATR-based position sizing using CASH (not portfolio_value)
- Limit orders for mean reversion (with ADX stability check)
- SPY market trend filter (blocks equity longs in downtrends)
- Daily timeframe trend confirmation
- Portfolio drawdown circuit breaker
- Sector exposure limits (max correlated positions)
- ADX transitional zone (20-25) signals at half size
- Diversified watchlist: commodities, bonds, defensives
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
                        peak_price: float, current_price: float,
                        entry_atr: float = 0.0):
        """Update or create position state and save immediately."""
        new_peak = max(peak_price, current_price)

        existing = self.positions.get(symbol, {})
        self.positions[symbol] = {
            'entry_price': entry_price,
            'entry_time': entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
            'peak_price': new_peak,
            'highest_watermark': new_peak,
            'entry_atr': entry_atr if entry_atr > 0 else existing.get('entry_atr', 0.0),
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

            # Safety check: ADX must be stable or falling for mean reversion buys.
            # Rising ADX = transitioning to trending regime = falling knife risk.
            prev_adx = previous['adx']
            adx_rising = adx > prev_adx + 1.0  # ADX increased by >1 point

            # Oversold at lower Bollinger Band - use LIMIT ORDER
            if rsi < cfg['rsi_oversold'] and current['close'] <= current['bb_lower']:
                if adx_rising:
                    signal = {
                        'signal': 'HOLD',
                        'reason': f'Mean Reversion blocked: ADX rising ({prev_adx:.1f}->{adx:.1f}), regime may be shifting',
                        'order_type': 'market',
                        'limit_price': None
                    }
                else:
                    signal = {
                        'signal': 'BUY',
                        'reason': f'Mean Reversion BUY: RSI={rsi:.1f}, at BB lower (ADX={adx:.1f}, stable)',
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

        # ========== TRANSITIONAL ZONE: ADX 20-25 (cautious trend-following) ==========
        elif not is_trending and not is_ranging:
            # ADX between thresholds - market is ambiguous.
            # Allow trend signals but require ALL confirmations and flag for half size.
            ema_cross_up = (
                previous['ema_short'] <= previous['ema_long'] and
                current['ema_short'] > current['ema_long']
            )
            macd_bullish = current['macd'] > current['macd_signal']
            rsi_not_overbought = current['rsi'] < cfg['rsi_overbought']

            if ema_cross_up and macd_bullish and volume_confirmed and rsi_not_overbought:
                signal = {
                    'signal': 'BUY',
                    'reason': f'Cautious BUY: EMA+MACD+Vol+RSI all confirm (ADX={adx:.1f}, transitional)',
                    'order_type': 'market',
                    'limit_price': None,
                    'half_size': True  # Flag for run_cycle to use half position size
                }

            ema_cross_down = (
                previous['ema_short'] >= previous['ema_long'] and
                current['ema_short'] < current['ema_long']
            )
            macd_bearish = current['macd'] < current['macd_signal']

            if ema_cross_down and macd_bearish:
                signal = {
                    'signal': 'SELL',
                    'reason': f'Cautious SELL: EMA crossover + MACD (ADX={adx:.1f}, transitional)',
                    'order_type': 'market',
                    'limit_price': None
                }

        return signal

    def calculate_position_size(self, symbol: str, df: pd.DataFrame, account_value: float) -> float:
        """
        Calculate position size using ATR-based scaling.
        - Low volatility stocks: up to max_position_pct (10%)
        - High volatility stocks: near base_position_pct (4%)
        """
        cfg = self.config['risk_management']

        base_pct = cfg['base_position_pct']  # 4%
        max_pct = cfg['max_position_pct']    # 10%

        # Get ATR from closed candle
        current = df.iloc[-2]
        atr = current['atr']
        price = current['close']

        if pd.isna(atr) or atr <= 0 or price <= 0:
            return account_value * base_pct

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

        position_value = account_value * position_pct

        logger.debug(f"{symbol}: ATR%={atr_pct:.2%}, Position={position_pct:.1%} (${position_value:,.0f})")

        return position_value

    def check_trailing_stop(self, symbol: str, current_price: float, position: dict,
                             market_bearish: bool = False) -> bool:
        """
        Two-phase stop system:
        1. INITIAL STOP: 2x ATR below entry (tight, protects fresh entries)
        2. TRAILING STOP: adaptive % from peak (kicks in once position is profitable)

        The initial stop prevents the old problem where buying into an immediate
        drop required waiting for a full 12% decline before exiting.

        Once price exceeds entry, the trailing stop takes over with adaptive logic:
        - Base: trailing_stop_pct from config (12%)
        - Tightens to 8% when SPY is bearish
        - Ratchets tighter for positions with >15% unrealized gains
        """
        cfg = self.config['risk_management']
        base_stop_pct = cfg['trailing_stop_pct']  # 0.12 = 12%
        atr_multiplier = cfg.get('atr_risk_multiplier', 2.0)

        entry_price = position['avg_entry_price']
        unrealized_plpc = position.get('unrealized_plpc', 0)

        # Get position state for entry_atr
        pos_state = self.position_state.get_position(symbol)
        entry_atr = pos_state.get('entry_atr', 0.0) if pos_state else 0.0

        # ---- PHASE 1: Initial ATR stop (for positions not yet in profit) ----
        if entry_atr > 0 and current_price <= entry_price:
            initial_stop = entry_price - (atr_multiplier * entry_atr)
            if current_price <= initial_stop:
                loss_pct = (entry_price - current_price) / entry_price
                logger.info(f"{symbol}: INITIAL STOP hit! Entry=${entry_price:.2f}, "
                           f"Current=${current_price:.2f}, Stop=${initial_stop:.2f} "
                           f"(2x ATR=${entry_atr:.2f}, loss={loss_pct:.1%})")
                return True

        # ---- PHASE 2: Adaptive trailing stop (for all positions) ----
        trailing_stop_pct = base_stop_pct

        # Tighten stop in bearish market: 12% -> 8%
        if market_bearish:
            trailing_stop_pct = min(trailing_stop_pct, 0.08)

        # Ratchet tighter for positions with gains > 15% (protect profits)
        if unrealized_plpc > 0.15:
            trailing_stop_pct = min(trailing_stop_pct, max(0.06, base_stop_pct - unrealized_plpc * 0.2))

        # Get or initialize peak price from persistent state
        peak_price = self.position_state.get_peak_price(symbol, current_price)

        # Update peak if current price is higher
        if current_price > peak_price:
            peak_price = current_price
            self.position_state.update_position(
                symbol=symbol,
                entry_price=entry_price,
                entry_time=datetime.now(),
                peak_price=peak_price,
                current_price=current_price
            )

        # Calculate trailing stop level
        stop_level = peak_price * (1 - trailing_stop_pct)

        if current_price <= stop_level:
            drawdown_from_peak = (peak_price - current_price) / peak_price
            logger.info(f"{symbol}: Trailing stop hit! Peak=${peak_price:.2f}, "
                       f"Current=${current_price:.2f}, Stop=${stop_level:.2f} "
                       f"(Down {drawdown_from_peak:.1%} from peak, "
                       f"adaptive stop={trailing_stop_pct:.0%}"
                       f"{', BEARISH MARKET' if market_bearish else ''})")
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

    def check_market_trend(self) -> dict:
        """
        Check broad market health using SPY daily bars.
        Returns dict with market state info used to gate new entries.
        """
        result = {
            'bullish': True,  # default permissive if data unavailable
            'spy_above_ema': True,
            'spy_ema_slope_up': True,
            'reason': ''
        }
        try:
            df = self.get_bars('SPY', timeframe='1Day', limit=50)
            if df is None or len(df) < 30:
                logger.warning("Could not fetch SPY daily data for market filter, defaulting to permissive")
                return result

            df = df.copy()
            close = df['close']
            ema_21 = TechnicalIndicators.ema(close, 21)
            ema_50 = TechnicalIndicators.ema(close, 50)

            # Use last closed daily bar
            current_close = close.iloc[-2]
            current_ema21 = ema_21.iloc[-2]
            current_ema50 = ema_50.iloc[-2]
            prev_ema21 = ema_21.iloc[-3]

            spy_above_ema = current_close > current_ema21
            ema_slope_up = current_ema21 > prev_ema21
            ema21_above_ema50 = current_ema21 > current_ema50

            # Market is bearish if price is below EMA21 AND EMA21 is sloping down
            is_bullish = spy_above_ema or ema_slope_up

            result = {
                'bullish': is_bullish,
                'spy_above_ema': spy_above_ema,
                'spy_ema_slope_up': ema_slope_up,
                'ema21_above_ema50': ema21_above_ema50,
                'reason': (
                    f"SPY={'above' if spy_above_ema else 'BELOW'} EMA21, "
                    f"slope={'up' if ema_slope_up else 'DOWN'}, "
                    f"EMA21 {'>' if ema21_above_ema50 else '<'} EMA50"
                )
            }

            if not is_bullish:
                logger.warning(f"MARKET FILTER: Bearish regime detected - {result['reason']}")
            else:
                logger.info(f"Market filter: {result['reason']}")

        except Exception as e:
            logger.warning(f"Market trend check failed: {e}, defaulting to permissive")

        return result

    def check_daily_trend(self, symbol: str) -> Optional[str]:
        """
        Check the daily trend direction for a symbol.
        Returns 'up', 'down', or None if data unavailable.
        Prevents buying 15-min signals against the daily trend.
        """
        try:
            df = self.get_bars(symbol, timeframe='1Day', limit=30)
            if df is None or len(df) < 22:
                return None

            df = df.copy()
            close = df['close']
            ema_21 = TechnicalIndicators.ema(close, 21)

            current_close = close.iloc[-2]
            current_ema = ema_21.iloc[-2]
            prev_ema = ema_21.iloc[-3]

            if current_close > current_ema and current_ema > prev_ema:
                return 'up'
            elif current_close < current_ema and current_ema < prev_ema:
                return 'down'
            return None  # indeterminate

        except Exception as e:
            logger.debug(f"Daily trend check failed for {symbol}: {e}")
            return None

    def check_drawdown_circuit_breaker(self, account: dict) -> bool:
        """
        Check if portfolio drawdown exceeds max_drawdown_pct.
        Returns True if new entries should be blocked.
        Uses the persistent metrics peak_portfolio_value to track the high-water mark.
        """
        cfg = self.config['risk_management']
        max_dd = cfg.get('max_drawdown_pct', 0.15)

        portfolio_value = account['portfolio_value']

        # Load metrics to get peak portfolio value
        metrics_file = self.config['persistence'].get('metrics_file', 'trading_metrics.json')
        peak_value = portfolio_value  # default if no history
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                peak_value = max(metrics.get('peak_portfolio_value', portfolio_value), portfolio_value)
            except (json.JSONDecodeError, IOError):
                pass

        if peak_value <= 0:
            return False

        current_dd = (peak_value - portfolio_value) / peak_value
        if current_dd >= max_dd:
            logger.warning(f"CIRCUIT BREAKER: Portfolio drawdown {current_dd:.1%} exceeds "
                          f"max {max_dd:.0%} (peak=${peak_value:,.0f}, current=${portfolio_value:,.0f}). "
                          f"Blocking new entries.")
            return True

        if current_dd > max_dd * 0.7:
            logger.info(f"Drawdown warning: {current_dd:.1%} approaching limit of {max_dd:.0%}")

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

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Map a symbol back to its watchlist sector for correlation limits."""
        for sector, symbols in self.config['watchlist'].items():
            if symbol in symbols:
                return sector
        return 'unknown'

    def _count_sector_positions(self, positions: Dict[str, dict]) -> Dict[str, int]:
        """Count how many current positions belong to each sector."""
        sector_counts: Dict[str, int] = {}
        for symbol in positions:
            # Normalize crypto symbols for lookup
            lookup = symbol if '/' in symbol else symbol
            # Also try with slash for crypto
            sector = self._get_sector_for_symbol(lookup)
            if sector == 'unknown':
                # Try crypto format
                for s in self.crypto_symbols:
                    if s.replace('/', '') == symbol:
                        sector = self._get_sector_for_symbol(s)
                        break
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts

    def run_cycle(self):
        """Run one complete trading cycle."""
        logger.info("=" * 60)
        logger.info("STARTING TRADING CYCLE")
        logger.info("=" * 60)

        # Get account info
        account = self.get_account()
        portfolio_value = account['portfolio_value']
        cash = account['cash']
        logger.info(f"Portfolio Value: ${portfolio_value:,.2f} | Cash: ${cash:,.2f}")

        # Get current positions
        positions = self.get_positions()
        current_position_count = len(positions)
        logger.info(f"Current Positions: {current_position_count}")

        # Calculate current exposure
        total_exposure = sum(p['market_value'] for p in positions.values())
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
        logger.info(f"Current Exposure: {exposure_pct:.1%}")

        cfg = self.config['risk_management']

        # MARKET FILTER: Check broad market trend (SPY) - used for both exits and entries
        market = self.check_market_trend()
        market_bearish = not market['bullish']

        # ========== CHECK EXISTING POSITIONS FOR EXITS ==========
        for symbol, position in positions.items():
            # Normalize symbol for matching
            lookup_symbol = symbol if '/' not in symbol else f"{symbol[:3]}/{symbol[3:]}"

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

            # ADAPTIVE trailing stop: tightens in bearish market
            if self.check_trailing_stop(lookup_symbol, current_price, position,
                                        market_bearish=market_bearish):
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

        # ========== PRE-ENTRY CHECKS ==========
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

        # CIRCUIT BREAKER: Check portfolio drawdown before opening new positions
        if self.check_drawdown_circuit_breaker(account):
            logger.warning("Circuit breaker active - only managing exits, no new entries")
            return

        # Skip equity trading during restricted hours
        skip_equities = self.should_skip_trading()

        # Get symbols with pending orders to avoid duplicates
        pending_symbols = self.get_pending_orders()

        # Sector exposure tracking
        sector_counts = self._count_sector_positions(positions)
        max_sector = cfg.get('max_correlated_positions', 5)

        # Refresh cash after exits (account may have changed)
        account = self.get_account()
        cash = account['cash']

        # ========== SCAN FOR NEW ENTRIES ==========
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

            # MARKET FILTER: Block new equity longs when SPY is bearish
            if not is_crypto and not market['bullish']:
                logger.debug(f"Skipping {symbol}: market filter bearish ({market['reason']})")
                continue

            # SECTOR LIMIT: Check if this sector is already at max
            sector = self._get_sector_for_symbol(symbol)
            if sector_counts.get(sector, 0) >= max_sector:
                logger.debug(f"Skipping {symbol}: sector '{sector}' at max ({max_sector} positions)")
                continue

            # Get data
            df = self.get_bars(symbol)
            if df is None or len(df) < 50:
                continue

            df = self.calculate_indicators(df)
            signal = self.generate_signal(df, symbol)

            if signal['signal'] == 'BUY':
                # DAILY TREND FILTER: Don't buy against daily downtrend
                daily_trend = self.check_daily_trend(symbol)
                if daily_trend == 'down':
                    logger.info(f"Skipping {symbol} BUY: daily trend is DOWN (counter-trend filter)")
                    continue

                # FIX: Size positions using CASH, not portfolio_value (avoid margin leverage)
                position_value = self.calculate_position_size(symbol, df, cash)

                # Half size for transitional zone (ADX 20-25) signals
                if signal.get('half_size', False):
                    position_value *= 0.5
                    logger.info(f"Half position for {symbol}: ADX transitional zone")

                # Reduce position size when market is uncertain (not strongly bullish)
                if not market.get('ema21_above_ema50', True):
                    position_value *= 0.6
                    logger.info(f"Reduced position size for {symbol}: SPY EMA21 < EMA50")

                # Check if this would exceed max exposure
                new_exposure = (total_exposure + position_value) / portfolio_value
                if new_exposure > max_exposure:
                    position_value = (max_exposure * portfolio_value) - total_exposure
                    if position_value <= 0:
                        continue

                # Don't exceed available cash
                if position_value > cash * 0.95:
                    position_value = cash * 0.95
                    if position_value <= 0:
                        continue

                # Calculate quantity (fractional shares supported)
                current_bar = df.iloc[-2]
                current_price = current_bar['close']  # Use closed candle price
                entry_atr = current_bar['atr'] if not pd.isna(current_bar['atr']) else 0.0
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
                        # Initialize position state with entry ATR for initial stop
                        self.position_state.update_position(
                            symbol=symbol,
                            entry_price=current_price,
                            entry_time=datetime.now(),
                            peak_price=current_price,
                            current_price=current_price,
                            entry_atr=entry_atr
                        )

                        current_position_count += 1
                        total_exposure += position_value
                        exposure_pct = total_exposure / portfolio_value
                        cash -= position_value
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1

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
