#!/usr/bin/env python3
"""
Enhanced Hybrid Trading Algorithm for Alpaca
============================================
Improvements implemented:
1. Trailing stops (configurable % from peak)
2. Volatility-adjusted position sizing (ATR-based)
3. Correlation filtering (reduces exposure when positions correlate)
4. ADX threshold hysteresis (separate entry/exit thresholds)
5. Volume confirmation (requires above-average volume)
6. Time-of-day filtering (avoids volatile open/close periods)
7. Comprehensive logging and performance metrics
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Central configuration for the trading algorithm."""
    
    # API Settings
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True
    
    # Instruments
    crypto_symbols: List[str] = field(default_factory=lambda: ["BTC/USD"])
    etf_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    stock_symbols: List[str] = field(default_factory=lambda: [
        "NVDA", "META", "AMZN", "GOOGL", "MSFT", 
        "AAPL", "TSLA", "AMD", "NFLX", "AVGO", "MSTR"
    ])
    
    # Position Sizing
    base_position_pct: float = 0.10  # 10% base position size
    max_positions: int = 8
    max_portfolio_exposure: float = 0.80  # 80% max exposure
    
    # ATR-based position sizing
    atr_period: int = 14
    atr_target_risk: float = 0.02  # Target 2% risk per trade based on ATR
    min_position_pct: float = 0.03  # Minimum 3% position
    max_position_pct: float = 0.15  # Maximum 15% position
    
    # Risk Management
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    take_profit_pct: float = 0.20  # 20% take profit
    
    # Trailing Stop Settings
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.12  # 12% trailing stop from peak
    trailing_stop_activation_pct: float = 0.05  # Activate after 5% profit
    
    # Technical Indicators
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # ADX Hysteresis Settings
    adx_period: int = 14
    adx_trending_entry: int = 25  # ADX above this = trending market
    adx_trending_exit: int = 20   # ADX must fall below this to switch back
    
    # Volume Confirmation
    volume_confirmation_enabled: bool = True
    volume_period: int = 20
    volume_multiplier: float = 1.2  # Require 1.2x average volume
    
    # Correlation Filtering
    correlation_enabled: bool = True
    correlation_lookback: int = 30  # Days for correlation calculation
    correlation_threshold: float = 0.7  # High correlation threshold
    max_correlated_positions: int = 3  # Max positions with high correlation
    
    # Time-of-Day Filtering (for equities)
    time_filter_enabled: bool = True
    market_open_buffer_minutes: int = 30  # Avoid first 30 mins
    market_close_buffer_minutes: int = 30  # Avoid last 30 mins
    
    # Logging
    log_file: str = "trading_bot.log"
    metrics_file: str = "trading_metrics.json"


# =============================================================================
# PERFORMANCE METRICS TRACKING
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    quantity: float
    pnl: float
    pnl_pct: float
    r_multiple: float  # Profit in terms of initial risk
    regime: str  # trending or ranging
    exit_reason: str  # take_profit, trailing_stop, signal_exit, etc.


@dataclass
class PositionTracker:
    """Tracks open position with trailing stop logic."""
    symbol: str
    entry_price: float
    entry_time: str
    quantity: float
    side: str  # 'long' or 'short'
    peak_price: float  # Highest price since entry (for longs)
    trough_price: float  # Lowest price since entry (for shorts)
    initial_stop: float
    regime: str
    trailing_active: bool = False


class MetricsTracker:
    """Tracks and calculates trading performance metrics."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trades: List[TradeRecord] = []
        self.daily_returns: List[float] = []
        self.peak_equity: float = 0
        self.current_drawdown: float = 0
        self.max_drawdown: float = 0
        self.metrics_by_symbol: Dict[str, Dict] = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_r': 0
        })
        self.metrics_by_regime: Dict[str, Dict] = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'total_pnl': 0
        })
        self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics from file."""
        if os.path.exists(self.config.metrics_file):
            try:
                with open(self.config.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data.get('trades', [])]
                    self.daily_returns = data.get('daily_returns', [])
                    self.peak_equity = data.get('peak_equity', 0)
                    self.max_drawdown = data.get('max_drawdown', 0)
                    
                    # Rebuild per-symbol and per-regime metrics
                    for trade in self.trades:
                        self._update_aggregates(trade)
            except Exception as e:
                logging.warning(f"Could not load metrics: {e}")
    
    def _update_aggregates(self, trade: TradeRecord):
        """Update aggregate metrics from a trade."""
        # Per-symbol
        sym = self.metrics_by_symbol[trade.symbol]
        sym['trades'] += 1
        sym['wins'] += 1 if trade.pnl > 0 else 0
        sym['total_pnl'] += trade.pnl
        sym['total_r'] += trade.r_multiple
        
        # Per-regime
        reg = self.metrics_by_regime[trade.regime]
        reg['trades'] += 1
        reg['wins'] += 1 if trade.pnl > 0 else 0
        reg['total_pnl'] += trade.pnl
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade."""
        self.trades.append(trade)
        self._update_aggregates(trade)
        self._save_metrics()
        
        logging.info(
            f"TRADE CLOSED: {trade.symbol} | PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2%}) | "
            f"R-Multiple: {trade.r_multiple:.2f} | Exit: {trade.exit_reason}"
        )
    
    def update_equity(self, current_equity: float):
        """Update equity tracking for drawdown calculation."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def add_daily_return(self, daily_return: float):
        """Add a daily return for Sharpe calculation."""
        self.daily_returns.append(daily_return)
        self._save_metrics()
    
    def calculate_sharpe(self, risk_free_rate: float = 0.05) -> float:
        """Calculate annualised Sharpe ratio."""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        excess_returns = returns - (risk_free_rate / 252)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def get_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        total_trades = len(self.trades)
        if total_trades == 0:
            return {'message': 'No trades recorded yet'}
        
        wins = sum(1 for t in self.trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self.trades)
        avg_r = np.mean([t.r_multiple for t in self.trades])
        
        return {
            'total_trades': total_trades,
            'win_rate': wins / total_trades,
            'total_pnl': total_pnl,
            'average_r_multiple': avg_r,
            'sharpe_ratio': self.calculate_sharpe(),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'by_symbol': dict(self.metrics_by_symbol),
            'by_regime': dict(self.metrics_by_regime)
        }
    
    def _save_metrics(self):
        """Save metrics to file."""
        data = {
            'trades': [asdict(t) for t in self.trades],
            'daily_returns': self.daily_returns,
            'peak_equity': self.peak_equity,
            'max_drawdown': self.max_drawdown,
            'summary': self.get_summary()
        }
        with open(self.config.metrics_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# CORRELATION MANAGER
# =============================================================================

class CorrelationManager:
    """Manages correlation calculations and filtering."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
    
    def update_correlations(self, price_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix from recent price data."""
        if not self.config.correlation_enabled:
            return
        
        # Build returns DataFrame
        returns_dict = {}
        for symbol, df in price_data.items():
            if len(df) >= self.config.correlation_lookback:
                returns_dict[symbol] = df['close'].pct_change().dropna().tail(
                    self.config.correlation_lookback
                )
        
        if len(returns_dict) < 2:
            return
        
        returns_df = pd.DataFrame(returns_dict).dropna()
        if len(returns_df) >= 10:
            self.correlation_matrix = returns_df.corr()
            self.last_update = datetime.now()
            logging.debug(f"Updated correlation matrix for {len(returns_dict)} symbols")
    
    def get_correlated_symbols(self, symbol: str, current_positions: List[str]) -> List[str]:
        """Get list of current positions highly correlated with symbol."""
        if self.correlation_matrix is None or symbol not in self.correlation_matrix.columns:
            return []
        
        correlated = []
        for pos_symbol in current_positions:
            if pos_symbol in self.correlation_matrix.columns and pos_symbol != symbol:
                corr = abs(self.correlation_matrix.loc[symbol, pos_symbol])
                if corr >= self.config.correlation_threshold:
                    correlated.append(pos_symbol)
        
        return correlated
    
    def can_open_position(self, symbol: str, current_positions: List[str]) -> Tuple[bool, str]:
        """Check if opening a position would violate correlation limits."""
        if not self.config.correlation_enabled:
            return True, ""
        
        correlated = self.get_correlated_symbols(symbol, current_positions)
        
        if len(correlated) >= self.config.max_correlated_positions:
            return False, f"Would exceed correlated position limit (correlated with: {correlated})"
        
        return True, ""


# =============================================================================
# MAIN TRADING BOT
# =============================================================================

class EnhancedTradingBot:
    """Enhanced hybrid trading algorithm with all improvements."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self._setup_logging()
        
        # API clients
        self.trading_client = TradingClient(
            config.api_key, 
            config.api_secret, 
            paper=config.paper
        )
        self.stock_client = StockHistoricalDataClient(config.api_key, config.api_secret)
        self.crypto_client = CryptoHistoricalDataClient(config.api_key, config.api_secret)
        
        # State tracking
        self.positions: Dict[str, PositionTracker] = {}
        self.regime_state: Dict[str, str] = {}  # 'trending' or 'ranging'
        self.metrics = MetricsTracker(config)
        self.correlation_manager = CorrelationManager(config)
        
        # All symbols
        self.all_equity_symbols = config.etf_symbols + config.stock_symbols
        self.all_symbols = config.crypto_symbols + self.all_equity_symbols
        
        logging.info("Enhanced Trading Bot initialised")
        logging.info(f"Trading {len(self.all_symbols)} instruments")
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
    
    # -------------------------------------------------------------------------
    # Data Fetching
    # -------------------------------------------------------------------------
    
    def fetch_stock_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Fetch historical stock/ETF data."""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=datetime.now() - timedelta(days=days),
                feed='iex'
            )
            bars = self.stock_client.get_stock_bars(request)
            
            if symbol not in bars.data or len(bars.data[symbol]) == 0:
                return None
            
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars.data[symbol]])
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def fetch_crypto_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Fetch historical crypto data."""
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=datetime.now() - timedelta(days=days)
            )
            bars = self.crypto_client.get_crypto_bars(request)
            
            if symbol not in bars.data or len(bars.data[symbol]) == 0:
                return None
            
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars.data[symbol]])
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"Error fetching crypto data for {symbol}: {e}")
            return None
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols."""
        data = {}
        
        for symbol in self.config.crypto_symbols:
            df = self.fetch_crypto_data(symbol)
            if df is not None:
                data[symbol] = df
        
        for symbol in self.all_equity_symbols:
            df = self.fetch_stock_data(symbol)
            if df is not None:
                data[symbol] = df
        
        return data
    
    # -------------------------------------------------------------------------
    # Technical Indicators
    # -------------------------------------------------------------------------
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.config.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config.ema_slow, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.config.bb_period).mean()
        bb_std = df['close'].rolling(window=self.config.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (self.config.bb_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (self.config.bb_std * bb_std)
        
        # MACD
        ema_fast = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX
        df = self._calculate_adx(df)
        
        # ATR
        df = self._calculate_atr(df)
        
        # Volume metrics
        df['volume_sma'] = df['volume'].rolling(window=self.config.volume_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX indicator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=self.config.adx_period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=self.config.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.config.adx_period).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
        df['adx'] = dx.rolling(window=self.config.adx_period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        df['atr'] = tr.rolling(window=self.config.atr_period).mean()
        df['atr_pct'] = df['atr'] / df['close']  # ATR as percentage of price
        
        return df
    
    # -------------------------------------------------------------------------
    # Regime Detection with Hysteresis
    # -------------------------------------------------------------------------
    
    def determine_regime(self, symbol: str, adx: float) -> str:
        """
        Determine market regime with hysteresis to prevent whipsawing.
        Uses separate entry/exit thresholds.
        """
        current_regime = self.regime_state.get(symbol, 'ranging')
        
        if current_regime == 'ranging':
            # Need ADX above entry threshold to switch to trending
            if adx >= self.config.adx_trending_entry:
                new_regime = 'trending'
                logging.info(f"{symbol}: Regime change RANGING -> TRENDING (ADX: {adx:.1f})")
            else:
                new_regime = 'ranging'
        else:  # currently trending
            # Need ADX below exit threshold to switch back to ranging
            if adx < self.config.adx_trending_exit:
                new_regime = 'ranging'
                logging.info(f"{symbol}: Regime change TRENDING -> RANGING (ADX: {adx:.1f})")
            else:
                new_regime = 'trending'
        
        self.regime_state[symbol] = new_regime
        return new_regime
    
    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[str]:
        """Generate trading signal based on regime and indicators."""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check volume confirmation
        if self.config.volume_confirmation_enabled:
            if latest['volume_ratio'] < self.config.volume_multiplier:
                logging.debug(f"{symbol}: Volume too low ({latest['volume_ratio']:.2f}x avg)")
                return None
        
        # Check time-of-day filter for equities
        if symbol in self.all_equity_symbols and self.config.time_filter_enabled:
            if not self._is_valid_trading_time():
                return None
        
        # Determine regime
        regime = self.determine_regime(symbol, latest['adx'])
        
        if regime == 'trending':
            return self._trending_signal(symbol, latest, prev)
        else:
            return self._ranging_signal(symbol, latest, prev)
    
    def _trending_signal(self, symbol: str, latest: pd.Series, prev: pd.Series) -> Optional[str]:
        """Generate signal for trending market (EMA crossover + MACD confirmation)."""
        # EMA crossover
        ema_cross_up = (prev['ema_fast'] <= prev['ema_slow']) and (latest['ema_fast'] > latest['ema_slow'])
        ema_cross_down = (prev['ema_fast'] >= prev['ema_slow']) and (latest['ema_fast'] < latest['ema_slow'])
        
        # MACD confirmation
        macd_bullish = latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0
        macd_bearish = latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0
        
        # Directional movement confirmation
        di_bullish = latest['plus_di'] > latest['minus_di']
        di_bearish = latest['minus_di'] > latest['plus_di']
        
        if ema_cross_up and macd_bullish and di_bullish:
            logging.info(f"{symbol} [TRENDING]: BUY signal - EMA crossover + MACD + DI confirm")
            return 'buy'
        elif ema_cross_down and macd_bearish and di_bearish:
            logging.info(f"{symbol} [TRENDING]: SELL signal - EMA crossover + MACD + DI confirm")
            return 'sell'
        
        return None
    
    def _ranging_signal(self, symbol: str, latest: pd.Series, prev: pd.Series) -> Optional[str]:
        """Generate signal for ranging market (RSI + Bollinger Bands mean reversion)."""
        # RSI conditions
        rsi_oversold = latest['rsi'] < self.config.rsi_oversold
        rsi_overbought = latest['rsi'] > self.config.rsi_overbought
        
        # Bollinger Band conditions
        price_below_lower = latest['close'] < latest['bb_lower']
        price_above_upper = latest['close'] > latest['bb_upper']
        
        # RSI turning (momentum shift)
        rsi_turning_up = prev['rsi'] < latest['rsi'] and latest['rsi'] < 40
        rsi_turning_down = prev['rsi'] > latest['rsi'] and latest['rsi'] > 60
        
        if rsi_oversold and price_below_lower and rsi_turning_up:
            logging.info(f"{symbol} [RANGING]: BUY signal - RSI oversold + below BB lower")
            return 'buy'
        elif rsi_overbought and price_above_upper and rsi_turning_down:
            logging.info(f"{symbol} [RANGING]: SELL signal - RSI overbought + above BB upper")
            return 'sell'
        
        return None
    
    def _is_valid_trading_time(self) -> bool:
        """Check if current time is within valid trading window (avoids open/close)."""
        now = datetime.now()
        current_time = now.time()
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # Buffer periods
        open_buffer_end = time(
            9, 30 + self.config.market_open_buffer_minutes
        )
        close_buffer_start = time(
            16 - (self.config.market_close_buffer_minutes // 60),
            60 - (self.config.market_close_buffer_minutes % 60) if self.config.market_close_buffer_minutes % 60 != 0 else 0
        )
        
        # Simplified: avoid first and last 30 minutes
        if current_time < open_buffer_end:
            logging.debug("Skipping trade: within market open buffer")
            return False
        if current_time > time(15, 30):  # After 3:30 PM
            logging.debug("Skipping trade: within market close buffer")
            return False
        
        return True
    
    # -------------------------------------------------------------------------
    # Position Sizing (Volatility-Adjusted)
    # -------------------------------------------------------------------------
    
    def calculate_position_size(self, symbol: str, df: pd.DataFrame, account_value: float) -> float:
        """
        Calculate position size adjusted for volatility (ATR-based).
        Higher volatility = smaller position size.
        """
        if len(df) < self.config.atr_period:
            return account_value * self.config.base_position_pct
        
        latest = df.iloc[-1]
        atr_pct = latest['atr_pct']
        
        if pd.isna(atr_pct) or atr_pct <= 0:
            return account_value * self.config.base_position_pct
        
        # Target risk-based position sizing
        # If ATR is 2% and we want 2% risk, position = 100%
        # If ATR is 4% and we want 2% risk, position = 50%
        raw_position_pct = self.config.atr_target_risk / atr_pct
        
        # Clamp to min/max bounds
        position_pct = max(
            self.config.min_position_pct,
            min(self.config.max_position_pct, raw_position_pct)
        )
        
        position_value = account_value * position_pct
        
        logging.debug(
            f"{symbol}: ATR={atr_pct:.2%}, Raw size={raw_position_pct:.2%}, "
            f"Clamped={position_pct:.2%}, Value=${position_value:.2f}"
        )
        
        return position_value
    
    # -------------------------------------------------------------------------
    # Trailing Stop Management
    # -------------------------------------------------------------------------
    
    def update_trailing_stops(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Update trailing stops and return list of symbols to close.
        """
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            if position.side == 'long':
                # Update peak price
                if current_price > position.peak_price:
                    position.peak_price = current_price
                
                # Check if trailing stop should activate
                profit_pct = (current_price - position.entry_price) / position.entry_price
                if profit_pct >= self.config.trailing_stop_activation_pct:
                    position.trailing_active = True
                
                # Check trailing stop hit
                if position.trailing_active and self.config.trailing_stop_enabled:
                    trailing_stop_price = position.peak_price * (1 - self.config.trailing_stop_pct)
                    if current_price <= trailing_stop_price:
                        logging.info(
                            f"{symbol}: Trailing stop triggered at ${current_price:.2f} "
                            f"(peak: ${position.peak_price:.2f}, stop: ${trailing_stop_price:.2f})"
                        )
                        symbols_to_close.append(symbol)
                        continue
                
                # Check take profit
                if profit_pct >= self.config.take_profit_pct:
                    logging.info(f"{symbol}: Take profit triggered at {profit_pct:.2%}")
                    symbols_to_close.append(symbol)
                    continue
                
                # Check initial stop loss
                loss_pct = (position.entry_price - current_price) / position.entry_price
                if loss_pct >= self.config.max_drawdown_pct:
                    logging.info(f"{symbol}: Stop loss triggered at {loss_pct:.2%}")
                    symbols_to_close.append(symbol)
        
        return symbols_to_close
    
    # -------------------------------------------------------------------------
    # Order Execution
    # -------------------------------------------------------------------------
    
    def execute_buy(self, symbol: str, quantity: float, regime: str) -> bool:
        """Execute a buy order."""
        try:
            # Determine if crypto or equity
            is_crypto = symbol in self.config.crypto_symbols
            
            order_request = MarketOrderRequest(
                symbol=symbol.replace("/", "") if is_crypto else symbol,
                qty=quantity if is_crypto else int(quantity),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            logging.info(f"BUY ORDER submitted: {symbol} x {quantity}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error executing buy for {symbol}: {e}")
            return False
    
    def execute_sell(self, symbol: str, quantity: float) -> bool:
        """Execute a sell order."""
        try:
            is_crypto = symbol in self.config.crypto_symbols
            
            order_request = MarketOrderRequest(
                symbol=symbol.replace("/", "") if is_crypto else symbol,
                qty=quantity if is_crypto else int(quantity),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            logging.info(f"SELL ORDER submitted: {symbol} x {quantity}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error executing sell for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, reason: str, current_price: float) -> bool:
        """Close a position and record the trade."""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # Execute the close
        success = self.execute_sell(symbol, position.quantity)
        
        if success:
            # Calculate PnL
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            # Calculate R-multiple (profit relative to initial risk)
            initial_risk = position.entry_price - position.initial_stop
            if initial_risk > 0:
                r_multiple = (current_price - position.entry_price) / initial_risk
            else:
                r_multiple = pnl_pct / self.config.max_drawdown_pct
            
            # Record the trade
            trade = TradeRecord(
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=current_price,
                entry_time=position.entry_time,
                exit_time=datetime.now().isoformat(),
                quantity=position.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                r_multiple=r_multiple,
                regime=position.regime,
                exit_reason=reason
            )
            self.metrics.record_trade(trade)
            
            # Remove from positions
            del self.positions[symbol]
        
        return success
    
    # -------------------------------------------------------------------------
    # Main Run Loop
    # -------------------------------------------------------------------------
    
    def run_once(self):
        """Execute one iteration of the trading logic."""
        logging.info("=" * 60)
        logging.info("Starting trading iteration")
        
        # Get account info
        account = self.trading_client.get_account()
        account_value = float(account.portfolio_value)
        buying_power = float(account.buying_power)
        
        # Update equity tracking
        self.metrics.update_equity(account_value)
        
        logging.info(f"Account value: ${account_value:,.2f} | Buying power: ${buying_power:,.2f}")
        logging.info(f"Current drawdown: {self.metrics.current_drawdown:.2%} | Max: {self.metrics.max_drawdown:.2%}")
        
        # Check max drawdown limit
        if self.metrics.current_drawdown >= self.config.max_drawdown_pct:
            logging.warning("MAX DRAWDOWN REACHED - Closing all positions")
            self._close_all_positions("max_drawdown")
            return
        
        # Fetch all data
        all_data = self.fetch_all_data()
        logging.info(f"Fetched data for {len(all_data)} symbols")
        
        # Update correlation matrix
        self.correlation_manager.update_correlations(all_data)
        
        # Get current prices
        current_prices = {
            symbol: df.iloc[-1]['close'] 
            for symbol, df in all_data.items() 
            if len(df) > 0
        }
        
        # Sync positions with broker
        self._sync_positions(current_prices)
        
        # Check trailing stops
        symbols_to_close = self.update_trailing_stops(current_prices)
        for symbol in symbols_to_close:
            if symbol in current_prices:
                self.close_position(symbol, "trailing_stop", current_prices[symbol])
        
        # Calculate current exposure
        current_positions = list(self.positions.keys())
        num_positions = len(current_positions)
        
        # Process each symbol
        for symbol, df in all_data.items():
            # Skip if we already have a position
            if symbol in self.positions:
                continue
            
            # Skip if at position limit
            if num_positions >= self.config.max_positions:
                logging.debug(f"Position limit reached ({num_positions}/{self.config.max_positions})")
                break
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Generate signal
            signal = self.generate_signal(symbol, df)
            
            if signal == 'buy':
                # Check correlation filter
                can_open, reason = self.correlation_manager.can_open_position(
                    symbol, current_positions
                )
                if not can_open:
                    logging.info(f"{symbol}: Skipping due to correlation filter - {reason}")
                    continue
                
                # Calculate position size
                position_value = self.calculate_position_size(symbol, df, account_value)
                
                # Check portfolio exposure limit
                current_exposure = sum(
                    p.quantity * current_prices.get(p.symbol, p.entry_price)
                    for p in self.positions.values()
                )
                if (current_exposure + position_value) / account_value > self.config.max_portfolio_exposure:
                    logging.info(f"{symbol}: Skipping - would exceed max portfolio exposure")
                    continue
                
                # Calculate quantity
                current_price = current_prices[symbol]
                quantity = position_value / current_price
                
                # Round appropriately
                if symbol not in self.config.crypto_symbols:
                    quantity = int(quantity)
                    if quantity < 1:
                        continue
                
                # Execute trade
                regime = self.regime_state.get(symbol, 'ranging')
                if self.execute_buy(symbol, quantity, regime):
                    # Track position
                    initial_stop = current_price * (1 - self.config.max_drawdown_pct)
                    self.positions[symbol] = PositionTracker(
                        symbol=symbol,
                        entry_price=current_price,
                        entry_time=datetime.now().isoformat(),
                        quantity=quantity,
                        side='long',
                        peak_price=current_price,
                        trough_price=current_price,
                        initial_stop=initial_stop,
                        regime=regime
                    )
                    num_positions += 1
                    current_positions.append(symbol)
        
        # Log summary
        logging.info(f"Iteration complete | Positions: {num_positions} | " 
                    f"Symbols: {list(self.positions.keys())}")
    
    def _sync_positions(self, current_prices: Dict[str, float]):
        """Sync internal position tracking with broker positions."""
        try:
            broker_positions = self.trading_client.get_all_positions()
            broker_symbols = set()
            
            for pos in broker_positions:
                symbol = pos.symbol
                # Handle crypto symbols
                if symbol == "BTCUSD":
                    symbol = "BTC/USD"
                
                broker_symbols.add(symbol)
                
                # Update our tracking if we don't have it
                if symbol not in self.positions:
                    self.positions[symbol] = PositionTracker(
                        symbol=symbol,
                        entry_price=float(pos.avg_entry_price),
                        entry_time=datetime.now().isoformat(),
                        quantity=float(pos.qty),
                        side='long' if float(pos.qty) > 0 else 'short',
                        peak_price=float(pos.current_price),
                        trough_price=float(pos.current_price),
                        initial_stop=float(pos.avg_entry_price) * (1 - self.config.max_drawdown_pct),
                        regime=self.regime_state.get(symbol, 'unknown')
                    )
            
            # Remove positions we're tracking but broker doesn't have
            closed = [s for s in self.positions if s not in broker_symbols]
            for symbol in closed:
                logging.info(f"Position {symbol} closed externally")
                del self.positions[symbol]
                
        except Exception as e:
            logging.error(f"Error syncing positions: {e}")
    
    def _close_all_positions(self, reason: str):
        """Close all positions."""
        for symbol in list(self.positions.keys()):
            try:
                self.trading_client.close_position(symbol.replace("/", ""))
                logging.info(f"Closed position: {symbol} ({reason})")
            except Exception as e:
                logging.error(f"Error closing {symbol}: {e}")
        
        self.positions.clear()
    
    def print_metrics_summary(self):
        """Print performance metrics summary."""
        summary = self.metrics.get_summary()
        
        logging.info("=" * 60)
        logging.info("PERFORMANCE SUMMARY")
        logging.info("=" * 60)
        
        if 'message' in summary:
            logging.info(summary['message'])
            return
        
        logging.info(f"Total Trades: {summary['total_trades']}")
        logging.info(f"Win Rate: {summary['win_rate']:.2%}")
        logging.info(f"Total PnL: ${summary['total_pnl']:,.2f}")
        logging.info(f"Average R-Multiple: {summary['average_r_multiple']:.2f}")
        logging.info(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        logging.info(f"Max Drawdown: {summary['max_drawdown']:.2%}")
        
        logging.info("\nPerformance by Regime:")
        for regime, stats in summary['by_regime'].items():
            if stats['trades'] > 0:
                wr = stats['wins'] / stats['trades']
                logging.info(f"  {regime}: {stats['trades']} trades, {wr:.2%} win rate, ${stats['total_pnl']:.2f} PnL")
        
        logging.info("\nTop/Bottom Symbols by PnL:")
        symbol_pnl = [(s, stats['total_pnl']) for s, stats in summary['by_symbol'].items() if stats['trades'] > 0]
        symbol_pnl.sort(key=lambda x: x[1], reverse=True)
        
        for symbol, pnl in symbol_pnl[:3]:
            logging.info(f"  +{symbol}: ${pnl:.2f}")
        for symbol, pnl in symbol_pnl[-3:]:
            if pnl < 0:
                logging.info(f"  -{symbol}: ${pnl:.2f}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    # Load config from environment or use defaults
    config = TradingConfig(
        api_key=os.environ.get('ALPACA_API_KEY', ''),
        api_secret=os.environ.get('ALPACA_SECRET_KEY', ''),
        paper=True
    )
    
    if not config.api_key or not config.api_secret:
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("\nExample:")
        print("  export ALPACA_API_KEY='your-api-key'")
        print("  export ALPACA_SECRET_KEY='your-secret-key'")
        return
    
    # Create and run bot
    bot = EnhancedTradingBot(config)
    
    print("\nEnhanced Trading Bot")
    print("=" * 40)
    print("Features enabled:")
    print(f"  - Trailing stops: {config.trailing_stop_enabled} ({config.trailing_stop_pct:.0%} from peak)")
    print(f"  - ATR position sizing: Target {config.atr_target_risk:.1%} risk")
    print(f"  - Correlation filter: {config.correlation_enabled} (threshold: {config.correlation_threshold})")
    print(f"  - ADX hysteresis: Entry>{config.adx_trending_entry}, Exit<{config.adx_trending_exit}")
    print(f"  - Volume confirmation: {config.volume_confirmation_enabled} ({config.volume_multiplier}x avg)")
    print(f"  - Time-of-day filter: {config.time_filter_enabled}")
    print("=" * 40)
    
    # Run one iteration
    bot.run_once()
    
    # Print metrics
    bot.print_metrics_summary()


if __name__ == "__main__":
    main()
