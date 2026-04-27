#!/usr/bin/env python3
"""
Backtester engine: replays historical bars through the live trading logic.

Reuses TechnicalIndicators and generate_signal() from enhanced_trading_bot.py
so backtest results match live behavior exactly.

Simulates:
- 15-min cycle execution (matching live bot's interval)
- Position sizing using cash (not portfolio value)
- Two-phase stops (initial 2x ATR + adaptive trailing)
- SPY market filter
- Sector limits, drawdown circuit breaker, cooldowns
- Limit order fills (only if next bar's low <= limit price)
- Conservative slippage on market orders
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from enhanced_trading_bot import TechnicalIndicators

logger = logging.getLogger(__name__)

SLIPPAGE_PCT = 0.0005  # 0.05% slippage on market orders
COOLDOWN_HOURS = 48
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
SKIP_FIRST_MINUTES = 30
SKIP_LAST_MINUTES = 30


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    entry_atr: float
    entry_type: str  # 'trend', 'mr', 'cautious'
    peak_price: float


@dataclass
class Trade:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    qty: float
    price: float
    time: datetime
    entry_type: str
    pnl: float = 0.0
    hold_hours: float = 0.0
    exit_reason: str = ""


@dataclass
class Snapshot:
    time: datetime
    portfolio_value: float
    cash: float
    n_positions: int
    exposure_pct: float


class Backtester:
    def __init__(self, config_path: str = "config.json", initial_cash: float = 100000.0):
        with open(config_path) as f:
            self.config = json.load(f)

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.cooldowns: Dict[str, datetime] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[Snapshot] = []

        # Load all data
        from backtest.data_loader import DataLoader
        self.loader = DataLoader.__new__(DataLoader)
        self.loader.config = self.config

        self.bars_15min: Dict[str, pd.DataFrame] = {}
        self.bars_daily: Dict[str, pd.DataFrame] = {}
        self.indicator_cache: Dict[str, pd.DataFrame] = {}

        self.watchlist = self._build_watchlist()
        self.crypto_symbols = set(s for s in self.watchlist if "/" in s)

    def _build_watchlist(self) -> List[str]:
        symbols = []
        for cat, syms in self.config["watchlist"].items():
            symbols.extend(syms)
        return symbols

    def _get_sector(self, symbol: str) -> str:
        for sector, syms in self.config["watchlist"].items():
            if symbol in syms:
                return sector
        return "unknown"

    def load_data(self, start: datetime, end: datetime):
        """Load all bars from parquet cache and slice to backtest window."""
        from backtest.data_loader import DataLoader
        loader = self.loader

        loaded_15 = 0
        loaded_d = 0
        symbols_to_load = self.watchlist + (["SPY"] if "SPY" not in self.watchlist else [])

        for symbol in symbols_to_load:
            df15 = loader.load_bars(symbol, "15Min")
            if df15 is not None:
                df15 = df15[(df15["timestamp"] >= start) & (df15["timestamp"] <= end)]
                if len(df15) > 0:
                    self.bars_15min[symbol] = df15.reset_index(drop=True)
                    loaded_15 += 1

            dfd = loader.load_bars(symbol, "1Day")
            if dfd is not None:
                # Daily needs more lookback for EMA calculation - keep extra history
                dfd = dfd[dfd["timestamp"] <= end]
                if len(dfd) > 0:
                    self.bars_daily[symbol] = dfd.reset_index(drop=True)
                    loaded_d += 1

        logger.info(f"Loaded data: {loaded_15} symbols (15Min), {loaded_d} symbols (1Day)")

    def _compute_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        """Compute indicators for a symbol's 15Min bars (cached)."""
        if symbol in self.indicator_cache:
            return self.indicator_cache[symbol]

        df = self.bars_15min.get(symbol)
        if df is None or len(df) < 50:
            return None

        cfg = self.config["strategy"]
        df = df.copy()
        df["ema_short"] = TechnicalIndicators.ema(df["close"], cfg["ema_short"])
        df["ema_long"] = TechnicalIndicators.ema(df["close"], cfg["ema_long"])
        df["macd"], df["macd_signal"], df["macd_hist"] = TechnicalIndicators.macd(
            df["close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]
        )
        df["rsi"] = TechnicalIndicators.rsi(df["close"], cfg["rsi_period"])
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = TechnicalIndicators.bollinger_bands(
            df["close"], cfg["bollinger_period"], cfg["bollinger_std"]
        )
        df["adx"] = TechnicalIndicators.adx(df["high"], df["low"], df["close"], cfg["adx_period"])
        df["atr"] = TechnicalIndicators.atr(df["high"], df["low"], df["close"], cfg["atr_period"])
        df["volume_ma"] = TechnicalIndicators.sma(df["volume"], cfg["volume_ma_period"])
        df["volume_ratio"] = df["volume"] / (df["volume_ma"] + 1e-10)

        self.indicator_cache[symbol] = df
        return df

    def _bar_at_or_before(self, df: pd.DataFrame, t: datetime) -> Optional[int]:
        """Return index of bar with timestamp <= t, or None."""
        mask = df["timestamp"] <= t
        if not mask.any():
            return None
        return mask[mask].index[-1]

    def _generate_signal(self, df: pd.DataFrame, idx: int) -> dict:
        """Generate signal using closed candle at idx-1 (matches live bot logic)."""
        if idx < 2:
            return {"signal": "HOLD", "reason": "Insufficient data"}

        cfg = self.config["strategy"]
        current = df.iloc[idx - 1]  # Last closed
        previous = df.iloc[idx - 2]

        if pd.isna(current["adx"]) or pd.isna(current["ema_short"]):
            return {"signal": "HOLD", "reason": "NaN indicators"}

        signal = {"signal": "HOLD", "reason": "", "order_type": "market",
                  "limit_price": None, "entry_type": None}

        adx = current["adx"]
        is_trending = adx > cfg["adx_trending_threshold"]
        is_ranging = adx < cfg["adx_ranging_threshold"]
        volume_confirmed = current["volume_ratio"] > cfg["volume_confirmation_multiplier"]

        if is_trending:
            ema_cross_up = (previous["ema_short"] <= previous["ema_long"]
                            and current["ema_short"] > current["ema_long"])
            macd_bullish = current["macd"] > current["macd_signal"]

            if ema_cross_up and macd_bullish and volume_confirmed:
                return {"signal": "BUY", "reason": f"Trending BUY (ADX={adx:.1f})",
                        "order_type": "market", "limit_price": None, "entry_type": "trend"}

            ema_cross_down = (previous["ema_short"] >= previous["ema_long"]
                              and current["ema_short"] < current["ema_long"])
            macd_bearish = current["macd"] < current["macd_signal"]
            if ema_cross_down and macd_bearish:
                return {"signal": "SELL", "reason": f"Trending SELL (ADX={adx:.1f})",
                        "order_type": "market", "limit_price": None, "entry_type": "trend"}

        elif is_ranging:
            rsi = current["rsi"]
            prev_adx = previous["adx"]
            adx_rising = adx > prev_adx + 1.0

            if rsi < cfg["rsi_oversold"] and current["close"] <= current["bb_lower"]:
                if not adx_rising:
                    return {"signal": "BUY", "reason": f"MR BUY: RSI={rsi:.1f}",
                            "order_type": "limit", "limit_price": current["bb_lower"],
                            "entry_type": "mr"}

            elif rsi > cfg["rsi_overbought"] and current["close"] >= current["bb_upper"]:
                return {"signal": "SELL", "reason": f"MR SELL: RSI={rsi:.1f}",
                        "order_type": "market", "limit_price": None, "entry_type": "mr"}

        else:  # Transitional zone
            ema_cross_up = (previous["ema_short"] <= previous["ema_long"]
                            and current["ema_short"] > current["ema_long"])
            macd_bullish = current["macd"] > current["macd_signal"]
            rsi_not_overbought = current["rsi"] < cfg["rsi_overbought"]

            if ema_cross_up and macd_bullish and volume_confirmed and rsi_not_overbought:
                return {"signal": "BUY", "reason": f"Cautious BUY (ADX={adx:.1f})",
                        "order_type": "market", "limit_price": None,
                        "entry_type": "cautious", "half_size": True}

        return signal

    def _check_market_trend(self, t: datetime) -> dict:
        """SPY daily trend filter."""
        result = {"bullish": True, "ema21_above_ema50": True, "reason": "no SPY data"}
        df = self.bars_daily.get("SPY")
        if df is None or len(df) < 50:
            return result

        idx = self._bar_at_or_before(df, t)
        if idx is None or idx < 30:
            return result

        # Use closed daily bars only - daily bar at idx is "today's" forming bar
        sub = df.iloc[:idx + 1].copy()
        ema_21 = TechnicalIndicators.ema(sub["close"], 21)
        ema_50 = TechnicalIndicators.ema(sub["close"], 50)

        if len(sub) < 3:
            return result

        current_close = sub["close"].iloc[-2]
        ema21_now = ema_21.iloc[-2]
        ema50_now = ema_50.iloc[-2]
        ema21_prev = ema_21.iloc[-3]

        spy_above_ema = current_close > ema21_now
        ema_slope_up = ema21_now > ema21_prev
        ema21_above_ema50 = ema21_now > ema50_now
        is_bullish = spy_above_ema or ema_slope_up

        return {
            "bullish": is_bullish,
            "ema21_above_ema50": ema21_above_ema50,
            "spy_above_ema": spy_above_ema,
            "reason": (f"SPY={'above' if spy_above_ema else 'BELOW'} EMA21, "
                       f"EMA21 {'>' if ema21_above_ema50 else '<'} EMA50"),
        }

    def _check_daily_trend(self, symbol: str, t: datetime) -> Optional[str]:
        df = self.bars_daily.get(symbol)
        if df is None or len(df) < 30:
            return None

        idx = self._bar_at_or_before(df, t)
        if idx is None or idx < 22:
            return None

        sub = df.iloc[:idx + 1].copy()
        ema_21 = TechnicalIndicators.ema(sub["close"], 21)

        if len(sub) < 3:
            return None

        current_close = sub["close"].iloc[-2]
        ema_now = ema_21.iloc[-2]
        ema_prev = ema_21.iloc[-3]

        if current_close > ema_now and ema_now > ema_prev:
            return "up"
        elif current_close < ema_now and ema_now < ema_prev:
            return "down"
        return None

    def _calculate_position_size(self, atr: float, price: float, cash: float) -> float:
        cfg = self.config["risk_management"]
        base_pct = cfg["base_position_pct"]
        max_pct = cfg["max_position_pct"]

        if atr <= 0 or price <= 0 or pd.isna(atr):
            return cash * base_pct

        atr_pct = atr / price
        if atr_pct <= 0.01:
            position_pct = max_pct
        elif atr_pct >= 0.04:
            position_pct = base_pct
        else:
            vol_position = (atr_pct - 0.01) / 0.03
            position_pct = max_pct - (vol_position * (max_pct - base_pct))
        return cash * position_pct

    def _portfolio_value(self, t: datetime) -> Tuple[float, float, float]:
        """Returns (portfolio_value, total_position_value, exposure_pct)."""
        position_value = 0.0
        for sym, pos in self.positions.items():
            df = self.bars_15min.get(sym)
            if df is None:
                position_value += pos.qty * pos.entry_price
                continue
            idx = self._bar_at_or_before(df, t)
            if idx is None:
                position_value += pos.qty * pos.entry_price
            else:
                position_value += pos.qty * df.iloc[idx]["close"]

        portfolio_value = self.cash + position_value
        exposure = position_value / portfolio_value if portfolio_value > 0 else 0
        return portfolio_value, position_value, exposure

    def _is_in_cooldown(self, symbol: str, t: datetime) -> bool:
        if symbol not in self.cooldowns:
            return False
        if t - self.cooldowns[symbol] < timedelta(hours=COOLDOWN_HOURS):
            return True
        del self.cooldowns[symbol]
        return False

    def _check_stops(self, symbol: str, current_price: float, market_bearish: bool) -> Optional[str]:
        """Returns stop reason if triggered, else None."""
        pos = self.positions[symbol]
        cfg = self.config["risk_management"]
        base_stop_pct = cfg["trailing_stop_pct"]
        atr_mult = cfg.get("atr_risk_multiplier", 2.0)

        # Phase 1: Initial ATR stop
        if pos.entry_atr > 0 and current_price <= pos.entry_price:
            initial_stop = pos.entry_price - (atr_mult * pos.entry_atr)
            if current_price <= initial_stop:
                return "initial_atr_stop"

        # Update peak
        if current_price > pos.peak_price:
            pos.peak_price = current_price

        # Phase 2: Adaptive trailing stop
        unrealized_plpc = (current_price - pos.entry_price) / pos.entry_price
        trailing_pct = base_stop_pct
        if market_bearish:
            trailing_pct = min(trailing_pct, 0.08)
        if unrealized_plpc > 0.15:
            trailing_pct = min(trailing_pct, max(0.06, base_stop_pct - unrealized_plpc * 0.2))

        stop_level = pos.peak_price * (1 - trailing_pct)
        if current_price <= stop_level:
            return "trailing_stop"

        return None

    def _close_position(self, symbol: str, price: float, t: datetime, reason: str):
        pos = self.positions[symbol]
        # Apply slippage on exit
        fill_price = price * (1 - SLIPPAGE_PCT)
        proceeds = pos.qty * fill_price
        cost_basis = pos.qty * pos.entry_price
        pnl = proceeds - cost_basis
        hold_hours = (t - pos.entry_time).total_seconds() / 3600

        self.cash += proceeds
        self.trades.append(Trade(
            symbol=symbol, side="SELL", qty=pos.qty, price=fill_price,
            time=t, entry_type=pos.entry_type, pnl=pnl,
            hold_hours=hold_hours, exit_reason=reason
        ))
        # Record cooldown only for stop-outs
        if reason in ("initial_atr_stop", "trailing_stop"):
            self.cooldowns[symbol] = t
        del self.positions[symbol]

    def _open_position(self, symbol: str, price: float, t: datetime, signal: dict, qty: float, atr: float):
        # Apply slippage on entry
        fill_price = price * (1 + SLIPPAGE_PCT)
        cost = qty * fill_price
        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol, qty=qty, entry_price=fill_price,
            entry_time=t, entry_atr=atr,
            entry_type=signal["entry_type"], peak_price=fill_price,
        )
        self.trades.append(Trade(
            symbol=symbol, side="BUY", qty=qty, price=fill_price,
            time=t, entry_type=signal["entry_type"]
        ))
        return True

    def _is_market_hours(self, t: datetime) -> bool:
        if t.weekday() >= 5:
            return False
        tt = t.time()
        skip_open = dtime(MARKET_OPEN.hour, MARKET_OPEN.minute + SKIP_FIRST_MINUTES)
        skip_close = dtime(MARKET_CLOSE.hour - 1, 60 - SKIP_LAST_MINUTES)
        return skip_open <= tt <= dtime(15, 30)

    def _check_drawdown_circuit_breaker(self, current_value: float) -> bool:
        cfg = self.config["risk_management"]
        max_dd = cfg.get("max_drawdown_pct", 0.15)
        peak = max((s.portfolio_value for s in self.snapshots), default=current_value)
        peak = max(peak, current_value)
        if peak <= 0:
            return False
        dd = (peak - current_value) / peak
        return dd >= max_dd

    def run(self, start: datetime, end: datetime):
        """Main backtest loop. Steps through 15-min intervals from start to end."""
        logger.info(f"Backtest: {start.date()} to {end.date()}, ${self.initial_cash:,.0f} starting")

        # Build timeline of 15-min cycle times from union of all 15Min bar timestamps
        all_times = set()
        for df in self.bars_15min.values():
            all_times.update(df["timestamp"].tolist())
        cycle_times = sorted(t for t in all_times if start <= t <= end)
        logger.info(f"Cycle timeline: {len(cycle_times)} timestamps")

        cfg = self.config["risk_management"]
        max_positions = cfg["max_positions"]
        max_exposure = cfg["max_portfolio_exposure"]
        max_sector = cfg.get("max_correlated_positions", 5)

        for i, t in enumerate(cycle_times):
            market = self._check_market_trend(t)
            market_bearish = not market["bullish"]

            # ====== EXIT CHECKS ======
            symbols_to_check = list(self.positions.keys())
            for symbol in symbols_to_check:
                df = self.bars_15min.get(symbol)
                if df is None:
                    continue
                idx = self._bar_at_or_before(df, t)
                if idx is None:
                    continue
                current_price = df.iloc[idx]["close"]

                stop_reason = self._check_stops(symbol, current_price, market_bearish)
                if stop_reason:
                    self._close_position(symbol, current_price, t, stop_reason)
                    continue

                # MR profit target at BB middle
                pos = self.positions[symbol]
                if pos.entry_type == "mr":
                    ind_df = self._compute_indicators(symbol)
                    if ind_df is not None and idx < len(ind_df):
                        bb_middle = ind_df.iloc[idx - 1]["bb_middle"] if idx > 0 else None
                        if bb_middle and current_price >= bb_middle:
                            self._close_position(symbol, current_price, t, "mr_target")
                            continue

                # SELL signal
                ind_df = self._compute_indicators(symbol)
                if ind_df is not None and idx < len(ind_df):
                    sig = self._generate_signal(ind_df, idx)
                    if sig["signal"] == "SELL":
                        self._close_position(symbol, current_price, t, "sell_signal")

            # ====== SNAPSHOT ======
            pv, pos_val, expo = self._portfolio_value(t)
            self.snapshots.append(Snapshot(
                time=t, portfolio_value=pv, cash=self.cash,
                n_positions=len(self.positions), exposure_pct=expo,
            ))

            # ====== ENTRY CHECKS ======
            if len(self.positions) >= max_positions:
                continue
            if expo >= max_exposure:
                continue
            if self._check_drawdown_circuit_breaker(pv):
                continue

            # Sector counts
            sector_counts: Dict[str, int] = {}
            for sym in self.positions:
                sec = self._get_sector(sym)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

            in_market_hours = self._is_market_hours(t)

            for symbol in self.watchlist:
                if symbol in self.positions:
                    continue
                if self._is_in_cooldown(symbol, t):
                    continue

                is_crypto = symbol in self.crypto_symbols
                if not is_crypto and not in_market_hours:
                    continue
                if not is_crypto and not market["bullish"]:
                    continue

                sector = self._get_sector(symbol)
                if sector_counts.get(sector, 0) >= max_sector:
                    continue

                ind_df = self._compute_indicators(symbol)
                if ind_df is None:
                    continue
                idx = self._bar_at_or_before(ind_df, t)
                if idx is None or idx < 50:
                    continue

                signal = self._generate_signal(ind_df, idx)
                if signal["signal"] != "BUY":
                    continue

                # Daily trend filter
                if self._check_daily_trend(symbol, t) == "down":
                    continue

                bar = ind_df.iloc[idx - 1]  # closed candle
                price = bar["close"]
                atr = bar["atr"] if not pd.isna(bar["atr"]) else 0.0

                position_value = self._calculate_position_size(atr, price, self.cash)
                if signal.get("half_size"):
                    position_value *= 0.5
                if not market.get("ema21_above_ema50", True):
                    position_value *= 0.6

                # Exposure cap
                new_exposure = (pos_val + position_value) / pv if pv > 0 else 0
                if new_exposure > max_exposure:
                    position_value = max_exposure * pv - pos_val
                if position_value <= 0:
                    continue
                if position_value > self.cash * 0.95:
                    position_value = self.cash * 0.95
                if position_value <= 0:
                    continue

                # Limit order fill check: only fill if next bar's low touches limit
                fill_price = price
                if signal["order_type"] == "limit":
                    limit_price = signal["limit_price"]
                    if idx + 1 >= len(ind_df):
                        continue
                    next_bar = ind_df.iloc[idx]
                    if next_bar["low"] > limit_price:
                        continue
                    fill_price = limit_price

                qty = position_value / fill_price
                if self._open_position(symbol, fill_price, t, signal, qty, atr):
                    pos_val += position_value
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    if len(self.positions) >= max_positions:
                        break

            # Progress log every ~1000 cycles
            if (i + 1) % 1000 == 0:
                logger.info(f"  [{i+1}/{len(cycle_times)}] {t.date()} | "
                            f"PV=${pv:,.0f} | positions={len(self.positions)}")

        # Final liquidation at last price for metrics
        final_t = cycle_times[-1] if cycle_times else end
        for symbol in list(self.positions.keys()):
            df = self.bars_15min.get(symbol)
            if df is None:
                continue
            idx = self._bar_at_or_before(df, final_t)
            if idx is None:
                continue
            self._close_position(symbol, df.iloc[idx]["close"], final_t, "backtest_end")

        final_pv, _, _ = self._portfolio_value(final_t)
        self.snapshots.append(Snapshot(
            time=final_t, portfolio_value=final_pv, cash=self.cash,
            n_positions=0, exposure_pct=0,
        ))
        logger.info(f"Backtest complete: final value ${final_pv:,.2f}")
