#!/usr/bin/env python3
"""
v5 Momentum Strategy Backtester
================================
Daily-bar momentum breakout with:
- Regime detection (bull/chop/bear) drives allocation
- 20-day high breakout + volume surge + EMA50 filter
- Relative strength ranking (top N by 20d return vs SPY)
- Rotation exits: RS decay kicks out weak positions
- Structural stops: 20-day low break
- SPXS hedge in bear regime

One decision per symbol per day. No 15-min chaos.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.regime import detect_regime, Regime

logger = logging.getLogger(__name__)

SLIPPAGE_PCT = 0.001  # 0.1% - realistic for market open orders on liquid stocks
HEDGE_SYMBOL = 'SQQQ'  # Inverse QQQ for bear hedge (or use SPXS)


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_date: datetime
    entry_atr: float
    peak_price: float
    is_hedge: bool = False
    weak_rs_days: int = 0


@dataclass
class Trade:
    symbol: str
    side: str
    qty: float
    price: float
    date: datetime
    pnl: float = 0.0
    hold_days: float = 0.0
    exit_reason: str = ""
    regime_at_entry: str = ""
    entry_rs_score: float = 0.0
    sector: str = ""


@dataclass
class Snapshot:
    date: datetime
    portfolio_value: float
    cash: float
    n_positions: int
    exposure_pct: float
    regime: str


class MomentumBacktester:
    def __init__(self, config_path: str = "config.json", initial_cash: float = 100000.0):
        with open(config_path) as f:
            self.config = json.load(f)

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[Snapshot] = []

        self.bars_daily: Dict[str, pd.DataFrame] = {}
        self.indicator_cache: Dict[str, pd.DataFrame] = {}

        self.watchlist = self._build_watchlist()
        self.crypto_symbols = set(s for s in self.watchlist if "/" in s)
        self.hedge_symbols = set(self.config["watchlist"].get("hedges", []))
        # Tradeable universe: exclude hedges (used only for regime hedging) and crypto (for now)
        self.tradeable = [s for s in self.watchlist
                          if s not in self.hedge_symbols and s not in self.crypto_symbols]

        # Momentum config with defaults
        mcfg = self.config.get('momentum', {})
        self.breakout_period = mcfg.get('breakout_period', 20)
        self.volume_period = mcfg.get('volume_period', 50)
        self.volume_multiplier = mcfg.get('volume_multiplier', 2.0)
        self.ema_filter_period = mcfg.get('ema_filter_period', 50)
        self.rs_period = mcfg.get('rs_period', 20)
        self.rotation_threshold = mcfg.get('rotation_threshold', 0.40)
        self.rotation_days = mcfg.get('rotation_days', 5)
        self.trailing_stop_pct = mcfg.get('trailing_stop_pct', 0.12)

        # Regime-driven allocation
        self.bull_max_positions = mcfg.get('bull_max_positions', 8)
        self.bull_position_pct = mcfg.get('bull_position_pct', 0.10)
        self.chop_max_positions = mcfg.get('chop_max_positions', 4)
        self.chop_position_pct = mcfg.get('chop_position_pct', 0.06)
        self.bear_hedge_pct = mcfg.get('bear_hedge_pct', 0.15)

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
        """Load daily bars for entire watchlist + SPY."""
        from backtest.data_loader import DataLoader
        loader = DataLoader.__new__(DataLoader)
        loader.config = self.config

        loaded = 0
        symbols_to_load = list(set(self.watchlist + ['SPY', HEDGE_SYMBOL]))

        for symbol in symbols_to_load:
            dfd = loader.load_bars(symbol, "1Day")
            if dfd is not None:
                # Keep all history for EMA200 warmup; slice at run time
                self.bars_daily[symbol] = dfd.reset_index(drop=True)
                loaded += 1

        logger.info(f"Loaded {loaded} symbols (daily bars)")
        if HEDGE_SYMBOL not in self.bars_daily:
            logger.warning(f"No data for {HEDGE_SYMBOL} — bear-regime hedge will be disabled. "
                           f"Run: python -m backtest.data_loader --symbol {HEDGE_SYMBOL}")

    def _compute_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self.indicator_cache:
            return self.indicator_cache[symbol]

        df = self.bars_daily.get(symbol)
        if df is None or len(df) < 100:
            return None

        df = df.copy()
        # ATR (14-day)
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # Rolling max (breakout) — use shift(1) so we don't include today's high
        df['high_20'] = df['high'].rolling(self.breakout_period).max().shift(1)
        df['low_20'] = df['low'].rolling(self.breakout_period).min().shift(1)

        # Volume average
        df['vol_avg'] = df['volume'].rolling(self.volume_period).mean().shift(1)

        # EMA filter
        df['ema_filter'] = df['close'].ewm(span=self.ema_filter_period, adjust=False).mean()

        # Return over rs_period for relative strength
        df['ret_rs'] = (df['close'] / df['close'].shift(self.rs_period)) - 1

        self.indicator_cache[symbol] = df
        return df

    def _bar_at_or_before(self, df: pd.DataFrame, d: datetime) -> Optional[int]:
        mask = df['timestamp'] <= d
        if not mask.any():
            return None
        return mask[mask].index[-1]

    def _get_rs_score(self, symbol: str, idx_map: Dict[str, int], spy_ret: float) -> Optional[float]:
        """Relative strength = symbol's 20d return - SPY 20d return."""
        df = self.indicator_cache.get(symbol)
        if df is None:
            return None
        idx = idx_map.get(symbol)
        if idx is None or idx < 1:
            return None
        sym_ret = df.iloc[idx - 1]['ret_rs']
        if pd.isna(sym_ret):
            return None
        return sym_ret - spy_ret

    def _breakout_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if breakout conditions are met on the closed bar at idx-1."""
        if idx < 2:
            return False
        bar = df.iloc[idx - 1]
        if pd.isna(bar['high_20']) or pd.isna(bar['vol_avg']) or pd.isna(bar['ema_filter']):
            return False
        close = bar['close']
        breakout = close > bar['high_20']
        volume_surge = bar['volume'] > (self.volume_multiplier * bar['vol_avg'])
        above_ema = close > bar['ema_filter']
        return breakout and volume_surge and above_ema

    def _portfolio_value(self, idx_map: Dict[str, int]) -> Tuple[float, float, float]:
        position_value = 0.0
        for sym, pos in self.positions.items():
            df = self.bars_daily.get(sym)
            idx = idx_map.get(sym)
            if df is None or idx is None:
                position_value += pos.qty * pos.entry_price
                continue
            position_value += pos.qty * df.iloc[idx]['close']
        pv = self.cash + position_value
        expo = position_value / pv if pv > 0 else 0
        return pv, position_value, expo

    def _check_exit(self, symbol: str, current_price: float, current_low: float,
                    idx: int, ind_df: pd.DataFrame, market_bearish: bool) -> Optional[str]:
        pos = self.positions[symbol]

        # Update peak
        if current_price > pos.peak_price:
            pos.peak_price = current_price

        # Trailing stop (adaptive)
        trailing_pct = self.trailing_stop_pct
        if market_bearish:
            trailing_pct = min(trailing_pct, 0.08)
        unrealized_pct = (current_price - pos.entry_price) / pos.entry_price
        if unrealized_pct > 0.20:
            trailing_pct = min(trailing_pct, max(0.06, self.trailing_stop_pct - unrealized_pct * 0.2))

        if current_price <= pos.peak_price * (1 - trailing_pct):
            return "trailing_stop"

        # Structural stop: close below 20-day low
        if idx > 0:
            bar = ind_df.iloc[idx - 1]
            if not pd.isna(bar['low_20']) and current_price < bar['low_20']:
                return "structural_stop"

        # Initial ATR stop (3.5x below entry, but only if underwater)
        if pos.entry_atr > 0 and current_price <= pos.entry_price:
            atr_stop = pos.entry_price - (3.5 * pos.entry_atr)
            if current_price <= atr_stop:
                return "initial_atr_stop"

        return None

    def _open_position(self, symbol: str, price: float, date: datetime, qty: float,
                       atr: float, regime: str, rs_score: float, is_hedge: bool = False):
        fill_price = price * (1 + SLIPPAGE_PCT)
        cost = qty * fill_price
        if cost > self.cash:
            return False
        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol, qty=qty, entry_price=fill_price,
            entry_date=date, entry_atr=atr, peak_price=fill_price,
            is_hedge=is_hedge,
        )
        self.trades.append(Trade(
            symbol=symbol, side="BUY", qty=qty, price=fill_price, date=date,
            regime_at_entry=regime, entry_rs_score=rs_score,
            sector=self._get_sector(symbol),
        ))
        return True

    def _close_position(self, symbol: str, price: float, date: datetime, reason: str):
        pos = self.positions[symbol]
        fill_price = price * (1 - SLIPPAGE_PCT)
        proceeds = pos.qty * fill_price
        pnl = proceeds - (pos.qty * pos.entry_price)
        hold_days = (date - pos.entry_date).total_seconds() / 86400
        self.cash += proceeds
        self.trades.append(Trade(
            symbol=symbol, side="SELL", qty=pos.qty, price=fill_price, date=date,
            pnl=pnl, hold_days=hold_days, exit_reason=reason,
            sector=self._get_sector(symbol),
        ))
        del self.positions[symbol]

    def run(self, start: datetime, end: datetime):
        logger.info(f"Momentum backtest: {start.date()} to {end.date()}, ${self.initial_cash:,.0f}")

        # Pre-compute indicators for entire watchlist
        for sym in self.watchlist + ['SPY', HEDGE_SYMBOL]:
            self._compute_indicators(sym)

        spy_df = self.bars_daily.get('SPY')
        if spy_df is None:
            logger.error("SPY data required for backtester")
            return

        # Build daily timeline from SPY bars in window
        s_ts = pd.Timestamp(start, tz="UTC") if getattr(start, 'tzinfo', None) is None else pd.Timestamp(start)
        e_ts = pd.Timestamp(end, tz="UTC") if getattr(end, 'tzinfo', None) is None else pd.Timestamp(end)
        if spy_df['timestamp'].dt.tz is None:
            s_ts = s_ts.tz_localize(None)
            e_ts = e_ts.tz_localize(None)

        spy_dates = spy_df[(spy_df['timestamp'] >= s_ts) & (spy_df['timestamp'] <= e_ts)]['timestamp'].tolist()
        logger.info(f"Daily timeline: {len(spy_dates)} trading days")

        prev_regime = None
        for i, d in enumerate(spy_dates):
            # Build idx map for all symbols at this date
            idx_map: Dict[str, int] = {}
            for sym in list(self.bars_daily.keys()):
                df = self.bars_daily[sym]
                idx = self._bar_at_or_before(df, d)
                if idx is not None:
                    idx_map[sym] = idx

            spy_idx = idx_map.get('SPY')
            if spy_idx is None:
                continue

            # Regime detection
            regime = detect_regime(spy_df, spy_idx + 1)
            market_bearish = regime.state == 'bear'

            # SPY return for RS baseline
            spy_ind = self.indicator_cache.get('SPY')
            spy_ret = 0.0
            if spy_ind is not None and spy_idx > 0:
                spy_ret_val = spy_ind.iloc[spy_idx - 1]['ret_rs']
                if not pd.isna(spy_ret_val):
                    spy_ret = spy_ret_val

            # === EXIT CHECKS ===
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                df = self.bars_daily.get(symbol)
                ind_df = self.indicator_cache.get(symbol)
                idx = idx_map.get(symbol)
                if df is None or ind_df is None or idx is None:
                    continue
                bar = df.iloc[idx]
                current_price = bar['close']
                current_low = bar['low']

                # Regime flip to bear: liquidate longs (keep hedge if it exists)
                if not pos.is_hedge and regime.state == 'bear' and prev_regime != 'bear':
                    self._close_position(symbol, current_price, d, "regime_bear")
                    continue

                # Skip exit checks for hedge — hedge managed separately
                if pos.is_hedge:
                    # Exit hedge if regime turns bull
                    if regime.state == 'bull':
                        self._close_position(symbol, current_price, d, "regime_bull")
                    continue

                exit_reason = self._check_exit(symbol, current_price, current_low, idx, ind_df, market_bearish)
                if exit_reason:
                    self._close_position(symbol, current_price, d, exit_reason)
                    continue

                # Rotation exit: RS decay
                rs = self._get_rs_score(symbol, idx_map, spy_ret)
                if rs is not None:
                    # Rank vs tradeable universe
                    all_rs = []
                    for w in self.tradeable:
                        w_rs = self._get_rs_score(w, idx_map, spy_ret)
                        if w_rs is not None:
                            all_rs.append(w_rs)
                    if all_rs:
                        threshold_val = pd.Series(all_rs).quantile(1 - self.rotation_threshold)
                        if rs < threshold_val:
                            pos.weak_rs_days += 1
                            if pos.weak_rs_days >= self.rotation_days:
                                self._close_position(symbol, current_price, d, "rs_rotation")
                                continue
                        else:
                            pos.weak_rs_days = 0

            # === SNAPSHOT ===
            pv, pos_val, expo = self._portfolio_value(idx_map)
            self.snapshots.append(Snapshot(
                date=d, portfolio_value=pv, cash=self.cash,
                n_positions=len(self.positions), exposure_pct=expo,
                regime=regime.state,
            ))

            # === ENTRIES ===
            if regime.state == 'bear':
                # Ensure hedge position
                if HEDGE_SYMBOL not in self.positions:
                    hedge_df = self.bars_daily.get(HEDGE_SYMBOL)
                    hedge_idx = idx_map.get(HEDGE_SYMBOL)
                    hedge_ind = self.indicator_cache.get(HEDGE_SYMBOL)
                    if hedge_df is not None and hedge_idx is not None:
                        price = hedge_df.iloc[hedge_idx]['close']
                        target_value = pv * self.bear_hedge_pct
                        qty = target_value / price
                        atr = hedge_ind.iloc[hedge_idx]['atr'] if hedge_ind is not None and not pd.isna(hedge_ind.iloc[hedge_idx]['atr']) else 0
                        self._open_position(HEDGE_SYMBOL, price, d, qty, atr, regime.state, 0.0, is_hedge=True)
            else:
                max_positions = self.bull_max_positions if regime.state == 'bull' else self.chop_max_positions
                position_pct = self.bull_position_pct if regime.state == 'bull' else self.chop_position_pct

                if len(self.positions) < max_positions:
                    # Find all breakout candidates and rank by RS
                    candidates = []
                    for symbol in self.tradeable:
                        if symbol in self.positions:
                            continue
                        ind_df = self.indicator_cache.get(symbol)
                        idx = idx_map.get(symbol)
                        if ind_df is None or idx is None:
                            continue
                        if not self._breakout_signal(ind_df, idx + 1):
                            continue
                        rs = self._get_rs_score(symbol, idx_map, spy_ret)
                        if rs is None:
                            continue
                        candidates.append((symbol, rs, idx))

                    # Sort by RS descending, take top slots
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    slots = max_positions - len(self.positions)

                    for symbol, rs, idx in candidates[:slots]:
                        ind_df = self.indicator_cache[symbol]
                        bar = ind_df.iloc[idx]
                        price = bar['close']
                        atr = bar['atr'] if not pd.isna(bar['atr']) else 0
                        target_value = pv * position_pct
                        if target_value > self.cash * 0.95:
                            target_value = self.cash * 0.95
                        if target_value <= 0:
                            continue
                        qty = target_value / price
                        self._open_position(symbol, price, d, qty, atr, regime.state, rs)

            prev_regime = regime.state

            if (i + 1) % 50 == 0:
                logger.info(f"  [{i+1}/{len(spy_dates)}] {d.date()} | "
                            f"regime={regime.state} | PV=${pv:,.0f} | positions={len(self.positions)}")

        # Final liquidation
        final_date = spy_dates[-1] if spy_dates else end
        final_idx_map = {}
        for sym in list(self.positions.keys()):
            df = self.bars_daily.get(sym)
            if df is None:
                continue
            idx = self._bar_at_or_before(df, final_date)
            if idx is None:
                continue
            final_idx_map[sym] = idx
            self._close_position(sym, df.iloc[idx]['close'], final_date, "backtest_end")

        # Final snapshot
        final_pv = self.cash
        self.snapshots.append(Snapshot(
            date=final_date, portfolio_value=final_pv, cash=self.cash,
            n_positions=0, exposure_pct=0, regime='end',
        ))
        logger.info(f"Momentum backtest complete: final value ${final_pv:,.2f}")
