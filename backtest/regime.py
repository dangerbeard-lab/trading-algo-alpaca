#!/usr/bin/env python3
"""
Market regime detection for the momentum strategy.

Classifies market state as bull/chop/bear based on SPY daily bars vs
50-day and 200-day EMAs. Regime drives allocation and entry aggressiveness.
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class Regime:
    state: str  # 'bull', 'chop', 'bear'
    spy_above_50: bool
    spy_above_200: bool
    spy_slope_up: bool  # 50-day EMA rising
    reason: str


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def detect_regime(spy_daily: pd.DataFrame, idx: int) -> Regime:
    """
    Detect market regime at SPY daily bar idx (uses closed data only).

    Bull: close > EMA50 AND slope up (also > EMA200 if available)
    Bear: close < EMA50 AND slope down
    Chop: everything else
    """
    if idx < 52:
        return Regime(state='chop', spy_above_50=False, spy_above_200=False,
                      spy_slope_up=False, reason='insufficient history for EMA50')

    sub = spy_daily.iloc[:idx + 1]
    close = sub['close']
    ema_50 = _ema(close, 50)
    have_200 = len(close) >= 200
    ema_200 = _ema(close, 200) if have_200 else None

    if len(close) < 3:
        return Regime(state='chop', spy_above_50=False, spy_above_200=False,
                      spy_slope_up=False, reason='no closed bar')

    close_now = close.iloc[-2]
    ema50_now = ema_50.iloc[-2]
    ema50_prev = ema_50.iloc[-3]

    above_50 = close_now > ema50_now
    slope_up = ema50_now > ema50_prev

    if have_200:
        ema200_now = ema_200.iloc[-2]
        above_200 = close_now > ema200_now
    else:
        # Fallback: use EMA50 direction alone when 200 isn't available
        above_200 = above_50

    if above_50 and above_200 and slope_up:
        state = 'bull'
        reason = f'SPY above EMA50 (slope up){", above EMA200" if have_200 else ""}'
    elif not above_50 and not above_200 and not slope_up:
        state = 'bear'
        reason = f'SPY below EMA50 (slope down){", below EMA200" if have_200 else ""}'
    else:
        state = 'chop'
        reason = f'SPY mixed (above_50={above_50}, above_200={above_200}, slope_up={slope_up})'

    return Regime(state=state, spy_above_50=above_50, spy_above_200=above_200,
                  spy_slope_up=slope_up, reason=reason)
