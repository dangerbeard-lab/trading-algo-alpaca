#!/usr/bin/env python3
"""
Calculate and display backtest performance metrics.
Supports both v4 (15-min) and v5 (daily momentum) strategies.
"""

import math
from collections import defaultdict
from typing import List

import pandas as pd


def _max_drawdown(values: pd.Series) -> float:
    cummax = values.cummax()
    dd = (values - cummax) / cummax
    return abs(dd.min()) if len(dd) > 0 else 0.0


def _sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * math.sqrt(periods_per_year)


def _get_time(x):
    """Return datetime from either .time (v4) or .date (v5) attribute."""
    return getattr(x, "time", None) or getattr(x, "date", None)


def _get_hold(t):
    """Hold time in hours (v4=hold_hours, v5=hold_days*24)."""
    if hasattr(t, "hold_hours") and t.hold_hours:
        return t.hold_hours
    if hasattr(t, "hold_days"):
        return t.hold_days * 24
    return 0.0


def calculate_metrics(snapshots: List, trades: List, initial_cash: float,
                      benchmark_return: float = None, strategy: str = "v4") -> dict:
    if not snapshots:
        return {}

    # Build value series from snapshots (handle both v4 and v5 field names)
    df = pd.DataFrame([{
        "time": _get_time(s),
        "value": s.portfolio_value,
        "regime": getattr(s, "regime", None),
    } for s in snapshots])
    df = df.set_index("time").sort_index()

    final_value = df["value"].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash

    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    years = days / 365.25 if days > 0 else 1
    ann_return = (final_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0

    max_dd = _max_drawdown(df["value"])

    # Sharpe: annualization factor depends on cadence
    returns = df["value"].pct_change().dropna()
    periods_per_year = 252 if strategy == "v5" else 252 * 26
    sharpe = _sharpe(returns, periods_per_year)

    # Trade metrics
    closed_trades = [t for t in trades if t.side == "SELL" and t.exit_reason != "backtest_end"]
    n_trades = len(closed_trades)
    wins = [t for t in closed_trades if t.pnl > 0]
    losses = [t for t in closed_trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))
                     if losses and sum(t.pnl for t in losses) != 0 else float("inf"))
    avg_hold_hours = sum(_get_hold(t) for t in closed_trades) / n_trades if n_trades > 0 else 0

    # Exit reason breakdown
    by_reason = defaultdict(list)
    for t in closed_trades:
        by_reason[t.exit_reason].append(t)
    exit_breakdown = {
        reason: {
            "n": len(tl),
            "total_pnl": sum(t.pnl for t in tl),
            "win_rate": sum(1 for t in tl if t.pnl > 0) / len(tl) if tl else 0,
        }
        for reason, tl in by_reason.items()
    }

    # Sector attribution
    sector_stats = defaultdict(list)
    for t in closed_trades:
        sector_stats[getattr(t, "sector", "unknown")].append(t)
    sector_breakdown = {}
    for sec, tl in sector_stats.items():
        w = [t for t in tl if t.pnl > 0]
        sector_breakdown[sec] = {
            "n": len(tl),
            "win_rate": len(w) / len(tl) if tl else 0,
            "total_pnl": sum(t.pnl for t in tl),
        }

    # v5-only: regime attribution (which regime state produced wins/losses)
    regime_breakdown = {}
    if strategy == "v5":
        by_regime = defaultdict(list)
        for t in closed_trades:
            by_regime[getattr(t, "regime_at_entry", "unknown")].append(t)
        for r, tl in by_regime.items():
            w = [t for t in tl if t.pnl > 0]
            regime_breakdown[r] = {
                "n": len(tl),
                "win_rate": len(w) / len(tl) if tl else 0,
                "total_pnl": sum(t.pnl for t in tl),
                "avg_pnl": sum(t.pnl for t in tl) / len(tl) if tl else 0,
            }

    # Time in each regime (v5)
    regime_time = {}
    if strategy == "v5" and "regime" in df.columns:
        rc = df["regime"].value_counts()
        total = rc.sum()
        regime_time = {r: n / total for r, n in rc.items() if r and r != "end"}

    # Monthly returns
    monthly = df["value"].resample("ME").last().pct_change().dropna()

    # Worst trades
    worst_trades = sorted(closed_trades, key=lambda t: t.pnl)[:10]
    worst_list = [{
        "symbol": t.symbol,
        "pnl": t.pnl,
        "time": _get_time(t),
        "hold_hours": _get_hold(t),
        "exit_reason": t.exit_reason,
        "sector": getattr(t, "sector", ""),
        "regime_at_entry": getattr(t, "regime_at_entry", ""),
    } for t in worst_trades]

    return {
        "strategy": strategy,
        "initial_cash": initial_cash,
        "final_value": final_value,
        "total_return": total_return,
        "ann_return": ann_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold_hours": avg_hold_hours,
        "exit_breakdown": exit_breakdown,
        "sector_attribution": sector_breakdown,
        "regime_breakdown": regime_breakdown,
        "regime_time": regime_time,
        "monthly_returns": monthly.to_dict(),
        "benchmark_return": benchmark_return,
        "worst_trades": worst_list,
    }


def print_metrics(metrics: dict):
    if not metrics:
        print("No metrics to display")
        return

    strategy = metrics.get("strategy", "v4")

    print("\n" + "=" * 70)
    print(f" BACKTEST RESULTS ({'v5 Momentum Daily' if strategy == 'v5' else 'v4 Trend 15min'})")
    print("=" * 70)
    print(f"  Starting capital:  ${metrics['initial_cash']:>14,.2f}")
    print(f"  Final value:       ${metrics['final_value']:>14,.2f}")
    print(f"  Total return:      {metrics['total_return']:>14.2%}")
    print(f"  Annualized return: {metrics['ann_return']:>14.2%}")
    print(f"  Max drawdown:      {metrics['max_drawdown']:>14.2%}")
    print(f"  Sharpe ratio:      {metrics['sharpe_ratio']:>14.2f}")
    if metrics.get("benchmark_return") is not None:
        diff = metrics["total_return"] - metrics["benchmark_return"]
        print(f"  Benchmark (SPY):   {metrics['benchmark_return']:>14.2%}")
        print(f"  Alpha vs SPY:      {diff:>14.2%}")

    print("\n" + "-" * 70)
    print(" TRADE STATISTICS")
    print("-" * 70)
    print(f"  Total trades:      {metrics['n_trades']:>14d}")
    print(f"  Win rate:          {metrics['win_rate']:>14.1%}")
    print(f"  Avg win:           ${metrics['avg_win']:>14,.2f}")
    print(f"  Avg loss:          ${metrics['avg_loss']:>14,.2f}")
    pf = metrics["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"  Profit factor:     {pf_str:>14}")
    if strategy == "v5":
        print(f"  Avg hold (days):   {metrics['avg_hold_hours']/24:>14.1f}")
    else:
        print(f"  Avg hold (hrs):    {metrics['avg_hold_hours']:>14.1f}")

    if metrics.get("regime_time"):
        print("\n" + "-" * 70)
        print(" TIME IN REGIME")
        print("-" * 70)
        for r, pct in sorted(metrics["regime_time"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {r:<10}  {pct:>7.1%}")

    if metrics.get("regime_breakdown"):
        print("\n" + "-" * 70)
        print(" TRADES BY REGIME AT ENTRY")
        print("-" * 70)
        print(f"  {'Regime':<10} {'N':>5} {'Win%':>8} {'Total PnL':>14} {'Avg PnL':>12}")
        for r, s in metrics["regime_breakdown"].items():
            print(f"  {r:<10} {s['n']:>5d} {s['win_rate']:>7.1%} "
                  f"${s['total_pnl']:>13,.2f} ${s['avg_pnl']:>11,.2f}")

    if metrics.get("exit_breakdown"):
        print("\n" + "-" * 70)
        print(" EXIT REASON BREAKDOWN")
        print("-" * 70)
        print(f"  {'Reason':<20} {'N':>5} {'Win%':>8} {'Total PnL':>14}")
        for reason, e in metrics["exit_breakdown"].items():
            print(f"  {reason:<20} {e['n']:>5d} {e['win_rate']:>7.1%} ${e['total_pnl']:>13,.2f}")

    if metrics.get("sector_attribution"):
        print("\n" + "-" * 70)
        print(" SECTOR ATTRIBUTION")
        print("-" * 70)
        print(f"  {'Sector':<20} {'N':>5} {'Win%':>8} {'Total PnL':>14}")
        for sec, s in sorted(metrics["sector_attribution"].items(), key=lambda x: x[1]["total_pnl"], reverse=True):
            print(f"  {sec:<20} {s['n']:>5d} {s['win_rate']:>7.1%} ${s['total_pnl']:>13,.2f}")

    if metrics.get("worst_trades"):
        print("\n" + "-" * 70)
        print(" WORST TRADES (Biggest Losses)")
        print("-" * 70)
        print(f"  {'Symbol':<8} {'Date':<12} {'PnL':>10} {'Exit':<18} {'Sector':<15}")
        for w in metrics["worst_trades"]:
            date_str = w["time"].strftime("%Y-%m-%d") if hasattr(w["time"], "strftime") else str(w["time"])[:10]
            print(f"  {w['symbol']:<8} {date_str:<12} ${w['pnl']:>9,.2f} "
                  f"{w['exit_reason']:<18} {w['sector']:<15}")

    if metrics.get("monthly_returns"):
        print("\n" + "-" * 70)
        print(" MONTHLY RETURNS")
        print("-" * 70)
        for date, ret in metrics["monthly_returns"].items():
            month_str = date.strftime("%Y-%m")
            bar = "+" if ret >= 0 else "-"
            magnitude = min(int(abs(ret) * 200), 30)
            print(f"  {month_str}  {ret:>7.2%}  {bar * magnitude}")

    print("=" * 70 + "\n")
