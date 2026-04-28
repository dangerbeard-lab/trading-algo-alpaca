#!/usr/bin/env python3
"""
Calculate and display backtest performance metrics.
"""

import math
from collections import defaultdict
from typing import List

import pandas as pd


def _max_drawdown(values: pd.Series) -> float:
    cummax = values.cummax()
    dd = (values - cummax) / cummax
    return abs(dd.min()) if len(dd) > 0 else 0.0


def _sharpe(returns: pd.Series, periods_per_year: float = 252 * 26) -> float:
    """Sharpe ratio. Default periods/year = 252 trading days * 26 fifteen-min bars/day."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * math.sqrt(periods_per_year)


def calculate_metrics(snapshots: List, trades: List, initial_cash: float, benchmark_return: float = None) -> dict:
    if not snapshots:
        return {}

    df = pd.DataFrame([{"time": s.time, "value": s.portfolio_value} for s in snapshots])
    df = df.set_index("time").sort_index()

    final_value = df["value"].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash

    # Annualized return
    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    years = days / 365.25 if days > 0 else 1
    ann_return = (final_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0

    max_dd = _max_drawdown(df["value"])

    # Returns for Sharpe (15-min returns)
    returns = df["value"].pct_change().dropna()
    sharpe = _sharpe(returns)

    # Trade-level metrics (closed positions only - SELL trades have pnl)
    closed_trades = [t for t in trades if t.side == "SELL" and t.exit_reason != "backtest_end"]
    n_trades = len(closed_trades)
    wins = [t for t in closed_trades if t.pnl > 0]
    losses = [t for t in closed_trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))
                     if losses and sum(t.pnl for t in losses) != 0 else float("inf"))
    avg_hold_hours = sum(t.hold_hours for t in closed_trades) / n_trades if n_trades > 0 else 0

    # Per-strategy breakdown
    by_type = defaultdict(list)
    for t in closed_trades:
        by_type[t.entry_type].append(t)

    strategy_breakdown = {}
    for entry_type, trades_list in by_type.items():
        n = len(trades_list)
        w = [t for t in trades_list if t.pnl > 0]
        strategy_breakdown[entry_type] = {
            "n_trades": n,
            "win_rate": len(w) / n if n > 0 else 0,
            "total_pnl": sum(t.pnl for t in trades_list),
            "avg_pnl": sum(t.pnl for t in trades_list) / n if n > 0 else 0,
            "avg_hold_hours": sum(t.hold_hours for t in trades_list) / n if n > 0 else 0,
        }

    # Exit reason breakdown
    by_reason = defaultdict(list)
    for t in closed_trades:
        by_reason[t.exit_reason].append(t)
    exit_breakdown = {
        reason: {
            "n": len(trades_list),
            "total_pnl": sum(t.pnl for t in trades_list),
            "win_rate": sum(1 for t in trades_list if t.pnl > 0) / len(trades_list) if trades_list else 0,
        }
        for reason, trades_list in by_reason.items()
    }

    # Monthly returns
    monthly = df["value"].resample("ME").last().pct_change().dropna()

    # Trade attribution: winners vs losers by ADX range and sector
    adx_buckets = {"low_25_30": [], "mid_30_40": [], "high_40+": []}
    for t in closed_trades:
        adx = getattr(t, "entry_adx", 0)
        if adx < 30:
            adx_buckets["low_25_30"].append(t)
        elif adx < 40:
            adx_buckets["mid_30_40"].append(t)
        else:
            adx_buckets["high_40+"].append(t)
    adx_attribution = {}
    for bucket, trades_list in adx_buckets.items():
        if trades_list:
            w = [t for t in trades_list if t.pnl > 0]
            adx_attribution[bucket] = {
                "n": len(trades_list),
                "win_rate": len(w) / len(trades_list),
                "total_pnl": sum(t.pnl for t in trades_list),
                "avg_pnl": sum(t.pnl for t in trades_list) / len(trades_list),
            }

    sector_attribution = defaultdict(list)
    for t in closed_trades:
        sec = getattr(t, "sector", "unknown")
        sector_attribution[sec].append(t)
    sector_stats = {}
    for sec, trades_list in sector_attribution.items():
        w = [t for t in trades_list if t.pnl > 0]
        sector_stats[sec] = {
            "n": len(trades_list),
            "win_rate": len(w) / len(trades_list) if trades_list else 0,
            "total_pnl": sum(t.pnl for t in trades_list),
        }

    return {
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
        "strategy_breakdown": strategy_breakdown,
        "exit_breakdown": exit_breakdown,
        "monthly_returns": monthly.to_dict(),
        "benchmark_return": benchmark_return,
        "adx_attribution": adx_attribution,
        "sector_attribution": sector_stats,
    }


def print_metrics(metrics: dict):
    if not metrics:
        print("No metrics to display")
        return

    print("\n" + "=" * 70)
    print(" BACKTEST RESULTS")
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
    print(f"  Avg hold (hrs):    {metrics['avg_hold_hours']:>14.1f}")

    if metrics.get("strategy_breakdown"):
        print("\n" + "-" * 70)
        print(" PER-STRATEGY BREAKDOWN")
        print("-" * 70)
        print(f"  {'Strategy':<12} {'N':>5} {'Win%':>8} {'Total PnL':>14} {'Avg PnL':>12} {'Hold(h)':>10}")
        for strat, s in metrics["strategy_breakdown"].items():
            print(f"  {strat:<12} {s['n_trades']:>5d} {s['win_rate']:>7.1%} "
                  f"${s['total_pnl']:>13,.2f} ${s['avg_pnl']:>11,.2f} {s['avg_hold_hours']:>10.1f}")

    if metrics.get("exit_breakdown"):
        print("\n" + "-" * 70)
        print(" EXIT REASON BREAKDOWN")
        print("-" * 70)
        print(f"  {'Reason':<20} {'N':>5} {'Win%':>8} {'Total PnL':>14}")
        for reason, e in metrics["exit_breakdown"].items():
            print(f"  {reason:<20} {e['n']:>5d} {e['win_rate']:>7.1%} ${e['total_pnl']:>13,.2f}")

    if metrics.get("adx_attribution"):
        print("\n" + "-" * 70)
        print(" ENTRY ADX ATTRIBUTION (Winners vs Losers)")
        print("-" * 70)
        print(f"  {'ADX Range':<15} {'N':>5} {'Win%':>8} {'Total PnL':>14} {'Avg PnL':>12}")
        for bucket, a in sorted(metrics["adx_attribution"].items()):
            print(f"  {bucket:<15} {a['n']:>5d} {a['win_rate']:>7.1%} "
                  f"${a['total_pnl']:>13,.2f} ${a['avg_pnl']:>11,.2f}")

    if metrics.get("sector_attribution"):
        print("\n" + "-" * 70)
        print(" SECTOR ATTRIBUTION")
        print("-" * 70)
        print(f"  {'Sector':<20} {'N':>5} {'Win%':>8} {'Total PnL':>14}")
        for sec, s in sorted(metrics["sector_attribution"].items(), key=lambda x: x[1]["total_pnl"], reverse=True):
            print(f"  {sec:<20} {s['n']:>5d} {s['win_rate']:>7.1%} ${s['total_pnl']:>13,.2f}")

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
