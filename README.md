# Enhanced Hybrid Trading Bot for Alpaca

A sophisticated algorithmic trading bot for Alpaca paper trading, featuring volatility-adjusted position sizing, correlation filtering, trailing stops, and comprehensive performance tracking.

## Features

| Feature | Description |
|---------|-------------|
| **Trailing Stops** | Activates after 5% profit, trails 12% from peak price |
| **ATR Position Sizing** | Volatile assets get smaller positions, steadier ones get larger |
| **Correlation Filter** | Limits highly-correlated positions to reduce drawdown risk |
| **ADX Hysteresis** | Prevents regime whipsawing with separate entry/exit thresholds |
| **Volume Confirmation** | Only triggers signals when volume exceeds average |
| **Time-of-Day Filter** | Avoids volatile market open/close periods |
| **Performance Metrics** | Tracks win rate, Sharpe ratio, R-multiples by symbol and regime |

## Instruments Traded

- **Crypto:** BTC/USD
- **ETFs:** SPY, QQQ, IWM
- **Stocks:** NVDA, META, AMZN, GOOGL, MSFT, AAPL, TSLA, AMD, NFLX, AVGO, MSTR

## Strategy Logic

The bot uses a **hybrid approach** that adapts to market conditions:

**Trending Markets (ADX > 25):**
- EMA(9)/EMA(21) crossover
- MACD confirmation
- Directional movement (+DI/-DI) confirmation

**Ranging Markets (ADX < 20):**
- RSI oversold/overbought conditions
- Bollinger Band touches
- RSI momentum shift confirmation

## File Structure

```
├── enhanced_trading_bot.py   # Main algorithm
├── runner.py                 # Continuous scheduler
├── config.json               # All configurable parameters
├── requirements.txt          # Python dependencies
├── trading_bot.log           # Runtime logs (created on first run)
└── trading_metrics.json      # Performance data (created on first run)
```

## Quick Start (Local)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export ALPACA_API_KEY='your-api-key'
export ALPACA_SECRET_KEY='your-secret-key'
```

To persist across terminal sessions, add these lines to `~/.zshrc` (Mac) or `~/.bashrc` (Linux).

### 3. Run

```bash
# Single test run
python runner.py --once

# Continuous (hourly)
python runner.py

# Every 30 minutes
python runner.py --interval 30

# View performance metrics
python runner.py --metrics
```

## Digital Ocean Deployment

See the deployment guide for running 24/7 on a $4/month droplet.

## Configuration

All parameters are in `config.json`. Key settings:

```json
{
    "position_sizing": {
        "max_positions": 8,
        "max_portfolio_exposure": 0.80,
        "atr_target_risk": 0.02
    },
    "trailing_stop": {
        "enabled": true,
        "trailing_stop_pct": 0.12,
        "activation_pct": 0.05
    },
    "adx_hysteresis": {
        "trending_entry_threshold": 25,
        "trending_exit_threshold": 20
    }
}
```

## Understanding the Logs

| Message | Meaning |
|---------|---------|
| `Regime change RANGING -> TRENDING` | Market conditions shifted |
| `Skipping due to correlation filter` | Would exceed correlated position limit |
| `Trailing stop triggered` | Position closed as price fell from peak |
| `Volume too low` | Signal rejected due to weak volume |
| `ATR=2.5%, Clamped=10%` | Position sized based on volatility |

## Performance Metrics

The bot tracks and saves:
- Win rate and total PnL
- Average R-multiple (profit relative to initial risk)
- Sharpe ratio (risk-adjusted returns)
- Max drawdown
- Performance breakdown by symbol and regime

View anytime with:
```bash
python runner.py --metrics
```

## Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max Drawdown | 10% | Halts trading if portfolio drops this much |
| Take Profit | 20% | Closes position at this gain |
| Trailing Stop | 12% | Distance from peak price |
| Max Positions | 8 | Maximum concurrent positions |
| Max Exposure | 80% | Maximum portfolio allocation |

## Disclaimer

This bot is for **paper trading and educational purposes only**. Past performance does not guarantee future results. Do not use real money without thorough backtesting and understanding of the risks involved.

Algorithmic trading carries significant risk of loss. Never trade with money you cannot afford to lose.
