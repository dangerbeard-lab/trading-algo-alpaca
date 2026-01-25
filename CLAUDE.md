# Trading Bot Project - Claude Code Context

## Overview

Automated algorithmic trading system targeting 30%+ annual returns through adaptive market regime strategies. Runs 24/7 on a DigitalOcean droplet, executing trades every 15 minutes during market hours via the Alpaca API.

**Portfolio size**: ~$100k
**Account type**: Margin (2:1 leverage available - intentionally avoided)
**Platform**: Alpaca (equities + crypto)
**Droplet**: root@24.199.109.2:/home/trader/trading-bot/

## Architecture
```
├── enhanced_trading_bot.py   # Core trading logic, strategy execution
├── runner.py                 # Scheduling, 15-minute execution cycles
├── config.json               # Trading parameters, thresholds, symbol universe
├── positions.json            # Persistent state tracking for positions
└── trading_bot.log           # Runtime logs
```

**Deployment**: systemd service (`trading-bot.service`) on DigitalOcean droplet
**Credentials**: Stored in /etc/systemd/system/trading-bot.service as environment variables - NEVER commit these

## Trading Strategy

### Regime Detection
- **ADX (Average Directional Index)** determines market regime
- ADX > threshold → Trending market
- ADX ≤ threshold → Ranging market

### Trend-Following Mode (High ADX)
- EMA crossover signals (fast/slow)
- MACD confirmation required
- Follows momentum

### Mean Reversion Mode (Low ADX)
- RSI for overbought/oversold conditions
- Bollinger Bands for price extremes
- Fades moves to the mean

## Risk Management - CRITICAL

| Parameter | Value | Notes |
|-----------|-------|-------|
| Risk per trade | 2% | Of portfolio value |
| Position sizing | 4-10% | Based on ATR volatility |
| Max single position | 10% | **Enforce strictly** |
| Max concurrent positions | 25 | |
| Max portfolio exposure | 90% | |
| Trailing stop | 12% | ATR-based |

### Position Sizing Logic
**IMPORTANT**: Size based on **cash balance**, not buying power or portfolio_value. Margin account provides 2:1 leverage by default - using buying power causes unintentional leverage.

## Critical Bug History

### Duplicate Order Bug (Previously Fixed)
**Problem**: Race conditions caused multiple trading cycles to execute simultaneously, placing duplicate orders. SOL reached 40% of portfolio instead of 10% maximum.

**Fixes implemented in runner.py**:
1. File locking prevents concurrent trading cycles
2. Last run time tracking prevents duplicate runs in same interval

**Still needed in enhanced_trading_bot.py**:
- Check for pending orders before placing new trades for same symbol

## Deployment Commands
```bash
# Restart service after changes
sudo systemctl restart trading-bot

# View logs
journalctl -u trading-bot -f

# Check status
sudo systemctl status trading-bot
```

## Syncing Repo and Droplet
```bash
# Pull from droplet to local
scp root@24.199.109.2:/home/trader/trading-bot/enhanced_trading_bot.py ./
scp root@24.199.109.2:/home/trader/trading-bot/runner.py ./
scp root@24.199.109.2:/home/trader/trading-bot/config.json ./

# Push to droplet
scp enhanced_trading_bot.py root@24.199.109.2:/home/trader/trading-bot/
scp runner.py root@24.199.109.2:/home/trader/trading-bot/
scp config.json root@24.199.109.2:/home/trader/trading-bot/
sudo systemctl restart trading-bot
```

## Warnings

1. **This trades real money** - be extremely careful with changes
2. **Never commit credentials** - API keys stay in systemd service only
3. **Never size positions using buying power** - use cash only to avoid leverage
4. **Always check for pending orders** before placing new ones for same symbol
5. **Crypto trades 24/7** - the bot doesn't stop on weekends for crypto positions
6. **Test changes carefully** - consider paper trading for significant logic changes
