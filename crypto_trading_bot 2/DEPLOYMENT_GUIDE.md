# 🚀 Deployment Guide - Crypto Trading Bot

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Configuration](#configuration)
4. [Training Models](#training-models)
5. [Running the Bot](#running-the-bot)
6. [Monitoring](#monitoring)
7. [Maintenance](#maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 22.04 recommended) or macOS
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 50GB+ free space
- **CPU**: 4+ cores recommended
- **Network**: Stable internet connection (low latency to Binance)

### Software Requirements
- **Docker**: 24.0+ and Docker Compose 2.0+
- **Python**: 3.11 (if running without Docker)
- **Git**: For version control

---

## Initial Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd crypto_trading_bot
```

### 2. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required Environment Variables:**
```bash
# Binance API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true  # Start with testnet!

# Database
DB_PASSWORD=your_secure_password
DB_NAME=trading_db
DB_USER=postgres

# Grafana
GRAFANA_PASSWORD=your_grafana_password
GRAFANA_USER=admin

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### 3. Create Binance API Keys

**⚠️ IMPORTANT SECURITY SETTINGS:**
1. Go to Binance → API Management
2. Create API Key with:
   - ✅ Enable Reading
   - ✅ Enable Futures
   - ❌ NO Enable Withdrawals
3. Whitelist your server IP
4. **START WITH TESTNET FIRST**

---

## Configuration

### 1. Main Configuration (`config/config.yaml`)
```yaml
capital:
  initial_capital: 10000  # Starting capital
  max_per_trade: 200      # Max $200 per trade (2%)

risk_limits:
  max_loss_per_trade: 0.02  # 2%
  max_loss_per_day: 0.05    # 5%
  max_drawdown: 0.20        # 20%
  max_open_positions: 8
  max_portfolio_leverage: 3
```

### 2. Strategy Parameters (`config/strategy_params.yaml`)
Adjust strategy-specific settings:
- Model hyperparameters
- Entry/exit thresholds
- Leverage limits per strategy

### 3. Risk Limits (`config/risk_limits.yaml`)
Fine-tune risk management rules:
- Position limits
- Daily/weekly loss limits
- Drawdown thresholds

---

## Training Models

### Option 1: Using Docker
```bash
# Build the image
docker-compose build trading_bot

# Run training
docker-compose run --rm trading_bot python train.py
```

### Option 2: Local Python
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --start-date 2022-01-01
```

**Training Duration:** 2-4 hours depending on hardware

**Output:**
- Trained models saved to `models/trained/`
- Training report: `models/trained/training_report.yaml`

---

## Running the Bot

### 🧪 Paper Trading (RECOMMENDED FIRST)

**Start with paper trading to test without risking real money:**

```bash
# Using Docker
docker-compose up -d
docker-compose logs -f trading_bot

# Or locally
python main.py --paper --testnet
```

**Paper Trading Checklist:**
- [ ] Run for at least 7 days
- [ ] Monitor performance metrics
- [ ] Check for errors in logs
- [ ] Verify risk limits work
- [ ] Test emergency shutdown

### 🔴 Live Trading (PRODUCTION)

**⚠️ ONLY after successful paper trading!**

```bash
# Update .env
BINANCE_TESTNET=false

# Start full stack
docker-compose up -d

# Check logs
docker-compose logs -f trading_bot
```

### Commands
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart trading bot only
docker-compose restart trading_bot

# View logs
docker-compose logs -f trading_bot

# Emergency stop
docker-compose stop trading_bot

# Complete shutdown
docker-compose down -v  # WARNING: Deletes all data!
```

---

## Monitoring

### Grafana Dashboard
**URL:** http://localhost:3000  
**Login:** admin / your_grafana_password

**Default Dashboards:**
1. **Trading Overview**
   - Open positions
   - P&L charts
   - Win rate
   - Strategy performance

2. **Risk Metrics**
   - Current drawdown
   - Position sizes
   - Leverage usage
   - Risk violations

3. **System Health**
   - API latency
   - Database performance
   - Error rates

### Prometheus Metrics
**URL:** http://localhost:9090

**Key Metrics:**
- `trades_total` - Total number of trades
- `pnl_total` - Total profit/loss
- `open_positions` - Current open positions
- `current_drawdown` - Current drawdown percentage

### Database Access
**PgAdmin URL:** http://localhost:5050  
(Start with: `docker-compose --profile tools up -d`)

### Telegram Alerts
Configure Telegram bot to receive:
- 🟢 Trade executions
- 🟡 Risk warnings
- 🔴 Critical errors
- 📊 Daily summaries

---

## Maintenance

### Daily Tasks
```bash
# Check bot status
docker-compose ps

# Review logs
docker-compose logs --tail=100 trading_bot

# Check positions
docker-compose exec trading_bot python -c "from execution.exchange_interface import ExchangeInterface; print(ExchangeInterface().fetch_positions())"
```

### Weekly Tasks
- Review performance metrics in Grafana
- Check for any risk violations
- Update strategy weights if needed
- Backup database

### Model Retraining
```bash
# Retrain models (recommended weekly)
docker-compose run --rm trading_bot python train.py

# Restart bot to load new models
docker-compose restart trading_bot
```

### Database Backup
```bash
# Backup database
docker-compose exec timescaledb pg_dump -U postgres trading_db > backup_$(date +%Y%m%d).sql

# Restore database
docker-compose exec -T timescaledb psql -U postgres trading_db < backup_20240101.sql
```

### Log Rotation
Logs automatically rotate when they reach 10MB. Old logs are kept for 30 days.

---

## Troubleshooting

### Bot Won't Start
```bash
# Check logs
docker-compose logs trading_bot

# Common issues:
# 1. Invalid API keys → Check .env
# 2. Database not ready → Wait 30s and retry
# 3. Missing models → Run training first
```

### No Trades Being Executed
1. Check if bot is in paper trading mode
2. Verify API keys have futures permission
3. Check risk limits aren't blocking trades
4. Review strategy signals in logs

### High API Latency
```bash
# Check API latency
docker-compose exec trading_bot python -c "from execution.exchange_interface import ExchangeInterface; print(f'Latency: {ExchangeInterface().get_api_latency()}ms')"

# Solutions:
# 1. Move server closer to Binance (Singapore region)
# 2. Check network connection
# 3. Reduce check interval in config
```

### Database Connection Errors
```bash
# Restart database
docker-compose restart timescaledb

# Check if database is healthy
docker-compose exec timescaledb pg_isready
```

### Out of Memory
```bash
# Check memory usage
docker stats

# Solutions:
# 1. Increase Docker memory limit
# 2. Reduce number of symbols
# 3. Reduce feature count
```

### Emergency Shutdown
```bash
# Stop trading immediately
docker-compose stop trading_bot

# Close all positions manually via Binance UI
# Review logs before restarting
```

---

## Performance Optimization

### 1. Reduce Latency
- Deploy server in Singapore (closest to Binance)
- Use dedicated server (not shared hosting)
- Optimize WebSocket connections

### 2. Improve Execution
- Use limit orders when possible
- Implement smart order splitting for large trades
- Monitor and adjust slippage limits

### 3. Model Performance
- Retrain models weekly with latest data
- Monitor walk-forward validation metrics
- Adjust strategy weights based on performance

---

## Safety Checklist

### Before Going Live
- [ ] Tested on testnet for 7+ days
- [ ] All risk limits configured correctly
- [ ] Stop loss logic verified
- [ ] Telegram alerts working
- [ ] Monitoring dashboards setup
- [ ] Backup and recovery tested
- [ ] Emergency shutdown procedure documented
- [ ] API keys have withdrawal disabled
- [ ] Starting with small capital (<$1000)

### Daily Checks
- [ ] Bot is running
- [ ] No critical errors in logs
- [ ] Risk limits not violated
- [ ] P&L within expectations
- [ ] All positions have stop losses

---

## Support & Updates

### Getting Help
1. Check logs: `docker-compose logs`
2. Review this guide
3. Check Grafana dashboards
4. Review code documentation

### Updating the Bot
```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose build

# Restart services
docker-compose down
docker-compose up -d
```

---

## ⚠️ DISCLAIMERS

1. **Trading Risk**: Cryptocurrency trading carries substantial risk. Never trade with money you cannot afford to lose.

2. **No Guarantees**: Past performance does not guarantee future results. This bot can and will lose money.

3. **Testing Required**: Always test thoroughly in paper trading mode before using real money.

4. **Monitoring Required**: This is NOT a "set and forget" system. Active monitoring is essential.

5. **Responsibility**: You are solely responsible for your trading decisions and any losses incurred.

---

## Quick Reference

### Essential Commands
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f trading_bot

# Restart
docker-compose restart trading_bot

# Emergency Stop
docker-compose stop trading_bot

# Train Models
docker-compose run --rm trading_bot python train.py

# Paper Trading
docker-compose run --rm trading_bot python main.py --paper
```

### Important URLs
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- PgAdmin: http://localhost:5050
- Logs: `logs/trading.log`

---

**Remember: Start small, test thoroughly, monitor actively!** 🚀

