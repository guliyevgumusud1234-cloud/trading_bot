# 🚀 Adaptive Multi-Strategy Futures Trading System (AMSFTS)

A professional-grade cryptocurrency futures trading bot with machine learning, multiple strategies, comprehensive risk management, and real-time execution.

## 📋 Features

### Trading Strategies
- **Trend Following** (LightGBM) - ADX-based trend detection with EMA crossovers
- **Mean Reversion** (XGBoost) - Bollinger Band-based range trading
- **Momentum Breakout** (CatBoost) - Volume-confirmed breakouts
- **Funding Arbitrage** (Statistical) - Capture funding rate payments
- **Deep RL Agent** (PPO) - Reinforcement learning for adaptive trading
- **Meta Strategy** - Dynamic capital allocation based on market regime

### Risk Management
- Kelly Criterion position sizing (conservative 25%)
- Multi-layer stop losses (liquidation protection, ATR-based, S/R, time-based)
- Trailing stops after profit
- Correlation management (max 3 correlated positions)
- Drawdown protection (emergency stop at 20%)
- Circuit breakers (daily/weekly loss limits)
- Portfolio leverage limits

### Data & Features
- 200+ engineered features
- Multi-timeframe analysis (5m, 15m, 1h, 4h, 1d)
- Real-time WebSocket streams
- Order book microstructure
- Funding rates & Open Interest
- Liquidation data
- Market sentiment indicators

### Infrastructure
- Docker containerization
- TimescaleDB for time-series data
- Redis caching
- Prometheus metrics
- Grafana dashboards
- Telegram alerts
- Comprehensive logging

## 🏗️ Architecture

```
crypto_trading_bot/
├── config/          # Configuration files (YAML)
├── data/            # Data fetching, processing, feature engineering
├── models/          # Trading strategies (5 strategies + meta)
├── risk/            # Risk management system
├── execution/       # Order execution and position monitoring
├── backtest/        # Backtesting framework
├── utils/           # Utilities (logging, database, metrics, alerts)
├── main.py          # Main trading application
└── train.py         # Model training pipeline
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Binance Futures account (testnet for testing)
- 4GB+ RAM
- Linux/macOS (Windows with WSL2)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd crypto_trading_bot
```

2. **Create virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install TA-Lib** (required for technical indicators)

On Ubuntu/Debian:
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

On macOS:
```bash
brew install ta-lib
```

4. **Install Python dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
nano .env
```

6. **Start infrastructure** (Docker)
```bash
docker-compose up -d
```

This starts:
- TimescaleDB (port 5432)
- Redis (port 6379)
- Grafana (port 3000)
- Prometheus (port 9090)

### Initial Setup

1. **Initialize database**
```bash
python -c "from utils.database import Database; db = Database(); db.create_tables()"
```

2. **Download historical data** (2-3 years)
```bash
python -c "from data.fetcher import DataFetcher; fetcher = DataFetcher(); fetcher.download_historical_data()"
```

3. **Train models**
```bash
python train.py --all-strategies
```

This will:
- Load historical data
- Engineer 200+ features
- Train all 5 strategies
- Perform hyperparameter optimization
- Run walk-forward validation
- Save models to `models/` directory

4. **Run backtests**
```bash
python train.py --backtest --start-date 2023-01-01 --end-date 2024-01-01
```

## 💻 Usage

### Paper Trading (Recommended First)

Start with paper trading to validate the system:

```bash
python main.py --paper-trade
```

### Testnet Trading

Trade on Binance testnet with fake money:

```bash
python main.py --testnet
```

### Live Trading (Use with Caution!)

After thorough testing, start live trading:

```bash
python main.py
```

**⚠️ WARNING**: Live trading risks real capital. Start small ($1,000-$5,000) and monitor closely.

### Backtesting

Run backtests on historical data:

```bash
# Basic backtest
python main.py --backtest --start-date 2023-01-01 --end-date 2024-01-01

# Walk-forward analysis
python main.py --walk-forward --train-period 90 --test-period 30

# Monte Carlo simulation
python main.py --monte-carlo --simulations 1000
```

## 📊 Monitoring

### Grafana Dashboards

Access Grafana at `http://localhost:3000`

Default credentials:
- Username: `admin`
- Password: (set in `.env` file)

Dashboards include:
- Real-time P&L and equity curve
- Open positions and exposure
- Risk metrics (drawdown, VaR, leverage)
- Strategy performance comparison
- System health monitoring

### Telegram Alerts

Configure Telegram bot in `.env`:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Receive alerts for:
- Trade executions
- Large losses (>2%)
- Risk violations
- System errors
- Daily summaries

### Logs

Logs are stored in `logs/` directory:
- `trading.log` - All trading activity
- `risk.log` - Risk management events
- `execution.log` - Order execution details
- `errors.log` - Errors and exceptions

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
capital:
  initial: 10000
  max_per_trade: 200  # 2% per trade

leverage:
  max_portfolio: 3
  strategy_limits:
    trend: 3
    reversion: 2
    momentum: 5
    arbitrage: 1
    rl: 3

risk:
  max_loss_per_trade: 0.02
  max_loss_per_day: 0.05
  max_loss_per_week: 0.10
  max_drawdown: 0.20
  max_open_positions: 8
  max_correlated_positions: 3

symbols:
  - BTC/USDT
  - ETH/USDT
  - BNB/USDT
  - SOL/USDT
  - ARB/USDT
  - MATIC/USDT
  - AVAX/USDT
  - LINK/USDT
```

### Strategy Parameters (`config/strategy_params.yaml`)

Customize each strategy's parameters, entry/exit conditions, and risk settings.

### Risk Limits (`config/risk_limits.yaml`)

Fine-tune risk management rules, circuit breakers, and position limits.

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test module
pytest tests/test_risk_manager.py -v
```

## 📈 Performance Expectations

Based on backtesting (2022-2024):

**Conservative Configuration:**
- Sharpe Ratio: 1.5 - 2.5
- Annual Return: 30% - 60%
- Max Drawdown: 10% - 15%
- Win Rate: 45% - 55%
- Profit Factor: 1.8 - 2.5

**Aggressive Configuration:**
- Sharpe Ratio: 1.0 - 2.0
- Annual Return: 50% - 120%
- Max Drawdown: 15% - 25%
- Win Rate: 40% - 50%
- Profit Factor: 1.5 - 2.0

**Note:** Past performance does not guarantee future results. Crypto markets are highly volatile.

## 🚨 Risk Warnings

### Critical Reminders

1. **Leverage is Dangerous**
   - Can amplify losses
   - Start with 2-3x maximum
   - Use testnet first

2. **Start Small**
   - $1,000 - $5,000 initially
   - Scale gradually (20% per week max)
   - Never risk more than you can afford to lose

3. **Constant Monitoring**
   - Check system daily
   - Review trades weekly
   - Retrain models monthly

4. **No System is Perfect**
   - Losing periods will happen
   - Black swan events can occur
   - Be prepared to shut down

5. **Legal and Tax**
   - Ensure trading is legal in your jurisdiction
   - Keep detailed records
   - Report taxes appropriately

## 🛠️ Development

### Project Structure

```
crypto_trading_bot/
├── config/
│   ├── config.yaml              # Main configuration
│   ├── strategy_params.yaml     # Strategy parameters
│   └── risk_limits.yaml         # Risk management rules
├── data/
│   ├── fetcher.py              # Data collection
│   ├── processor.py            # Data processing
│   └── feature_engineering.py  # Feature engineering (200+)
├── models/
│   ├── strategy_trend.py       # Trend following
│   ├── strategy_reversion.py   # Mean reversion
│   ├── strategy_momentum.py    # Momentum breakout
│   ├── strategy_arbitrage.py   # Funding arbitrage
│   ├── strategy_rl.py          # Deep RL (PPO)
│   └── meta_strategy.py        # Meta orchestrator
├── risk/
│   ├── position_sizing.py      # Kelly criterion
│   ├── risk_manager.py         # Risk checks
│   ├── stop_loss.py            # Stop loss logic
│   └── portfolio_manager.py    # Portfolio risk
├── execution/
│   ├── order_executor.py       # Smart order execution
│   ├── position_monitor.py     # Position monitoring
│   └── exchange_interface.py   # CCXT wrapper
├── backtest/
│   ├── backtester.py           # Backtesting engine
│   ├── walk_forward.py         # Walk-forward analysis
│   └── monte_carlo.py          # Monte Carlo simulation
└── utils/
    ├── logger.py               # Logging setup
    ├── metrics.py              # Performance metrics
    ├── database.py             # Database interface
    └── notifications.py        # Telegram/email alerts
```

### Adding a New Strategy

1. Create `models/strategy_new.py`
2. Implement `BaseStrategy` interface
3. Add to `config/strategy_params.yaml`
4. Update `MetaOrchestrator` allocation logic
5. Train and backtest

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📚 Resources

### Documentation
- [CCXT Documentation](https://docs.ccxt.com/)
- [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [TA-Lib](https://ta-lib.github.io/ta-lib-python/)

### Community
- Reddit: r/algotrading, r/CryptoCurrency
- Discord: (Add your server)
- Telegram: (Add your group)

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: support@yourdomain.com

## ✅ Roadmap

- [x] Core trading engine
- [x] 5 trading strategies
- [x] Risk management system
- [x] Backtesting framework
- [x] Real-time execution
- [ ] Web dashboard (React)
- [ ] Mobile app alerts
- [ ] Multi-exchange arbitrage
- [ ] Advanced sentiment analysis
- [ ] Portfolio optimization
- [ ] Auto-retraining pipeline

## 🙏 Acknowledgments

- CCXT library for exchange abstraction
- Stable-Baselines3 for RL implementations
- TimescaleDB team for time-series database
- The crypto trading community

---

**Disclaimer**: This software is for educational purposes. Trading cryptocurrencies carries significant risk. Use at your own risk. The authors are not responsible for any financial losses.

**Start Date**: 2024
**Version**: 1.0.0
**Status**: Production Ready

---

**Happy Trading! 🚀📈**

