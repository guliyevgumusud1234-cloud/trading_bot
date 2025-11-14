"""
Main Trading Application.

This is the heart of the trading bot:
- Loads all configurations and models
- Initializes all systems
- Runs main trading loop
- Monitors positions in real-time
- Handles errors and shutdown gracefully
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import yaml
import pandas as pd

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Data
from data.fetcher import DataFetcher
from data.processor import DataProcessor
from data.feature_engineering import FeatureEngineer

# Strategies
from models.strategy_trend import TrendFollowingStrategy
from models.strategy_reversion import MeanReversionStrategy
from models.strategy_momentum import MomentumBreakoutStrategy
from models.strategy_arbitrage import FundingArbitrageStrategy
from models.strategy_rl import DeepRLStrategy
from models.meta_strategy import MetaOrchestrator

# Risk & Execution
from risk.risk_manager import RiskManager
from execution.exchange_interface import ExchangeInterface
from execution.order_executor import SmartOrderExecutor
from execution.position_monitor import RealTimeMonitor

# Utils
from utils.logger import get_logger
from utils.database import DatabaseManager
from utils.notifications import NotificationManager, AlertLevel

logger = get_logger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.
    """
    
    def __init__(
        self,
        config_path: str = 'config/config.yaml',
        paper_trading: bool = False,
        testnet: bool = False
    ):
        """
        Initialize trading bot.
        
        Args:
            config_path: Path to configuration
            paper_trading: Run in paper trading mode
            testnet: Use testnet/sandbox
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paper_trading = paper_trading
        self.testnet = testnet
        self.is_running = False
        
        # Components (will be initialized)
        self.exchange = None
        self.order_executor = None
        self.position_monitor = None
        self.risk_manager = None
        self.data_fetcher = None
        self.feature_engineer = None
        self.strategies = {}
        self.meta_orchestrator = None
        self.notifier = None
        self.db = None
        
        logger.info("="*80)
        logger.info("TRADING BOT INITIALIZING")
        logger.info(f"Mode: {'PAPER TRADING' if paper_trading else 'LIVE'}")
        logger.info(f"Environment: {'TESTNET' if testnet else 'PRODUCTION'}")
        logger.info("="*80)
    
    def initialize(self):
        """Initialize all components."""
        logger.info("\n[1/9] Initializing Exchange Interface...")
        if self.paper_trading:
            # Paper trading: Try to initialize exchange, but don't fail if it doesn't work
            try:
                self.exchange = ExchangeInterface(
                    exchange_id='binance',
                    api_key=os.getenv('BINANCE_API_KEY'),
                    api_secret=os.getenv('BINANCE_API_SECRET'),
                    testnet=self.testnet
                )
                logger.info("✅ Exchange interface initialized (optional for paper trading)")
            except Exception as e:
                logger.warning(f"⚠️  Exchange interface failed (paper trading can continue): {e}")
                self.exchange = None
        else:
            # Live trading: Exchange is required
            self.exchange = ExchangeInterface(
                exchange_id='binance',
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_API_SECRET'),
                testnet=self.testnet
            )
        
        logger.info("[2/9] Initializing Order Executor...")
        self.order_executor = SmartOrderExecutor(
            exchange=self.exchange,
            max_slippage=0.002,
            use_twap=True
        )
        
        logger.info("[3/9] Initializing Position Monitor...")
        self.position_monitor = RealTimeMonitor(
            exchange=self.exchange,
            check_interval=1.0
        )
        
        logger.info("[4/9] Initializing Risk Manager...")
        initial_balance = self.config.get('capital', {}).get('initial', 10000)
        self.risk_manager = RiskManager(initial_balance=initial_balance)
        
        logger.info("[5/9] Initializing Data Systems...")
        self.data_fetcher = DataFetcher(
            exchange_id='binance',
            testnet=self.testnet
        )
        self.feature_engineer = FeatureEngineer()
        
        logger.info("[6/9] Loading Trading Strategies...")
        self._load_strategies()
        
        logger.info("[7/9] Initializing Meta Orchestrator...")
        self.meta_orchestrator = MetaOrchestrator()
        
        logger.info("[8/9] Initializing Notification System...")
        telegram_enabled = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
        self.notifier = NotificationManager(
            telegram_enabled=telegram_enabled,
            email_enabled=False
        )
        
        logger.info("[9/9] Initializing Database...")
        self.db = DatabaseManager(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'trading_db'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD')
        )
        
        logger.info("\n✅ All components initialized successfully")
        
        # Send startup notification
        try:
            self.notifier.send_alert(
                AlertLevel.INFO,
                'Trading Bot Started',
                f"Mode: {'Paper' if self.paper_trading else 'Live'}\n"
                f"Environment: {'Testnet' if self.testnet else 'Production'}\n"
                f"Strategies: {len(self.strategies)}"
            )
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")
    
    def _load_strategies(self):
        """Load all trained strategy models."""
        models_dir = 'models/trained'
        
        # Try to load each strategy
        strategy_classes = {
            'trend': TrendFollowingStrategy,
            'reversion': MeanReversionStrategy,
            'momentum': MomentumBreakoutStrategy,
            'arbitrage': FundingArbitrageStrategy,
            'rl': DeepRLStrategy
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            try:
                strategy = strategy_class()
                
                # Try different possible model file names and extensions
                possible_paths = [
                    f"{models_dir}/{strategy_name}_model.pkl",
                    f"{models_dir}/{strategy_name}_model.cbm",  # CatBoost
                    f"{models_dir}/{strategy_name}_model.txt",  # LightGBM/XGBoost
                    f"{models_dir}/{strategy_name}_breakout.cbm",  # Momentum specific
                    f"{models_dir}/{strategy_name}_following.txt",  # Trend specific
                    f"{models_dir}/{strategy_name}_reversion.txt",  # Reversion specific
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path:
                    strategy.load_model(model_path)
                    logger.info(f"  ✅ Loaded {strategy_name} from {model_path}")
                else:
                    logger.warning(f"  ⚠️  Model file not found for {strategy_name}, using untrained model")
                
                self.strategies[strategy_name] = strategy
                
            except Exception as e:
                logger.error(f"  ❌ Failed to load {strategy_name}: {e}")
    
    def run(self):
        """Run the main trading loop."""
        self.is_running = True
        
        # Start position monitor
        self.position_monitor.start()
        
        logger.info("\n" + "="*80)
        logger.info("TRADING BOT RUNNING")
        logger.info("="*80)
        
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                cycle_start = time.time()
                
                logger.info(f"\n{'='*40} Iteration {iteration} {'='*40}")
                
                try:
                    # 1. Fetch latest market data
                    market_data = self._fetch_market_data()
                    
                    # 2. Calculate features
                    features = self._calculate_features(market_data)
                    
                    # 3. Detect market regime
                    market_regime = self.meta_orchestrator.detect_market_regime(
                        features.iloc[-1]
                    )
                    logger.info(f"Market Regime: {market_regime}")
                    
                    # 4. Get strategy allocations
                    strategy_weights = self.meta_orchestrator.allocate_strategies(
                        market_regime=market_regime
                    )
                    logger.info(f"Strategy Weights: {strategy_weights}")
                    
                    # 5. Generate signals from all strategies
                    signals = self._generate_signals(features, market_data)
                    
                    # 6. Filter and validate signals
                    approved_signals = self._validate_signals(signals, strategy_weights)
                    
                    # 7. Execute approved signals
                    if approved_signals:
                        self._execute_signals(approved_signals)
                    
                    # 8. Update metrics
                    self._update_metrics()
                    
                    # 9. Check for alerts
                    self._check_alerts()
                    
                except Exception as e:
                    logger.error(f"Error in trading loop iteration {iteration}: {e}")
                    self.notifier.send_alert(AlertLevel.CRITICAL, 'Trading Loop Error', str(e))
                
                # Sleep until next iteration (15 seconds)
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, 15 - cycle_duration)
                
                logger.info(f"Cycle completed in {cycle_duration:.2f}s, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("\n\nKeyboard interrupt received, shutting down...")
        except Exception as e:
            logger.critical(f"Fatal error in main loop: {e}")
            self.notifier.send_alert(AlertLevel.CRITICAL, 'Fatal Error', str(e))
        finally:
            self.shutdown()
    
    def _fetch_market_data(self) -> Dict:
        """Fetch latest market data for all symbols."""
        market_data = {}
        
        # Get all symbols from config (primary + secondary)
        symbols = []
        symbols_config = self.config.get('symbols', {})
        
        # Handle different config structures
        if isinstance(symbols_config, dict):
            if 'primary' in symbols_config and isinstance(symbols_config['primary'], list):
                symbols.extend(symbols_config['primary'])
            if 'secondary' in symbols_config and isinstance(symbols_config['secondary'], list):
                symbols.extend(symbols_config['secondary'])
        elif isinstance(symbols_config, list):
            symbols = symbols_config
        
        # If no symbols found, use default
        if not symbols:
            symbols = ['BTC/USDT', 'ETH/USDT']
            logger.warning("No symbols found in config, using defaults")
        
        for symbol in symbols:
            try:
                if self.exchange is None:
                    # Paper trading: Use mock data
                    market_data[symbol] = {
                        'price': 50000.0 if 'BTC' in symbol else 3000.0,
                        'bid': 50000.0 if 'BTC' in symbol else 3000.0,
                        'ask': 50001.0 if 'BTC' in symbol else 3001.0,
                        'volume': 1000000.0,
                        'timestamp': datetime.now()
                    }
                else:
                    ticker = self.exchange.fetch_ticker(symbol)
                    market_data[symbol] = {
                        'price': ticker['last'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'volume': ticker['volume'],
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                # Fallback to mock data
                market_data[symbol] = {
                    'price': 50000.0 if 'BTC' in symbol else 3000.0,
                    'bid': 50000.0 if 'BTC' in symbol else 3000.0,
                    'ask': 50001.0 if 'BTC' in symbol else 3001.0,
                    'volume': 1000000.0,
                    'timestamp': datetime.now()
                }
        
        return market_data
    
    def _calculate_features(self, market_data: Dict) -> pd.DataFrame:
        """Calculate features from market data."""
        # Fetch historical data and calculate real features
        try:
            # Use primary symbol (BTC/USDT) and primary timeframe (15m)
            primary_symbol = 'BTC/USDT'
            primary_timeframe = '15m'
            
            # Fetch last 500 bars for feature calculation
            if self.data_fetcher is not None:
                since = datetime.now() - timedelta(days=30)  # Last 30 days
                df = self.data_fetcher.fetch_ohlcv(
                    symbol=primary_symbol,
                    timeframe=primary_timeframe,
                    since=since,
                    limit=500
                )
            else:
                # Fallback: create minimal OHLCV from current price
                current_price = market_data.get(primary_symbol, {}).get('price', 50000)
                dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
                df = pd.DataFrame({
                    'open': current_price * (1 + np.random.randn(100) * 0.001),
                    'high': current_price * (1 + np.abs(np.random.randn(100) * 0.002)),
                    'low': current_price * (1 - np.abs(np.random.randn(100) * 0.002)),
                    'close': current_price * (1 + np.random.randn(100) * 0.001),
                    'volume': np.random.uniform(100, 10000, 100)
                }, index=dates)
            
            if df.empty:
                logger.warning("No historical data available, using placeholder features")
                return self._create_placeholder_features(market_data)
            
            # Calculate all features using FeatureEngineer
            features_df = self.feature_engineer.create_all_features(df)
            
            # Return only the latest row (most recent features)
            if len(features_df) > 0:
                return features_df.iloc[-1:].copy()
            else:
                return self._create_placeholder_features(market_data)
                
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return self._create_placeholder_features(market_data)
    
    def _create_placeholder_features(self, market_data: Dict) -> pd.DataFrame:
        """Create placeholder features when real data is unavailable."""
        import pandas as pd
        import numpy as np
        
        current_price = market_data.get('BTC/USDT', {}).get('price', 50000)
        
        # Create minimal feature set
        features = pd.DataFrame({
            'close': [current_price],
            'adx_14': [25.0],
            'ema_8': [current_price * 1.01],
            'ema_50': [current_price * 0.99],
            'atr_14_pct': [0.03],
            'volume_ratio_20': [1.2],
            'funding_rate': [0.0003]
        })
        
        return features
    
    def _generate_signals(self, features: pd.DataFrame, market_data: Dict) -> Dict:
        """Generate signals from all strategies."""
        signals = {}
        
        current_price = market_data['BTC/USDT']['price']
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy_name == 'arbitrage':
                    # Arbitrage strategy needs funding rate
                    signal, confidence, params = strategy.generate_signal(
                        current_funding_rate=features.iloc[-1].get('funding_rate', 0),
                        symbol='BTC/USDT',
                        current_price=current_price
                    )
                elif strategy_name == 'rl':
                    # RL strategy needs current_position, not current_price
                    current_position = {}  # Empty for paper trading
                    signal, confidence, params = strategy.generate_signal(
                        features=features,
                        current_position=current_position
                    )
                else:
                    signal, confidence, params = strategy.generate_signal(
                        features=features,
                        current_price=current_price
                    )
                
                if signal != 'NEUTRAL' and confidence > 0.6:
                    signals[strategy_name] = {
                        'signal': signal,
                        'confidence': confidence,
                        'params': params
                    }
                    logger.info(f"  {strategy_name}: {signal} (confidence: {confidence:.2f})")
                    
            except Exception as e:
                logger.error(f"Error generating signal from {strategy_name}: {e}")
        
        return signals
    
    def _validate_signals(self, signals: Dict, strategy_weights: Dict) -> List:
        """Validate signals through risk manager."""
        approved_signals = []
        
        for strategy_name, signal_info in signals.items():
            # Check strategy weight
            weight = strategy_weights.get(strategy_name, 0)
            if weight < 0.05:  # Skip if weight < 5%
                logger.info(f"  Skipping {strategy_name}: weight too low ({weight:.1%})")
                continue
            
            # Validate through risk manager
            trade_params = signal_info['params']
            if trade_params:
                is_approved, reasons = self.risk_manager.validate_trade(trade_params)
                
                if is_approved:
                    logger.info(f"  ✅ {strategy_name} signal approved")
                    approved_signals.append({
                        'strategy': strategy_name,
                        'signal': signal_info['signal'],
                        'params': trade_params,
                        'weight': weight
                    })
                else:
                    logger.warning(f"  ❌ {strategy_name} signal rejected: {reasons}")
        
        return approved_signals
    
    def _execute_signals(self, approved_signals: List):
        """Execute approved trading signals."""
        for signal_info in approved_signals:
            if self.paper_trading:
                logger.info(f"[PAPER] Would execute {signal_info['signal']} for {signal_info['strategy']}")
                continue
            
            try:
                logger.info(f"Executing {signal_info['signal']} for {signal_info['strategy']}...")
                
                params = signal_info['params']
                
                result = self.order_executor.execute_order(
                    symbol='BTC/USDT',  # Simplified
                    side='buy' if signal_info['signal'] == 'LONG' else 'sell',
                    amount=0.001,  # Simplified
                    leverage=params.get('max_leverage', 2),
                    urgency='normal'
                )
                
                if result['success']:
                    logger.info(f"✅ Order executed successfully")
                    self.notifier.send_alert(
                        AlertLevel.INFO,
                        'Trade Executed',
                        f"Strategy: {signal_info['strategy']}\n"
                        f"Side: {signal_info['signal']}\n"
                        f"Price: ${result.get('filled_price', 0):.2f}"
                    )
                else:
                    logger.error(f"❌ Order execution failed: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
    
    def _update_metrics(self):
        """Update performance metrics."""
        try:
            # Get position summary
            summary = self.position_monitor.get_positions_summary()
            
            # Log metrics
            logger.info(f"\nPositions: {summary['num_positions']}")
            logger.info(f"Total P&L: ${summary.get('total_pnl', 0):.2f}")
            logger.info(f"Exposure: ${summary.get('total_exposure', 0):,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions."""
        try:
            # Check risk violations
            # Get current positions (empty list for paper trading without exchange)
            current_positions = []
            if self.exchange is not None:
                try:
                    current_positions = self.exchange.fetch_positions()
                except:
                    pass
            
            violations = self.risk_manager.check_risk_violations(current_positions)
            
            if violations:
                for violation in violations:
                    logger.warning(f"⚠️  Risk violation: {violation}")
                    self.notifier.send_alert(
                        AlertLevel.WARNING,
                        'Risk Violation',
                        f"Violation: {violation}"
                    )
                    
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("\n" + "="*80)
        logger.info("SHUTTING DOWN TRADING BOT")
        logger.info("="*80)
        
        self.is_running = False
        
        # Stop position monitor
        if self.position_monitor:
            self.position_monitor.stop()
        
        # Get final stats
        if self.order_executor:
            stats = self.order_executor.get_execution_stats()
            logger.info(f"\nExecution Stats:")
            logger.info(f"  Total executions: {stats.get('total_executions', 0)}")
            logger.info(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        
        # Send shutdown notification
        if self.notifier:
            self.notifier.send_alert(
                AlertLevel.INFO,
                'Trading Bot Stopped',
                'Bot has been shut down gracefully'
            )
        
        logger.info("\n✅ Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Crypto Futures Trading Bot')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--testnet', action='store_true', help='Use testnet/sandbox')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = TradingBot(
        config_path=args.config,
        paper_trading=args.paper,
        testnet=args.testnet
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutdown signal received")
        bot.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run
    try:
        bot.initialize()
        bot.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

