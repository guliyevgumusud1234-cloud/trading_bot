"""
Backtest Script - RL Olmadan
RL stratejisi olmadan diğer 4 stratejiyi test eder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml

from data.fetcher import DataFetcher
from data.processor import DataProcessor
from data.feature_engineering import FeatureEngineer
from models.strategy_trend import TrendFollowingStrategy
from models.strategy_reversion import MeanReversionStrategy
from models.strategy_momentum import MomentumBreakoutStrategy
from models.strategy_arbitrage import FundingArbitrageStrategy
from models.meta_strategy import MetaOrchestrator
from backtest.backtester import RealisticBacktester
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_data(n_days=90):
    """Generate synthetic OHLCV data for backtesting."""
    logger.info(f"Generating {n_days} days of synthetic data...")
    
    # Generate timestamps (15-minute intervals)
    n_bars = n_days * 24 * 4  # 4 bars per hour
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_days),
        periods=n_bars,
        freq='15min'
    )
    
    # Generate realistic price data (random walk with trend)
    np.random.seed(42)
    base_price = 50000.0
    returns = np.random.normal(0.0001, 0.01, n_bars)  # Small positive drift
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
        'close': prices,
        'volume': np.random.uniform(1000000, 5000000, n_bars)
    }, index=dates)
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    logger.info(f"✅ Generated {len(df)} bars")
    return df


def main():
    """Run backtest without RL strategy."""
    logger.info("="*80)
    logger.info("BACKTEST - RL OLMAYAN STRATEJİLER")
    logger.info("="*80)
    logger.info("Stratejiler: Trend, Reversion, Momentum, Arbitrage")
    logger.info("RL: DEVRE DIŞI")
    logger.info("")
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate or load data
    logger.info("📊 Preparing data...")
    try:
        # Try to use real data if available
        data_fetcher = DataFetcher(exchange_id='binance', testnet=False)
        df = data_fetcher.fetch_ohlcv(
            symbol='BTC/USDT',
            timeframe='15m',
            limit=5000
        )
        logger.info(f"✅ Loaded {len(df)} bars from exchange")
    except Exception as e:
        logger.warning(f"Could not fetch real data: {e}")
        logger.info("Using synthetic data instead...")
        df = generate_synthetic_data(n_days=90)
    
    # Process data
    logger.info("🔧 Processing data...")
    processor = DataProcessor()
    df = processor.process_ohlcv(df)
    
    # Create features
    logger.info("🔧 Creating features...")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_all_features(df)
    
    logger.info(f"✅ Data ready: {len(df)} rows, {len(df.columns)} features")
    logger.info("")
    
    # Load strategies (without RL)
    logger.info("📦 Loading strategies...")
    strategies = {}
    
    strategy_configs = {
        'trend': TrendFollowingStrategy,
        'reversion': MeanReversionStrategy,
        'momentum': MomentumBreakoutStrategy,
        'arbitrage': FundingArbitrageStrategy
    }
    
    models_dir = 'models/trained'
    for strategy_name, strategy_class in strategy_configs.items():
        try:
            strategy = strategy_class()
            
            # Try to load model
            possible_paths = [
                f"{models_dir}/{strategy_name}_model.pkl",
                f"{models_dir}/{strategy_name}_model.cbm",
                f"{models_dir}/{strategy_name}_model.txt",
                f"{models_dir}/{strategy_name}_breakout.cbm",
                f"{models_dir}/{strategy_name}_following.txt",
                f"{models_dir}/{strategy_name}_reversion.txt",
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                strategy.load_model(model_path)
                logger.info(f"  ✅ Loaded {strategy_name}")
            else:
                logger.warning(f"  ⚠️  {strategy_name} model not found, using untrained")
            
            strategies[strategy_name] = strategy
            
        except Exception as e:
            logger.error(f"  ❌ Failed to load {strategy_name}: {e}")
    
    logger.info("")
    
    # Initialize MetaOrchestrator
    logger.info("🎯 Initializing MetaOrchestrator...")
    meta_orchestrator = MetaOrchestrator()
    logger.info("")
    
    # Initialize backtester
    logger.info("📊 Initializing backtester...")
    backtester = RealisticBacktester(
        data=df,
        initial_balance=config['capital']['initial']
    )
    logger.info("")
    
    # Run backtest
    logger.info("🚀 Starting backtest...")
    logger.info("="*80)
    
    # Simulate trading
    open_positions = {}
    equity_curve = [config['capital']['initial']]
    trades = []
    
    # Strategy weights (RL = 0)
    strategy_weights = {
        'trend': 0.35,
        'reversion': 0.30,
        'momentum': 0.25,
        'arbitrage': 0.10,
        'rl': 0.00
    }
    
    for i in range(100, len(df)):  # Start from 100 to have enough history
        current_data = df.iloc[:i+1]
        current_features = df.iloc[i]
        current_price = current_data.iloc[-1]['close']
        
        # Detect market regime
        market_regime = meta_orchestrator.detect_market_regime(current_features)
        
        # Get strategy allocation
        weights = meta_orchestrator.allocate_strategies(market_regime)
        # Override RL weight to 0 and redistribute
        weights['rl'] = 0.0
        total_weight = sum(w for k, w in weights.items() if k != 'rl')
        if total_weight > 0:
            weights = {k: (w/total_weight if k != 'rl' else 0.0) for k, w in weights.items()}
        
        # Suppress logging for every bar (too verbose)
        if i % 100 != 0:
            import logging
            logging.getLogger('trading_bot.models.meta_strategy').setLevel(logging.WARNING)
        
        # Generate signals from each strategy
        signals = {}
        for strategy_name, strategy in strategies.items():
            if weights.get(strategy_name, 0) < 0.05:
                continue
            
            try:
                if strategy_name == 'arbitrage':
                    signal, confidence, params = strategy.generate_signal(
                        current_funding_rate=current_features.get('funding_rate', 0),
                        symbol='BTC/USDT',
                        current_price=current_price
                    )
                else:
                    signal, confidence, params = strategy.generate_signal(
                        features=current_data,
                        current_price=current_price
                    )
                
                if signal != 'NEUTRAL' and confidence > 0.6:
                    signals[strategy_name] = {
                        'signal': signal,
                        'confidence': confidence,
                        'params': params,
                        'weight': weights.get(strategy_name, 0)
                    }
            except Exception as e:
                logger.debug(f"Error in {strategy_name}: {e}")
        
        # Execute signals (simplified - just track trades)
        for strategy_name, signal_info in signals.items():
            if strategy_name in open_positions:
                continue  # Already have position
            
            # Open position
            weight = signal_info['weight']
            position_size = backtester.balance * weight * 0.5  # Use 50% of allocated capital
            
            if position_size > 10:  # Min trade size
                open_positions[strategy_name] = {
                    'entry_price': current_price,
                    'side': signal_info['signal'],
                    'size': position_size,
                    'entry_bar': i,
                    'strategy': strategy_name
                }
        
        # Check exits
        for strategy_name, position in list(open_positions.items()):
            # Simple exit: 2% profit or 1% loss
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            if pnl_pct >= 0.02 or pnl_pct <= -0.01:
                # Close position
                pnl = position['size'] * pnl_pct
                backtester.balance += pnl
                backtester.balance -= position['size'] * 0.0004  # Fee
                
                trades.append({
                    'entry': position['entry_price'],
                    'exit': current_price,
                    'side': position['side'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'strategy': strategy_name,
                    'bars_held': i - position['entry_bar']
                })
                
                del open_positions[strategy_name]
        
        # Update equity curve
        unrealized_pnl = sum(
            pos['size'] * ((current_price - pos['entry_price']) / pos['entry_price'] if pos['side'] == 'LONG' else (pos['entry_price'] - current_price) / pos['entry_price'])
            for pos in open_positions.values()
        )
        equity_curve.append(backtester.balance + unrealized_pnl)
        
        if i % 500 == 0:
            logger.info(f"Bar {i}/{len(df)} | Balance: ${backtester.balance:.2f} | Positions: {len(open_positions)} | Trades: {len(trades)}")
    
    # Close remaining positions
    final_price = df.iloc[-1]['close']
    for strategy_name, position in open_positions.items():
        if position['side'] == 'LONG':
            pnl_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = position['size'] * pnl_pct
        backtester.balance += pnl
        backtester.balance -= position['size'] * 0.0004
        
        trades.append({
            'entry': position['entry_price'],
            'exit': final_price,
            'side': position['side'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'strategy': strategy_name,
            'bars_held': len(df) - position['entry_bar']
        })
    
    # Calculate metrics
    logger.info("")
    logger.info("="*80)
    logger.info("📊 BACKTEST SONUÇLARI")
    logger.info("="*80)
    
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)) if returns.std() > 0 else 0
    max_dd = ((equity_series / equity_series.expanding().max()) - 1).min() * 100
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl'] > 0).mean() * 100 if len(trades_df) > 0 else 0
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        trades_df = pd.DataFrame()
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    logger.info("")
    logger.info("💰 PERFORMANS METRİKLERİ:")
    logger.info(f"   • Başlangıç Bakiyesi: ${equity_series.iloc[0]:,.2f}")
    logger.info(f"   • Final Bakiyesi: ${equity_series.iloc[-1]:,.2f}")
    logger.info(f"   • Toplam Getiri: {total_return:.2f}%")
    logger.info(f"   • Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"   • Max Drawdown: {max_dd:.2f}%")
    logger.info("")
    logger.info("📈 TRADE İSTATİSTİKLERİ:")
    logger.info(f"   • Toplam Trade: {len(trades)}")
    logger.info(f"   • Win Rate: {win_rate:.1f}%")
    logger.info(f"   • Ortalama Kazanç: ${avg_win:.2f}")
    logger.info(f"   • Ortalama Kayıp: ${avg_loss:.2f}")
    logger.info(f"   • Profit Factor: {profit_factor:.2f}")
    logger.info("")
    
    # Strategy breakdown
    if len(trades_df) > 0:
        logger.info("🎯 STRATEJİ BAZINDA PERFORMANS:")
        for strategy_name in strategies.keys():
            strategy_trades = trades_df[trades_df['strategy'] == strategy_name]
            if len(strategy_trades) > 0:
                strategy_pnl = strategy_trades['pnl'].sum()
                strategy_win_rate = (strategy_trades['pnl'] > 0).mean() * 100
                logger.info(f"   • {strategy_name.upper()}: {len(strategy_trades)} trades, ${strategy_pnl:.2f} PnL, {strategy_win_rate:.1f}% win rate")
    
    logger.info("")
    logger.info("="*80)
    logger.info("✅ BACKTEST TAMAMLANDI")
    logger.info("="*80)


if __name__ == "__main__":
    main()

