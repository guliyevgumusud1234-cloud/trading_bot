"""
Train only Momentum Breakout Strategy.
This script loads data, creates features, and trains only the momentum model.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.feature_engineering import FeatureEngineer
from models.strategy_momentum import MomentumBreakoutStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Train only Momentum Breakout strategy."""
    logger.info("="*80)
    logger.info("TRAINING MOMENTUM BREAKOUT STRATEGY ONLY")
    logger.info("="*80)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse symbols
    symbols_config = config.get('symbols', {})
    symbols = []
    if isinstance(symbols_config, dict):
        if 'primary' in symbols_config and isinstance(symbols_config['primary'], list):
            symbols.extend(symbols_config['primary'])
        if 'secondary' in symbols_config and isinstance(symbols_config['secondary'], list):
            symbols.extend(symbols_config['secondary'])
    if not symbols:
        symbols = ['BTC/USDT', 'ETH/USDT']
    
    # Parse timeframes
    timeframes_config = config.get('timeframes', {})
    if isinstance(timeframes_config, dict):
        timeframes = []
        if 'primary' in timeframes_config:
            timeframes.append(timeframes_config['primary'])
        if 'secondary' in timeframes_config and isinstance(timeframes_config['secondary'], list):
            timeframes.extend(timeframes_config['secondary'])
    else:
        timeframes = ['15m', '1h', '4h', '1d']
    
    # Step 1: Load data (use synthetic data)
    logger.info("\n[1/4] Loading data...")
    
    # Use BTC 15m as primary
    primary_symbol = 'BTC/USDT'
    primary_timeframe = '15m'
    
    # Generate synthetic data
    start_date = '2022-01-01'
    end_date = None
    
    logger.info(f"Generating synthetic data for {primary_symbol} {primary_timeframe}...")
    df = _generate_synthetic_data(primary_symbol, primary_timeframe, start_date, end_date)
    logger.info(f"Generated {len(df)} bars")
    
    # Step 2: Engineer features
    logger.info("\n[2/4] Engineering features...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_all_features(df)
    logger.info(f"Generated {len(features_df.columns)} features")
    
    # Step 3: Prepare splits
    logger.info("\n[3/4] Preparing train/val/test splits...")
    features_df = features_df.dropna()
    
    # Create labels (simplified: 0=Short, 1=Neutral, 2=Long)
    # Based on future returns
    future_returns = features_df['close'].pct_change(5).shift(-5)
    labels = pd.cut(future_returns, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    labels = labels.fillna(1).astype(int)
    
    # Remove rows with NaN labels
    valid_mask = ~labels.isna()
    features_df = features_df[valid_mask]
    labels = labels[valid_mask]
    
    # Split: 70% train, 15% val, 15% test
    n = len(features_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train = features_df.iloc[:train_end].select_dtypes(include=[np.number])
    y_train = labels.iloc[:train_end]
    X_val = features_df.iloc[train_end:val_end].select_dtypes(include=[np.number])
    y_val = labels.iloc[train_end:val_end]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Step 4: Train Momentum
    logger.info("\n[4/4] Training Momentum Breakout Strategy...")
    try:
        momentum_strategy = MomentumBreakoutStrategy()
        metrics = momentum_strategy.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        logger.info(f"✅ Momentum Breakout trained: {metrics}")
        
        # Save model
        os.makedirs('models/trained', exist_ok=True)
        momentum_strategy.save_model('models/trained/momentum_breakout.cbm')
        logger.info("✅ Model saved to models/trained/momentum_breakout.cbm")
        
    except Exception as e:
        logger.error(f"❌ Momentum training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("\n" + "="*80)
    logger.info("MOMENTUM TRAINING COMPLETE")
    logger.info("="*80)
    
    return True


def _generate_synthetic_data(symbol: str, timeframe: str, start_date: str, end_date: str = None):
    """Generate synthetic OHLCV data."""
    from datetime import timedelta
    
    # Calculate number of bars
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
    
    start = pd.to_datetime(start_date)
    
    # Timeframe to minutes
    tf_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
    minutes = tf_minutes.get(timeframe, 15)
    
    n_bars = int((end_date - start).total_seconds() / 60 / minutes)
    
    # Generate dates
    dates = pd.date_range(start=start, periods=n_bars, freq=f'{minutes}min')
    
    # Generate price returns (random walk with drift)
    np.random.seed(42)
    returns = np.random.randn(n_bars) * 0.01 + 0.0001
    
    # Base prices
    base_prices = {
        'BTC/USDT': 50000,
        'ETH/USDT': 3000,
        'BNB/USDT': 400,
        'SOL/USDT': 100,
        'ARB/USDT': 2,
        'MATIC/USDT': 1,
        'AVAX/USDT': 40,
        'LINK/USDT': 15
    }
    
    base_price = base_prices.get(symbol, 50000)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_bars) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(n_bars) * 0.005)),
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_bars)
    })
    
    df.set_index('timestamp', inplace=True)
    
    return df


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

