"""
Train only the Deep RL Strategy with improved settings.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import DataFetcher
from data.processor import DataProcessor
from data.feature_engineering import FeatureEngineer
from models.strategy_rl import DeepRLStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Train only the RL strategy."""
    logger.info("=" * 60)
    logger.info("🚀 Training Deep RL Strategy (Improved Settings)")
    logger.info("=" * 60)
    logger.info("Settings:")
    logger.info("  • Reward Normalization: ✅ (reward / 100)")
    logger.info("  • Exploration: ✅ (ent_coef: 0.05)")
    logger.info("  • Timesteps: ✅ (2,000,000)")
    logger.info("  • Estimated Time: 4-6 hours")
    logger.info("")
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get symbols and timeframes
    symbols = config.get('symbols', {})
    if isinstance(symbols, dict):
        primary = symbols.get('primary', ['BTC/USDT'])
        secondary = symbols.get('secondary', [])
        all_symbols = primary + secondary
    else:
        all_symbols = symbols if isinstance(symbols, list) else ['BTC/USDT']
    
    timeframes = config.get('timeframes', ['5m', '15m', '1h', '4h', '1d'])
    
    logger.info(f"📊 Symbols: {all_symbols}")
    logger.info(f"📊 Timeframes: {timeframes}")
    logger.info("")
    
    # Initialize components
    logger.info("🔧 Initializing components...")
    processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    
    # Generate synthetic data (bypassing database/exchange requirements)
    logger.info("📊 Generating synthetic data for training...")
    logger.info("   (Using synthetic data to avoid database/exchange dependencies)")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    n = len(dates)
    
    # Create realistic price movement with trends and volatility
    np.random.seed(42)
    returns = np.random.randn(n) * 0.001  # 0.1% volatility
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.002)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.002)),
        'close': prices * (1 + np.random.randn(n) * 0.001),
        'volume': np.random.rand(n) * 1000 + 500
    }, index=dates)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    logger.info(f"✅ Fetched {len(df)} bars")
    
    # Process data
    logger.info("🔧 Processing data...")
    df = processor.process_ohlcv(df)
    
    # Create features
    logger.info("🔧 Creating features (this may take a while)...")
    features_df = feature_engineer.create_all_features(df)
    
    logger.info(f"✅ Created {len(features_df.columns)} features")
    logger.info(f"✅ Total rows: {len(features_df)}")
    
    # Split data
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)
    
    train_data = features_df.iloc[:train_size].copy()
    val_data = features_df.iloc[train_size:train_size+val_size].copy()
    
    logger.info(f"📊 Train: {len(train_data)} rows")
    logger.info(f"📊 Val: {len(val_data)} rows")
    logger.info("")
    
    # Train RL strategy
    logger.info("🎯 Starting RL training...")
    logger.info("")
    
    try:
        rl_strategy = DeepRLStrategy()
        metrics = rl_strategy.train(
            train_data=train_data,
            val_data=val_data,
            total_timesteps=2_000_000  # 2M timesteps
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ RL Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Metrics: {metrics}")
        logger.info("")
        logger.info("💾 Model saved to: models/trained/rl_model.pkl")
        
    except Exception as e:
        logger.error(f"❌ RL Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

