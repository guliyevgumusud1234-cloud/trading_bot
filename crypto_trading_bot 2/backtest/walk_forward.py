"""
Walk-Forward Analysis.

Prevents overfitting by:
- Sliding window training and testing
- Continuous retraining
- Out-of-sample validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import timedelta

from backtest.backtester import RealisticBacktester
from utils.logger import get_logger

logger = get_logger(__name__)


class WalkForwardOptimizer:
    """
    Walk-forward analysis for strategy validation.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        train_period: int = 90,  # days
        test_period: int = 30,  # days
        bars_per_day: int = 96  # 15min bars
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            data: Full dataset with OHLCV and features
            train_period: Training period in days
            test_period: Testing period in days
            bars_per_day: Number of bars per day
        """
        self.data = data.reset_index(drop=True)
        self.train_period = train_period
        self.test_period = test_period
        self.bars_per_day = bars_per_day
        
        self.train_bars = train_period * bars_per_day
        self.test_bars = test_period * bars_per_day
        
        logger.info(
            f"WalkForwardOptimizer initialized: "
            f"{train_period}d train, {test_period}d test"
        )
    
    def run_walk_forward(
        self,
        strategy_class,
        initial_balance: float = 10000
    ) -> Dict:
        """
        Run walk-forward analysis.
        
        Args:
            strategy_class: Strategy class to test (must have train() and generate_signal())
            initial_balance: Starting capital
            
        Returns:
            Walk-forward results dict
        """
        logger.info("Starting walk-forward analysis...")
        
        results = []
        n_windows = self._calculate_num_windows()
        
        logger.info(f"Total windows: {n_windows}")
        
        for window_idx in range(n_windows):
            logger.info(f"\n=== Window {window_idx + 1}/{n_windows} ===")
            
            # Define window boundaries
            train_start = window_idx * self.test_bars
            train_end = train_start + self.train_bars
            test_start = train_end
            test_end = test_start + self.test_bars
            
            # Check bounds
            if test_end > len(self.data):
                logger.warning("Reached end of data")
                break
            
            # Split data
            train_data = self.data.iloc[train_start:train_end].copy()
            test_data = self.data.iloc[test_start:test_end].copy()
            
            if len(train_data) < 1000 or len(test_data) < 100:
                logger.warning("Insufficient data for window")
                continue
            
            logger.info(
                f"Train: {train_start} to {train_end} ({len(train_data)} bars)"
            )
            logger.info(
                f"Test: {test_start} to {test_end} ({len(test_data)} bars)"
            )
            
            try:
                # Initialize and train strategy
                strategy = strategy_class()
                
                # Prepare training data
                X_train, y_train = self._prepare_training_data(train_data)
                X_val, y_val = self._prepare_training_data(test_data.iloc[:len(test_data)//4])
                
                # Train model
                logger.info("Training model...")
                train_metrics = strategy.train(X_train, y_train, X_val, y_val)
                logger.info(f"Training metrics: {train_metrics}")
                
                # Backtest on test period
                logger.info("Running backtest on test period...")
                backtester = RealisticBacktester(
                    data=test_data,
                    initial_balance=initial_balance
                )
                
                test_metrics = backtester.run_backtest(strategy)
                
                # Store results
                window_result = {
                    'window': window_idx,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }
                
                results.append(window_result)
                
                logger.info(
                    f"Window {window_idx + 1} results: "
                    f"Return={test_metrics.get('total_return', 0):.2%}, "
                    f"Sharpe={test_metrics.get('sharpe_ratio', 0):.2f}, "
                    f"Trades={test_metrics.get('total_trades', 0)}"
                )
                
            except Exception as e:
                logger.error(f"Error in window {window_idx}: {e}")
                continue
        
        # Aggregate results
        aggregate_results = self._aggregate_results(results)
        
        logger.info("\n=== Walk-Forward Analysis Complete ===")
        logger.info(f"Total windows: {len(results)}")
        logger.info(f"Avg Sharpe: {aggregate_results.get('avg_sharpe', 0):.2f}")
        logger.info(f"Avg Return: {aggregate_results.get('avg_return', 0):.2%}")
        logger.info(f"Consistency: {aggregate_results.get('consistency', 0):.2%}")
        
        return {
            'window_results': results,
            'aggregate': aggregate_results
        }
    
    def _calculate_num_windows(self) -> int:
        """Calculate number of walk-forward windows."""
        total_bars = len(self.data)
        available_for_windows = total_bars - self.train_bars
        n_windows = available_for_windows // self.test_bars
        
        return max(1, n_windows)
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data and labels.
        
        Args:
            data: Raw data with features
            
        Returns:
            Tuple of (X, y)
        """
        # Create labels (forward-looking returns)
        horizon = 12  # 12 bars ahead (3 hours for 15min bars)
        
        if 'close' in data.columns:
            future_returns = data['close'].pct_change(horizon).shift(-horizon)
            
            # Classify returns
            labels = pd.Series(1, index=data.index)  # Neutral
            labels[future_returns > 0.015] = 2  # Long (>1.5% profit)
            labels[future_returns < -0.015] = 0  # Short
            
            # Remove last rows with NaN labels
            valid_idx = labels.notna()
            
            # Features (exclude target and non-feature columns)
            exclude_cols = ['close', 'open', 'high', 'low', 'timestamp']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            X = data.loc[valid_idx, feature_cols].fillna(0)
            y = labels[valid_idx]
            
            return X, y
        else:
            logger.error("No 'close' column found in data")
            return pd.DataFrame(), pd.Series()
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across all windows."""
        if not results:
            return {}
        
        # Extract test metrics
        test_metrics = [r['test_metrics'] for r in results]
        
        # Calculate aggregates
        sharpe_ratios = [m.get('sharpe_ratio', 0) for m in test_metrics]
        returns = [m.get('total_return', 0) for m in test_metrics]
        win_rates = [m.get('win_rate', 0) for m in test_metrics]
        max_drawdowns = [m.get('max_drawdown', 0) for m in test_metrics]
        
        # Consistency: std dev of Sharpe ratios (lower is better)
        sharpe_std = np.std(sharpe_ratios)
        consistency_score = max(0, 1 - sharpe_std)  # Simple metric
        
        # Profitable windows
        profitable_windows = sum(1 for r in returns if r > 0)
        profitability_ratio = profitable_windows / len(returns)
        
        aggregate = {
            'total_windows': len(results),
            'avg_sharpe': np.mean(sharpe_ratios),
            'sharpe_std': sharpe_std,
            'avg_return': np.mean(returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'consistency': consistency_score,
            'profitable_windows': profitable_windows,
            'profitability_ratio': profitability_ratio,
            'best_window_return': max(returns),
            'worst_window_return': min(returns)
        }
        
        return aggregate


if __name__ == "__main__":
    print("Testing Walk-Forward Optimizer...")
    
    # Create synthetic data (1 year = ~35k bars of 15min data)
    n_bars = 35000
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='15min')
    
    returns = np.random.randn(n_bars) * 0.002
    prices = 50000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })
    
    # Add features
    for i in range(20):
        data[f'feature_{i}'] = np.random.randn(n_bars)
    
    # Mock strategy class
    class MockStrategy:
        def train(self, X_train, y_train, X_val, y_val):
            return {'train_accuracy': 0.55, 'val_accuracy': 0.52}
        
        def generate_signal(self, features, current_price):
            if np.random.random() > 0.98:
                signal = np.random.choice(['LONG', 'SHORT'])
                return signal, 0.7, {
                    'stop_loss': current_price * 0.98,
                    'take_profit': current_price * 1.04,
                    'max_leverage': 2,
                    'strategy': 'mock'
                }
            return 'NEUTRAL', 0.0, None
    
    # Initialize optimizer
    optimizer = WalkForwardOptimizer(
        data=data,
        train_period=90,
        test_period=30
    )
    
    print("\nRunning walk-forward analysis (this may take a while)...")
    print("Note: Using mock strategy with random signals for testing\n")
    
    # Run with small subset for testing
    test_data = data.iloc[:10000]  # Use only first 10k bars for quick test
    test_optimizer = WalkForwardOptimizer(
        data=test_data,
        train_period=30,  # Shorter for testing
        test_period=10
    )
    
    results = test_optimizer.run_walk_forward(MockStrategy)
    
    print("\n=== Aggregate Results ===")
    agg = results['aggregate']
    print(f"Total Windows: {agg['total_windows']}")
    print(f"Avg Sharpe: {agg['avg_sharpe']:.2f}")
    print(f"Sharpe Std Dev: {agg['sharpe_std']:.2f}")
    print(f"Avg Return: {agg['avg_return']:.2%}")
    print(f"Consistency Score: {agg['consistency']:.2%}")
    print(f"Profitable Windows: {agg['profitable_windows']}/{agg['total_windows']}")
    
    print("\n✅ Walk-Forward Optimizer test complete")

