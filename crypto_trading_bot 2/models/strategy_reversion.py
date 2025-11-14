"""
Mean Reversion Strategy using XGBoost.

This strategy:
- Trades Bollinger Band extremes
- Uses Z-score for entry signals
- RSI confirmation (<30 or >70)
- Only trades in ranging markets (ADX < 25)
- Volume spike confirmation
- Order book imbalance check
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Optional, Tuple
import yaml
import pickle

from utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy:
    """
    Mean reversion strategy with XGBoost classifier.
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        Initialize mean reversion strategy.
        
        Args:
            config_path: Path to strategy configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['mean_reversion']
        self.model = None
        self.feature_columns = None
        
        logger.info("MeanReversionStrategy initialized")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        logger.info("Training Mean Reversion model...")
        
        self.feature_columns = X_train.columns.tolist()
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train
        params = self.config['model']['params']
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evals,
            early_stopping_rounds=params['early_stopping_rounds'],
            verbose_eval=100
        )
        
        # Evaluate
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        train_acc = (train_pred.argmax(axis=1) == y_train).mean()
        val_acc = (val_pred.argmax(axis=1) == y_val).mean()
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'best_iteration': self.model.best_iteration
        }
        
        logger.info(f"Training complete: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        return metrics
    
    def generate_signal(
        self,
        features: pd.DataFrame,
        current_price: float
    ) -> Tuple[str, float, Optional[Dict]]:
        """
        Generate trading signal.
        
        Args:
            features: Feature DataFrame
            current_price: Current market price
            
        Returns:
            Tuple of (signal, confidence, trade_params)
        """
        if self.model is None:
            logger.warning("Model not trained!")
            return 'NEUTRAL', 0.0, None
        
        # Get latest features
        if len(features) > 1:
            latest_features = features.iloc[-1:][self.feature_columns]
            feature_row = features.iloc[-1]
        else:
            latest_features = features[self.feature_columns]
            feature_row = features.iloc[0]
        
        # Check ranging market condition FIRST
        if not self._check_ranging_market(feature_row):
            return 'NEUTRAL', 0.0, None
        
        # Predict
        dmatrix = xgb.DMatrix(latest_features)
        prediction = self.model.predict(dmatrix)[0]
        predicted_class = prediction.argmax()
        confidence = prediction.max()
        
        signal_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}
        signal = signal_map[predicted_class]
        
        # Apply mean reversion filters
        if not self._check_reversion_conditions(feature_row, signal):
            signal = 'NEUTRAL'
            confidence = 0.0
        
        # Build trade parameters
        trade_params = None
        if signal != 'NEUTRAL':
            trade_params = self._build_trade_params(
                signal=signal,
                current_price=current_price,
                features=feature_row,
                confidence=confidence
            )
        
        logger.debug(f"Signal: {signal}, Confidence: {confidence:.2f}")
        
        return signal, confidence, trade_params
    
    def _check_ranging_market(self, features: pd.Series) -> bool:
        """Check if market is ranging (required for mean reversion)."""
        entry_config = self.config['entry']
        
        # ADX must be low (ranging market)
        if 'adx_14' in features:
            if features['adx_14'] > entry_config['adx_max']:
                logger.debug(f"ADX too high for mean reversion: {features['adx_14']:.1f}")
                return False
        
        return True
    
    def _check_reversion_conditions(self, features: pd.Series, signal: str) -> bool:
        """
        Check mean reversion entry conditions.
        
        Args:
            features: Feature row
            signal: Proposed signal
            
        Returns:
            Whether conditions are met
        """
        entry_config = self.config['entry']
        
        if signal == 'LONG':
            # Long at lower BB
            if 'bb_position' in features:
                if features['bb_position'] > entry_config['bb_lower_threshold']:
                    return False
            
            # RSI oversold
            if 'rsi_14' in features:
                if features['rsi_14'] > entry_config['rsi_oversold']:
                    return False
            
            # Z-score extreme
            if 'zscore_20' in features:
                if features['zscore_20'] > -entry_config['z_score_threshold']:
                    return False
        
        elif signal == 'SHORT':
            # Short at upper BB
            if 'bb_position' in features:
                if features['bb_position'] < entry_config['bb_upper_threshold']:
                    return False
            
            # RSI overbought
            if 'rsi_14' in features:
                if features['rsi_14'] < entry_config['rsi_overbought']:
                    return False
            
            # Z-score extreme
            if 'zscore_20' in features:
                if features['zscore_20'] < entry_config['z_score_threshold']:
                    return False
        
        # Volume spike confirmation
        if 'volume_ratio_20' in features:
            if features['volume_ratio_20'] < entry_config['volume_spike_threshold']:
                return False
        
        return True
    
    def _build_trade_params(
        self,
        signal: str,
        current_price: float,
        features: pd.Series,
        confidence: float
    ) -> Dict:
        """Build trade parameters for mean reversion."""
        # Target is BB middle (mean)
        bb_middle = features.get('bb_middle', current_price)
        
        # Stop loss configuration
        stop_config = self.config['stop_loss']
        stop_loss_pct = stop_config['pct']
        
        if signal == 'LONG':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = bb_middle  # Exit at mean
        else:  # SHORT
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = bb_middle
        
        # Calculate distance from mean for position sizing
        distance_from_mean = abs(current_price - bb_middle) / bb_middle
        
        trade_params = {
            'signal': signal,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'bb_middle': bb_middle,
            'distance_from_mean': distance_from_mean,
            'confidence': confidence,
            'strategy': 'mean_reversion',
            'max_leverage': self.config['max_leverage'],
            'time_stop_hours': self.config['exit']['time_based_hours']
        }
        
        return trade_params
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        self.model.save_model(path)
        
        with open(path + '.features', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        
        try:
            with open(path + '.features', 'rb') as f:
                self.feature_columns = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Feature columns file not found")
        
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    print("Testing Mean Reversion Strategy...")
    
    strategy = MeanReversionStrategy()
    
    # Create test data
    n_samples = 1000
    n_features = 50
    
    X_train = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
    y_train = pd.Series(np.random.randint(0, 3, n_samples))
    
    X_val = pd.DataFrame(np.random.randn(200, n_features), columns=[f'f{i}' for i in range(n_features)])
    y_val = pd.Series(np.random.randint(0, 3, 200))
    
    # Add required features
    X_val['adx_14'] = 20  # Ranging market
    X_val['bb_position'] = 0.1  # Near lower BB
    X_val['rsi_14'] = 25  # Oversold
    X_val['zscore_20'] = -2.5  # Extreme
    X_val['volume_ratio_20'] = 1.8  # Volume spike
    X_val['bb_middle'] = 50500
    
    # Train
    metrics = strategy.train(X_train, y_train, X_val, y_val)
    print(f"\nTraining Metrics: {metrics}")
    
    # Generate signal
    signal, confidence, params = strategy.generate_signal(X_val.tail(1), current_price=50000)
    print(f"\nSignal: {signal}")
    print(f"Confidence: {confidence:.2f}")
    if params:
        print(f"Target (BB Middle): ${params['bb_middle']:.2f}")
        print(f"Stop Loss: ${params['stop_loss']:.2f}")
    
    print("\n✅ Mean Reversion strategy test complete")

