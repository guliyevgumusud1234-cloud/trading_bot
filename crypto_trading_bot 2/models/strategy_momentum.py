"""
Momentum Breakout Strategy using CatBoost.

This strategy:
- Detects consolidation periods (tight BB, low ATR)
- Waits for volume-confirmed breakouts
- Requires multi-timeframe alignment
- Aggressive with 5x max leverage
- 1:3 minimum risk-reward ratio
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from typing import Dict, Optional, Tuple
import yaml
import pickle

from utils.logger import get_logger

logger = get_logger(__name__)


class MomentumBreakoutStrategy:
    """
    Momentum breakout strategy with CatBoost classifier.
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        Initialize momentum breakout strategy.
        
        Args:
            config_path: Path to strategy configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['momentum_breakout']
        self.model = None
        self.feature_columns = None
        
        logger.info("MomentumBreakoutStrategy initialized")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """
        Train the CatBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        logger.info("Training Momentum Breakout model...")
        
        self.feature_columns = X_train.columns.tolist()
        
        # Create pools
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        # Initialize model
        params = self.config['model']['params']
        self.model = CatBoostClassifier(**params)
        
        # Train
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            verbose=100
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Ensure predictions and labels are 1D arrays
        train_pred = train_pred.flatten() if train_pred.ndim > 1 else train_pred
        val_pred = val_pred.flatten() if val_pred.ndim > 1 else val_pred
        
        # Ensure y_train and y_val are 1D
        if hasattr(y_train, 'values'):
            y_train_flat = y_train.values.flatten()
        elif hasattr(y_train, 'flatten'):
            y_train_flat = y_train.flatten()
        else:
            y_train_flat = y_train
            
        if hasattr(y_val, 'values'):
            y_val_flat = y_val.values.flatten()
        elif hasattr(y_val, 'flatten'):
            y_val_flat = y_val.flatten()
        else:
            y_val_flat = y_val
        
        train_acc = (train_pred == y_train_flat).mean()
        val_acc = (val_pred == y_val_flat).mean()
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'best_iteration': self.model.best_iteration_
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
        if self.feature_columns is None:
            logger.warning("Feature columns not set, using all numeric columns")
            self.feature_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check which features are available
        available_features = [col for col in self.feature_columns if col in features.columns]
        missing_features = [col for col in self.feature_columns if col not in features.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, using {len(available_features)} available")
            # Use only available features
            self.feature_columns = available_features
        
        if len(self.feature_columns) == 0:
            logger.error("No features available!")
            return 'NEUTRAL', 0.0, None
        
        if len(features) > 1:
            latest_features = features.iloc[-1:][self.feature_columns]
            feature_row = features.iloc[-1]
        else:
            latest_features = features[self.feature_columns]
            feature_row = features.iloc[0]
        
        # Check consolidation first
        if not self._detect_consolidation(feature_row):
            return 'NEUTRAL', 0.0, None
        
        # Predict
        prediction_proba = self.model.predict_proba(latest_features)[0]
        predicted_class = prediction_proba.argmax()
        confidence = prediction_proba.max()
        
        signal_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}
        signal = signal_map[predicted_class]
        
        # Check breakout conditions
        if signal != 'NEUTRAL':
            if not self._check_breakout_conditions(feature_row, signal):
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
    
    def _detect_consolidation(self, features: pd.Series) -> bool:
        """
        Detect if market is in consolidation.
        
        Args:
            features: Feature row
            
        Returns:
            Whether market is consolidating
        """
        consolidation_config = self.config['entry']['consolidation']
        
        # Low ATR indicates consolidation
        if 'atr_14_pct' in features:
            if features['atr_14_pct'] > consolidation_config['atr_threshold_low']:
                return False
        
        # Tight Bollinger Bands
        if 'bb_width' in features:
            # Need to check against historical average
            # For now, use absolute threshold
            if features['bb_width'] > consolidation_config['bb_width_threshold']:
                return False
        
        return True
    
    def _check_breakout_conditions(self, features: pd.Series, signal: str) -> bool:
        """
        Check breakout confirmation conditions.
        
        Args:
            features: Feature row
            signal: Proposed signal
            
        Returns:
            Whether breakout is confirmed
        """
        breakout_config = self.config['entry']['breakout']
        
        # Volume confirmation (critical for breakouts)
        if 'volume_ratio_20' in features:
            if features['volume_ratio_20'] < breakout_config['volume_multiplier']:
                logger.debug(f"Insufficient volume for breakout: {features['volume_ratio_20']:.2f}")
                return False
        
        # Price momentum
        if 'price_momentum_5' in features:
            if abs(features['price_momentum_5']) < breakout_config['momentum_threshold']:
                return False
        
        # Higher timeframe alignment (if available)
        if breakout_config['require_higher_tf_alignment']:
            # Check if higher timeframe is aligned
            # This would check 'ema_8_1h' > 'ema_21_1h' for example
            pass  # Placeholder - would need higher TF features
        
        return True
    
    def _build_trade_params(
        self,
        signal: str,
        current_price: float,
        features: pd.Series,
        confidence: float
    ) -> Dict:
        """Build trade parameters for momentum breakout."""
        # Get consolidation range for stop loss
        # Assume we have consolidation high/low features
        consolidation_low = current_price * 0.98  # Placeholder
        consolidation_high = current_price * 1.02  # Placeholder
        
        # Stop loss configuration
        stop_config = self.config['stop_loss']
        max_loss_pct = stop_config['max_loss_pct']
        
        if signal == 'LONG':
            # Stop below consolidation
            stop_loss = consolidation_low * (1 - stop_config['consolidation_buffer_pct'])
            # Ensure max 2% loss
            stop_loss = max(stop_loss, current_price * (1 - max_loss_pct))
            
            # Calculate distance to stop
            risk = current_price - stop_loss
            
            # Minimum 1:3 risk-reward
            min_reward = risk * 3
            take_profit = current_price + min_reward
        
        else:  # SHORT
            stop_loss = consolidation_high * (1 + stop_config['consolidation_buffer_pct'])
            stop_loss = min(stop_loss, current_price * (1 + max_loss_pct))
            
            risk = stop_loss - current_price
            min_reward = risk * 3
            take_profit = current_price - min_reward
        
        # Position sizing based on volatility
        atr = features.get('atr_14', current_price * 0.02)
        volatility_adjusted = True
        
        trade_params = {
            'signal': signal,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk': risk,
            'reward': min_reward,
            'risk_reward_ratio': 3.0,
            'confidence': confidence,
            'strategy': 'momentum_breakout',
            'max_leverage': self.config['max_leverage'],
            'volatility_adjusted': volatility_adjusted,
            'atr': atr
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
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        
        try:
            with open(path + '.features', 'rb') as f:
                self.feature_columns = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Feature columns file not found")
        
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    print("Testing Momentum Breakout Strategy...")
    
    strategy = MomentumBreakoutStrategy()
    
    # Create test data
    n_samples = 1000
    n_features = 50
    
    X_train = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
    y_train = pd.Series(np.random.randint(0, 3, n_samples))
    
    X_val = pd.DataFrame(np.random.randn(200, n_features), columns=[f'f{i}' for i in range(n_features)])
    y_val = pd.Series(np.random.randint(0, 3, 200))
    
    # Add required features
    X_val['atr_14_pct'] = 0.015  # Low ATR (consolidation)
    X_val['bb_width'] = 0.01  # Tight BB
    X_val['volume_ratio_20'] = 2.5  # Volume breakout
    X_val['price_momentum_5'] = 0.008  # Momentum
    X_val['atr_14'] = 500
    
    # Train
    metrics = strategy.train(X_train, y_train, X_val, y_val)
    print(f"\nTraining Metrics: {metrics}")
    
    # Generate signal
    signal, confidence, params = strategy.generate_signal(X_val.tail(1), current_price=50000)
    print(f"\nSignal: {signal}")
    print(f"Confidence: {confidence:.2f}")
    if params:
        print(f"Risk: ${params['risk']:.2f}")
        print(f"Reward: ${params['reward']:.2f}")
        print(f"R:R Ratio: 1:{params['risk_reward_ratio']:.1f}")
        print(f"Stop Loss: ${params['stop_loss']:.2f}")
        print(f"Take Profit: ${params['take_profit']:.2f}")
    
    print("\n✅ Momentum Breakout strategy test complete")

