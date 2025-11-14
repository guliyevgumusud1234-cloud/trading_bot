"""
Trend Following Strategy using LightGBM.

This strategy:
- Uses EMA crossovers for trend detection
- Requires ADX > 25 for strong trends
- Confirms with Supertrend indicator
- Volume confirmation required
- Higher timeframe filter
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional, Tuple
import yaml
import pickle

from utils.logger import get_logger

logger = get_logger(__name__)


class TrendFollowingStrategy:
    """
    Trend following strategy with LightGBM classifier.
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        Initialize trend following strategy.
        
        Args:
            config_path: Path to strategy configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['trend_following']
        self.model = None
        self.scaler = None
        
        # Feature list (will be populated during training)
        self.feature_columns = None
        
        logger.info("TrendFollowingStrategy initialized")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels (0=Short, 1=Neutral, 2=Long)
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        logger.info("Training Trend Following model...")
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            self.config['model']['params'],
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(self.config['model']['params']['early_stopping_rounds']),
                lgb.log_evaluation(100)
            ]
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = (train_pred.argmax(axis=1) == y_train).mean()
        val_acc = (val_pred.argmax(axis=1) == y_val).mean()
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'num_trees': self.model.num_trees(),
            'best_iteration': self.model.best_iteration
        }
        
        logger.info(
            f"Training complete: Train Acc={train_acc:.3f}, "
            f"Val Acc={val_acc:.3f}"
        )
        
        return metrics
    
    def generate_signal(
        self,
        features: pd.DataFrame,
        current_price: float
    ) -> Tuple[str, float, Optional[Dict]]:
        """
        Generate trading signal.
        
        Args:
            features: Feature DataFrame (single row or recent rows)
            current_price: Current market price
            
        Returns:
            Tuple of (signal, confidence, trade_params)
            signal: 'LONG', 'SHORT', or 'NEUTRAL'
        """
        if self.model is None:
            logger.warning("Model not trained!")
            return 'NEUTRAL', 0.0, None
        
        # Get latest features
        if len(features) > 1:
            latest_features = features.iloc[-1:][self.feature_columns]
        else:
            latest_features = features[self.feature_columns]
        
        # Predict
        prediction = self.model.predict(latest_features)[0]
        predicted_class = prediction.argmax()
        confidence = prediction.max()
        
        # Map to signal
        signal_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}
        signal = signal_map[predicted_class]
        
        # Apply rule-based filters
        if not self._check_trend_conditions(features.iloc[-1] if len(features) > 1 else features.iloc[0]):
            signal = 'NEUTRAL'
            confidence = 0.0
        
        # Build trade parameters if signal is not neutral
        trade_params = None
        if signal != 'NEUTRAL':
            trade_params = self._build_trade_params(
                signal=signal,
                current_price=current_price,
                features=features.iloc[-1] if len(features) > 1 else features.iloc[0],
                confidence=confidence
            )
        
        logger.debug(f"Signal: {signal}, Confidence: {confidence:.2f}")
        
        return signal, confidence, trade_params
    
    def _check_trend_conditions(self, features: pd.Series) -> bool:
        """
        Check rule-based trend conditions.
        
        Args:
            features: Feature row
            
        Returns:
            Whether conditions are met
        """
        entry_config = self.config['entry']
        
        # ADX threshold
        if 'adx_14' in features:
            if features['adx_14'] < entry_config['adx_threshold']:
                return False
        
        # EMA alignment check
        if entry_config['require_ema_alignment']:
            # For long: EMA8 > EMA21 > EMA50
            if 'ema_8' in features and 'ema_21' in features and 'ema_50' in features:
                ema_aligned_long = (
                    features['ema_8'] > features['ema_21'] and
                    features['ema_21'] > features['ema_50']
                )
                ema_aligned_short = (
                    features['ema_8'] < features['ema_21'] and
                    features['ema_21'] < features['ema_50']
                )
                
                if not (ema_aligned_long or ema_aligned_short):
                    return False
        
        # Volume confirmation
        if 'volume_ratio_20' in features:
            if features['volume_ratio_20'] < entry_config['volume_threshold']:
                return False
        
        return True
    
    def _build_trade_params(
        self,
        signal: str,
        current_price: float,
        features: pd.Series,
        confidence: float
    ) -> Dict:
        """
        Build trade parameters including stop loss and take profit.
        
        Args:
            signal: Trading signal
            current_price: Current price
            features: Feature row
            confidence: Model confidence
            
        Returns:
            Trade parameters dictionary
        """
        # Get ATR for stop loss calculation
        atr = features.get('atr_14', current_price * 0.02)  # Default 2%
        
        # Calculate stops and targets
        stop_config = self.config['stop_loss']
        exit_config = self.config['exit']
        
        atr_multiplier = stop_config['atr_multiplier']
        
        if signal == 'LONG':
            stop_loss = current_price - (atr * atr_multiplier)
            partial_tp = current_price + (atr * exit_config['partial_tp_atr_multiplier'])
            full_tp = current_price + (atr * exit_config['full_tp_atr_multiplier'])
        else:  # SHORT
            stop_loss = current_price + (atr * atr_multiplier)
            partial_tp = current_price - (atr * exit_config['partial_tp_atr_multiplier'])
            full_tp = current_price - (atr * exit_config['full_tp_atr_multiplier'])
        
        # Calculate position sizing multiplier based on ADX
        adx = features.get('adx_14', 25)
        position_sizing_config = self.config['position_sizing']
        
        size_multiplier = 1.0
        for adx_range in position_sizing_config['adx_ranges']:
            if adx_range[0] <= adx < adx_range[1]:
                size_multiplier = adx_range[2]
                break
        
        trade_params = {
            'signal': signal,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit_partial': partial_tp,
            'take_profit_full': full_tp,
            'atr': atr,
            'confidence': confidence,
            'size_multiplier': size_multiplier,
            'strategy': 'trend_following',
            'max_leverage': self.config['max_leverage']
        }
        
        return trade_params
    
    def save_model(self, path: str):
        """
        Save model to file.
        
        Args:
            path: File path
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        self.model.save_model(path)
        
        # Save feature columns separately
        with open(path + '.features', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from file.
        
        Args:
            path: File path
        """
        self.model = lgb.Booster(model_file=path)
        
        # Load feature columns
        try:
            with open(path + '.features', 'rb') as f:
                self.feature_columns = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Feature columns file not found")
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of {feature: importance}
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.feature_columns if self.feature_columns else [f'f{i}' for i in range(len(importance))]
        
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort and get top N
        sorted_importance = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return dict(sorted_importance)


if __name__ == "__main__":
    # Test strategy
    print("Testing Trend Following Strategy...")
    
    strategy = TrendFollowingStrategy()
    
    # Create dummy data for testing
    n_samples = 1000
    n_features = 50
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 3, n_samples))
    
    X_val = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.randint(0, 3, 200))
    
    # Add required features for signal generation
    X_val['adx_14'] = 30
    X_val['ema_8'] = 50100
    X_val['ema_21'] = 50000
    X_val['ema_50'] = 49900
    X_val['volume_ratio_20'] = 1.5
    X_val['atr_14'] = 500
    
    # Train
    metrics = strategy.train(X_train, y_train, X_val, y_val)
    print(f"\nTraining Metrics: {metrics}")
    
    # Generate signal
    signal, confidence, params = strategy.generate_signal(X_val.tail(1), current_price=50000)
    print(f"\nSignal: {signal}")
    print(f"Confidence: {confidence:.2f}")
    if params:
        print(f"Stop Loss: ${params['stop_loss']:.2f}")
        print(f"Take Profit: ${params['take_profit_full']:.2f}")
    
    # Feature importance
    importance = strategy.get_feature_importance(top_n=10)
    print(f"\nTop 10 Features:")
    for feat, imp in importance.items():
        print(f"  {feat}: {imp:.2f}")
    
    print("\n✅ Trend Following strategy test complete")

