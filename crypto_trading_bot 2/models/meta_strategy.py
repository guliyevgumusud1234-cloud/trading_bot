"""
Meta Strategy Orchestrator.

This component:
- Allocates capital across all 5 strategies
- Detects market regime (trending/ranging/volatile)
- Adjusts strategy weights dynamically
- Monitors individual strategy performance
- Rebalances based on recent performance
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import yaml
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)


class MetaOrchestrator:
    """
    Master orchestrator that allocates capital to strategies based on market regime.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize meta orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.default_weights = config['strategy_weights']
        
        self.model = None  # LightGBM model for dynamic allocation
        self.strategy_performance = {}  # Track recent performance
        
        logger.info("MetaOrchestrator initialized")
    
    def detect_market_regime(self, features: pd.Series) -> Dict:
        """
        Detect current market regime.
        
        Args:
            features: Latest market features
            
        Returns:
            Market regime characteristics
        """
        regime = {}
        
        # 1. Trend Strength (ADX-based)
        if 'adx_14' in features:
            adx = features['adx_14']
            if adx > 30:
                regime['trend_strength'] = 'strong'
            elif adx < 20:
                regime['trend_strength'] = 'weak'
            else:
                regime['trend_strength'] = 'medium'
            regime['adx'] = adx
        else:
            regime['trend_strength'] = 'unknown'
        
        # 2. Trend Direction
        if 'ema_8' in features and 'ema_50' in features:
            if features['ema_8'] > features['ema_50']:
                regime['direction'] = 'bullish'
            else:
                regime['direction'] = 'bearish'
        else:
            regime['direction'] = 'neutral'
        
        # 3. Volatility Level
        if 'atr_14_pct' in features:
            atr_pct = features['atr_14_pct']
            if atr_pct > 0.05:  # >5%
                regime['volatility'] = 'high'
            elif atr_pct < 0.02:  # <2%
                regime['volatility'] = 'low'
            else:
                regime['volatility'] = 'medium'
            regime['atr_pct'] = atr_pct
        else:
            regime['volatility'] = 'unknown'
        
        # 4. Volume Regime
        if 'volume_ratio_20' in features:
            vol_ratio = features['volume_ratio_20']
            if vol_ratio > 1.5:
                regime['volume'] = 'high'
            elif vol_ratio < 0.7:
                regime['volume'] = 'low'
            else:
                regime['volume'] = 'normal'
        else:
            regime['volume'] = 'unknown'
        
        # 5. Funding Rate Extreme (if available)
        if 'funding_rate' in features:
            funding = features['funding_rate']
            if abs(funding) > 0.001:  # >0.1%
                regime['funding_extreme'] = True
                regime['funding_rate'] = funding
            else:
                regime['funding_extreme'] = False
        else:
            regime['funding_extreme'] = False
        
        logger.debug(f"Market Regime: {regime}")
        
        return regime
    
    def allocate_strategies(
        self,
        market_regime: Dict,
        recent_performance: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Allocate weights to strategies based on market regime.
        
        Args:
            market_regime: Current market regime
            recent_performance: Recent performance metrics by strategy
            
        Returns:
            Weight allocation dict {strategy: weight}
        """
        weights = {
            'trend': 0.0,
            'reversion': 0.0,
            'momentum': 0.0,
            'arbitrage': 0.0,
            'rl': 0.0
        }
        
        # Base allocation by regime
        trend_strength = market_regime.get('trend_strength', 'unknown')
        
        if trend_strength == 'strong':
            # Trending market: favor trend & momentum
            weights['trend'] = 0.40
            weights['momentum'] = 0.30
            weights['rl'] = 0.20
            weights['reversion'] = 0.05
            weights['arbitrage'] = 0.05
            
        elif trend_strength == 'weak':
            # Ranging market: favor mean reversion & arbitrage
            weights['reversion'] = 0.40
            weights['arbitrage'] = 0.25
            weights['rl'] = 0.25
            weights['trend'] = 0.05
            weights['momentum'] = 0.05
            
        else:
            # Medium trend or unknown: balanced
            weights['trend'] = 0.25
            weights['reversion'] = 0.25
            weights['momentum'] = 0.20
            weights['arbitrage'] = 0.15
            weights['rl'] = 0.15
        
        # Adjust for volatility
        volatility = market_regime.get('volatility', 'medium')
        if volatility == 'high':
            # Reduce leverage-dependent strategies in high vol
            weights['momentum'] *= 0.5
            weights['trend'] *= 0.7
            # Increase arbitrage (low risk)
            weights['arbitrage'] *= 1.5
        
        # Adjust for funding extremes
        if market_regime.get('funding_extreme', False):
            # Maximize arbitrage
            weights['arbitrage'] = 0.50
            # Reduce directional strategies
            for strategy in ['trend', 'reversion', 'momentum', 'rl']:
                weights[strategy] *= 0.5
        
        # Performance-based adjustment
        if recent_performance:
            weights = self._adjust_for_performance(weights, recent_performance)
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        logger.info(f"Strategy Allocation: {weights}")
        
        return weights
    
    def _adjust_for_performance(
        self,
        weights: Dict[str, float],
        performance: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Adjust weights based on recent strategy performance.
        
        Args:
            weights: Current weight allocation
            performance: Recent performance metrics {strategy: {'sharpe': x, 'win_rate': y}}
            
        Returns:
            Adjusted weights
        """
        adjusted_weights = weights.copy()
        
        for strategy, perf in performance.items():
            if strategy not in adjusted_weights:
                continue
            
            sharpe = perf.get('sharpe', 0)
            win_rate = perf.get('win_rate', 0.5)
            
            # Penalize negative Sharpe
            if sharpe < 0:
                adjusted_weights[strategy] *= 0.5
                logger.warning(f"{strategy} has negative Sharpe, reducing allocation")
            
            # Penalize low win rate
            elif win_rate < 0.4:
                adjusted_weights[strategy] *= 0.7
                logger.warning(f"{strategy} has low win rate, reducing allocation")
            
            # Reward excellent performance
            elif sharpe > 2.0:
                adjusted_weights[strategy] *= 1.3
                logger.info(f"{strategy} has excellent Sharpe, increasing allocation")
        
        return adjusted_weights
    
    def calculate_total_exposure(self, active_positions: List[Dict]) -> Dict:
        """
        Calculate total portfolio exposure.
        
        Args:
            active_positions: List of active position dicts
            
        Returns:
            Exposure metrics
        """
        if not active_positions:
            return {
                'total_exposure': 0.0,
                'total_leverage': 0.0,
                'net_exposure': 0.0,
                'num_positions': 0
            }
        
        total_long = 0.0
        total_short = 0.0
        
        for pos in active_positions:
            size = pos.get('size', 0)
            leverage = pos.get('leverage', 1)
            exposure = size * leverage
            
            if pos.get('side') == 'LONG':
                total_long += exposure
            else:
                total_short += exposure
        
        total_exposure = total_long + total_short
        net_exposure = total_long - total_short
        
        return {
            'total_exposure': total_exposure,
            'net_exposure': net_exposure,
            'total_long': total_long,
            'total_short': total_short,
            'total_leverage': total_exposure / self.config['capital']['initial_capital'],
            'num_positions': len(active_positions)
        }
    
    def should_reduce_risk(self, market_regime: Dict, portfolio_metrics: Dict) -> Tuple[bool, Optional[str]]:
        """
        Determine if risk should be reduced.
        
        Args:
            market_regime: Current market regime
            portfolio_metrics: Current portfolio metrics
            
        Returns:
            Tuple of (should_reduce, reason)
        """
        # Check volatility spike
        if market_regime.get('volatility') == 'high':
            atr_pct = market_regime.get('atr_pct', 0)
            if atr_pct > 0.08:  # >8% ATR
                return True, 'extreme_volatility'
        
        # Check leverage
        if portfolio_metrics.get('total_leverage', 0) > self.config['risk_limits']['max_portfolio_leverage']:
            return True, 'leverage_exceeded'
        
        # Check drawdown
        if portfolio_metrics.get('current_drawdown', 0) > 0.15:
            return True, 'high_drawdown'
        
        # Check correlation
        if portfolio_metrics.get('correlation_risk', 0) > 0.7:
            return True, 'high_correlation'
        
        return False, None
    
    def update_performance(self, strategy: str, metrics: Dict):
        """
        Update performance tracking for a strategy.
        
        Args:
            strategy: Strategy name
            metrics: Performance metrics
        """
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'history': [],
                'recent_7d': None,
                'recent_30d': None
            }
        
        # Add timestamp
        metrics['timestamp'] = datetime.now()
        
        # Add to history
        self.strategy_performance[strategy]['history'].append(metrics)
        
        # Keep only last 90 days
        cutoff = datetime.now() - timedelta(days=90)
        self.strategy_performance[strategy]['history'] = [
            m for m in self.strategy_performance[strategy]['history']
            if m['timestamp'] > cutoff
        ]
        
        # Calculate recent averages
        self._calculate_recent_performance(strategy)
    
    def _calculate_recent_performance(self, strategy: str):
        """Calculate 7-day and 30-day performance averages."""
        history = self.strategy_performance[strategy]['history']
        
        if not history:
            return
        
        now = datetime.now()
        
        # 7-day performance
        recent_7d = [m for m in history if (now - m['timestamp']).days <= 7]
        if recent_7d:
            self.strategy_performance[strategy]['recent_7d'] = {
                'sharpe': np.mean([m.get('sharpe', 0) for m in recent_7d]),
                'win_rate': np.mean([m.get('win_rate', 0.5) for m in recent_7d]),
                'avg_pnl': np.mean([m.get('pnl', 0) for m in recent_7d])
            }
        
        # 30-day performance
        recent_30d = [m for m in history if (now - m['timestamp']).days <= 30]
        if recent_30d:
            self.strategy_performance[strategy]['recent_30d'] = {
                'sharpe': np.mean([m.get('sharpe', 0) for m in recent_30d]),
                'win_rate': np.mean([m.get('win_rate', 0.5) for m in recent_30d]),
                'avg_pnl': np.mean([m.get('pnl', 0) for m in recent_30d])
            }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for all strategies."""
        summary = {}
        
        for strategy, perf in self.strategy_performance.items():
            summary[strategy] = {
                'total_trades': len(perf['history']),
                'recent_7d': perf.get('recent_7d', {}),
                'recent_30d': perf.get('recent_30d', {})
            }
        
        return summary


if __name__ == "__main__":
    print("Testing Meta Strategy Orchestrator...")
    
    orchestrator = MetaOrchestrator()
    
    # Test 1: Detect market regime
    print("\n=== Test 1: Market Regime Detection ===")
    
    # Trending market
    features_trending = pd.Series({
        'adx_14': 35,
        'ema_8': 51000,
        'ema_50': 50000,
        'atr_14_pct': 0.03,
        'volume_ratio_20': 1.2,
        'funding_rate': 0.0005
    })
    
    regime = orchestrator.detect_market_regime(features_trending)
    print(f"Trending Market Regime: {regime}")
    
    # Test 2: Strategy allocation
    print("\n=== Test 2: Strategy Allocation ===")
    weights = orchestrator.allocate_strategies(regime)
    print("Weights for trending market:")
    for strategy, weight in weights.items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Ranging market
    features_ranging = pd.Series({
        'adx_14': 18,
        'ema_8': 50000,
        'ema_50': 50100,
        'atr_14_pct': 0.015,
        'volume_ratio_20': 0.9,
        'funding_rate': 0.0002
    })
    
    regime_ranging = orchestrator.detect_market_regime(features_ranging)
    weights_ranging = orchestrator.allocate_strategies(regime_ranging)
    print("\nWeights for ranging market:")
    for strategy, weight in weights_ranging.items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Test 3: Performance-based adjustment
    print("\n=== Test 3: Performance-Based Adjustment ===")
    
    # Simulate poor performance for trend strategy
    recent_performance = {
        'trend': {'sharpe': -0.5, 'win_rate': 0.35},
        'reversion': {'sharpe': 1.8, 'win_rate': 0.58},
        'momentum': {'sharpe': 1.2, 'win_rate': 0.52}
    }
    
    weights_adjusted = orchestrator.allocate_strategies(regime, recent_performance)
    print("Adjusted weights (trend underperforming):")
    for strategy, weight in weights_adjusted.items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Test 4: Risk reduction check
    print("\n=== Test 4: Risk Reduction Check ===")
    
    high_vol_regime = {
        'volatility': 'high',
        'atr_pct': 0.09
    }
    
    portfolio_metrics = {
        'total_leverage': 2.5,
        'current_drawdown': 0.12
    }
    
    should_reduce, reason = orchestrator.should_reduce_risk(high_vol_regime, portfolio_metrics)
    print(f"Should reduce risk: {should_reduce}, Reason: {reason}")
    
    # Test 5: Exposure calculation
    print("\n=== Test 5: Portfolio Exposure ===")
    
    active_positions = [
        {'side': 'LONG', 'size': 1000, 'leverage': 3},
        {'side': 'SHORT', 'size': 500, 'leverage': 2},
        {'side': 'LONG', 'size': 800, 'leverage': 2}
    ]
    
    exposure = orchestrator.calculate_total_exposure(active_positions)
    print(f"Total Exposure: ${exposure['total_exposure']:,.0f}")
    print(f"Net Exposure: ${exposure['net_exposure']:,.0f}")
    print(f"Total Leverage: {exposure['total_leverage']:.2f}x")
    print(f"Open Positions: {exposure['num_positions']}")
    
    print("\n✅ Meta Strategy Orchestrator test complete")

