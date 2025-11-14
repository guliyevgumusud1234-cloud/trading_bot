"""
Position sizing module for the crypto trading bot.

This module implements:
- Kelly Criterion for optimal position sizing
- Volatility-based adjustments
- Market regime adjustments
- Conservative position sizing (25% of Kelly)
- Leverage calculation
"""

import numpy as np
from typing import Dict, Optional, Tuple
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class KellyPositionSizer:
    """
    Position sizing using Kelly Criterion with conservative adjustments.
    
    Kelly Formula: f* = (bp - q) / b
    where:
    - f* = fraction of capital to bet
    - b = odds received on the bet (avg_win / avg_loss)
    - p = probability of winning (win_rate)
    - q = probability of losing (1 - p)
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize position sizer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.max_position_pct = self.config['capital']['max_per_trade'] / self.config['capital']['initial']
        self.kelly_fraction = 0.25  # Conservative: use 25% of full Kelly
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (absolute value)
            
        Returns:
            Kelly fraction (0-1)
        """
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win rate: {win_rate}")
            return 0.01  # Default to 1%
        
        if avg_win <= 0 or avg_loss <= 0:
            logger.warning(f"Invalid avg_win or avg_loss: {avg_win}, {avg_loss}")
            return 0.01
        
        # Kelly formula
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly = (b * p - q) / b
        
        # Ensure positive and reasonable
        kelly = max(0, min(kelly, 1.0))
        
        # Apply conservative fraction (25% of Kelly)
        conservative_kelly = kelly * self.kelly_fraction
        
        logger.debug(
            f"Kelly calculation: win_rate={win_rate:.2%}, "
            f"avg_win={avg_win:.2f}, avg_loss={avg_loss:.2f}, "
            f"kelly={kelly:.4f}, conservative={conservative_kelly:.4f}"
        )
        
        return conservative_kelly
    
    def apply_volatility_adjustment(
        self,
        base_size: float,
        current_volatility: float,
        avg_volatility: float
    ) -> float:
        """
        Adjust position size based on volatility.
        
        Higher volatility = smaller position size
        
        Args:
            base_size: Base position size
            current_volatility: Current volatility (e.g., ATR)
            avg_volatility: Average volatility
            
        Returns:
            Adjusted position size
        """
        if avg_volatility <= 0:
            return base_size
        
        vol_ratio = current_volatility / avg_volatility
        
        # Inverse relationship: higher vol = smaller size
        adjustment_factor = 1 / (1 + (vol_ratio - 1))
        
        # Clamp adjustment between 0.5 and 1.5
        adjustment_factor = max(0.5, min(adjustment_factor, 1.5))
        
        adjusted_size = base_size * adjustment_factor
        
        logger.debug(
            f"Volatility adjustment: vol_ratio={vol_ratio:.2f}, "
            f"factor={adjustment_factor:.2f}, "
            f"base={base_size:.2f}, adjusted={adjusted_size:.2f}"
        )
        
        return adjusted_size
    
    def apply_regime_adjustment(
        self,
        base_size: float,
        market_regime: str
    ) -> float:
        """
        Adjust position size based on market regime.
        
        Args:
            base_size: Base position size
            market_regime: 'favorable', 'neutral', or 'unfavorable'
            
        Returns:
            Adjusted position size
        """
        regime_multipliers = {
            'favorable': 1.0,
            'neutral': 0.7,
            'unfavorable': 0.4
        }
        
        multiplier = regime_multipliers.get(market_regime, 0.7)
        adjusted_size = base_size * multiplier
        
        logger.debug(
            f"Regime adjustment: regime={market_regime}, "
            f"multiplier={multiplier}, "
            f"adjusted={adjusted_size:.2f}"
        )
        
        return adjusted_size
    
    def calculate_position_size(
        self,
        balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_volatility: float,
        avg_volatility: float,
        market_regime: str = 'neutral',
        confidence_score: Optional[float] = None
    ) -> float:
        """
        Calculate final position size with all adjustments.
        
        Args:
            balance: Current account balance
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            current_volatility: Current market volatility
            avg_volatility: Average market volatility
            market_regime: Market regime
            confidence_score: Model confidence (0-1), optional
            
        Returns:
            Position size in base currency
        """
        # Calculate Kelly fraction
        kelly = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Base position size
        base_size = balance * kelly
        
        # Apply volatility adjustment
        base_size = self.apply_volatility_adjustment(
            base_size,
            current_volatility,
            avg_volatility
        )
        
        # Apply regime adjustment
        base_size = self.apply_regime_adjustment(base_size, market_regime)
        
        # Apply confidence adjustment if provided
        if confidence_score is not None:
            confidence_adjustment = 0.5 + (confidence_score * 0.5)  # 0.5 to 1.0
            base_size *= confidence_adjustment
            logger.debug(f"Confidence adjustment: {confidence_score:.2%} -> {confidence_adjustment:.2f}")
        
        # Enforce maximum position size (2% of balance)
        max_size = balance * self.max_position_pct
        final_size = min(base_size, max_size)
        
        logger.info(
            f"Position size calculated: ${final_size:.2f} "
            f"({final_size/balance:.2%} of balance)"
        )
        
        return final_size


class LeverageCalculator:
    """
    Calculate appropriate leverage based on volatility and confidence.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize leverage calculator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.max_leverage = self.config['leverage']['max_portfolio']
        self.strategy_limits = self.config['leverage']['strategy_limits']
    
    def calculate_leverage(
        self,
        strategy: str,
        volatility: str,  # 'low', 'medium', 'high'
        confidence_score: float,
        current_portfolio_leverage: float = 0
    ) -> float:
        """
        Calculate appropriate leverage.
        
        Args:
            strategy: Strategy name
            volatility: Volatility regime
            confidence_score: Model confidence (0-1)
            current_portfolio_leverage: Current portfolio leverage
            
        Returns:
            Leverage multiplier
        """
        # Base leverage by strategy
        strategy_max = self.strategy_limits.get(strategy, 3)
        
        # Adjust for volatility
        volatility_multipliers = {
            'low': 1.2,     # Can use more leverage in low vol
            'medium': 1.0,
            'high': 0.6     # Reduce leverage in high vol
        }
        
        vol_multiplier = volatility_multipliers.get(volatility, 1.0)
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + (confidence_score * 0.5)  # 0.5 to 1.0
        
        # Calculate leverage
        leverage = strategy_max * vol_multiplier * confidence_multiplier
        
        # Ensure we don't exceed portfolio max
        available_leverage = self.max_leverage - current_portfolio_leverage
        leverage = min(leverage, available_leverage)
        
        # Floor at 1x
        leverage = max(1.0, leverage)
        
        logger.info(
            f"Leverage calculated: {leverage:.1f}x "
            f"(strategy_max={strategy_max}, vol={volatility}, "
            f"confidence={confidence_score:.2%})"
        )
        
        return leverage
    
    def calculate_position_value(
        self,
        position_size: float,
        leverage: float,
        entry_price: float
    ) -> Dict[str, float]:
        """
        Calculate position details.
        
        Args:
            position_size: Position size in base currency
            leverage: Leverage multiplier
            entry_price: Entry price
            
        Returns:
            Dict with position details
        """
        notional_value = position_size * leverage
        quantity = notional_value / entry_price
        margin_required = position_size  # Without leverage, this is the margin
        
        return {
            'position_size': position_size,
            'leverage': leverage,
            'notional_value': notional_value,
            'quantity': quantity,
            'margin_required': margin_required,
            'entry_price': entry_price
        }


if __name__ == "__main__":
    # Test position sizing
    sizer = KellyPositionSizer()
    
    # Example trade statistics
    balance = 10000
    win_rate = 0.55
    avg_win = 100
    avg_loss = 50
    current_vol = 0.03
    avg_vol = 0.025
    
    position_size = sizer.calculate_position_size(
        balance=balance,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        current_volatility=current_vol,
        avg_volatility=avg_vol,
        market_regime='favorable',
        confidence_score=0.75
    )
    
    print(f"\nPosition Size: ${position_size:.2f}")
    print(f"Percentage of Balance: {position_size/balance:.2%}")
    
    # Test leverage
    lev_calc = LeverageCalculator()
    leverage = lev_calc.calculate_leverage(
        strategy='trend',
        volatility='medium',
        confidence_score=0.75
    )
    
    print(f"\nLeverage: {leverage:.1f}x")
    
    # Position details
    details = lev_calc.calculate_position_value(
        position_size=position_size,
        leverage=leverage,
        entry_price=50000
    )
    
    print(f"\nPosition Details:")
    for key, value in details.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✅ Position sizing test complete")

