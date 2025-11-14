"""
Funding Rate Arbitrage Strategy.

This strategy:
- Detects extreme funding rates (>0.1% or <-0.1%)
- Opens hedged positions to collect funding
- Long spot + Short futures (or vice versa)
- Low leverage (1x), low risk
- Minimum $10,000 position size
- Exits when funding normalizes
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import yaml
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)


class FundingArbitrageStrategy:
    """
    Funding rate arbitrage strategy (statistical model).
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        Initialize funding arbitrage strategy.
        
        Args:
            config_path: Path to strategy configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['funding_arbitrage']
        self.funding_history = {}  # Track funding rates
        
        logger.info("FundingArbitrageStrategy initialized")
    
    def generate_signal(
        self,
        current_funding_rate: float,
        symbol: str,
        current_price: float,
        funding_history: Optional[pd.DataFrame] = None
    ) -> Tuple[str, float, Optional[Dict]]:
        """
        Generate arbitrage signal based on funding rate.
        
        Args:
            current_funding_rate: Current 8h funding rate (as decimal, e.g., 0.001 = 0.1%)
            symbol: Trading symbol
            current_price: Current market price
            funding_history: Historical funding rates
            
        Returns:
            Tuple of (signal, confidence, trade_params)
        """
        signal = 'NEUTRAL'
        confidence = 0.0
        trade_params = None
        
        # Check if funding rate is extreme
        entry_threshold = self.config.get('entry', {}).get('extreme_funding_threshold', 0.001)  # Default 0.1%
        
        if current_funding_rate > entry_threshold:
            # Funding rate is very positive (longs paying shorts)
            # Strategy: Short futures + Long spot to collect funding
            signal = 'SHORT_FUTURES_LONG_SPOT'
            confidence = min(abs(current_funding_rate) / entry_threshold, 1.0)
            
        elif current_funding_rate < -entry_threshold:
            # Funding rate is very negative (shorts paying longs)
            # Strategy: Long futures + Short spot to collect funding
            signal = 'LONG_FUTURES_SHORT_SPOT'
            confidence = min(abs(current_funding_rate) / entry_threshold, 1.0)
        
        # Build trade parameters if signal exists
        if signal != 'NEUTRAL':
            # Calculate expected funding collection
            expected_funding = self._calculate_expected_funding(
                current_funding_rate,
                funding_history
            )
            
            trade_params = {
                'signal': signal,
                'entry_price': current_price,
                'funding_rate': current_funding_rate,
                'expected_funding_8h': expected_funding,
                'expected_daily_return': expected_funding * 3,  # 3x per day
                'confidence': confidence,
                'strategy': 'funding_arbitrage',
                'max_leverage': self.config['max_leverage'],
                'min_position_size': self.config['execution']['min_position_size_usd']
            }
            
            # Add exit conditions
            trade_params.update(self._calculate_exit_conditions(current_funding_rate))
        
        logger.debug(f"Funding Rate: {current_funding_rate:.4f}%, Signal: {signal}")
        
        return signal, confidence, trade_params
    
    def should_exit(
        self,
        position: Dict,
        current_funding_rate: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if arbitrage position should be exited.
        
        Args:
            position: Position dict with entry details
            current_funding_rate: Current funding rate
            
        Returns:
            Tuple of (should_exit, reason)
        """
        exit_config = self.config['exit']
        
        # Exit if funding normalized
        if abs(current_funding_rate) < exit_config['normalize_threshold']:
            return True, 'funding_normalized'
        
        # Exit if funding reversed significantly
        entry_funding = position.get('funding_rate', 0)
        funding_change = current_funding_rate - entry_funding
        
        # If we were short futures (positive funding), and it turned negative
        if position['signal'] == 'SHORT_FUTURES_LONG_SPOT':
            if current_funding_rate < -exit_config['reverse_threshold']:
                return True, 'funding_reversed'
        
        # If we were long futures (negative funding), and it turned positive
        elif position['signal'] == 'LONG_FUTURES_SHORT_SPOT':
            if current_funding_rate > exit_config['reverse_threshold']:
                return True, 'funding_reversed'
        
        # Time-based exit (max holding period)
        if 'entry_time' in position:
            holding_hours = (datetime.now() - position['entry_time']).total_seconds() / 3600
            if holding_hours > exit_config['max_holding_hours']:
                return True, 'max_holding_time'
        
        return False, None
    
    def _calculate_expected_funding(
        self,
        current_funding: float,
        funding_history: Optional[pd.DataFrame]
    ) -> float:
        """
        Calculate expected funding payment.
        
        Args:
            current_funding: Current funding rate
            funding_history: Historical funding rates
            
        Returns:
            Expected funding rate
        """
        if funding_history is None or len(funding_history) < 10:
            # Use current funding if no history
            return abs(current_funding)
        
        # Use exponentially weighted average of recent funding
        recent_funding = funding_history.tail(24)  # Last 24 periods (8 days)
        weights = np.exp(np.linspace(-1, 0, len(recent_funding)))
        weights = weights / weights.sum()
        
        expected = (recent_funding['funding_rate'].values * weights).sum()
        
        return abs(expected)
    
    def _calculate_exit_conditions(self, current_funding: float) -> Dict:
        """Calculate exit thresholds for the position."""
        exit_config = self.config['exit']
        
        return {
            'exit_funding_threshold': exit_config['normalize_threshold'],
            'reverse_threshold': exit_config['reverse_threshold'],
            'max_holding_hours': exit_config['max_holding_hours'],
            'target_collections': exit_config['min_collections']  # Collect at least N times
        }
    
    def calculate_position_size(
        self,
        available_capital: float,
        current_price: float,
        expected_funding: float
    ) -> float:
        """
        Calculate optimal position size for arbitrage.
        
        Args:
            available_capital: Available capital
            current_price: Current asset price
            expected_funding: Expected funding rate per 8h
            
        Returns:
            Position size in USD
        """
        min_size = self.config['execution']['min_position_size_usd']
        
        # For arbitrage, we want large positions (since risk is low)
        # Use up to 30% of capital per arbitrage
        max_allocation = available_capital * 0.3
        
        # Ensure minimum size
        position_size = max(min_size, max_allocation)
        
        # Cap at available capital
        position_size = min(position_size, available_capital)
        
        return position_size
    
    def calculate_expected_profit(
        self,
        position_size: float,
        funding_rate: float,
        num_collections: int = 3
    ) -> Dict:
        """
        Calculate expected profit from arbitrage.
        
        Args:
            position_size: Position size in USD
            funding_rate: 8h funding rate
            num_collections: Number of funding collections expected
            
        Returns:
            Profit metrics
        """
        # Profit per collection (both sides collect/pay)
        profit_per_collection = position_size * abs(funding_rate)
        
        # Total expected profit
        total_profit = profit_per_collection * num_collections
        
        # Costs
        # Entry: 2 orders (spot buy + futures short, or vice versa)
        # Exit: 2 orders
        # Total: 4 orders at maker fee (0.02%)
        maker_fee = 0.0002
        total_fees = position_size * maker_fee * 4
        
        # Net profit
        net_profit = total_profit - total_fees
        
        # ROI
        roi = net_profit / position_size
        
        return {
            'profit_per_collection': profit_per_collection,
            'expected_collections': num_collections,
            'total_profit': total_profit,
            'total_fees': total_fees,
            'net_profit': net_profit,
            'roi': roi,
            'annual_roi': roi * 365 / (num_collections * 8 / 24)  # Annualized
        }


if __name__ == "__main__":
    print("Testing Funding Arbitrage Strategy...")
    
    strategy = FundingArbitrageStrategy()
    
    # Test 1: Extreme positive funding (longs paying shorts)
    print("\n=== Test 1: Extreme Positive Funding ===")
    funding_rate = 0.0015  # 0.15% per 8h (very high)
    signal, confidence, params = strategy.generate_signal(
        current_funding_rate=funding_rate,
        symbol='BTC/USDT',
        current_price=50000
    )
    
    print(f"Funding Rate: {funding_rate:.4f}% ({funding_rate*100:.2f}%)")
    print(f"Signal: {signal}")
    print(f"Confidence: {confidence:.2f}")
    if params:
        print(f"Expected Daily Return: {params['expected_daily_return']:.4f}%")
        print(f"Min Position Size: ${params['min_position_size']:,.0f}")
    
    # Test 2: Calculate position size and profit
    print("\n=== Test 2: Position Sizing & Profit Calculation ===")
    available_capital = 100000  # $100k
    position_size = strategy.calculate_position_size(available_capital, 50000, 0.0015)
    print(f"Available Capital: ${available_capital:,.0f}")
    print(f"Position Size: ${position_size:,.0f}")
    
    profit_metrics = strategy.calculate_expected_profit(position_size, 0.0015, num_collections=6)
    print(f"\nExpected Profit (6 collections = 2 days):")
    print(f"  Profit per collection: ${profit_metrics['profit_per_collection']:.2f}")
    print(f"  Total profit: ${profit_metrics['total_profit']:.2f}")
    print(f"  Total fees: ${profit_metrics['total_fees']:.2f}")
    print(f"  Net profit: ${profit_metrics['net_profit']:.2f}")
    print(f"  ROI: {profit_metrics['roi']:.2%}")
    print(f"  Annualized ROI: {profit_metrics['annual_roi']:.2%}")
    
    # Test 3: Exit conditions
    print("\n=== Test 3: Exit Conditions ===")
    mock_position = {
        'signal': 'SHORT_FUTURES_LONG_SPOT',
        'funding_rate': 0.0015,
        'entry_time': datetime.now() - timedelta(hours=48)
    }
    
    # Funding normalized
    should_exit, reason = strategy.should_exit(mock_position, current_funding_rate=0.0002)
    print(f"Funding normalized to 0.02%: Exit={should_exit}, Reason={reason}")
    
    # Funding reversed
    should_exit, reason = strategy.should_exit(mock_position, current_funding_rate=-0.0008)
    print(f"Funding reversed to -0.08%: Exit={should_exit}, Reason={reason}")
    
    print("\n✅ Funding Arbitrage strategy test complete")

