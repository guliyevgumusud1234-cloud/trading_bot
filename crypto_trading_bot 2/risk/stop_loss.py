"""
Stop loss management for the crypto trading bot.

This module implements:
- Multi-layer stop loss system
- Trailing stops
- Time-based exits
- Support/Resistance based stops
- Liquidation protection
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class StopLossManager:
    """
    Manage multi-layer stop loss system.
    
    Implements:
    1. Hard stop (liquidation protection)
    2. ATR-based stop
    3. Support/Resistance stop
    4. Time-based stop
    5. Trailing stop (activated after profit threshold)
    """
    
    def __init__(
        self,
        position: Dict,
        atr: float,
        config_path: str = 'config/risk_limits.yaml'
    ):
        """
        Initialize stop loss manager.
        
        Args:
            position: Position dictionary with entry details
            atr: Current ATR value
            config_path: Path to risk configuration
        """
        self.position = position
        self.atr = atr
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.entry_price = position['entry_price']
        self.entry_time = position.get('entry_time', datetime.now())
        self.size = position['size']
        self.side = position['side']  # 'LONG' or 'SHORT'
        self.leverage = position.get('leverage', 1)
        
        # Initialize stops
        self.hard_stop = None
        self.atr_stop = None
        self.sr_stop = None
        self.trailing_stop = None
        self.time_stop_active = False
        
        # Calculate initial stops
        self.calculate_all_stops()
    
    def calculate_all_stops(self) -> Dict[str, float]:
        """
        Calculate all stop loss levels.
        
        Returns:
            Dictionary of stop loss levels
        """
        # Layer 1: Hard stop (liquidation protection)
        self.hard_stop = self.calculate_hard_stop()
        
        # Layer 2: ATR-based stop
        self.atr_stop = self.calculate_atr_stop()
        
        # Layer 3: Support/Resistance stop (placeholder - needs S/R levels)
        self.sr_stop = self.calculate_sr_stop()
        
        # Select tightest stop (closest to entry, but not liquidation)
        stops = [self.atr_stop, self.sr_stop]
        stops = [s for s in stops if s is not None]
        
        if self.side == 'LONG':
            # For longs, use highest stop (furthest from price, tightest)
            final_stop = max(stops) if stops else self.atr_stop
            # But ensure it's below entry
            final_stop = min(final_stop, self.entry_price * 0.98)
        else:  # SHORT
            # For shorts, use lowest stop
            final_stop = min(stops) if stops else self.atr_stop
            # But ensure it's above entry
            final_stop = max(final_stop, self.entry_price * 1.02)
        
        # Ensure final stop is safer than hard stop
        if self.side == 'LONG':
            final_stop = max(final_stop, self.hard_stop)
        else:
            final_stop = min(final_stop, self.hard_stop)
        
        logger.info(
            f"Stop losses calculated: Hard={self.hard_stop:.2f}, "
            f"ATR={self.atr_stop:.2f}, Final={final_stop:.2f}"
        )
        
        return {
            'hard_stop': self.hard_stop,
            'atr_stop': self.atr_stop,
            'sr_stop': self.sr_stop,
            'final_stop': final_stop
        }
    
    def calculate_hard_stop(self) -> float:
        """
        Calculate hard stop (liquidation protection).
        
        Returns:
            Hard stop price
        """
        # Calculate liquidation price
        if self.side == 'LONG':
            # Long liquidation = entry * (1 - 1/leverage)
            liq_price = self.entry_price * (1 - 0.9 / self.leverage)
            # Hard stop 10% above liquidation
            hard_stop = liq_price * 1.10
        else:  # SHORT
            # Short liquidation = entry * (1 + 1/leverage)
            liq_price = self.entry_price * (1 + 0.9 / self.leverage)
            # Hard stop 10% below liquidation
            hard_stop = liq_price * 0.90
        
        logger.debug(
            f"Liquidation protection: liq_price={liq_price:.2f}, "
            f"hard_stop={hard_stop:.2f}"
        )
        
        return hard_stop
    
    def calculate_atr_stop(self, atr_multiplier: float = 2.0) -> float:
        """
        Calculate ATR-based stop loss.
        
        Args:
            atr_multiplier: ATR multiplier (default 2.0)
            
        Returns:
            ATR stop price
        """
        if self.side == 'LONG':
            atr_stop = self.entry_price - (atr_multiplier * self.atr)
        else:  # SHORT
            atr_stop = self.entry_price + (atr_multiplier * self.atr)
        
        # Ensure max 2% loss
        max_loss_pct = self.config['loss_limits']['per_trade']['max_loss_pct']
        
        if self.side == 'LONG':
            max_loss_stop = self.entry_price * (1 - max_loss_pct)
            atr_stop = max(atr_stop, max_loss_stop)
        else:
            max_loss_stop = self.entry_price * (1 + max_loss_pct)
            atr_stop = min(atr_stop, max_loss_stop)
        
        return atr_stop
    
    def calculate_sr_stop(self, sr_level: Optional[float] = None) -> Optional[float]:
        """
        Calculate support/resistance based stop.
        
        Args:
            sr_level: S/R level (optional)
            
        Returns:
            S/R stop price or None
        """
        if sr_level is None:
            return None
        
        # Add small buffer (0.5%) beyond S/R level
        buffer = 0.005
        
        if self.side == 'LONG':
            sr_stop = sr_level * (1 - buffer)
        else:
            sr_stop = sr_level * (1 + buffer)
        
        return sr_stop
    
    def update_trailing_stop(
        self,
        current_price: float,
        current_profit_pct: float,
        activation_threshold: float = 0.015
    ) -> Optional[float]:
        """
        Update trailing stop if profit threshold reached.
        
        Args:
            current_price: Current market price
            current_profit_pct: Current profit percentage
            activation_threshold: Profit threshold to activate trailing (default 1.5%)
            
        Returns:
            New trailing stop price or None
        """
        # Check if trailing should be activated
        if current_profit_pct < activation_threshold:
            return self.trailing_stop
        
        # Calculate trailing distance (1x ATR)
        trail_distance = self.atr
        
        if self.side == 'LONG':
            new_trailing_stop = current_price - trail_distance
            
            # Only move stop up, never down
            if self.trailing_stop is None:
                self.trailing_stop = new_trailing_stop
                logger.info(f"Trailing stop activated at {self.trailing_stop:.2f}")
            elif new_trailing_stop > self.trailing_stop:
                logger.info(
                    f"Trailing stop moved: {self.trailing_stop:.2f} -> {new_trailing_stop:.2f}"
                )
                self.trailing_stop = new_trailing_stop
        
        else:  # SHORT
            new_trailing_stop = current_price + trail_distance
            
            # Only move stop down, never up
            if self.trailing_stop is None:
                self.trailing_stop = new_trailing_stop
                logger.info(f"Trailing stop activated at {self.trailing_stop:.2f}")
            elif new_trailing_stop < self.trailing_stop:
                logger.info(
                    f"Trailing stop moved: {self.trailing_stop:.2f} -> {new_trailing_stop:.2f}"
                )
                self.trailing_stop = new_trailing_stop
        
        return self.trailing_stop
    
    def check_time_stop(
        self,
        max_duration_hours: int = 24,
        min_profit_threshold: float = 0.0
    ) -> bool:
        """
        Check if time-based stop should trigger.
        
        Args:
            max_duration_hours: Maximum position duration
            min_profit_threshold: Minimum profit to avoid time stop
            
        Returns:
            Whether time stop should trigger
        """
        duration = datetime.now() - self.entry_time
        duration_hours = duration.total_seconds() / 3600
        
        if duration_hours > max_duration_hours:
            logger.info(
                f"Time stop check: Position held for {duration_hours:.1f}h "
                f"(max: {max_duration_hours}h)"
            )
            return True
        
        return False
    
    def should_exit(
        self,
        current_price: float,
        current_profit_pct: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if any stop loss should trigger.
        
        Args:
            current_price: Current market price
            current_profit_pct: Current profit percentage
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Check hard stop (liquidation protection)
        if self.side == 'LONG' and current_price <= self.hard_stop:
            return True, "hard_stop_liquidation_protection"
        elif self.side == 'SHORT' and current_price >= self.hard_stop:
            return True, "hard_stop_liquidation_protection"
        
        # Check ATR stop
        if self.side == 'LONG' and current_price <= self.atr_stop:
            return True, "atr_stop_loss"
        elif self.side == 'SHORT' and current_price >= self.atr_stop:
            return True, "atr_stop_loss"
        
        # Check trailing stop (if activated)
        if self.trailing_stop is not None:
            if self.side == 'LONG' and current_price <= self.trailing_stop:
                return True, "trailing_stop"
            elif self.side == 'SHORT' and current_price >= self.trailing_stop:
                return True, "trailing_stop"
        
        # Check time stop
        if self.check_time_stop() and current_profit_pct < 0.01:
            return True, "time_stop_no_progress"
        
        return False, None
    
    def get_current_stops(self) -> Dict[str, Optional[float]]:
        """
        Get all current stop levels.
        
        Returns:
            Dictionary of stop levels
        """
        return {
            'hard_stop': self.hard_stop,
            'atr_stop': self.atr_stop,
            'sr_stop': self.sr_stop,
            'trailing_stop': self.trailing_stop
        }


class DynamicStopLoss:
    """
    Dynamic stop loss adjustment based on market conditions.
    """
    
    @staticmethod
    def adjust_stop_for_volatility(
        stop_price: float,
        entry_price: float,
        current_volatility: float,
        avg_volatility: float,
        side: str
    ) -> float:
        """
        Adjust stop loss for volatility changes.
        
        In high volatility, widen stops to avoid whipsaws.
        
        Args:
            stop_price: Current stop price
            entry_price: Entry price
            current_volatility: Current volatility
            avg_volatility: Average volatility
            side: 'LONG' or 'SHORT'
            
        Returns:
            Adjusted stop price
        """
        if avg_volatility <= 0:
            return stop_price
        
        vol_ratio = current_volatility / avg_volatility
        
        if vol_ratio > 1.5:  # High volatility
            # Widen stop by 50%
            distance = abs(entry_price - stop_price)
            new_distance = distance * 1.5
            
            if side == 'LONG':
                adjusted_stop = entry_price - new_distance
            else:
                adjusted_stop = entry_price + new_distance
            
            logger.info(
                f"Stop widened for volatility: {stop_price:.2f} -> {adjusted_stop:.2f}"
            )
            return adjusted_stop
        
        return stop_price
    
    @staticmethod
    def calculate_breakeven_stop(
        entry_price: float,
        side: str,
        buffer_pct: float = 0.001
    ) -> float:
        """
        Calculate breakeven stop price.
        
        Args:
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            buffer_pct: Small buffer to ensure breakeven (0.1%)
            
        Returns:
            Breakeven stop price
        """
        if side == 'LONG':
            return entry_price * (1 + buffer_pct)
        else:
            return entry_price * (1 - buffer_pct)
    
    @staticmethod
    def calculate_profit_protection_stop(
        entry_price: float,
        current_price: float,
        side: str,
        protection_pct: float = 0.5
    ) -> float:
        """
        Calculate stop to protect percentage of profit.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            side: 'LONG' or 'SHORT'
            protection_pct: Percentage of profit to protect (0.5 = 50%)
            
        Returns:
            Profit protection stop price
        """
        if side == 'LONG':
            profit = current_price - entry_price
            protected_profit = profit * protection_pct
            return entry_price + protected_profit
        else:
            profit = entry_price - current_price
            protected_profit = profit * protection_pct
            return entry_price - protected_profit


class PartialExitManager:
    """
    Manage partial position exits for taking profits.
    """
    
    def __init__(self, position: Dict):
        """
        Initialize partial exit manager.
        
        Args:
            position: Position dictionary
        """
        self.position = position
        self.entry_price = position['entry_price']
        self.original_size = position['size']
        self.remaining_size = position['size']
        self.side = position['side']
        
        self.exits_taken = []
    
    def should_take_partial(
        self,
        current_price: float,
        current_profit_pct: float
    ) -> Optional[Dict]:
        """
        Check if partial exit should be taken.
        
        Args:
            current_price: Current market price
            current_profit_pct: Current profit percentage
            
        Returns:
            Partial exit details or None
        """
        # Level 1: 25% at 1.5% profit
        if current_profit_pct >= 0.015 and not self._exit_taken_at_level(1):
            return {
                'level': 1,
                'percentage': 0.25,
                'size': self.original_size * 0.25,
                'price': current_price,
                'reason': 'first_target_1.5pct'
            }
        
        # Level 2: 25% at 3% profit
        if current_profit_pct >= 0.03 and not self._exit_taken_at_level(2):
            return {
                'level': 2,
                'percentage': 0.25,
                'size': self.original_size * 0.25,
                'price': current_price,
                'reason': 'second_target_3pct'
            }
        
        # Level 3: 25% at 5% profit
        if current_profit_pct >= 0.05 and not self._exit_taken_at_level(3):
            return {
                'level': 3,
                'percentage': 0.25,
                'size': self.original_size * 0.25,
                'price': current_price,
                'reason': 'third_target_5pct'
            }
        
        return None
    
    def record_partial_exit(self, exit_details: Dict):
        """
        Record a partial exit.
        
        Args:
            exit_details: Exit details
        """
        self.exits_taken.append(exit_details)
        self.remaining_size -= exit_details['size']
        
        logger.info(
            f"Partial exit taken: Level {exit_details['level']}, "
            f"Size: {exit_details['size']:.4f}, "
            f"Remaining: {self.remaining_size:.4f}"
        )
    
    def _exit_taken_at_level(self, level: int) -> bool:
        """Check if exit already taken at level."""
        return any(e['level'] == level for e in self.exits_taken)
    
    def get_remaining_percentage(self) -> float:
        """Get remaining position percentage."""
        return self.remaining_size / self.original_size


if __name__ == "__main__":
    # Test stop loss manager
    position = {
        'entry_price': 50000,
        'entry_time': datetime.now() - timedelta(hours=2),
        'size': 0.1,
        'side': 'LONG',
        'leverage': 3
    }
    
    sl_manager = StopLossManager(position=position, atr=500)
    
    stops = sl_manager.calculate_all_stops()
    print("\nStop Loss Levels:")
    for stop_type, price in stops.items():
        print(f"  {stop_type}: ${price:.2f}")
    
    # Test trailing stop
    current_price = 51000
    profit_pct = (current_price - position['entry_price']) / position['entry_price']
    
    trailing = sl_manager.update_trailing_stop(current_price, profit_pct)
    print(f"\nTrailing Stop: ${trailing:.2f if trailing else 'Not activated'}")
    
    # Test exit check
    should_exit, reason = sl_manager.should_exit(current_price, profit_pct)
    print(f"\nShould Exit: {should_exit} ({reason})")
    
    # Test partial exits
    partial_manager = PartialExitManager(position)
    partial = partial_manager.should_take_partial(current_price, profit_pct)
    
    if partial:
        print(f"\nPartial Exit Available:")
        print(f"  Level: {partial['level']}")
        print(f"  Size: {partial['size']:.4f}")
        print(f"  Reason: {partial['reason']}")
    
    print("\n✅ Stop loss test complete")

