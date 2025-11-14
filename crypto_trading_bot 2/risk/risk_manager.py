"""
Master risk manager for the crypto trading bot.

This module coordinates all risk management:
- Pre-trade risk checks
- Position limits
- Drawdown protection
- Loss limits (circuit breakers)
- Margin management
- Real-time risk monitoring
- Emergency procedures
"""

import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from risk.position_sizing import KellyPositionSizer, LeverageCalculator
from risk.stop_loss import StopLossManager
from risk.portfolio_manager import PortfolioManager, ExposureManager
from utils.logger import get_logger
from utils.notifications import get_alert_manager

logger = get_logger(__name__)


class RiskAction(Enum):
    """Risk action types."""
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    EMERGENCY_STOP = "emergency_stop"


class DrawdownProtection:
    """
    Drawdown protection system.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize drawdown protection.
        
        Args:
            config: Risk configuration
        """
        self.config = config['drawdown_protection']
        self.peak_balance = 0
        self.current_drawdown = 0
    
    def update(self, current_balance: float):
        """
        Update drawdown calculation.
        
        Args:
            current_balance: Current balance
        """
        # Update peak
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        else:
            self.current_drawdown = 0
    
    def get_action(self) -> Tuple[str, Dict]:
        """
        Get action based on drawdown level.
        
        Returns:
            Tuple of (action, parameters)
        """
        levels = self.config['levels']
        
        # Check each level
        if self.current_drawdown >= levels['emergency']['threshold']:
            return 'emergency_stop', levels['emergency']
        elif self.current_drawdown >= levels['warning']['threshold']:
            return 'significant_reduction', levels['warning']
        elif self.current_drawdown >= levels['caution']['threshold']:
            return 'reduce_risk', levels['caution']
        else:
            return 'normal', levels['normal']
    
    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on drawdown.
        
        Returns:
            Multiplier (0-1)
        """
        action, params = self.get_action()
        return params.get('position_size_multiplier', 1.0)
    
    def get_leverage_multiplier(self) -> float:
        """
        Get leverage multiplier based on drawdown.
        
        Returns:
            Multiplier (0-1)
        """
        action, params = self.get_action()
        return params.get('leverage_multiplier', 1.0)


class RiskManager:
    """
    Master risk manager.
    
    Coordinates all risk management and enforces limits.
    """
    
    def __init__(
        self,
        initial_balance: float,
        config_path: str = 'config/risk_limits.yaml'
    ):
        """
        Initialize risk manager.
        
        Args:
            initial_balance: Starting balance
            config_path: Path to risk configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Initialize components
        self.position_sizer = KellyPositionSizer()
        self.leverage_calc = LeverageCalculator()
        self.portfolio_manager = PortfolioManager()
        self.drawdown_protection = DrawdownProtection(self.config)
        
        # Alert manager
        self.alerts = get_alert_manager()
        
        # State tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.last_reset = datetime.now()
        
        # Emergency state
        self.emergency_stop_active = False
        self.pause_until = None
        
        logger.info(f"RiskManager initialized with balance: ${initial_balance}")
    
    def validate_new_trade(
        self,
        symbol: str,
        side: str,
        strategy: str,
        confidence_score: float,
        current_positions: List[Dict],
        market_data: Dict
    ) -> Tuple[RiskAction, Optional[str], Optional[Dict]]:
        """
        Validate a new trade against all risk limits.
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            strategy: Strategy name
            confidence_score: Model confidence
            current_positions: List of current positions
            market_data: Market data
            
        Returns:
            Tuple of (action, reason, trade_params)
        """
        # Check emergency stop
        if self.emergency_stop_active:
            return RiskAction.BLOCK, "emergency_stop_active", None
        
        # Check pause
        if self.pause_until and datetime.now() < self.pause_until:
            return RiskAction.BLOCK, f"trading_paused_until_{self.pause_until}", None
        
        # Reset daily counters if needed
        self._check_and_reset_counters()
        
        # Check daily limits
        if not self._check_daily_limits():
            return RiskAction.BLOCK, "daily_limits_exceeded", None
        
        # Check weekly limits
        if not self._check_weekly_limits():
            return RiskAction.BLOCK, "weekly_limits_exceeded", None
        
        # Check position limits
        if not self._check_position_limits(current_positions):
            return RiskAction.BLOCK, "position_limits_exceeded", None
        
        # Check portfolio leverage
        current_leverage = self.portfolio_manager.calculate_portfolio_leverage(current_positions)
        if current_leverage >= self.config['leverage_limits']['max_portfolio_leverage']:
            return RiskAction.BLOCK, "portfolio_leverage_limit", None
        
        # Check margin
        if not self._check_margin_available(current_positions):
            return RiskAction.BLOCK, "insufficient_margin", None
        
        # Check correlation
        if len(current_positions) > 0:
            # This would need actual returns data in production
            pass  # Placeholder for correlation check
        
        # Check drawdown
        dd_action, dd_params = self.drawdown_protection.get_action()
        if dd_action == 'emergency_stop':
            self.trigger_emergency_stop("max_drawdown_exceeded")
            return RiskAction.EMERGENCY_STOP, "max_drawdown", None
        
        # Calculate position size
        position_size = self._calculate_safe_position_size(
            strategy=strategy,
            confidence_score=confidence_score,
            market_data=market_data
        )
        
        if position_size <= 0:
            return RiskAction.BLOCK, "position_size_too_small", None
        
        # Calculate leverage
        leverage = self._calculate_safe_leverage(
            strategy=strategy,
            confidence_score=confidence_score,
            market_data=market_data,
            current_leverage=current_leverage
        )
        
        # Apply drawdown adjustments
        if dd_action != 'normal':
            position_size *= dd_params.get('position_size_multiplier', 1.0)
            leverage *= dd_params.get('leverage_multiplier', 1.0)
            
            logger.warning(
                f"Drawdown adjustment applied: action={dd_action}, "
                f"size_mult={dd_params.get('position_size_multiplier', 1.0):.2f}, "
                f"lev_mult={dd_params.get('leverage_multiplier', 1.0):.2f}"
            )
        
        # Build trade parameters
        trade_params = {
            'symbol': symbol,
            'side': side,
            'strategy': strategy,
            'position_size': position_size,
            'leverage': leverage,
            'confidence_score': confidence_score
        }
        
        logger.info(
            f"Trade validated: {side} {symbol} | "
            f"Size: ${position_size:.2f} | Leverage: {leverage:.1f}x"
        )
        
        return RiskAction.ALLOW, None, trade_params
    
    def record_trade(
        self,
        pnl: float,
        fees: float = 0,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a completed trade.
        
        Args:
            pnl: Trade P&L
            fees: Trading fees
            timestamp: Trade timestamp
        """
        net_pnl = pnl - fees
        
        self.daily_trades += 1
        self.daily_pnl += net_pnl
        self.weekly_pnl += net_pnl
        self.current_balance += net_pnl
        
        # Update drawdown
        self.drawdown_protection.update(self.current_balance)
        
        logger.info(
            f"Trade recorded: P&L=${net_pnl:.2f}, "
            f"Daily P&L=${self.daily_pnl:.2f}, "
            f"Balance=${self.current_balance:.2f}"
        )
        
        # Check for large loss
        loss_pct = abs(net_pnl / self.current_balance)
        if net_pnl < 0 and loss_pct > 0.02:
            self.alerts.alert_large_loss(
                symbol="PORTFOLIO",
                loss_amount=abs(net_pnl),
                loss_pct=loss_pct
            )
    
    def check_risk_violations(
        self,
        current_positions: List[Dict]
    ) -> List[str]:
        """
        Check for any risk violations.
        
        Args:
            current_positions: Current positions
            
        Returns:
            List of violations
        """
        violations = []
        
        # Daily loss check
        daily_loss_limit = self.config['loss_limits']['per_day']['max_loss_pct']
        if self.daily_pnl < 0 and abs(self.daily_pnl / self.current_balance) > daily_loss_limit:
            violations.append('daily_loss_limit')
            self._handle_daily_loss_limit()
        
        # Weekly loss check
        weekly_loss_limit = self.config['loss_limits']['per_week']['max_loss_pct']
        if self.weekly_pnl < 0 and abs(self.weekly_pnl / self.current_balance) > weekly_loss_limit:
            violations.append('weekly_loss_limit')
            self._handle_weekly_loss_limit()
        
        # Drawdown check
        dd_action, dd_params = self.drawdown_protection.get_action()
        if dd_action == 'emergency_stop':
            violations.append('max_drawdown')
            self.trigger_emergency_stop("max_drawdown_exceeded")
        
        # Portfolio leverage check
        leverage = self.portfolio_manager.calculate_portfolio_leverage(current_positions)
        if leverage > self.config['leverage_limits']['max_portfolio_leverage']:
            violations.append('portfolio_leverage')
        
        # Position count check
        max_positions = self.config['position_limits']['max_total_positions']
        if len(current_positions) > max_positions:
            violations.append('max_positions')
        
        return violations
    
    def trigger_emergency_stop(self, reason: str):
        """
        Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            
            logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
            
            self.alerts.alert_emergency_stop(
                reason=reason,
                positions_closed=0  # Updated by position closer
            )
    
    def reset_emergency_stop(self):
        """Reset emergency stop (requires manual intervention)."""
        self.emergency_stop_active = False
        logger.info("Emergency stop reset")
    
    def _check_daily_limits(self) -> bool:
        """Check daily limits."""
        config = self.config['loss_limits']['per_day']
        
        # Max trades
        if self.daily_trades >= config['max_trades']:
            logger.warning(f"Daily trade limit reached: {self.daily_trades}")
            return False
        
        # Max loss
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl / self.current_balance)
            if loss_pct >= config['max_loss_pct']:
                logger.warning(f"Daily loss limit reached: {loss_pct:.2%}")
                return False
        
        return True
    
    def _check_weekly_limits(self) -> bool:
        """Check weekly limits."""
        config = self.config['loss_limits']['per_week']
        
        if self.weekly_pnl < 0:
            loss_pct = abs(self.weekly_pnl / self.current_balance)
            if loss_pct >= config['max_loss_pct']:
                logger.warning(f"Weekly loss limit reached: {loss_pct:.2%}")
                return False
        
        return True
    
    def _check_position_limits(self, positions: List[Dict]) -> bool:
        """Check position limits."""
        max_positions = self.config['position_limits']['max_total_positions']
        
        if len(positions) >= max_positions:
            logger.warning(f"Max positions reached: {len(positions)}/{max_positions}")
            return False
        
        return True
    
    def _check_margin_available(self, positions: List[Dict]) -> bool:
        """Check if sufficient margin available."""
        # Calculate used margin
        used_margin = sum(
            pos['size'] * pos.get('current_price', pos['entry_price'])
            for pos in positions
        )
        
        # Require buffer
        buffer = self.config['margin_management']['maintenance_margin_buffer']
        required_margin = used_margin * buffer
        
        available = self.current_balance - required_margin
        
        if available <= 0:
            logger.warning("Insufficient margin available")
            return False
        
        return True
    
    def _calculate_safe_position_size(
        self,
        strategy: str,
        confidence_score: float,
        market_data: Dict
    ) -> float:
        """Calculate safe position size with all adjustments."""
        # Get historical stats (would come from database in production)
        win_rate = 0.55  # Placeholder
        avg_win = 100  # Placeholder
        avg_loss = 50  # Placeholder
        
        position_size = self.position_sizer.calculate_position_size(
            balance=self.current_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_volatility=market_data.get('current_volatility', 0.03),
            avg_volatility=market_data.get('avg_volatility', 0.025),
            market_regime='neutral',
            confidence_score=confidence_score
        )
        
        return position_size
    
    def _calculate_safe_leverage(
        self,
        strategy: str,
        confidence_score: float,
        market_data: Dict,
        current_leverage: float
    ) -> float:
        """Calculate safe leverage."""
        volatility = market_data.get('volatility_regime', 'medium')
        
        leverage = self.leverage_calc.calculate_leverage(
            strategy=strategy,
            volatility=volatility,
            confidence_score=confidence_score,
            current_portfolio_leverage=current_leverage
        )
        
        return leverage
    
    def _check_and_reset_counters(self):
        """Reset daily/weekly counters if needed."""
        now = datetime.now()
        
        # Daily reset
        if now.date() != self.last_reset.date():
            logger.info(
                f"Daily reset: Trades={self.daily_trades}, "
                f"P&L=${self.daily_pnl:.2f}"
            )
            self.daily_trades = 0
            self.daily_pnl = 0
            self.last_reset = now
        
        # Weekly reset (every Monday)
        if now.weekday() == 0 and self.last_reset.weekday() != 0:
            logger.info(f"Weekly reset: P&L=${self.weekly_pnl:.2f}")
            self.weekly_pnl = 0
    
    def _handle_daily_loss_limit(self):
        """Handle daily loss limit breach."""
        pause_hours = 24
        self.pause_until = datetime.now() + timedelta(hours=pause_hours)
        
        logger.critical(f"Daily loss limit breached! Paused until {self.pause_until}")
        
        self.alerts.alert_risk_violation(
            violation_type="daily_loss_limit",
            current_value=abs(self.daily_pnl),
            limit_value=self.config['loss_limits']['per_day']['max_loss_pct'] * self.current_balance,
            action_taken=f"trading_paused_{pause_hours}h"
        )
    
    def _handle_weekly_loss_limit(self):
        """Handle weekly loss limit breach."""
        pause_days = 7
        self.pause_until = datetime.now() + timedelta(days=pause_days)
        
        logger.critical(f"Weekly loss limit breached! Paused until {self.pause_until}")
        
        self.alerts.alert_risk_violation(
            violation_type="weekly_loss_limit",
            current_value=abs(self.weekly_pnl),
            limit_value=self.config['loss_limits']['per_week']['max_loss_pct'] * self.current_balance,
            action_taken=f"trading_paused_{pause_days}d"
        )
    
    def get_risk_summary(self) -> Dict:
        """
        Get risk summary.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'current_balance': self.current_balance,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'current_drawdown': self.drawdown_protection.current_drawdown,
            'peak_balance': self.drawdown_protection.peak_balance,
            'emergency_stop': self.emergency_stop_active,
            'paused_until': self.pause_until
        }


if __name__ == "__main__":
    # Test risk manager
    risk_manager = RiskManager(initial_balance=10000)
    
    # Test trade validation
    action, reason, params = risk_manager.validate_new_trade(
        symbol="BTC/USDT",
        side="LONG",
        strategy="trend",
        confidence_score=0.75,
        current_positions=[],
        market_data={
            'current_volatility': 0.03,
            'avg_volatility': 0.025,
            'volatility_regime': 'medium'
        }
    )
    
    print(f"\nTrade Validation:")
    print(f"  Action: {action}")
    print(f"  Reason: {reason}")
    if params:
        print(f"  Position Size: ${params['position_size']:.2f}")
        print(f"  Leverage: {params['leverage']:.1f}x")
    
    # Test risk summary
    summary = risk_manager.get_risk_summary()
    print(f"\nRisk Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Risk manager test complete")

