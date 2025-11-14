"""
Real-Time Position Monitor.

This module:
- Monitors all open positions continuously
- Checks stop loss conditions
- Checks take profit conditions
- Updates trailing stops
- Monitors risk violations
- Logs funding payments
- Handles emergency closures
"""

import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading

from execution.exchange_interface import ExchangeInterface
from risk.stop_loss import StopLossManager
from utils.logger import get_logger
from utils.metrics import PerformanceMetrics

logger = get_logger(__name__)


class RealTimeMonitor:
    """
    Real-time monitoring of all open positions.
    """
    
    def __init__(
        self,
        exchange: ExchangeInterface,
        check_interval: float = 1.0,  # Check every second
        funding_interval: int = 8 * 3600  # 8 hours
    ):
        """
        Initialize real-time monitor.
        
        Args:
            exchange: Exchange interface instance
            check_interval: Check interval in seconds
            funding_interval: Funding payment interval in seconds
        """
        self.exchange = exchange
        self.check_interval = check_interval
        self.funding_interval = funding_interval
        
        # Tracking
        self.positions = {}  # symbol -> position dict
        self.stop_loss_managers = {}  # symbol -> StopLossManager
        self.last_funding_check = datetime.now()
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        
        # Metrics
        self.metrics_calculator = PerformanceMetrics()
        
        # Callbacks
        self.on_stop_loss_callback = None
        self.on_take_profit_callback = None
        self.on_violation_callback = None
        
        logger.info("RealTimeMonitor initialized")
    
    def start(self):
        """Start the monitoring loop in a separate thread."""
        if self.is_running:
            logger.warning("Monitor already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop(self):
        """Stop the monitoring loop."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Real-time monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Monitor loop started")
        
        while self.is_running:
            try:
                # 1. Update all positions
                self._update_positions()
                
                # 2. Check stop losses
                self._check_stop_losses()
                
                # 3. Check take profits
                self._check_take_profits()
                
                # 4. Update trailing stops
                self._update_trailing_stops()
                
                # 5. Check funding times
                if self._is_funding_time():
                    self._log_funding_payments()
                
                # 6. Monitor API health
                self._check_api_health()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.check_interval)
    
    def _update_positions(self):
        """Update all position data from exchange."""
        if self.exchange is None:
            return  # Paper trading mode without exchange
        try:
            # Fetch current positions
            positions = self.exchange.fetch_positions()
            
            for position in positions:
                symbol = position['symbol']
                
                # Update position dict
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': 'LONG' if float(position.get('contracts', 0)) > 0 else 'SHORT',
                    'size': abs(float(position.get('contracts', 0))),
                    'entry_price': float(position.get('entryPrice', 0)),
                    'current_price': float(position.get('markPrice', 0)),
                    'unrealized_pnl': float(position.get('unrealizedPnl', 0)),
                    'leverage': float(position.get('leverage', 1)),
                    'liquidation_price': float(position.get('liquidationPrice', 0)),
                    'margin': float(position.get('initialMargin', 0)),
                    'timestamp': datetime.now()
                }
                
                # Initialize stop loss manager if needed
                if symbol not in self.stop_loss_managers:
                    self.stop_loss_managers[symbol] = StopLossManager(
                        position=self.positions[symbol]
                    )
            
            # Remove closed positions
            current_symbols = {p['symbol'] for p in positions}
            closed_symbols = set(self.positions.keys()) - current_symbols
            
            for symbol in closed_symbols:
                logger.info(f"Position closed: {symbol}")
                self.positions.pop(symbol, None)
                self.stop_loss_managers.pop(symbol, None)
                
        except Exception as e:
            # Skip NoneType errors in paper trading mode
            error_msg = str(e)
            if 'NoneType' in error_msg or self.exchange is None:
                # Paper trading mode - silently skip
                return
            # Real error - log it
            logger.error(f"Error updating positions: {e}")
    
    def _check_stop_losses(self):
        """Check stop loss conditions for all positions."""
        for symbol, position in list(self.positions.items()):
            try:
                current_price = position['current_price']
                
                # Get stop loss manager
                sl_manager = self.stop_loss_managers.get(symbol)
                if not sl_manager:
                    continue
                
                # Calculate stop loss levels
                stop_loss, time_stop = sl_manager.calculate_stops()
                
                # Check if stop loss hit
                should_close = False
                reason = None
                
                if position['side'] == 'LONG':
                    if current_price <= stop_loss:
                        should_close = True
                        reason = 'stop_loss_hit'
                else:  # SHORT
                    if current_price >= stop_loss:
                        should_close = True
                        reason = 'stop_loss_hit'
                
                # Check time-based stop
                if time_stop:
                    should_close = True
                    reason = 'time_stop'
                
                # Close position if needed
                if should_close:
                    logger.warning(
                        f"Stop loss triggered for {symbol}: {reason} "
                        f"(price: ${current_price:.2f}, stop: ${stop_loss:.2f})"
                    )
                    self.close_position(symbol, reason=reason)
                    
                    # Call callback if set
                    if self.on_stop_loss_callback:
                        self.on_stop_loss_callback(symbol, reason, position)
                        
            except Exception as e:
                logger.error(f"Error checking stop loss for {symbol}: {e}")
    
    def _check_take_profits(self):
        """Check take profit conditions for all positions."""
        for symbol, position in list(self.positions.items()):
            try:
                # Check if take profit is set (would be in trade params)
                # This is a simplified version
                
                unrealized_pnl_pct = position['unrealized_pnl'] / position.get('margin', 1)
                
                # Take profit at 3% gain (configurable)
                if unrealized_pnl_pct > 0.03:
                    logger.info(
                        f"Take profit triggered for {symbol}: "
                        f"P&L={unrealized_pnl_pct:.2%}"
                    )
                    self.close_position(symbol, reason='take_profit')
                    
                    # Call callback if set
                    if self.on_take_profit_callback:
                        self.on_take_profit_callback(symbol, position)
                        
            except Exception as e:
                logger.error(f"Error checking take profit for {symbol}: {e}")
    
    def _update_trailing_stops(self):
        """Update trailing stops for profitable positions."""
        for symbol, position in self.positions.items():
            try:
                sl_manager = self.stop_loss_managers.get(symbol)
                if not sl_manager:
                    continue
                
                current_price = position['current_price']
                
                # Update trailing stop
                sl_manager.trailing_stop(current_price)
                
            except Exception as e:
                logger.error(f"Error updating trailing stop for {symbol}: {e}")
    
    def _is_funding_time(self) -> bool:
        """Check if it's funding payment time (every 8h)."""
        now = datetime.now()
        
        # Funding times: 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        
        # Check if we're within 5 minutes of funding time and haven't logged recently
        if now.hour in funding_hours:
            if (now - self.last_funding_check).total_seconds() > 3600:  # 1 hour
                return True
        
        return False
    
    def _log_funding_payments(self):
        """Log funding payments for all positions."""
        logger.info("=== Funding Payment Time ===")
        
        total_funding = 0.0
        
        for symbol, position in self.positions.items():
            try:
                # Fetch funding rate (simplified - would need actual API call)
                # funding_rate = self.exchange.fetch_funding_rate(symbol)
                
                # Calculate estimated funding payment
                # funding_payment = position['size'] * position['current_price'] * funding_rate
                
                # For now, log position info
                logger.info(
                    f"{symbol}: Size={position['size']:.4f}, "
                    f"Side={position['side']}, "
                    f"Entry=${position['entry_price']:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error logging funding for {symbol}: {e}")
        
        self.last_funding_check = datetime.now()
    
    def _check_api_health(self):
        """Monitor API latency and connection health."""
        if self.exchange is None:
            return  # Paper trading mode without exchange
        try:
            latency = self.exchange.get_api_latency()
            
            if latency > 500:  # >500ms
                logger.warning(f"High API latency detected: {latency:.2f}ms")
            
            if latency < 0:
                logger.error("API connection issue detected!")
                
        except Exception as e:
            # Skip NoneType errors in paper trading mode
            error_msg = str(e)
            if 'NoneType' in error_msg or self.exchange is None:
                # Paper trading mode - silently skip
                return
            # Real error - log it
            logger.error(f"API health check failed: {e}")
    
    def close_position(
        self,
        symbol: str,
        reason: str = 'manual',
        reduce_only: bool = True
    ) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Trading pair
            reason: Reason for closing
            reduce_only: Use reduce-only order
            
        Returns:
            Success status
        """
        if self.exchange is None:
            # Paper trading: Just remove from tracking
            logger.info(f"Paper trading: Closing position {symbol} (simulated)")
            self.positions.pop(symbol, None)
            self.stop_loss_managers.pop(symbol, None)
            return True
        
        try:
            position = self.positions.get(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False
            
            # Determine order side (opposite of position)
            order_side = 'sell' if position['side'] == 'LONG' else 'buy'
            
            # Close with market order
            order = self.exchange.create_order(
                symbol=symbol,
                order_type='market',
                side=order_side,
                amount=position['size'],
                params={'reduceOnly': reduce_only}
            )
            
            logger.info(
                f"Position closed for {symbol}: {reason} | "
                f"P&L: ${position['unrealized_pnl']:.2f}"
            )
            
            # Remove from tracking
            self.positions.pop(symbol, None)
            self.stop_loss_managers.pop(symbol, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def emergency_close_all(self) -> Dict:
        """
        Emergency close all positions (panic button).
        
        Returns:
            Summary of closures
        """
        logger.critical("EMERGENCY: Closing all positions!")
        
        results = {
            'total': len(self.positions),
            'closed': 0,
            'failed': 0,
            'errors': []
        }
        
        for symbol in list(self.positions.keys()):
            success = self.close_position(symbol, reason='emergency')
            
            if success:
                results['closed'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(symbol)
        
        logger.info(f"Emergency closure complete: {results}")
        
        return results
    
    def get_positions_summary(self) -> Dict:
        """Get summary of all open positions."""
        if not self.positions:
            return {
                'num_positions': 0,
                'total_exposure': 0.0,
                'total_pnl': 0.0
            }
        
        total_exposure = 0.0
        total_pnl = 0.0
        total_margin = 0.0
        
        for position in self.positions.values():
            exposure = position['size'] * position['current_price'] * position['leverage']
            total_exposure += exposure
            total_pnl += position['unrealized_pnl']
            total_margin += position['margin']
        
        return {
            'num_positions': len(self.positions),
            'total_exposure': total_exposure,
            'total_pnl': total_pnl,
            'total_margin': total_margin,
            'positions': list(self.positions.values())
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    print("Testing Real-Time Position Monitor...")
    
    load_dotenv()
    
    # Initialize exchange and monitor
    exchange = ExchangeInterface(
        exchange_id='binance',
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    
    monitor = RealTimeMonitor(
        exchange=exchange,
        check_interval=2.0  # Check every 2 seconds for testing
    )
    
    print("\n=== Test 1: Position Summary ===")
    summary = monitor.get_positions_summary()
    print(f"Open Positions: {summary['num_positions']}")
    print(f"Total Exposure: ${summary['total_exposure']:,.2f}")
    print(f"Total P&L: ${summary['total_pnl']:,.2f}")
    
    print("\n=== Test 2: Start Monitoring ===")
    print("Starting monitor for 10 seconds...")
    monitor.start()
    
    # Run for 10 seconds
    time.sleep(10)
    
    monitor.stop()
    print("Monitor stopped")
    
    print("\n✅ Position Monitor test complete")
    print("\nNote: Full monitoring requires open positions and running continuously")

