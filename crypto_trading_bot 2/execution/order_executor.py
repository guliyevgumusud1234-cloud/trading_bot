"""
Smart Order Executor.

This module:
- Executes orders with pre-checks
- Implements order splitting (TWAP)
- Monitors slippage
- Handles execution errors
- Logs all execution details
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from execution.exchange_interface import ExchangeInterface
from utils.logger import get_logger

logger = get_logger(__name__)


class SmartOrderExecutor:
    """
    Smart order execution with pre-checks, splitting, and slippage monitoring.
    """
    
    def __init__(
        self,
        exchange: ExchangeInterface,
        max_slippage: float = 0.002,  # 0.2%
        use_twap: bool = True,
        twap_splits: int = 5,
        twap_interval: int = 60  # seconds
    ):
        """
        Initialize smart order executor.
        
        Args:
            exchange: Exchange interface instance
            max_slippage: Maximum acceptable slippage
            use_twap: Use TWAP for large orders
            twap_splits: Number of TWAP splits
            twap_interval: Interval between TWAP orders (seconds)
        """
        self.exchange = exchange
        self.max_slippage = max_slippage
        self.use_twap = use_twap
        self.twap_splits = twap_splits
        self.twap_interval = twap_interval
        
        # Execution tracking
        self.execution_history = []
        
        logger.info("SmartOrderExecutor initialized")
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = 'limit',
        leverage: int = 1,
        urgency: str = 'normal'
    ) -> Dict:
        """
        Execute an order with smart routing.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount in base currency
            order_type: 'market' or 'limit'
            leverage: Leverage to use
            urgency: 'low', 'normal', or 'high'
            
        Returns:
            Execution result dict
        """
        logger.info(f"Executing order: {side.upper()} {amount} {symbol} @ {leverage}x leverage")
        
        # Pre-execution checks
        checks_passed, check_results = self._pre_execution_checks(
            symbol=symbol,
            amount=amount,
            leverage=leverage
        )
        
        if not checks_passed:
            logger.error(f"Pre-execution checks failed: {check_results}")
            return {
                'success': False,
                'error': 'pre_checks_failed',
                'details': check_results
            }
        
        # Get current price
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Determine execution strategy
        if self.use_twap and amount * current_price > 5000:  # Large order
            logger.info(f"Large order detected, using TWAP execution")
            result = self._execute_twap(
                symbol=symbol,
                side=side,
                amount=amount,
                leverage=leverage
            )
        else:
            # Single order execution
            result = self._execute_single_order(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                leverage=leverage,
                urgency=urgency,
                expected_price=current_price
            )
        
        # Track execution
        self._track_execution(result)
        
        return result
    
    def _pre_execution_checks(
        self,
        symbol: str,
        amount: float,
        leverage: int
    ) -> Tuple[bool, Dict]:
        """
        Perform pre-execution checks.
        
        Args:
            symbol: Trading pair
            amount: Order amount
            leverage: Leverage
            
        Returns:
            Tuple of (checks_passed, check_results)
        """
        checks = {}
        
        try:
            # Check 1: Liquidity check
            orderbook = self.exchange.fetch_order_book(symbol, limit=20)
            
            # Calculate available liquidity in first 20 levels
            total_liquidity = sum([bid[1] for bid in orderbook['bids'][:20]])
            
            if amount > total_liquidity * 0.5:
                checks['liquidity'] = {
                    'passed': False,
                    'reason': f'Order too large: {amount:.4f} vs available {total_liquidity:.4f}'
                }
            else:
                checks['liquidity'] = {'passed': True}
            
            # Check 2: Margin check
            balance = self.exchange.fetch_balance()
            available_margin = balance.get('USDT', {}).get('free', 0)
            
            ticker = self.exchange.fetch_ticker(symbol)
            required_margin = (amount * ticker['last']) / leverage
            
            if required_margin > available_margin * 0.95:  # Leave 5% buffer
                checks['margin'] = {
                    'passed': False,
                    'reason': f'Insufficient margin: need {required_margin:.2f}, have {available_margin:.2f}'
                }
            else:
                checks['margin'] = {'passed': True}
            
            # Check 3: Leverage within limits
            if leverage > 20:  # Conservative max
                checks['leverage'] = {
                    'passed': False,
                    'reason': f'Leverage too high: {leverage}x'
                }
            else:
                checks['leverage'] = {'passed': True}
            
        except Exception as e:
            logger.error(f"Pre-execution checks error: {e}")
            checks['error'] = {'passed': False, 'reason': str(e)}
        
        # Determine if all checks passed
        all_passed = all(check.get('passed', False) for check in checks.values())
        
        return all_passed, checks
    
    def _execute_single_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str,
        leverage: int,
        urgency: str,
        expected_price: float
    ) -> Dict:
        """Execute a single order."""
        try:
            # Set leverage
            self.exchange.set_leverage(symbol, leverage)
            
            # Determine order parameters based on urgency
            if urgency == 'high' or order_type == 'market':
                # Market order for immediate execution
                order = self.exchange.create_order(
                    symbol=symbol,
                    order_type='market',
                    side=side,
                    amount=amount,
                    params={'leverage': leverage}
                )
            else:
                # Limit order with competitive price
                ticker = self.exchange.fetch_ticker(symbol)
                
                if side == 'buy':
                    # Buy at slightly above bid
                    limit_price = ticker['bid'] * 1.0005
                else:
                    # Sell at slightly below ask
                    limit_price = ticker['ask'] * 0.9995
                
                order = self.exchange.create_order(
                    symbol=symbol,
                    order_type='limit',
                    side=side,
                    amount=amount,
                    price=limit_price,
                    params={'leverage': leverage}
                )
                
                # Wait for fill (with timeout)
                timeout = 300  # 5 minutes
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    order_status = self.exchange.fetch_order(order['id'], symbol)
                    
                    if order_status['status'] == 'closed':
                        order = order_status
                        break
                    elif order_status['status'] == 'canceled':
                        logger.warning("Order was canceled")
                        return {
                            'success': False,
                            'error': 'order_canceled',
                            'order': order_status
                        }
                    
                    time.sleep(1)
                
                # If not filled, cancel and retry with market order
                if order['status'] != 'closed':
                    logger.warning("Limit order timeout, switching to market order")
                    self.exchange.cancel_order(order['id'], symbol)
                    
                    order = self.exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side=side,
                        amount=amount,
                        params={'leverage': leverage}
                    )
            
            # Calculate slippage
            filled_price = float(order.get('average', order.get('price', expected_price)))
            slippage = abs(filled_price - expected_price) / expected_price
            
            if slippage > self.max_slippage:
                logger.warning(
                    f"High slippage detected: {slippage:.2%} "
                    f"(expected: ${expected_price:.2f}, filled: ${filled_price:.2f})"
                )
            
            return {
                'success': True,
                'order': order,
                'filled_price': filled_price,
                'expected_price': expected_price,
                'slippage': slippage,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _execute_twap(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: int
    ) -> Dict:
        """
        Execute order using TWAP (Time-Weighted Average Price) strategy.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Total amount
            leverage: Leverage
            
        Returns:
            Execution result
        """
        logger.info(f"Executing TWAP: {self.twap_splits} splits over {self.twap_splits * self.twap_interval}s")
        
        # Split amount
        split_amount = amount / self.twap_splits
        
        filled_orders = []
        total_filled = 0.0
        total_cost = 0.0
        
        for i in range(self.twap_splits):
            logger.info(f"TWAP split {i+1}/{self.twap_splits}")
            
            # Execute split
            result = self._execute_single_order(
                symbol=symbol,
                side=side,
                amount=split_amount,
                order_type='limit',
                leverage=leverage,
                urgency='low',
                expected_price=self.exchange.fetch_ticker(symbol)['last']
            )
            
            if result['success']:
                filled_orders.append(result)
                filled_price = result['filled_price']
                total_filled += split_amount
                total_cost += split_amount * filled_price
            else:
                logger.error(f"TWAP split {i+1} failed: {result.get('error')}")
            
            # Wait between splits (except last one)
            if i < self.twap_splits - 1:
                time.sleep(self.twap_interval)
        
        # Calculate average fill price
        if total_filled > 0:
            avg_fill_price = total_cost / total_filled
        else:
            avg_fill_price = 0
        
        return {
            'success': len(filled_orders) > 0,
            'strategy': 'TWAP',
            'splits_executed': len(filled_orders),
            'total_splits': self.twap_splits,
            'total_filled': total_filled,
            'target_amount': amount,
            'avg_fill_price': avg_fill_price,
            'orders': filled_orders,
            'timestamp': datetime.now()
        }
    
    def _track_execution(self, result: Dict):
        """Track execution for monitoring."""
        self.execution_history.append(result)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        if not self.execution_history:
            return {}
        
        successful = [e for e in self.execution_history if e.get('success', False)]
        
        slippages = [e['slippage'] for e in successful if 'slippage' in e]
        
        return {
            'total_executions': len(self.execution_history),
            'successful': len(successful),
            'failed': len(self.execution_history) - len(successful),
            'success_rate': len(successful) / len(self.execution_history),
            'avg_slippage': np.mean(slippages) if slippages else 0,
            'max_slippage': max(slippages) if slippages else 0,
            'high_slippage_count': sum(1 for s in slippages if s > self.max_slippage)
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    print("Testing Smart Order Executor...")
    
    load_dotenv()
    
    # Initialize exchange and executor
    exchange = ExchangeInterface(
        exchange_id='binance',
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    
    executor = SmartOrderExecutor(
        exchange=exchange,
        max_slippage=0.002,
        use_twap=True
    )
    
    print("\n=== Test 1: Pre-execution Checks ===")
    checks_passed, checks = executor._pre_execution_checks(
        symbol='BTC/USDT',
        amount=0.001,
        leverage=3
    )
    print(f"Checks passed: {checks_passed}")
    for check_name, check_result in checks.items():
        status = "✅" if check_result.get('passed') else "❌"
        print(f"  {status} {check_name}: {check_result}")
    
    print("\n=== Test 2: Execution Stats ===")
    # Add some mock execution history
    executor.execution_history = [
        {'success': True, 'slippage': 0.0005},
        {'success': True, 'slippage': 0.0012},
        {'success': False},
        {'success': True, 'slippage': 0.0008}
    ]
    
    stats = executor.get_execution_stats()
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Slippage: {stats['avg_slippage']:.2%}")
    print(f"Max Slippage: {stats['max_slippage']:.2%}")
    
    print("\n✅ Order Executor test complete")
    print("\nNote: Actual order execution requires valid API keys and balance")

