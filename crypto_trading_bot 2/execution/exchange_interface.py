"""
Exchange Interface - CCXT Wrapper for Binance Futures.

This module:
- Wraps CCXT for unified exchange interaction
- Handles rate limiting
- Implements retry logic
- Provides error handling
- Manages API credentials
- Supports both REST and WebSocket
"""

import ccxt
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from functools import wraps

from utils.logger import get_logger

logger = get_logger(__name__)


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry failed API calls.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API call failed after {max_retries} attempts")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error in API call: {e}")
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator


class ExchangeInterface:
    """
    Unified interface for exchange interactions using CCXT.
    """
    
    def __init__(
        self,
        exchange_id: str = 'binance',
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        enable_rate_limit: bool = True
    ):
        """
        Initialize exchange interface.
        
        Args:
            exchange_id: Exchange identifier (default: binance)
            api_key: API key
            api_secret: API secret
            testnet: Use testnet/sandbox
            enable_rate_limit: Enable built-in rate limiting
        """
        self.exchange_id = exchange_id
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': enable_rate_limit,
            'options': {
                'defaultType': 'future',  # Use futures
                'adjustForTimeDifference': True
            }
        }
        
        if testnet:
            if exchange_id == 'binance':
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com',
                        'private': 'https://testnet.binancefuture.com'
                    }
                }
        
        self.exchange = exchange_class(config)
        
        # Load markets
        try:
            self.exchange.load_markets()
            logger.info(f"Exchange {exchange_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
        
        # API call counters for monitoring
        self.api_calls = {'total': 0, 'errors': 0}
        self.last_api_call = None
    
    @retry_on_error(max_retries=3, delay=1.0)
    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Create an order.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            order_type: 'market' or 'limit'
            side: 'buy' or 'sell'
            amount: Order amount in base currency
            price: Limit price (required for limit orders)
            params: Additional parameters (e.g., {'leverage': 3})
            
        Returns:
            Order dict
        """
        try:
            self._track_api_call()
            
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
            
            logger.info(
                f"Order created: {side.upper()} {amount} {symbol} "
                f"@ {price if price else 'MARKET'}"
            )
            
            return order
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            raise
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order: {e}")
            raise
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Order creation failed: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair
            
        Returns:
            Cancellation result
        """
        try:
            self._track_api_call()
            
            result = self.exchange.cancel_order(order_id, symbol)
            
            logger.info(f"Order {order_id} cancelled for {symbol}")
            
            return result
            
        except ccxt.OrderNotFound as e:
            logger.warning(f"Order not found: {e}")
            return None
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Order cancellation failed: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """
        Fetch order details.
        
        Args:
            order_id: Order ID
            symbol: Trading pair
            
        Returns:
            Order details
        """
        try:
            self._track_api_call()
            
            order = self.exchange.fetch_order(order_id, symbol)
            
            return order
            
        except ccxt.OrderNotFound as e:
            logger.warning(f"Order not found: {e}")
            return None
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Fetch order failed: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def fetch_balance(self) -> Dict:
        """
        Fetch account balance.
        
        Returns:
            Balance dict with 'total', 'free', 'used' for each currency
        """
        try:
            self._track_api_call()
            
            balance = self.exchange.fetch_balance()
            
            return balance
            
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Fetch balance failed: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch open positions.
        
        Args:
            symbols: List of symbols to fetch (None = all)
            
        Returns:
            List of position dicts
        """
        try:
            self._track_api_call()
            
            positions = self.exchange.fetch_positions(symbols)
            
            # Filter out zero positions
            open_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]
            
            return open_positions
            
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Fetch positions failed: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch ticker for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Ticker dict with bid, ask, last price, etc.
        """
        try:
            self._track_api_call()
            
            ticker = self.exchange.fetch_ticker(symbol)
            
            return ticker
            
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Fetch ticker failed for {symbol}: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Fetch order book.
        
        Args:
            symbol: Trading pair
            limit: Depth limit
            
        Returns:
            Order book with 'bids' and 'asks'
        """
        try:
            self._track_api_call()
            
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            
            return orderbook
            
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Fetch order book failed for {symbol}: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading pair
            leverage: Leverage value (1-125)
            
        Returns:
            Result dict
        """
        try:
            self._track_api_call()
            
            result = self.exchange.set_leverage(leverage, symbol)
            
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            
            return result
            
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.error(f"Set leverage failed: {e}")
            raise
    
    @retry_on_error(max_retries=3, delay=1.0)
    def set_margin_mode(self, symbol: str, margin_mode: str = 'isolated') -> Dict:
        """
        Set margin mode (isolated or cross).
        
        Args:
            symbol: Trading pair
            margin_mode: 'isolated' or 'cross'
            
        Returns:
            Result dict
        """
        try:
            self._track_api_call()
            
            result = self.exchange.set_margin_mode(margin_mode, symbol)
            
            logger.info(f"Margin mode set to {margin_mode} for {symbol}")
            
            return result
            
        except Exception as e:
            self.api_calls['errors'] += 1
            logger.warning(f"Set margin mode failed: {e}")
            return None
    
    def get_api_latency(self) -> float:
        """
        Measure API latency.
        
        Returns:
            Latency in milliseconds
        """
        try:
            start = time.time()
            self.exchange.fetch_time()
            latency = (time.time() - start) * 1000
            
            return latency
            
        except Exception as e:
            logger.error(f"Latency check failed: {e}")
            return -1
    
    def _track_api_call(self):
        """Track API call for monitoring."""
        self.api_calls['total'] += 1
        self.last_api_call = datetime.now()
    
    def get_api_stats(self) -> Dict:
        """Get API call statistics."""
        return {
            'total_calls': self.api_calls['total'],
            'total_errors': self.api_calls['errors'],
            'error_rate': self.api_calls['errors'] / max(self.api_calls['total'], 1),
            'last_call': self.last_api_call
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    print("Testing Exchange Interface...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize interface (testnet mode)
    interface = ExchangeInterface(
        exchange_id='binance',
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        testnet=True  # Use testnet for testing
    )
    
    # Test 1: Check latency
    print("\n=== Test 1: API Latency ===")
    latency = interface.get_api_latency()
    print(f"API Latency: {latency:.2f}ms")
    
    # Test 2: Fetch ticker
    print("\n=== Test 2: Fetch Ticker ===")
    try:
        ticker = interface.fetch_ticker('BTC/USDT')
        print(f"BTC/USDT Price: ${ticker['last']:,.2f}")
        print(f"Bid: ${ticker['bid']:,.2f}")
        print(f"Ask: ${ticker['ask']:,.2f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Fetch balance
    print("\n=== Test 3: Fetch Balance ===")
    try:
        balance = interface.fetch_balance()
        print(f"USDT Balance: {balance['USDT']['free']:.2f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Fetch order book
    print("\n=== Test 4: Fetch Order Book ===")
    try:
        orderbook = interface.fetch_order_book('BTC/USDT', limit=5)
        print("Top 5 Bids:")
        for price, amount in orderbook['bids'][:5]:
            print(f"  ${price:,.2f} x {amount:.4f}")
        print("Top 5 Asks:")
        for price, amount in orderbook['asks'][:5]:
            print(f"  ${price:,.2f} x {amount:.4f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: API Stats
    print("\n=== Test 5: API Statistics ===")
    stats = interface.get_api_stats()
    print(f"Total API Calls: {stats['total_calls']}")
    print(f"Total Errors: {stats['total_errors']}")
    print(f"Error Rate: {stats['error_rate']:.2%}")
    
    print("\n✅ Exchange Interface test complete")

