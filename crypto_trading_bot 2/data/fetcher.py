"""
Data fetcher for the crypto trading bot.

This module handles:
- Historical OHLCV data fetching via CCXT
- Real-time WebSocket streams (klines, orderbook, trades, liquidations)
- Additional data sources (funding rates, open interest, liquidations, sentiment)
- Data validation and error handling
"""

import ccxt
import asyncio
import websockets
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from threading import Thread
import pandas as pd
import requests

from utils.logger import get_logger
from utils.database import Database

logger = get_logger(__name__)


class DataFetcher:
    """
    Main data fetcher for historical and real-time data.
    """
    
    def __init__(
        self,
        exchange_id: str = 'binance',
        testnet: bool = False
    ):
        """
        Initialize data fetcher.
        
        Args:
            exchange_id: Exchange ID (default: binance)
            testnet: Whether to use testnet
        """
        self.exchange_id = exchange_id
        self.testnet = testnet
        
        # Initialize CCXT exchange
        self.exchange = self._initialize_exchange()
        
        # WebSocket manager
        self.ws_manager = WebSocketManager(exchange_id=exchange_id)
        
        # Database
        self.db = Database()
        
        logger.info(f"DataFetcher initialized: {exchange_id} (testnet={testnet})")
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialize CCXT exchange.
        
        Returns:
            CCXT exchange instance
        """
        exchange_class = getattr(ccxt, self.exchange_id)
        
        config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # For futures trading
            }
        }
        
        if self.testnet:
            # Binance testnet configuration
            if self.exchange_id == 'binance':
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com',
                        'private': 'https://testnet.binancefuture.com'
                    }
                }
        
        exchange = exchange_class(config)
        
        # Load markets
        exchange.load_markets()
        
        logger.info(f"Exchange initialized: {len(exchange.markets)} markets loaded")
        
        return exchange
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '15m',
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '5m', '15m', '1h')
            since: Start time
            limit: Maximum number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert since to timestamp
            since_ms = int(since.timestamp() * 1000) if since else None
            
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit
            )
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def download_historical_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        save_to_db: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download historical data for multiple symbols and timeframes.
        
        Args:
            symbols: List of symbols
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date (default: now)
            save_to_db: Whether to save to database
            
        Returns:
            Dictionary of {symbol: {timeframe: DataFrame}}
        """
        if end_date is None:
            end_date = datetime.utcnow()
        
        all_data = {}
        
        for symbol in symbols:
            all_data[symbol] = {}
            
            for timeframe in timeframes:
                logger.info(f"Downloading {symbol} {timeframe}...")
                
                # Calculate timeframe in milliseconds
                tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
                
                # Fetch in chunks
                all_candles = []
                current_since = start_date
                
                while current_since < end_date:
                    df = self.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=1000
                    )
                    
                    if df.empty:
                        break
                    
                    all_candles.append(df)
                    
                    # Move to next chunk
                    last_timestamp = df.index[-1]
                    current_since = last_timestamp + timedelta(milliseconds=tf_ms)
                    
                    # Rate limiting
                    time.sleep(0.2)
                
                # Combine all chunks
                if all_candles:
                    combined_df = pd.concat(all_candles)
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df = combined_df.sort_index()
                    
                    all_data[symbol][timeframe] = combined_df
                    
                    logger.info(
                        f"Downloaded {len(combined_df)} candles for {symbol} {timeframe} "
                        f"({combined_df.index[0]} to {combined_df.index[-1]})"
                    )
                    
                    # Save to database
                    if save_to_db:
                        data_list = [
                            {
                                'timestamp': idx,
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'volume': row['volume']
                            }
                            for idx, row in combined_df.iterrows()
                        ]
                        self.db.insert_ohlcv(symbol, timeframe, data_list)
        
        logger.info("Historical data download complete")
        return all_data
    
    def fetch_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current funding rate.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Funding rate data
        """
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return {
                'symbol': symbol,
                'funding_rate': funding['fundingRate'],
                'funding_timestamp': pd.to_datetime(funding['fundingTimestamp'], unit='ms'),
                'mark_price': funding.get('markPrice'),
                'index_price': funding.get('indexPrice')
            }
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """
        Fetch open interest.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Open interest value
        """
        try:
            oi = self.exchange.fetch_open_interest(symbol)
            return float(oi['openInterest']) if oi else None
        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            return None
    
    def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        Fetch order book.
        
        Args:
            symbol: Trading symbol
            limit: Depth limit
            
        Returns:
            Order book data
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            return {
                'symbol': symbol,
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'timestamp': pd.to_datetime(orderbook['timestamp'], unit='ms')
            }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None
    
    def start_websocket_streams(self, symbols: List[str], timeframes: List[str]):
        """
        Start WebSocket streams for real-time data.
        
        Args:
            symbols: List of symbols to stream
            timeframes: List of timeframes to stream
        """
        self.ws_manager.start_streams(symbols, timeframes)
        logger.info("WebSocket streams started")
    
    def stop_websocket_streams(self):
        """Stop all WebSocket streams."""
        self.ws_manager.stop_streams()
        logger.info("WebSocket streams stopped")
    
    def get_latest_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Get latest data from WebSocket buffer.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Latest candle data
        """
        return self.ws_manager.get_latest_candle(symbol, timeframe)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time data.
    """
    
    def __init__(self, exchange_id: str = 'binance'):
        """
        Initialize WebSocket manager.
        
        Args:
            exchange_id: Exchange ID
        """
        self.exchange_id = exchange_id
        self.streams = {}
        self.data_buffer = {}
        self.running = False
        self.threads = []
        
        # WebSocket URL for Binance Futures
        self.ws_base_url = "wss://fstream.binance.com/ws"
    
    def start_streams(self, symbols: List[str], timeframes: List[str]):
        """
        Start WebSocket streams.
        
        Args:
            symbols: Symbols to stream
            timeframes: Timeframes to stream
        """
        self.running = True
        
        # Start kline streams
        for symbol in symbols:
            for timeframe in timeframes:
                thread = Thread(
                    target=self._run_kline_stream,
                    args=(symbol, timeframe),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
        
        logger.info(f"Started {len(self.threads)} WebSocket streams")
    
    def stop_streams(self):
        """Stop all streams."""
        self.running = False
        
        for thread in self.threads:
            thread.join(timeout=1)
        
        self.threads.clear()
    
    def _run_kline_stream(self, symbol: str, timeframe: str):
        """
        Run kline WebSocket stream.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        # Convert symbol format (BTC/USDT -> btcusdt)
        stream_symbol = symbol.replace('/', '').lower()
        
        # Convert timeframe (15m -> 15m)
        stream_tf = timeframe.lower()
        
        stream_name = f"{stream_symbol}@kline_{stream_tf}"
        url = f"{self.ws_base_url}/{stream_name}"
        
        asyncio.run(self._kline_stream_loop(url, symbol, timeframe))
    
    async def _kline_stream_loop(self, url: str, symbol: str, timeframe: str):
        """
        Async WebSocket loop for kline data.
        
        Args:
            url: WebSocket URL
            symbol: Trading symbol
            timeframe: Timeframe
        """
        reconnect_delay = 5
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"WebSocket connected: {symbol} {timeframe}")
                    
                    while self.running:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        if 'k' in data:
                            kline = data['k']
                            
                            # Store in buffer
                            key = f"{symbol}:{timeframe}"
                            self.data_buffer[key] = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                                'open': float(kline['o']),
                                'high': float(kline['h']),
                                'low': float(kline['l']),
                                'close': float(kline['c']),
                                'volume': float(kline['v']),
                                'is_closed': kline['x']
                            }
            
            except Exception as e:
                logger.error(f"WebSocket error for {symbol} {timeframe}: {e}")
                await asyncio.sleep(reconnect_delay)
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Get latest candle from buffer.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Latest candle data
        """
        key = f"{symbol}:{timeframe}"
        return self.data_buffer.get(key)


class ExternalDataFetcher:
    """
    Fetch data from external sources (liquidations, sentiment, etc.).
    """
    
    @staticmethod
    def fetch_fear_greed_index() -> Optional[int]:
        """
        Fetch Fear & Greed Index from alternative.me.
        
        Returns:
            Index value (0-100)
        """
        try:
            response = requests.get(
                "https://api.alternative.me/fng/",
                timeout=10
            )
            data = response.json()
            value = int(data['data'][0]['value'])
            logger.debug(f"Fear & Greed Index: {value}")
            return value
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return None
    
    @staticmethod
    def fetch_btc_dominance() -> Optional[float]:
        """
        Fetch Bitcoin dominance.
        
        Returns:
            BTC dominance percentage
        """
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10
            )
            data = response.json()
            dominance = data['data']['market_cap_percentage']['btc']
            logger.debug(f"BTC Dominance: {dominance}%")
            return float(dominance)
        except Exception as e:
            logger.error(f"Error fetching BTC dominance: {e}")
            return None
    
    @staticmethod
    def fetch_liquidations(
        symbol: str,
        api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch liquidation data from Coinglass.
        
        Args:
            symbol: Trading symbol
            api_key: Coinglass API key
            
        Returns:
            Liquidation data
        """
        if not api_key:
            logger.warning("Coinglass API key not provided")
            return None
        
        try:
            # Coinglass API endpoint
            headers = {
                'coinglassSecret': api_key
            }
            
            response = requests.get(
                f"https://open-api.coinglass.com/public/v2/liquidation",
                headers=headers,
                params={'symbol': symbol.replace('/', '')},
                timeout=10
            )
            
            data = response.json()
            
            if data.get('success'):
                return data['data']
            else:
                logger.error(f"Coinglass API error: {data.get('msg')}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching liquidations: {e}")
            return None


if __name__ == "__main__":
    # Test data fetcher
    fetcher = DataFetcher(testnet=False)
    
    # Test OHLCV fetch
    df = fetcher.fetch_ohlcv('BTC/USDT', '15m', limit=10)
    print("\nLatest 10 candles:")
    print(df.tail())
    
    # Test funding rate
    funding = fetcher.fetch_funding_rate('BTC/USDT')
    print(f"\nFunding Rate: {funding}")
    
    # Test Fear & Greed
    fgi = ExternalDataFetcher.fetch_fear_greed_index()
    print(f"\nFear & Greed Index: {fgi}")
    
    print("\n✅ Data fetcher test complete")

