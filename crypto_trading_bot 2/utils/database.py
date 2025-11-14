"""
Database interface for the crypto trading bot.

This module provides:
- TimescaleDB connection and management
- Table creation and schema management
- CRUD operations for all data types
- Redis cache integration
- Connection pooling
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import json

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor, execute_batch
import redis
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Database:
    """
    Main database interface for TimescaleDB.
    
    Handles:
    - Connection pooling
    - Table creation
    - CRUD operations
    - Query execution
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        pool_size: int = 10
    ):
        """
        Initialize database connection.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            pool_size: Connection pool size
        """
        # Get from environment if not provided
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', 5432))
        self.database = database or os.getenv('DB_NAME', 'trading_db')
        self.user = user or os.getenv('DB_USER', 'postgres')
        self.password = password or os.getenv('DB_PASSWORD', 'password')
        
        # Connection pool
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=pool_size,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"Database connection pool created: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
        
        # SQLAlchemy engine for ORM operations
        self.engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}",
            pool_size=pool_size,
            max_overflow=20,
            pool_pre_ping=True  # Verify connections before using
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.metadata = MetaData()
    
    @contextmanager
    def get_connection(self):
        """
        Get database connection from pool.
        
        Yields:
            Database connection
        """
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = False):
        """
        Get database cursor.
        
        Args:
            dict_cursor: Whether to use dictionary cursor
            
        Yields:
            Database cursor
        """
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                cursor.close()
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get SQLAlchemy session.
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all required tables and hypertables."""
        logger.info("Creating database tables...")
        
        with self.get_cursor() as cursor:
            # Enable TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # OHLCV table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    PRIMARY KEY (time, symbol, timeframe)
                );
            """)
            
            # Convert to hypertable
            cursor.execute("""
                SELECT create_hypertable('ohlcv', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    features JSONB,
                    PRIMARY KEY (time, symbol, timeframe)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('features', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    size DOUBLE PRECISION,
                    leverage DOUBLE PRECISION,
                    pnl DOUBLE PRECISION,
                    pnl_pct DOUBLE PRECISION,
                    duration_minutes INTEGER,
                    strategy TEXT,
                    entry_reason TEXT,
                    exit_reason TEXT,
                    fees DOUBLE PRECISION,
                    slippage DOUBLE PRECISION,
                    metadata JSONB
                );
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);")
            
            # Positions table (current open positions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL UNIQUE,
                    side TEXT NOT NULL,
                    entry_price DOUBLE PRECISION,
                    current_price DOUBLE PRECISION,
                    size DOUBLE PRECISION,
                    leverage DOUBLE PRECISION,
                    entry_time TIMESTAMPTZ,
                    unrealized_pnl DOUBLE PRECISION,
                    stop_loss DOUBLE PRECISION,
                    take_profit DOUBLE PRECISION,
                    strategy TEXT,
                    metadata JSONB,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    period TEXT NOT NULL,
                    total_pnl DOUBLE PRECISION,
                    win_rate DOUBLE PRECISION,
                    sharpe_ratio DOUBLE PRECISION,
                    sortino_ratio DOUBLE PRECISION,
                    max_drawdown DOUBLE PRECISION,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    avg_win DOUBLE PRECISION,
                    avg_loss DOUBLE PRECISION,
                    profit_factor DOUBLE PRECISION,
                    strategy TEXT,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, period, strategy)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('performance_metrics', 'timestamp',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Risk events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT,
                    current_value DOUBLE PRECISION,
                    limit_value DOUBLE PRECISION,
                    action_taken TEXT,
                    metadata JSONB
                );
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events(timestamp DESC);")
            
            # System logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    level TEXT,
                    component TEXT,
                    message TEXT,
                    metadata JSONB
                );
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp DESC);")
            
            # Funding rates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate DOUBLE PRECISION,
                    mark_price DOUBLE PRECISION,
                    index_price DOUBLE PRECISION,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('funding_rates', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Open interest table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS open_interest (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open_interest DOUBLE PRECISION,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            cursor.execute("""
                SELECT create_hypertable('open_interest', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            logger.info("✅ All tables created successfully")
    
    def insert_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]]
    ):
        """
        Insert OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '15m', '1h')
            data: List of OHLCV dictionaries
        """
        if not data:
            return
        
        with self.get_cursor() as cursor:
            insert_query = """
                INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume;
            """
            
            values = [
                (
                    row['timestamp'],
                    symbol,
                    timeframe,
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                )
                for row in data
            ]
            
            execute_batch(cursor, insert_query, values, page_size=1000)
            logger.debug(f"Inserted {len(values)} OHLCV records for {symbol} {timeframe}")
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            limit: Maximum number of records
            
        Returns:
            List of OHLCV dictionaries
        """
        with self.get_cursor(dict_cursor=True) as cursor:
            query = "SELECT * FROM ohlcv WHERE symbol = %s AND timeframe = %s"
            params = [symbol, timeframe]
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            query += " ORDER BY time DESC"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def insert_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Insert a trade record.
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            Trade ID
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, side, entry_price, exit_price,
                    size, leverage, pnl, pnl_pct, duration_minutes,
                    strategy, entry_reason, exit_reason, fees, slippage, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                trade_data.get('timestamp', datetime.utcnow()),
                trade_data['symbol'],
                trade_data['side'],
                trade_data['entry_price'],
                trade_data.get('exit_price'),
                trade_data['size'],
                trade_data['leverage'],
                trade_data.get('pnl'),
                trade_data.get('pnl_pct'),
                trade_data.get('duration_minutes'),
                trade_data['strategy'],
                trade_data.get('entry_reason'),
                trade_data.get('exit_reason'),
                trade_data.get('fees', 0),
                trade_data.get('slippage', 0),
                json.dumps(trade_data.get('metadata', {}))
            ))
            
            trade_id = cursor.fetchone()[0]
            logger.info(f"Trade recorded: ID {trade_id}")
            return trade_id
    
    def get_trades(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: Optional[int] = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade records.
        
        Args:
            start_time: Start time
            end_time: End time
            symbol: Filter by symbol
            strategy: Filter by strategy
            limit: Maximum records
            
        Returns:
            List of trades
        """
        with self.get_cursor(dict_cursor=True) as cursor:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time)
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if strategy:
                query += " AND strategy = %s"
                params.append(strategy)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def close(self):
        """Close database connections."""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connections closed")


class RedisCache:
    """
    Redis cache interface.
    
    Provides caching for frequently accessed data.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None
    ):
        """
        Initialize Redis connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.db = db
        self.password = password or os.getenv('REDIS_PASSWORD')
        
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.client.ping()
            logger.info(f"Redis connected: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.client = None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300
    ) -> bool:
        """
        Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        if not self.client:
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., 'ohlcv:*')
            
        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0


if __name__ == "__main__":
    # Test database
    db = Database()
    db.create_tables()
    
    # Test Redis
    cache = RedisCache()
    cache.set("test_key", {"data": "test_value"}, ttl=60)
    print(cache.get("test_key"))
    
    print("\n✅ Database test complete")


# Alias for backward compatibility
DatabaseManager = Database

