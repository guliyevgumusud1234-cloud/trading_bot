-- Database Initialization for Crypto Trading Bot
-- Creates all necessary tables with TimescaleDB hypertables

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV Data Table
CREATE TABLE IF NOT EXISTS ohlcv_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);

SELECT create_hypertable('ohlcv_data', 'time', if_not_exists => TRUE);

-- Feature Data Table
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    feature_name VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, feature_name)
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);

-- Trades Table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    size DOUBLE PRECISION NOT NULL,
    leverage INTEGER NOT NULL,
    pnl DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    holding_time_minutes INTEGER,
    exit_reason VARCHAR(50),
    status VARCHAR(20) DEFAULT 'open'
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_strategy ON trades(strategy);
CREATE INDEX idx_trades_symbol ON trades(symbol);

-- Positions Table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL,
    leverage INTEGER NOT NULL,
    unrealized_pnl DOUBLE PRECISION NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    last_update TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    status VARCHAR(20) DEFAULT 'open'
);

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    strategy VARCHAR(50),
    symbol VARCHAR(20),
    PRIMARY KEY (time, metric_name, strategy, symbol)
);

SELECT create_hypertable('performance_metrics', 'time', if_not_exists => TRUE);

-- Risk Events Table
CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    symbol VARCHAR(20),
    strategy VARCHAR(50),
    action_taken VARCHAR(100)
);

CREATE INDEX idx_risk_events_timestamp ON risk_events(timestamp DESC);
CREATE INDEX idx_risk_events_type ON risk_events(event_type);

-- System Logs Table
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    log_level VARCHAR(20) NOT NULL,
    component VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    error_details JSONB
);

CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX idx_system_logs_level ON system_logs(log_level);

-- Account Balance Table
CREATE TABLE IF NOT EXISTS account_balance (
    time TIMESTAMPTZ NOT NULL,
    balance DOUBLE PRECISION NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    margin_used DOUBLE PRECISION NOT NULL,
    margin_available DOUBLE PRECISION NOT NULL,
    unrealized_pnl DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time)
);

SELECT create_hypertable('account_balance', 'time', if_not_exists => TRUE);

-- Funding Payments Table
CREATE TABLE IF NOT EXISTS funding_payments (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    funding_rate DOUBLE PRECISION NOT NULL,
    payment_amount DOUBLE PRECISION NOT NULL,
    position_size DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_funding_payments_timestamp ON funding_payments(timestamp DESC);
CREATE INDEX idx_funding_payments_symbol ON funding_payments(symbol);

-- Model Performance Table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION
);

CREATE INDEX idx_model_performance_strategy ON model_performance(strategy);
CREATE INDEX idx_model_performance_timestamp ON model_performance(timestamp DESC);

-- Create views for common queries
CREATE OR REPLACE VIEW daily_performance AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as max_win,
    MIN(pnl) as max_loss
FROM trades
WHERE exit_time IS NOT NULL
GROUP BY DATE(timestamp)
ORDER BY date DESC;

CREATE OR REPLACE VIEW strategy_performance AS
SELECT
    strategy,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as total_wins,
    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as total_losses,
    CASE
        WHEN SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) > 0
        THEN SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) / SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END)
        ELSE 0
    END as profit_factor
FROM trades
WHERE exit_time IS NOT NULL
GROUP BY strategy;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;

-- Continuous aggregates for performance (TimescaleDB feature)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_metrics
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    AVG(close) as avg_price,
    MAX(high) as max_price,
    MIN(low) as min_price,
    SUM(volume) as total_volume
FROM ohlcv_data
WHERE timeframe = '15m'
GROUP BY bucket, symbol;

SELECT add_continuous_aggregate_policy('hourly_metrics',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Insert initial system log
INSERT INTO system_logs (log_level, component, message)
VALUES ('INFO', 'DATABASE', 'Database initialized successfully');

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Trading Bot Database Initialized Successfully!';
    RAISE NOTICE 'Tables created: ohlcv_data, features, trades, positions, performance_metrics, risk_events, system_logs, account_balance, funding_payments, model_performance';
    RAISE NOTICE 'Views created: daily_performance, strategy_performance';
    RAISE NOTICE 'Continuous aggregate: hourly_metrics';
END $$;

