"""
Logging configuration for the crypto trading bot.

This module sets up structured logging with:
- JSON formatting for easy parsing
- Log rotation for disk space management
- Multiple handlers (console, file, error file)
- Different log levels for different components
"""

import logging
import logging.handlers
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import traceback


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    
    Makes logs more readable in the terminal with color coding.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Colored log string
        """
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format: [TIMESTAMP] LEVEL - logger - message
        log_fmt = (
            f"{color}[%(asctime)s]{reset} "
            f"{color}%(levelname)-8s{reset} - "
            f"%(name)s - "
            f"%(message)s"
        )
        
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class TradingLogger:
    """
    Main logger class for the trading bot.
    
    Provides:
    - Multiple log files (main, trading, risk, execution, errors)
    - Console output with colors
    - JSON formatting for structured logging
    - Log rotation
    - Context managers for trading operations
    """
    
    def __init__(
        self,
        name: str = "trading_bot",
        log_dir: str = "./logs",
        log_level: str = "INFO",
        console_output: bool = True,
        json_format: bool = False
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output to console
            json_format: Whether to use JSON formatting
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.console_output = console_output
        self.json_format = json_format
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger with handlers and formatters.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Choose formatter
        if self.json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(
                ColoredFormatter() if not self.json_format else formatter
            )
            logger.addHandler(console_handler)
        
        # Main log file (rotating)
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trading.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10
        )
        main_handler.setLevel(self.log_level)
        main_handler.setFormatter(formatter)
        logger.addHandler(main_handler)
        
        # Error log file (errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            module_name: Name of the module (creates child logger)
            
        Returns:
            Logger instance
        """
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger
    
    def create_component_logger(self, component: str) -> logging.Logger:
        """
        Create a dedicated logger for a component with its own log file.
        
        Args:
            component: Component name (e.g., 'trading', 'risk', 'execution')
            
        Returns:
            Logger instance for the component
        """
        logger = logging.getLogger(f"{self.name}.{component}")
        logger.setLevel(self.log_level)
        
        # Component-specific log file
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{component}.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        handler.setLevel(self.log_level)
        
        formatter = JSONFormatter() if self.json_format else logging.Formatter(
            '[%(asctime)s] %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        return logger
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        leverage: float,
        strategy: str,
        reason: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Log a trade execution.
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            size: Position size
            leverage: Leverage used
            strategy: Strategy name
            reason: Reason for trade
            extra: Additional data
        """
        trade_data = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "size": size,
            "leverage": leverage,
            "strategy": strategy,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if extra:
            trade_data.update(extra)
        
        # Log to main logger
        self.logger.info(
            f"TRADE: {side} {symbol} @ {entry_price} | Size: {size} | "
            f"Leverage: {leverage}x | Strategy: {strategy}",
            extra={'extra_data': trade_data}
        )
        
        # Also log to dedicated trading log file
        trading_logger = self.create_component_logger("trading")
        trading_logger.info(json.dumps(trade_data))
    
    def log_position_close(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        pnl_pct: float,
        duration_minutes: int,
        reason: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Log a position close.
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            pnl: Profit/Loss in USDT
            pnl_pct: P&L percentage
            duration_minutes: Position duration
            reason: Reason for close
            extra: Additional data
        """
        close_data = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "duration_minutes": duration_minutes,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if extra:
            close_data.update(extra)
        
        level = logging.INFO if pnl >= 0 else logging.WARNING
        
        self.logger.log(
            level,
            f"POSITION CLOSED: {side} {symbol} | P&L: ${pnl:.2f} ({pnl_pct:.2%}) | "
            f"Duration: {duration_minutes}min | Reason: {reason}",
            extra={'extra_data': close_data}
        )
        
        trading_logger = self.create_component_logger("trading")
        trading_logger.log(level, json.dumps(close_data))
    
    def log_risk_violation(
        self,
        violation_type: str,
        current_value: float,
        limit_value: float,
        action_taken: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Log a risk limit violation.
        
        Args:
            violation_type: Type of violation
            current_value: Current value that triggered violation
            limit_value: The limit that was breached
            action_taken: Action taken in response
            extra: Additional data
        """
        risk_data = {
            "violation_type": violation_type,
            "current_value": current_value,
            "limit_value": limit_value,
            "action_taken": action_taken,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if extra:
            risk_data.update(extra)
        
        self.logger.warning(
            f"RISK VIOLATION: {violation_type} | "
            f"Current: {current_value} | Limit: {limit_value} | "
            f"Action: {action_taken}",
            extra={'extra_data': risk_data}
        )
        
        risk_logger = self.create_component_logger("risk")
        risk_logger.warning(json.dumps(risk_data))
    
    def log_performance(
        self,
        period: str,
        total_pnl: float,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
        total_trades: int,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Log performance metrics.
        
        Args:
            period: Period (e.g., 'daily', 'weekly')
            total_pnl: Total P&L
            win_rate: Win rate
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            total_trades: Number of trades
            extra: Additional metrics
        """
        perf_data = {
            "period": period,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if extra:
            perf_data.update(extra)
        
        self.logger.info(
            f"PERFORMANCE ({period}): P&L: ${total_pnl:.2f} | "
            f"Win Rate: {win_rate:.1%} | Sharpe: {sharpe_ratio:.2f} | "
            f"Max DD: {max_drawdown:.2%} | Trades: {total_trades}",
            extra={'extra_data': perf_data}
        )


# Global logger instance
_global_logger: Optional[TradingLogger] = None


def setup_logger(
    name: str = "trading_bot",
    log_dir: str = "./logs",
    log_level: str = "INFO",
    console_output: bool = True,
    json_format: bool = False
) -> TradingLogger:
    """
    Setup global logger instance.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        console_output: Whether to output to console
        json_format: Whether to use JSON formatting
        
    Returns:
        Logger instance
    """
    global _global_logger
    _global_logger = TradingLogger(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        console_output=console_output,
        json_format=json_format
    )
    return _global_logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        module_name: Module name for child logger
        
    Returns:
        Logger instance
        
    Raises:
        RuntimeError: If logger not initialized
    """
    global _global_logger
    if _global_logger is None:
        # Auto-initialize with defaults
        _global_logger = setup_logger()
    return _global_logger.get_logger(module_name)


# Convenience functions
def debug(msg: str, *args, **kwargs):
    """Log debug message."""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log info message."""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log warning message."""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log error message."""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log critical message."""
    get_logger().critical(msg, *args, **kwargs)


if __name__ == "__main__":
    # Test logger
    logger = setup_logger(log_level="DEBUG")
    
    test_logger = logger.get_logger("test")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    
    # Test trade logging
    logger.log_trade(
        symbol="BTC/USDT",
        side="LONG",
        entry_price=50000.0,
        size=0.1,
        leverage=3.0,
        strategy="trend_following",
        reason="EMA crossover + ADX > 25"
    )
    
    # Test position close logging
    logger.log_position_close(
        symbol="BTC/USDT",
        side="LONG",
        entry_price=50000.0,
        exit_price=51000.0,
        size=0.1,
        pnl=100.0,
        pnl_pct=0.02,
        duration_minutes=120,
        reason="Take profit hit"
    )
    
    print("\n✅ Logger test complete. Check ./logs/ directory for output.")

