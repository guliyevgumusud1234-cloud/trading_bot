"""
Notification system for the crypto trading bot.

This module provides:
- Telegram bot integration for alerts
- Email notifications (optional)
- Alert management and rate limiting
- Formatted message templates
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from telegram import Bot
from telegram.error import TelegramError

from utils.logger import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "ℹ️"
    WARNING = "⚠️"
    CRITICAL = "🚨"
    SUCCESS = "✅"
    ERROR = "❌"


class TelegramNotifier:
    """
    Telegram notification handler.
    
    Sends alerts via Telegram bot.
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = enabled and self.bot_token and self.chat_id
        
        if self.enabled:
            try:
                self.bot = Bot(token=self.bot_token)
                logger.info("Telegram notifier initialized")
            except Exception as e:
                logger.error(f"Telegram initialization failed: {e}")
                self.enabled = False
        else:
            self.bot = None
            if not self.bot_token or not self.chat_id:
                logger.warning("Telegram credentials not provided")
        
        # Rate limiting
        self.last_sent = {}
        self.min_interval = timedelta(seconds=60)  # Min 1 min between same alerts
    
    async def send_message(
        self,
        message: str,
        parse_mode: str = 'Markdown',
        disable_notification: bool = False
    ) -> bool:
        """
        Send message via Telegram.
        
        Args:
            message: Message text
            parse_mode: Parse mode ('Markdown' or 'HTML')
            disable_notification: Silent notification
            
        Returns:
            Success status
        """
        if not self.enabled or not self.bot:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification
            )
            return True
        except TelegramError as e:
            logger.error(f"Telegram send error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram: {e}")
            return False
    
    def send_message_sync(self, message: str, **kwargs) -> bool:
        """
        Send message synchronously.
        
        Args:
            message: Message text
            **kwargs: Additional arguments for send_message
            
        Returns:
            Success status
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new task
                asyncio.create_task(self.send_message(message, **kwargs))
                return True
            else:
                return loop.run_until_complete(self.send_message(message, **kwargs))
        except Exception as e:
            logger.error(f"Sync send error: {e}")
            return False
    
    def should_send(self, alert_key: str) -> bool:
        """
        Check if alert should be sent (rate limiting).
        
        Args:
            alert_key: Unique key for this alert type
            
        Returns:
            Whether alert should be sent
        """
        now = datetime.now()
        
        if alert_key in self.last_sent:
            time_since_last = now - self.last_sent[alert_key]
            if time_since_last < self.min_interval:
                return False
        
        self.last_sent[alert_key] = now
        return True


class AlertManager:
    """
    Main alert management system.
    
    Handles alert formatting, routing, and delivery.
    """
    
    def __init__(
        self,
        telegram_enabled: bool = True,
        email_enabled: bool = False
    ):
        """
        Initialize alert manager.
        
        Args:
            telegram_enabled: Enable Telegram alerts
            email_enabled: Enable email alerts
        """
        self.telegram = TelegramNotifier(enabled=telegram_enabled)
        self.email_enabled = email_enabled
        
        if email_enabled:
            self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
            self.smtp_port = int(os.getenv('SMTP_PORT', 587))
            self.smtp_user = os.getenv('SMTP_USER')
            self.smtp_password = os.getenv('SMTP_PASSWORD')
            self.alert_email = os.getenv('ALERT_EMAIL')
    
    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None,
        rate_limit: bool = True
    ):
        """
        Send alert via all enabled channels.
        
        Args:
            level: Alert level
            title: Alert title
            message: Alert message
            extra_data: Additional data to include
            rate_limit: Whether to apply rate limiting
        """
        # Create alert key for rate limiting
        alert_key = f"{level.name}:{title}"
        
        if rate_limit and not self.telegram.should_send(alert_key):
            logger.debug(f"Alert rate limited: {alert_key}")
            return
        
        # Format message
        formatted_message = self._format_message(level, title, message, extra_data)
        
        # Send via Telegram
        if self.telegram.enabled:
            self.telegram.send_message_sync(formatted_message)
        
        # Send via Email (if critical)
        if self.email_enabled and level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
            self._send_email(title, formatted_message)
    
    def _format_message(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format alert message for Telegram.
        
        Args:
            level: Alert level
            title: Title
            message: Message
            extra_data: Extra data
            
        Returns:
            Formatted message
        """
        emoji = level.value
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        formatted = f"{emoji} *{title}*\n\n"
        formatted += f"{message}\n\n"
        
        if extra_data:
            formatted += "*Details:*\n"
            for key, value in extra_data.items():
                # Format key nicely
                key_formatted = key.replace('_', ' ').title()
                
                # Format value
                if isinstance(value, float):
                    if abs(value) < 0.01 or abs(value) > 1000:
                        value_formatted = f"{value:.6f}"
                    else:
                        value_formatted = f"{value:.2f}"
                else:
                    value_formatted = str(value)
                
                formatted += f"• {key_formatted}: `{value_formatted}`\n"
            formatted += "\n"
        
        formatted += f"🕐 {timestamp}"
        
        return formatted
    
    def _send_email(self, subject: str, body: str):
        """
        Send email alert.
        
        Args:
            subject: Email subject
            body: Email body
        """
        if not all([self.smtp_user, self.smtp_password, self.alert_email]):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.alert_email
            msg['Subject'] = f"[Trading Bot] {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent: {subject}")
        except Exception as e:
            logger.error(f"Email send error: {e}")
    
    # Convenience methods for common alerts
    
    def alert_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        leverage: float,
        strategy: str
    ):
        """Alert when trade is opened."""
        self.send_alert(
            level=AlertLevel.INFO,
            title="Trade Opened",
            message=f"{side} {symbol}",
            extra_data={
                'entry_price': entry_price,
                'size': size,
                'leverage': f"{leverage}x",
                'strategy': strategy
            }
        )
    
    def alert_trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        duration_minutes: int
    ):
        """Alert when trade is closed."""
        level = AlertLevel.SUCCESS if pnl >= 0 else AlertLevel.WARNING
        
        self.send_alert(
            level=level,
            title="Trade Closed",
            message=f"{side} {symbol}",
            extra_data={
                'pnl': f"${pnl:.2f}",
                'pnl_pct': f"{pnl_pct:.2%}",
                'duration': f"{duration_minutes} min"
            }
        )
    
    def alert_large_loss(
        self,
        symbol: str,
        loss_amount: float,
        loss_pct: float
    ):
        """Alert on large loss."""
        self.send_alert(
            level=AlertLevel.WARNING,
            title="Large Loss Detected",
            message=f"Position: {symbol}",
            extra_data={
                'loss': f"${loss_amount:.2f}",
                'loss_pct': f"{loss_pct:.2%}"
            },
            rate_limit=False  # Always send
        )
    
    def alert_risk_violation(
        self,
        violation_type: str,
        current_value: float,
        limit_value: float,
        action_taken: str
    ):
        """Alert on risk limit violation."""
        self.send_alert(
            level=AlertLevel.CRITICAL,
            title="Risk Violation",
            message=f"Type: {violation_type}",
            extra_data={
                'current': current_value,
                'limit': limit_value,
                'action': action_taken
            },
            rate_limit=False
        )
    
    def alert_drawdown_limit(
        self,
        current_dd: float,
        limit: float,
        action: str
    ):
        """Alert on drawdown limit."""
        self.send_alert(
            level=AlertLevel.CRITICAL,
            title="Drawdown Limit Reached",
            message="Trading paused!",
            extra_data={
                'current_drawdown': f"{current_dd:.2%}",
                'limit': f"{limit:.2%}",
                'action': action
            },
            rate_limit=False
        )
    
    def alert_system_error(
        self,
        error_type: str,
        error_message: str,
        component: str
    ):
        """Alert on system error."""
        self.send_alert(
            level=AlertLevel.ERROR,
            title="System Error",
            message=f"Component: {component}",
            extra_data={
                'error_type': error_type,
                'error_message': error_message[:200]  # Truncate
            },
            rate_limit=False
        )
    
    def alert_daily_summary(
        self,
        daily_pnl: float,
        trades: int,
        win_rate: float,
        sharpe: float,
        current_balance: float
    ):
        """Send daily performance summary."""
        level = AlertLevel.SUCCESS if daily_pnl >= 0 else AlertLevel.INFO
        
        self.send_alert(
            level=level,
            title="Daily Summary",
            message="Performance Report",
            extra_data={
                'pnl': f"${daily_pnl:.2f}",
                'trades': trades,
                'win_rate': f"{win_rate:.1%}",
                'sharpe': f"{sharpe:.2f}",
                'balance': f"${current_balance:.2f}"
            },
            rate_limit=False
        )
    
    def alert_strategy_performance(
        self,
        strategy: str,
        performance: Dict[str, Any]
    ):
        """Alert on strategy performance."""
        self.send_alert(
            level=AlertLevel.INFO,
            title=f"Strategy: {strategy}",
            message="Performance Update",
            extra_data=performance
        )
    
    def alert_high_volatility(
        self,
        symbol: str,
        current_atr: float,
        avg_atr: float
    ):
        """Alert on high volatility."""
        self.send_alert(
            level=AlertLevel.WARNING,
            title="High Volatility Detected",
            message=f"Symbol: {symbol}",
            extra_data={
                'current_atr': f"{current_atr:.2%}",
                'average_atr': f"{avg_atr:.2%}",
                'spike': f"{(current_atr/avg_atr):.1f}x"
            }
        )
    
    def alert_funding_extreme(
        self,
        symbol: str,
        funding_rate: float
    ):
        """Alert on extreme funding rate."""
        self.send_alert(
            level=AlertLevel.INFO,
            title="Extreme Funding Rate",
            message=f"Symbol: {symbol}",
            extra_data={
                'funding_rate': f"{funding_rate:.4%}",
                'annual_rate': f"{funding_rate * 365 * 3:.2%}"
            }
        )
    
    def alert_liquidation_risk(
        self,
        symbol: str,
        distance_pct: float,
        liquidation_price: float
    ):
        """Alert on liquidation risk."""
        self.send_alert(
            level=AlertLevel.CRITICAL,
            title="Liquidation Risk",
            message=f"Position: {symbol}",
            extra_data={
                'distance': f"{distance_pct:.2%}",
                'liquidation_price': liquidation_price
            },
            rate_limit=False
        )
    
    def alert_api_failure(
        self,
        exchange: str,
        error_message: str
    ):
        """Alert on API failure."""
        self.send_alert(
            level=AlertLevel.ERROR,
            title="API Failure",
            message=f"Exchange: {exchange}",
            extra_data={
                'error': error_message
            },
            rate_limit=False
        )
    
    def alert_emergency_stop(
        self,
        reason: str,
        positions_closed: int
    ):
        """Alert on emergency stop."""
        self.send_alert(
            level=AlertLevel.CRITICAL,
            title="🚨 EMERGENCY STOP",
            message="All trading paused!",
            extra_data={
                'reason': reason,
                'positions_closed': positions_closed,
                'status': 'Manual restart required'
            },
            rate_limit=False
        )


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def setup_alerts(
    telegram_enabled: bool = True,
    email_enabled: bool = False
) -> AlertManager:
    """
    Setup global alert manager.
    
    Args:
        telegram_enabled: Enable Telegram
        email_enabled: Enable email
        
    Returns:
        AlertManager instance
    """
    global _global_alert_manager
    _global_alert_manager = AlertManager(
        telegram_enabled=telegram_enabled,
        email_enabled=email_enabled
    )
    return _global_alert_manager


def get_alert_manager() -> AlertManager:
    """
    Get global alert manager.
    
    Returns:
        AlertManager instance
    """
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = setup_alerts()
    return _global_alert_manager


if __name__ == "__main__":
    # Test alerts
    print("Testing alert system...")
    print("Make sure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env")
    
    alerts = setup_alerts()
    
    # Test different alert types
    alerts.alert_trade_opened(
        symbol="BTC/USDT",
        side="LONG",
        entry_price=50000,
        size=0.1,
        leverage=3,
        strategy="trend_following"
    )
    
    alerts.alert_daily_summary(
        daily_pnl=150.50,
        trades=5,
        win_rate=0.6,
        sharpe=1.8,
        current_balance=10150.50
    )
    
    print("\n✅ Alerts sent (check Telegram)")


# Alias for backward compatibility
NotificationManager = AlertManager

