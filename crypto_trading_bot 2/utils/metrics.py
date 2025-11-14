"""
Performance metrics calculation for the crypto trading bot.

This module provides:
- Comprehensive trading metrics (Sharpe, Sortino, Calmar, etc.)
- Risk metrics (VaR, CVaR, drawdown)
- Trade statistics
- Strategy performance analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats

from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            trades: DataFrame with trade history
            equity_curve: Time series of equity values
            
        Returns:
            Dictionary of metrics
        """
        if trades.empty:
            return self._empty_metrics()
        
        metrics = {
            **self.calculate_basic_metrics(trades),
            **self.calculate_risk_metrics(trades, equity_curve),
            **self.calculate_trade_statistics(trades),
            **self.calculate_time_metrics(trades)
        }
        
        return metrics
    
    def calculate_basic_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic trading metrics.
        
        Args:
            trades: DataFrame with trade history (must have 'pnl' column)
            
        Returns:
            Dictionary of basic metrics
        """
        if trades.empty:
            return {}
        
        # Total metrics
        total_pnl = trades['pnl'].sum()
        total_trades = len(trades)
        
        # Win/Loss analysis
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        win_rate = num_wins / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if num_losses > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if num_wins > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': winning_trades['pnl'].max() if num_wins > 0 else 0,
            'largest_loss': losing_trades['pnl'].min() if num_losses > 0 else 0,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def calculate_risk_metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted metrics.
        
        Args:
            trades: DataFrame with trade history
            equity_curve: Time series of equity
            
        Returns:
            Dictionary of risk metrics
        """
        if trades.empty:
            return {}
        
        # Calculate returns
        if equity_curve is not None and len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
        else:
            # Use trade PnL as proxy
            returns = trades['pnl_pct'] if 'pnl_pct' in trades.columns else trades['pnl']
            returns = pd.Series(returns)
        
        if returns.empty or len(returns) < 2:
            return self._empty_risk_metrics()
        
        # Sharpe Ratio (annualized)
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Assuming daily returns, annualize
        periods_per_year = 365
        sharpe_ratio = (
            (mean_return - self.risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)
            if std_return > 0 else 0
        )
        
        # Sortino Ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
        
        sortino_ratio = (
            (mean_return - self.risk_free_rate / periods_per_year) / downside_std * np.sqrt(periods_per_year)
            if downside_std > 0 else 0
        )
        
        # Maximum Drawdown
        if equity_curve is not None:
            cumulative = equity_curve
        else:
            cumulative = (1 + returns).cumprod()
        
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Average Drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Calmar Ratio (return / max drawdown)
        total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1 if len(cumulative) > 0 else 0
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = returns.quantile(0.05)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else 0
        
        # Volatility (annualized)
        volatility = std_return * np.sqrt(periods_per_year)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'volatility': volatility,
            'downside_deviation': downside_std * np.sqrt(periods_per_year),
            'total_return': total_return
        }
    
    def calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate detailed trade statistics.
        
        Args:
            trades: DataFrame with trade history
            
        Returns:
            Dictionary of trade statistics
        """
        if trades.empty:
            return {}
        
        # Consecutive wins/losses
        is_win = (trades['pnl'] > 0).astype(int)
        
        # Find consecutive sequences
        consecutive_wins = self._max_consecutive(is_win, 1)
        consecutive_losses = self._max_consecutive(is_win, 0)
        
        # Average trade duration
        if 'duration_minutes' in trades.columns:
            avg_duration = trades['duration_minutes'].mean()
            max_duration = trades['duration_minutes'].max()
            min_duration = trades['duration_minutes'].min()
        else:
            avg_duration = max_duration = min_duration = 0
        
        # Best/worst streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for win in is_win:
            if win:
                current_streak = current_streak + 1 if current_streak >= 0 else 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        return {
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'avg_trade_duration_minutes': avg_duration,
            'max_trade_duration_minutes': max_duration,
            'min_trade_duration_minutes': min_duration
        }
    
    def calculate_time_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate time-based metrics.
        
        Args:
            trades: DataFrame with trade history
            
        Returns:
            Dictionary of time-based metrics
        """
        if trades.empty or 'timestamp' not in trades.columns:
            return {}
        
        trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        
        # Trades per day/week/month
        days = (trades['timestamp'].max() - trades['timestamp'].min()).days + 1
        trades_per_day = len(trades) / days if days > 0 else 0
        
        # Profitable days
        if days > 0:
            daily_pnl = trades.set_index('timestamp')['pnl'].resample('D').sum()
            profitable_days = (daily_pnl > 0).sum()
            profitable_days_pct = profitable_days / len(daily_pnl) if len(daily_pnl) > 0 else 0
        else:
            profitable_days = 0
            profitable_days_pct = 0
        
        return {
            'total_days': days,
            'trades_per_day': trades_per_day,
            'profitable_days': profitable_days,
            'profitable_days_pct': profitable_days_pct
        }
    
    def calculate_strategy_comparison(
        self,
        trades: pd.DataFrame,
        strategy_column: str = 'strategy'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across strategies.
        
        Args:
            trades: DataFrame with trade history
            strategy_column: Column name for strategy
            
        Returns:
            Dictionary of metrics per strategy
        """
        if trades.empty or strategy_column not in trades.columns:
            return {}
        
        results = {}
        
        for strategy in trades[strategy_column].unique():
            strategy_trades = trades[trades[strategy_column] == strategy]
            results[strategy] = self.calculate_all_metrics(strategy_trades)
        
        return results
    
    def _max_consecutive(self, series: pd.Series, value: int) -> int:
        """
        Find maximum consecutive occurrences of a value.
        
        Args:
            series: Series to search
            value: Value to count
            
        Returns:
            Maximum consecutive count
        """
        max_count = 0
        current_count = 0
        
        for v in series:
            if v == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'total_pnl': 0,
            'total_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    def _empty_risk_metrics(self) -> Dict[str, float]:
        """Return empty risk metrics dictionary."""
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'avg_drawdown': 0,
            'var_95': 0,
            'cvar_95': 0,
            'volatility': 0,
            'downside_deviation': 0,
            'total_return': 0
        }


class RealTimeMetrics:
    """
    Calculate real-time metrics for monitoring.
    """
    
    def __init__(self, initial_balance: float = 10000):
        """
        Initialize real-time metrics tracker.
        
        Args:
            initial_balance: Starting balance
        """
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        
        self.trades_today = 0
        self.pnl_today = 0.0
        self.pnl_week = 0.0
        self.pnl_month = 0.0
        
        self.last_reset_day = datetime.now().date()
        self.last_reset_week = datetime.now().isocalendar()[1]
        self.last_reset_month = datetime.now().month
    
    def update_balance(self, new_balance: float):
        """
        Update current balance and calculate metrics.
        
        Args:
            new_balance: New balance value
        """
        self.current_balance = new_balance
        
        # Update peak
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
    
    def get_current_drawdown(self) -> float:
        """
        Get current drawdown from peak.
        
        Returns:
            Drawdown percentage (negative value)
        """
        if self.peak_balance == 0:
            return 0
        
        return (self.current_balance - self.peak_balance) / self.peak_balance
    
    def get_total_return(self) -> float:
        """
        Get total return percentage.
        
        Returns:
            Return percentage
        """
        if self.initial_balance == 0:
            return 0
        
        return (self.current_balance - self.initial_balance) / self.initial_balance
    
    def record_trade(self, pnl: float):
        """
        Record a trade and update period metrics.
        
        Args:
            pnl: Trade P&L
        """
        # Check if we need to reset periods
        today = datetime.now().date()
        current_week = datetime.now().isocalendar()[1]
        current_month = datetime.now().month
        
        if today != self.last_reset_day:
            self.trades_today = 0
            self.pnl_today = 0
            self.last_reset_day = today
        
        if current_week != self.last_reset_week:
            self.pnl_week = 0
            self.last_reset_week = current_week
        
        if current_month != self.last_reset_month:
            self.pnl_month = 0
            self.last_reset_month = current_month
        
        # Update counts
        self.trades_today += 1
        self.pnl_today += pnl
        self.pnl_week += pnl
        self.pnl_month += pnl
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': self.get_current_drawdown(),
            'total_return': self.get_total_return(),
            'trades_today': self.trades_today,
            'pnl_today': self.pnl_today,
            'pnl_week': self.pnl_week,
            'pnl_month': self.pnl_month
        }


if __name__ == "__main__":
    # Test metrics
    trades_data = {
        'pnl': [100, -50, 150, -30, 200, -40, 80],
        'pnl_pct': [0.02, -0.01, 0.03, -0.006, 0.04, -0.008, 0.016],
        'duration_minutes': [120, 90, 180, 60, 240, 75, 150],
        'timestamp': pd.date_range(start='2024-01-01', periods=7, freq='D'),
        'strategy': ['trend', 'reversion', 'trend', 'momentum', 'trend', 'reversion', 'momentum']
    }
    
    trades_df = pd.DataFrame(trades_data)
    
    metrics = PerformanceMetrics()
    results = metrics.calculate_all_metrics(trades_df)
    
    print("Performance Metrics:")
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    print("\n✅ Metrics test complete")

