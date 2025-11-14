"""
Realistic Backtesting Engine.

This module:
- Bar-by-bar simulation
- Realistic cost modeling (slippage, fees, funding)
- Position simulation with margin calls
- Equity curve tracking
- Comprehensive performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from utils.logger import get_logger
from utils.metrics import PerformanceMetrics

logger = get_logger(__name__)


class Position:
    """Represents a trading position."""
    
    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        leverage: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = 'unknown'
    ):
        self.symbol = symbol
        self.side = side  # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.size = size
        self.leverage = leverage
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy = strategy
        
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.holding_time = 0
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L."""
        self.current_price = current_price
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if self.side == 'LONG':
            self.unrealized_pnl = self.size * price_change * self.leverage
        else:  # SHORT
            self.unrealized_pnl = self.size * (-price_change) * self.leverage
    
    def check_exit(self, current_price: float) -> Tuple[bool, Optional[str]]:
        """Check if position should be exited."""
        # Stop loss check
        if self.stop_loss:
            if self.side == 'LONG' and current_price <= self.stop_loss:
                return True, 'stop_loss'
            elif self.side == 'SHORT' and current_price >= self.stop_loss:
                return True, 'stop_loss'
        
        # Take profit check
        if self.take_profit:
            if self.side == 'LONG' and current_price >= self.take_profit:
                return True, 'take_profit'
            elif self.side == 'SHORT' and current_price <= self.take_profit:
                return True, 'take_profit'
        
        return False, None


class RealisticBacktester:
    """
    Realistic backtesting engine with proper cost modeling.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000,
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0004,  # 0.04%
        slippage_model: str = 'volume_based'
    ):
        """
        Initialize backtester.
        
        Args:
            data: OHLCV data with features
            initial_balance: Starting capital
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            slippage_model: 'fixed' or 'volume_based'
        """
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_model = slippage_model
        
        # State
        self.reset()
        
        logger.info(f"Backtester initialized with {len(data)} bars")
    
    def reset(self):
        """Reset backtest state."""
        self.balance = self.initial_balance
        self.equity_curve = [self.balance]
        self.equity_timestamps = [self.data.index[0] if hasattr(self.data, 'index') else 0]
        
        self.open_positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        
        self.current_step = 0
        self.peak_balance = self.balance
        self.max_drawdown = 0.0
    
    def run_backtest(
        self,
        strategy,
        use_maker_orders: bool = True
    ) -> Dict:
        """
        Run backtest.
        
        Args:
            strategy: Strategy object with generate_signal() method
            use_maker_orders: Use maker fees (True) or taker fees (False)
            
        Returns:
            Performance metrics dict
        """
        logger.info("Starting backtest...")
        
        self.reset()
        
        for i in range(len(self.data)):
            self.current_step = i
            current_bar = self.data.iloc[i]
            
            # Get current price
            current_price = current_bar['close']
            timestamp = current_bar.get('timestamp', datetime.now())
            
            # 1. Update open positions
            for position in self.open_positions[:]:
                position.update_pnl(current_price)
                position.holding_time += 1
                
                # Check exit conditions
                should_exit, reason = position.check_exit(current_price)
                if should_exit:
                    self._close_position(position, current_price, timestamp, reason)
            
            # 2. Generate new signal
            if i > 50:  # Need historical data for features
                try:
                    signal, confidence, trade_params = strategy.generate_signal(
                        features=self.data.iloc[:i+1],
                        current_price=current_price
                    )
                    
                    # Execute signal if confidence is high enough
                    if signal in ['LONG', 'SHORT'] and confidence > 0.6:
                        self._open_position(
                            signal=signal,
                            price=current_price,
                            timestamp=timestamp,
                            trade_params=trade_params,
                            use_maker_orders=use_maker_orders
                        )
                        
                except Exception as e:
                    logger.error(f"Error generating signal at step {i}: {e}")
            
            # 3. Update equity
            total_equity = self.balance + sum(p.unrealized_pnl for p in self.open_positions)
            self.equity_curve.append(total_equity)
            self.equity_timestamps.append(timestamp)
            
            # 4. Check margin call
            if total_equity < self.initial_balance * 0.2:
                logger.warning(f"Margin call at step {i}!")
                self._liquidate_all_positions(current_price, timestamp)
                break
            
            # 5. Update peak and drawdown
            if total_equity > self.peak_balance:
                self.peak_balance = total_equity
            
            current_dd = (self.peak_balance - total_equity) / self.peak_balance
            if current_dd > self.max_drawdown:
                self.max_drawdown = current_dd
        
        # Close any remaining positions
        if self.open_positions:
            final_price = self.data.iloc[-1]['close']
            final_timestamp = self.data.iloc[-1].get('timestamp', datetime.now())
            for position in self.open_positions[:]:
                self._close_position(position, final_price, final_timestamp, 'backtest_end')
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Backtest complete: {len(self.closed_trades)} trades")
        
        return metrics
    
    def _open_position(
        self,
        signal: str,
        price: float,
        timestamp: datetime,
        trade_params: Optional[Dict],
        use_maker_orders: bool
    ):
        """Open a new position."""
        # Position sizing (2% of balance)
        position_size = self.balance * 0.02
        
        # Get leverage from trade params
        leverage = trade_params.get('max_leverage', 2) if trade_params else 2
        leverage = min(leverage, 5)  # Cap at 5x
        
        # Apply entry cost
        fee_rate = self.maker_fee if use_maker_orders else self.taker_fee
        slippage = self._calculate_slippage(position_size, price)
        
        entry_price = price * (1 + slippage) if signal == 'LONG' else price * (1 - slippage)
        entry_cost = position_size * (fee_rate + slippage)
        
        # Check if we have enough balance
        required_margin = position_size / leverage
        if required_margin + entry_cost > self.balance * 0.95:
            logger.debug("Insufficient balance for position")
            return
        
        # Deduct cost
        self.balance -= entry_cost
        
        # Get stop loss and take profit from trade params
        stop_loss = trade_params.get('stop_loss') if trade_params else None
        take_profit = trade_params.get('take_profit') if trade_params else None
        strategy_name = trade_params.get('strategy', 'unknown') if trade_params else 'unknown'
        
        # Create position
        position = Position(
            symbol='BACKTEST',
            side=signal,
            entry_price=entry_price,
            size=position_size,
            leverage=leverage,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy_name
        )
        
        self.open_positions.append(position)
        
        logger.debug(
            f"Opened {signal} position: ${position_size:.2f} @ ${entry_price:.2f} "
            f"({leverage}x leverage)"
        )
    
    def _close_position(
        self,
        position: Position,
        price: float,
        timestamp: datetime,
        reason: str
    ):
        """Close a position."""
        # Update final P&L
        position.update_pnl(price)
        
        # Apply exit cost
        slippage = self._calculate_slippage(position.size, price)
        exit_price = price * (1 - slippage) if position.side == 'LONG' else price * (1 + slippage)
        
        # Recalculate P&L with actual exit price
        price_change = (exit_price - position.entry_price) / position.entry_price
        if position.side == 'LONG':
            realized_pnl = position.size * price_change * position.leverage
        else:
            realized_pnl = position.size * (-price_change) * position.leverage
        
        # Subtract exit fee
        exit_cost = position.size * self.taker_fee
        realized_pnl -= exit_cost
        
        # Calculate funding cost (simplified: 0.01% per 8h)
        hours_held = position.holding_time / 4  # Assuming 15min bars
        funding_payments = int(hours_held / 8)
        funding_cost = position.size * 0.0001 * funding_payments
        realized_pnl -= funding_cost
        
        # Update balance
        self.balance += realized_pnl
        
        # Record trade
        trade = {
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'leverage': position.leverage,
            'pnl': realized_pnl,
            'pnl_pct': realized_pnl / position.size,
            'holding_time': position.holding_time,
            'exit_reason': reason,
            'strategy': position.strategy
        }
        
        self.closed_trades.append(trade)
        self.open_positions.remove(position)
        
        logger.debug(
            f"Closed {position.side} position: P&L=${realized_pnl:.2f} "
            f"({reason})"
        )
    
    def _liquidate_all_positions(self, price: float, timestamp: datetime):
        """Liquidate all positions (margin call)."""
        logger.critical("LIQUIDATING ALL POSITIONS!")
        
        for position in self.open_positions[:]:
            self._close_position(position, price, timestamp, 'liquidation')
    
    def _calculate_slippage(self, order_size: float, price: float) -> float:
        """Calculate slippage based on order size."""
        if self.slippage_model == 'fixed':
            return 0.0002  # 0.02%
        
        elif self.slippage_model == 'volume_based':
            # Get current bar volume
            if self.current_step < len(self.data):
                bar_volume = self.data.iloc[self.current_step].get('volume', 1000000)
            else:
                bar_volume = 1000000
            
            # Slippage increases with order size relative to volume
            volume_impact = (order_size / price) / bar_volume
            
            base_slippage = 0.0002
            impact_slippage = volume_impact * 0.001
            
            total_slippage = base_slippage + impact_slippage
            
            return min(total_slippage, 0.005)  # Cap at 0.5%
        
        return 0.0
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'final_balance': self.balance,
                'total_return': 0.0
            }
        
        trades_df = pd.DataFrame(self.closed_trades)
        equity_series = pd.Series(self.equity_curve)
        
        # Calculate metrics using utility
        calculator = PerformanceMetrics()
        
        returns = equity_series.pct_change().dropna()
        
        metrics = {
            # Overview
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': (self.balance / self.initial_balance) - 1,
            'total_trades': len(trades_df),
            
            # Risk-adjusted returns
            'sharpe_ratio': calculator.sharpe_ratio(returns, periods=365*24*4),
            'sortino_ratio': calculator.sortino_ratio(returns, periods=365*24*4),
            'calmar_ratio': calculator.calmar_ratio(returns),
            
            # Drawdown
            'max_drawdown': self.max_drawdown,
            
            # Trade statistics
            'win_rate': (trades_df['pnl'] > 0).sum() / len(trades_df),
            'profit_factor': calculator.profit_factor(trades_df['pnl'].values),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean(),
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_holding_time': trades_df['holding_time'].mean(),
            
            # Strategy breakdown
            'trades_by_strategy': trades_df.groupby('strategy').size().to_dict(),
            'pnl_by_strategy': trades_df.groupby('strategy')['pnl'].sum().to_dict()
        }
        
        return metrics
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame({
            'timestamp': self.equity_timestamps,
            'equity': self.equity_curve
        })
    
    def get_trades(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        return pd.DataFrame(self.closed_trades)


if __name__ == "__main__":
    print("Testing Realistic Backtester...")
    
    # Create synthetic data
    n_bars = 10000
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')
    
    # Generate realistic price data
    returns = np.random.randn(n_bars) * 0.002
    prices = 50000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars),
        'rsi_14': np.random.uniform(30, 70, n_bars),
        'adx_14': np.random.uniform(15, 40, n_bars)
    })
    
    # Add more features
    for i in range(20):
        data[f'feature_{i}'] = np.random.randn(n_bars)
    
    # Initialize backtester
    backtester = RealisticBacktester(
        data=data,
        initial_balance=10000,
        slippage_model='volume_based'
    )
    
    # Simple mock strategy
    class MockStrategy:
        def generate_signal(self, features, current_price):
            # Random signals for testing
            if np.random.random() > 0.95:
                signal = np.random.choice(['LONG', 'SHORT'])
                return signal, 0.8, {
                    'stop_loss': current_price * 0.98,
                    'take_profit': current_price * 1.04,
                    'max_leverage': 3,
                    'strategy': 'mock'
                }
            return 'NEUTRAL', 0.0, None
    
    strategy = MockStrategy()
    
    # Run backtest
    print("\nRunning backtest...")
    metrics = backtester.run_backtest(strategy)
    
    print("\n=== Backtest Results ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Final Balance: ${metrics['final_balance']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    print("\n✅ Backtester test complete")

