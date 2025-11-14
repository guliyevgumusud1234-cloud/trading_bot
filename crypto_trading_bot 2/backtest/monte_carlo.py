"""
Monte Carlo Simulation.

This module:
- Shuffles trade sequences
- Simulates multiple possible outcomes
- Calculates probability distributions
- Identifies worst-case scenarios
- Estimates risk of ruin
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt

from utils.logger import get_logger

logger = get_logger(__name__)


def monte_carlo_simulation(
    trades: pd.DataFrame,
    initial_balance: float = 10000,
    n_simulations: int = 1000,
    plot: bool = False
) -> Dict:
    """
    Run Monte Carlo simulation on trade sequence.
    
    Args:
        trades: DataFrame with trade results (must have 'pnl' column)
        initial_balance: Starting capital
        n_simulations: Number of simulations
        plot: Whether to plot results
        
    Returns:
        Monte Carlo results dict
    """
    logger.info(f"Running Monte Carlo simulation with {n_simulations} iterations...")
    
    if 'pnl' not in trades.columns:
        logger.error("Trades DataFrame must have 'pnl' column")
        return {}
    
    if len(trades) < 10:
        logger.warning("Too few trades for meaningful Monte Carlo")
        return {}
    
    # Storage for simulation results
    final_balances = []
    max_drawdowns = []
    returns_list = []
    equity_curves = []
    
    for sim in range(n_simulations):
        # Shuffle trade order
        shuffled_trades = trades.sample(frac=1, replace=False).reset_index(drop=True)
        
        # Calculate equity curve
        equity = initial_balance
        equity_curve = [equity]
        peak_equity = equity
        max_dd = 0.0
        
        for trade_pnl in shuffled_trades['pnl']:
            equity += trade_pnl
            equity_curve.append(equity)
            
            # Update peak and drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            current_dd = (peak_equity - equity) / peak_equity
            if current_dd > max_dd:
                max_dd = current_dd
        
        # Store results
        final_balances.append(equity)
        max_drawdowns.append(max_dd)
        returns_list.append((equity / initial_balance) - 1)
        equity_curves.append(equity_curve)
    
    # Convert to arrays
    final_balances = np.array(final_balances)
    max_drawdowns = np.array(max_drawdowns)
    returns_array = np.array(returns_list)
    
    # Calculate statistics
    results = {
        # Final balance stats
        'mean_final_balance': float(np.mean(final_balances)),
        'median_final_balance': float(np.median(final_balances)),
        'std_final_balance': float(np.std(final_balances)),
        'min_final_balance': float(np.min(final_balances)),
        'max_final_balance': float(np.max(final_balances)),
        
        # Return stats
        'mean_return': float(np.mean(returns_array)),
        'median_return': float(np.median(returns_array)),
        'std_return': float(np.std(returns_array)),
        
        # Percentiles
        'return_5th_percentile': float(np.percentile(returns_array, 5)),
        'return_25th_percentile': float(np.percentile(returns_array, 25)),
        'return_75th_percentile': float(np.percentile(returns_array, 75)),
        'return_95th_percentile': float(np.percentile(returns_array, 95)),
        
        # Drawdown stats
        'mean_max_drawdown': float(np.mean(max_drawdowns)),
        'median_max_drawdown': float(np.median(max_drawdowns)),
        'worst_drawdown': float(np.max(max_drawdowns)),
        'best_drawdown': float(np.min(max_drawdowns)),
        
        # Probability metrics
        'prob_profit': float((returns_array > 0).sum() / len(returns_array)),
        'prob_loss': float((returns_array < 0).sum() / len(returns_array)),
        'prob_ruin': float((final_balances < initial_balance * 0.2).sum() / len(final_balances)),
        
        # Risk metrics
        'var_95': float(np.percentile(returns_array, 5)),  # Value at Risk
        'cvar_95': float(returns_array[returns_array <= np.percentile(returns_array, 5)].mean()),  # Conditional VaR
        
        # Additional info
        'n_simulations': n_simulations,
        'n_trades': len(trades),
        'initial_balance': initial_balance
    }
    
    logger.info("Monte Carlo simulation complete")
    logger.info(f"Mean return: {results['mean_return']:.2%}")
    logger.info(f"Probability of profit: {results['prob_profit']:.2%}")
    logger.info(f"Worst case (5th percentile): {results['return_5th_percentile']:.2%}")
    logger.info(f"Best case (95th percentile): {results['return_95th_percentile']:.2%}")
    
    # Plot if requested
    if plot:
        _plot_monte_carlo_results(results, equity_curves, returns_array, max_drawdowns)
    
    return results


def _plot_monte_carlo_results(
    results: Dict,
    equity_curves: List[List[float]],
    returns: np.ndarray,
    drawdowns: np.ndarray
):
    """Plot Monte Carlo simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Equity curves (sample)
    ax = axes[0, 0]
    sample_curves = equity_curves[:100]  # Plot first 100
    for curve in sample_curves:
        ax.plot(curve, alpha=0.1, color='blue')
    ax.set_title('Sample Equity Curves (100 simulations)')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Equity ($)')
    ax.grid(True, alpha=0.3)
    
    # 2. Return distribution
    ax = axes[0, 1]
    ax.hist(returns * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(results['mean_return'] * 100, color='red', linestyle='--', label=f"Mean: {results['mean_return']:.2%}")
    ax.axvline(results['return_5th_percentile'] * 100, color='orange', linestyle='--', label=f"5th %ile: {results['return_5th_percentile']:.2%}")
    ax.axvline(results['return_95th_percentile'] * 100, color='purple', linestyle='--', label=f"95th %ile: {results['return_95th_percentile']:.2%}")
    ax.set_title('Return Distribution')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Drawdown distribution
    ax = axes[1, 0]
    ax.hist(drawdowns * 100, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax.axvline(results['mean_max_drawdown'] * 100, color='blue', linestyle='--', label=f"Mean: {results['mean_max_drawdown']:.2%}")
    ax.set_title('Maximum Drawdown Distribution')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Monte Carlo Simulation Results
    {'='*40}
    
    Simulations: {results['n_simulations']:,}
    Trades: {results['n_trades']}
    
    Returns:
    Mean: {results['mean_return']:.2%}
    Median: {results['median_return']:.2%}
    Std Dev: {results['std_return']:.2%}
    
    Worst Case (5th %ile): {results['return_5th_percentile']:.2%}
    Best Case (95th %ile): {results['return_95th_percentile']:.2%}
    
    Drawdown:
    Mean Max DD: {results['mean_max_drawdown']:.2%}
    Worst DD: {results['worst_drawdown']:.2%}
    
    Probabilities:
    Profit: {results['prob_profit']:.2%}
    Loss: {results['prob_loss']:.2%}
    Ruin: {results['prob_ruin']:.2%}
    
    Risk Metrics:
    VaR (95%): {results['var_95']:.2%}
    CVaR (95%): {results['cvar_95']:.2%}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
    logger.info("Plot saved to monte_carlo_results.png")
    plt.show()


def analyze_trade_sequence_risk(
    trades: pd.DataFrame,
    initial_balance: float = 10000,
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Analyze risk characteristics of trade sequence.
    
    Args:
        trades: DataFrame with trade results
        initial_balance: Starting capital
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Risk analysis dict
    """
    if 'pnl' not in trades.columns:
        return {}
    
    # Calculate equity curve
    equity_curve = [initial_balance]
    for pnl in trades['pnl']:
        equity_curve.append(equity_curve[-1] + pnl)
    
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    
    # Drawdown analysis
    peak_equity = equity_series.expanding().max()
    drawdown = (equity_series - peak_equity) / peak_equity
    
    # Consecutive losses
    consecutive_losses = []
    current_streak = 0
    for pnl in trades['pnl']:
        if pnl < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                consecutive_losses.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        consecutive_losses.append(current_streak)
    
    # Analyze win/loss sequences
    wins = (trades['pnl'] > 0).astype(int)
    win_sequences = []
    loss_sequences = []
    
    current_win_streak = 0
    current_loss_streak = 0
    
    for win in wins:
        if win == 1:
            current_win_streak += 1
            if current_loss_streak > 0:
                loss_sequences.append(current_loss_streak)
                current_loss_streak = 0
        else:
            current_loss_streak += 1
            if current_win_streak > 0:
                win_sequences.append(current_win_streak)
                current_win_streak = 0
    
    analysis = {
        # Drawdown metrics
        'max_drawdown': float(drawdown.min()),
        'avg_drawdown': float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0,
        'drawdown_duration_max': int((drawdown < -0.05).sum()),  # Bars in >5% DD
        
        # Consecutive losses
        'max_consecutive_losses': int(max(consecutive_losses)) if consecutive_losses else 0,
        'avg_consecutive_losses': float(np.mean(consecutive_losses)) if consecutive_losses else 0.0,
        
        # Win/loss streaks
        'max_win_streak': int(max(win_sequences)) if win_sequences else 0,
        'max_loss_streak': int(max(loss_sequences)) if loss_sequences else 0,
        'avg_win_streak': float(np.mean(win_sequences)) if win_sequences else 0.0,
        'avg_loss_streak': float(np.mean(loss_sequences)) if loss_sequences else 0.0,
        
        # Return metrics
        'sharpe_ratio': float((returns.mean() - risk_free_rate) / returns.std()) if returns.std() > 0 else 0.0,
        'sortino_ratio': float((returns.mean() - risk_free_rate) / returns[returns < 0].std()) if (returns < 0).any() else 0.0,
        
        # Recovery metrics
        'avg_recovery_time': float(len(equity_curve) / len(consecutive_losses)) if consecutive_losses else 0.0
    }
    
    return analysis


if __name__ == "__main__":
    print("Testing Monte Carlo Simulation...")
    
    # Create synthetic trade data
    n_trades = 200
    
    # Simulate realistic trades (win rate ~50%, profit factor ~1.5)
    wins = np.random.uniform(50, 200, int(n_trades * 0.5))  # Winning trades
    losses = np.random.uniform(-100, -50, int(n_trades * 0.5))  # Losing trades
    
    pnl_values = np.concatenate([wins, losses])
    np.random.shuffle(pnl_values)
    
    trades = pd.DataFrame({
        'pnl': pnl_values,
        'timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='1h')
    })
    
    print(f"\nTest Data: {len(trades)} trades")
    print(f"Win Rate: {(trades['pnl'] > 0).sum() / len(trades):.2%}")
    print(f"Total P&L: ${trades['pnl'].sum():,.2f}")
    
    # Run Monte Carlo
    print("\n=== Running Monte Carlo Simulation ===")
    results = monte_carlo_simulation(
        trades=trades,
        initial_balance=10000,
        n_simulations=1000,
        plot=False  # Set to True to generate plots
    )
    
    print("\n=== Results ===")
    print(f"Mean Return: {results['mean_return']:.2%}")
    print(f"Std Dev: {results['std_return']:.2%}")
    print(f"\nWorst Case (5th percentile): {results['return_5th_percentile']:.2%}")
    print(f"Expected (50th percentile): {results['median_return']:.2%}")
    print(f"Best Case (95th percentile): {results['return_95th_percentile']:.2%}")
    print(f"\nProbability of Profit: {results['prob_profit']:.2%}")
    print(f"Probability of Ruin (<20% capital): {results['prob_ruin']:.2%}")
    print(f"\nMean Max Drawdown: {results['mean_max_drawdown']:.2%}")
    print(f"Worst Drawdown: {results['worst_drawdown']:.2%}")
    
    # Risk analysis
    print("\n=== Risk Analysis ===")
    risk_analysis = analyze_trade_sequence_risk(trades)
    print(f"Max Consecutive Losses: {risk_analysis['max_consecutive_losses']}")
    print(f"Max Loss Streak: {risk_analysis['max_loss_streak']}")
    print(f"Sharpe Ratio: {risk_analysis['sharpe_ratio']:.2f}")
    
    print("\n✅ Monte Carlo simulation test complete")

