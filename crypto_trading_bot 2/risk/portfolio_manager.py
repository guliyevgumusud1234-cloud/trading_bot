"""
Portfolio risk management for the crypto trading bot.

This module handles:
- Correlation management
- Portfolio leverage calculation
- Diversification scoring
- Exposure management
- Position concentration limits
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """
    Manage portfolio-level risk.
    """
    
    def __init__(self, config_path: str = 'config/risk_limits.yaml'):
        """
        Initialize portfolio manager.
        
        Args:
            config_path: Path to risk configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.max_portfolio_leverage = self.config['leverage_limits']['max_portfolio_leverage']
        self.max_correlated_positions = self.config['correlation_management']['max_highly_correlated_positions']
        self.correlation_threshold = self.config['correlation_management']['high_correlation_threshold']
    
    def calculate_portfolio_leverage(self, positions: List[Dict]) -> float:
        """
        Calculate total portfolio leverage.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Total portfolio leverage
        """
        if not positions:
            return 0.0
        
        total_notional = sum(
            pos['size'] * pos.get('leverage', 1) * pos.get('current_price', pos['entry_price'])
            for pos in positions
        )
        
        total_margin = sum(pos['size'] * pos.get('current_price', pos['entry_price']) for pos in positions)
        
        if total_margin == 0:
            return 0.0
        
        portfolio_leverage = total_notional / total_margin
        
        logger.debug(f"Portfolio leverage: {portfolio_leverage:.2f}x")
        
        return portfolio_leverage
    
    def check_correlation(
        self,
        positions: List[Dict],
        returns_data: pd.DataFrame,
        lookback: int = 100
    ) -> Dict[str, any]:
        """
        Check correlation between positions.
        
        Args:
            positions: List of positions
            returns_data: DataFrame with returns for each symbol
            lookback: Lookback period for correlation
            
        Returns:
            Dictionary with correlation analysis
        """
        if len(positions) < 2:
            return {
                'high_correlation_pairs': [],
                'max_correlation': 0,
                'avg_correlation': 0
            }
        
        # Get symbols
        symbols = [pos['symbol'] for pos in positions]
        
        # Calculate correlation matrix
        corr_matrix = returns_data[symbols].tail(lookback).corr()
        
        # Find high correlation pairs
        high_corr_pairs = []
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], start=i+1):
                correlation = abs(corr_matrix.loc[sym1, sym2])
                
                if correlation > self.correlation_threshold:
                    high_corr_pairs.append({
                        'symbol1': sym1,
                        'symbol2': sym2,
                        'correlation': correlation
                    })
        
        # Calculate average correlation
        corr_values = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr_values.append(abs(corr_matrix.iloc[i, j]))
        
        avg_correlation = np.mean(corr_values) if corr_values else 0
        max_correlation = max(corr_values) if corr_values else 0
        
        result = {
            'high_correlation_pairs': high_corr_pairs,
            'max_correlation': max_correlation,
            'avg_correlation': avg_correlation,
            'correlation_matrix': corr_matrix
        }
        
        if high_corr_pairs:
            logger.warning(
                f"Found {len(high_corr_pairs)} high correlation pairs "
                f"(threshold: {self.correlation_threshold})"
            )
        
        return result
    
    def check_correlation_limit(
        self,
        positions: List[Dict],
        returns_data: pd.DataFrame
    ) -> bool:
        """
        Check if correlation limit is breached.
        
        Args:
            positions: List of positions
            returns_data: Returns data
            
        Returns:
            Whether limit is breached
        """
        corr_analysis = self.check_correlation(positions, returns_data)
        
        num_high_corr = len(corr_analysis['high_correlation_pairs'])
        
        if num_high_corr > self.max_correlated_positions:
            logger.warning(
                f"Correlation limit breached: {num_high_corr} highly correlated pairs "
                f"(max: {self.max_correlated_positions})"
            )
            return True
        
        return False
    
    def reduce_correlated_positions(
        self,
        positions: List[Dict],
        correlation_pairs: List[Dict],
        reduction_factor: float = 0.3
    ) -> List[Dict]:
        """
        Reduce size of correlated positions.
        
        Args:
            positions: List of positions
            correlation_pairs: List of high correlation pairs
            reduction_factor: Percentage to reduce (0.3 = 30%)
            
        Returns:
            List of positions to reduce
        """
        positions_to_reduce = []
        
        # Get unique symbols from correlated pairs
        correlated_symbols = set()
        for pair in correlation_pairs:
            correlated_symbols.add(pair['symbol1'])
            correlated_symbols.add(pair['symbol2'])
        
        # Reduce positions in correlated symbols
        for pos in positions:
            if pos['symbol'] in correlated_symbols:
                new_size = pos['size'] * (1 - reduction_factor)
                positions_to_reduce.append({
                    'symbol': pos['symbol'],
                    'old_size': pos['size'],
                    'new_size': new_size,
                    'reduction': pos['size'] - new_size
                })
        
        logger.info(
            f"Reducing {len(positions_to_reduce)} correlated positions "
            f"by {reduction_factor:.1%}"
        )
        
        return positions_to_reduce
    
    def calculate_diversification_score(
        self,
        positions: List[Dict],
        returns_data: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio diversification score.
        
        Higher score = better diversification
        
        Args:
            positions: List of positions
            returns_data: Returns data
            
        Returns:
            Diversification score (0-1)
        """
        if len(positions) < 2:
            return 0.0
        
        # Get correlation analysis
        corr_analysis = self.check_correlation(positions, returns_data)
        avg_corr = corr_analysis['avg_correlation']
        
        # Score based on average correlation (lower is better)
        # Score = 1 - avg_correlation
        diversification_score = 1 - avg_corr
        
        # Adjust for number of positions
        n_positions = len(positions)
        position_factor = min(n_positions / 8, 1.0)  # Optimal at 8 positions
        
        final_score = diversification_score * position_factor
        
        logger.debug(f"Diversification score: {final_score:.2f}")
        
        return final_score
    
    def check_concentration_limits(
        self,
        positions: List[Dict],
        total_balance: float
    ) -> Dict[str, bool]:
        """
        Check concentration limits.
        
        Args:
            positions: List of positions
            total_balance: Total account balance
            
        Returns:
            Dictionary of limit violations
        """
        violations = {
            'symbol_concentration': False,
            'strategy_concentration': False,
            'sector_concentration': False,
            'net_exposure': False
        }
        
        if not positions:
            return violations
        
        # Check per-symbol concentration
        symbol_exposure = {}
        for pos in positions:
            symbol = pos['symbol']
            exposure = pos['size'] * pos.get('current_price', pos['entry_price'])
            symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + exposure
        
        max_symbol_pct = self.config['concentration_limits']['max_exposure_per_symbol_pct']
        for symbol, exposure in symbol_exposure.items():
            if exposure / total_balance > max_symbol_pct:
                violations['symbol_concentration'] = True
                logger.warning(
                    f"Symbol concentration limit breached: {symbol} "
                    f"{exposure/total_balance:.1%} (max: {max_symbol_pct:.1%})"
                )
        
        # Check per-strategy concentration
        strategy_exposure = {}
        for pos in positions:
            strategy = pos.get('strategy', 'unknown')
            exposure = pos['size'] * pos.get('current_price', pos['entry_price'])
            strategy_exposure[strategy] = strategy_exposure.get(strategy, 0) + exposure
        
        max_strategy_pct = self.config['concentration_limits']['max_exposure_per_strategy_pct']
        for strategy, exposure in strategy_exposure.items():
            if exposure / total_balance > max_strategy_pct:
                violations['strategy_concentration'] = True
                logger.warning(
                    f"Strategy concentration limit breached: {strategy} "
                    f"{exposure/total_balance:.1%} (max: {max_strategy_pct:.1%})"
                )
        
        # Check net exposure (long vs short)
        long_exposure = sum(
            pos['size'] * pos.get('current_price', pos['entry_price'])
            for pos in positions if pos['side'] == 'LONG'
        )
        short_exposure = sum(
            pos['size'] * pos.get('current_price', pos['entry_price'])
            for pos in positions if pos['side'] == 'SHORT'
        )
        
        net_exposure = abs(long_exposure - short_exposure) / total_balance
        max_net_exposure = self.config['concentration_limits']['max_net_exposure_pct']
        
        if net_exposure > max_net_exposure:
            violations['net_exposure'] = True
            logger.warning(
                f"Net exposure limit breached: {net_exposure:.1%} "
                f"(max: {max_net_exposure:.1%})"
            )
        
        return violations
    
    def calculate_position_weights(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate weight of each position in portfolio.
        
        Args:
            positions: List of positions
            
        Returns:
            Dictionary of {symbol: weight}
        """
        if not positions:
            return {}
        
        total_value = sum(
            pos['size'] * pos.get('current_price', pos['entry_price'])
            for pos in positions
        )
        
        weights = {}
        for pos in positions:
            value = pos['size'] * pos.get('current_price', pos['entry_price'])
            weights[pos['symbol']] = value / total_value if total_value > 0 else 0
        
        return weights
    
    def suggest_rebalancing(
        self,
        positions: List[Dict],
        target_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Suggest rebalancing actions.
        
        Args:
            positions: Current positions
            target_weights: Target weights (optional)
            
        Returns:
            List of rebalancing actions
        """
        if not target_weights:
            # Equal weight as default
            n = len(positions)
            target_weights = {pos['symbol']: 1/n for pos in positions}
        
        current_weights = self.calculate_position_weights(positions)
        
        rebalancing_actions = []
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            # Only rebalance if difference > 5%
            if abs(weight_diff) > 0.05:
                action = 'increase' if weight_diff > 0 else 'decrease'
                rebalancing_actions.append({
                    'symbol': symbol,
                    'action': action,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'difference': weight_diff
                })
        
        return rebalancing_actions


class ExposureManager:
    """
    Manage market exposure.
    """
    
    @staticmethod
    def calculate_gross_exposure(positions: List[Dict]) -> float:
        """
        Calculate gross exposure (sum of all positions).
        
        Args:
            positions: List of positions
            
        Returns:
            Gross exposure
        """
        return sum(
            abs(pos['size'] * pos.get('current_price', pos['entry_price']) * pos.get('leverage', 1))
            for pos in positions
        )
    
    @staticmethod
    def calculate_net_exposure(positions: List[Dict]) -> float:
        """
        Calculate net exposure (long - short).
        
        Args:
            positions: List of positions
            
        Returns:
            Net exposure
        """
        long_exposure = sum(
            pos['size'] * pos.get('current_price', pos['entry_price']) * pos.get('leverage', 1)
            for pos in positions if pos['side'] == 'LONG'
        )
        
        short_exposure = sum(
            pos['size'] * pos.get('current_price', pos['entry_price']) * pos.get('leverage', 1)
            for pos in positions if pos['side'] == 'SHORT'
        )
        
        return long_exposure - short_exposure
    
    @staticmethod
    def calculate_beta_exposure(
        positions: List[Dict],
        market_returns: pd.Series,
        position_returns: Dict[str, pd.Series],
        lookback: int = 100
    ) -> float:
        """
        Calculate portfolio beta to market.
        
        Args:
            positions: List of positions
            market_returns: Market returns (e.g., BTC)
            position_returns: Returns for each position
            lookback: Lookback period
            
        Returns:
            Portfolio beta
        """
        weights = {}
        total_value = sum(
            pos['size'] * pos.get('current_price', pos['entry_price'])
            for pos in positions
        )
        
        for pos in positions:
            value = pos['size'] * pos.get('current_price', pos['entry_price'])
            weights[pos['symbol']] = value / total_value if total_value > 0 else 0
        
        # Calculate weighted beta
        portfolio_beta = 0
        
        for pos in positions:
            symbol = pos['symbol']
            if symbol in position_returns:
                # Beta = Cov(asset, market) / Var(market)
                asset_returns = position_returns[symbol].tail(lookback)
                market_rets = market_returns.tail(lookback)
                
                covariance = asset_returns.cov(market_rets)
                market_variance = market_rets.var()
                
                if market_variance > 0:
                    beta = covariance / market_variance
                    portfolio_beta += beta * weights[symbol]
        
        return portfolio_beta


if __name__ == "__main__":
    # Test portfolio manager
    positions = [
        {
            'symbol': 'BTC/USDT',
            'side': 'LONG',
            'entry_price': 50000,
            'current_price': 51000,
            'size': 0.1,
            'leverage': 3,
            'strategy': 'trend'
        },
        {
            'symbol': 'ETH/USDT',
            'side': 'LONG',
            'entry_price': 3000,
            'current_price': 3100,
            'size': 1.0,
            'leverage': 2,
            'strategy': 'momentum'
        },
        {
            'symbol': 'BNB/USDT',
            'side': 'SHORT',
            'entry_price': 400,
            'current_price': 390,
            'size': 5.0,
            'leverage': 2,
            'strategy': 'reversion'
        }
    ]
    
    pm = PortfolioManager()
    
    # Test portfolio leverage
    leverage = pm.calculate_portfolio_leverage(positions)
    print(f"\nPortfolio Leverage: {leverage:.2f}x")
    
    # Test concentration checks
    violations = pm.check_concentration_limits(positions, total_balance=10000)
    print(f"\nConcentration Violations:")
    for check, violated in violations.items():
        print(f"  {check}: {violated}")
    
    # Test position weights
    weights = pm.calculate_position_weights(positions)
    print(f"\nPosition Weights:")
    for symbol, weight in weights.items():
        print(f"  {symbol}: {weight:.2%}")
    
    # Test exposure
    gross = ExposureManager.calculate_gross_exposure(positions)
    net = ExposureManager.calculate_net_exposure(positions)
    print(f"\nExposure:")
    print(f"  Gross: ${gross:.2f}")
    print(f"  Net: ${net:.2f}")
    
    print("\n✅ Portfolio manager test complete")

