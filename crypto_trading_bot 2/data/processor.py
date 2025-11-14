"""
Data processor for the crypto trading bot.

This module handles:
- Data cleaning and validation
- Outlier detection and handling
- Missing data interpolation
- Data normalization
- Multi-timeframe alignment
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import RobustScaler

from utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Process and clean trading data.
    """
    
    def __init__(
        self,
        max_missing_pct: float = 0.05,
        outlier_std: float = 5.0
    ):
        """
        Initialize data processor.
        
        Args:
            max_missing_pct: Maximum allowed missing data percentage
            outlier_std: Standard deviations for outlier detection
        """
        self.max_missing_pct = max_missing_pct
        self.outlier_std = outlier_std
        self.scaler = RobustScaler()
    
    def process_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str = ""
    ) -> pd.DataFrame:
        """
        Process OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            symbol: Symbol name (for logging)
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return df
        
        df = df.copy()
        
        # Validate data
        df = self.validate_ohlcv(df, symbol)
        
        # Handle missing data
        df = self.handle_missing_data(df, symbol)
        
        # Detect and handle outliers
        df = self.handle_outliers(df, symbol)
        
        # Ensure proper sorting
        df = df.sort_index()
        
        return df
    
    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str = ""
    ) -> pd.DataFrame:
        """
        Validate OHLCV data integrity.
        
        Args:
            df: OHLCV DataFrame
            symbol: Symbol name
            
        Returns:
            Validated DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for {symbol}: {missing_cols}")
        
        # Check for negative values
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                logger.warning(f"Negative or zero values found in {col} for {symbol}")
                df = df[df[col] > 0]
        
        # Validate high/low logic
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            logger.warning(f"Found {invalid_hl.sum()} candles with high < low for {symbol}")
            # Fix by swapping
            df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
        
        # Validate OHLC logic
        invalid_ohlc = (
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} candles with invalid OHLC for {symbol}")
            df = df[~invalid_ohlc]
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps for {symbol}")
            df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def handle_missing_data(
        self,
        df: pd.DataFrame,
        symbol: str = ""
    ) -> pd.DataFrame:
        """
        Handle missing data.
        
        Args:
            df: DataFrame with potential missing data
            symbol: Symbol name
            
        Returns:
            DataFrame with filled data
        """
        # Calculate missing percentage
        missing_pct = df.isnull().sum() / len(df)
        
        # Check if within threshold
        for col in df.columns:
            if missing_pct[col] > self.max_missing_pct:
                logger.error(
                    f"Too much missing data in {col} for {symbol}: "
                    f"{missing_pct[col]:.2%} (max: {self.max_missing_pct:.2%})"
                )
                raise ValueError(f"Excessive missing data in {col}")
        
        # Forward fill for price data (use last known value)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Fill remaining with backward fill
        df[price_cols] = df[price_cols].fillna(method='bfill')
        
        # Volume: fill with 0 or median
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(df['volume'].median())
        
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        symbol: str = ""
    ) -> pd.DataFrame:
        """
        Detect and handle outliers.
        
        Args:
            df: DataFrame
            symbol: Symbol name
            
        Returns:
            DataFrame with outliers handled
        """
        # Calculate returns for outlier detection
        df['returns'] = df['close'].pct_change()
        
        # Z-score method
        mean_return = df['returns'].mean()
        std_return = df['returns'].std()
        
        z_scores = np.abs((df['returns'] - mean_return) / std_return)
        
        # Identify outliers
        outliers = z_scores > self.outlier_std
        
        if outliers.any():
            logger.warning(f"Found {outliers.sum()} outliers for {symbol}")
            
            # Cap outliers at threshold
            threshold_upper = mean_return + (self.outlier_std * std_return)
            threshold_lower = mean_return - (self.outlier_std * std_return)
            
            df.loc[outliers & (df['returns'] > threshold_upper), 'returns'] = threshold_upper
            df.loc[outliers & (df['returns'] < threshold_lower), 'returns'] = threshold_lower
            
            # Recalculate prices based on capped returns
            df['close'] = df['close'].iloc[0] * (1 + df['returns']).cumprod()
        
        # Drop temporary column
        df = df.drop('returns', axis=1)
        
        return df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe (e.g., '1h', '4h')
            
        Returns:
            Resampled DataFrame
        """
        # Mapping for pandas resample
        resample_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        rule = resample_map.get(target_timeframe)
        if not rule:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove any NaN rows (incomplete periods)
        resampled = resampled.dropna()
        
        return resampled
    
    def align_multi_timeframe(
        self,
        data_dict: Dict[str, pd.DataFrame],
        primary_timeframe: str
    ) -> pd.DataFrame:
        """
        Align multiple timeframes to primary timeframe.
        
        Args:
            data_dict: Dict of {timeframe: DataFrame}
            primary_timeframe: Primary timeframe to align to
            
        Returns:
            Aligned DataFrame with multi-timeframe data
        """
        if primary_timeframe not in data_dict:
            raise ValueError(f"Primary timeframe {primary_timeframe} not in data")
        
        # Start with primary timeframe
        primary_df = data_dict[primary_timeframe].copy()
        
        # Add data from other timeframes
        for tf, df in data_dict.items():
            if tf == primary_timeframe:
                continue
            
            # Merge with forward fill (use most recent value)
            df_renamed = df.add_suffix(f'_{tf}')
            
            primary_df = primary_df.join(
                df_renamed,
                how='left'
            )
            
            # Forward fill to align
            primary_df[df_renamed.columns] = primary_df[df_renamed.columns].fillna(method='ffill')
        
        return primary_df
    
    def calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate data quality metrics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict of quality metrics
        """
        metrics = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'days': (df.index.max() - df.index.min()).days
            },
            'missing_data': {
                col: df[col].isnull().sum()
                for col in df.columns
            },
            'missing_pct': {
                col: df[col].isnull().sum() / len(df)
                for col in df.columns
            },
            'duplicates': df.index.duplicated().sum(),
            'gaps': self._detect_gaps(df)
        }
        
        return metrics
    
    def _detect_gaps(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """
        Detect gaps in time series.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            List of (start, end) tuples for gaps
        """
        # Calculate time differences
        time_diff = df.index.to_series().diff()
        
        # Expected frequency (most common diff)
        expected_freq = time_diff.mode()[0] if len(time_diff) > 0 else timedelta(minutes=15)
        
        # Find gaps (diff > 2x expected)
        gaps = time_diff[time_diff > expected_freq * 2]
        
        gap_list = [
            (df.index[i-1], df.index[i])
            for i in gaps.index
        ]
        
        return gap_list
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using RobustScaler.
        
        Args:
            df: DataFrame to normalize
            columns: Columns to normalize (default: all numeric)
            fit: Whether to fit scaler (True for train, False for test)
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            df[columns] = self.scaler.transform(df[columns])
        
        return df
    
    def split_train_val_test(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets (time-based).
        
        Args:
            df: DataFrame to split
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert train_size + val_size + test_size == 1.0, "Sizes must sum to 1.0"
        
        n = len(df)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(
            f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        
        return train_df, val_df, test_df


class MarketRegimeDetector:
    """
    Detect market regimes for adaptive strategy allocation.
    """
    
    @staticmethod
    def detect_volatility_regime(
        df: pd.DataFrame,
        atr_column: str = 'atr_14',
        lookback: int = 100
    ) -> str:
        """
        Detect volatility regime.
        
        Args:
            df: DataFrame with ATR data
            atr_column: ATR column name
            lookback: Lookback period
            
        Returns:
            'low', 'medium', or 'high'
        """
        if atr_column not in df.columns or len(df) < lookback:
            return 'medium'
        
        current_atr = df[atr_column].iloc[-1]
        avg_atr = df[atr_column].iloc[-lookback:].mean()
        
        if current_atr < avg_atr * 0.8:
            return 'low'
        elif current_atr > avg_atr * 1.5:
            return 'high'
        else:
            return 'medium'
    
    @staticmethod
    def detect_trend_regime(
        df: pd.DataFrame,
        adx_column: str = 'adx_14'
    ) -> str:
        """
        Detect trend regime.
        
        Args:
            df: DataFrame with ADX data
            adx_column: ADX column name
            
        Returns:
            'trending' or 'ranging'
        """
        if adx_column not in df.columns:
            return 'ranging'
        
        current_adx = df[adx_column].iloc[-1]
        
        if current_adx > 25:
            return 'trending'
        else:
            return 'ranging'
    
    @staticmethod
    def detect_market_phase(
        df: pd.DataFrame,
        lookback: int = 50
    ) -> str:
        """
        Detect market phase (Wyckoff).
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period
            
        Returns:
            'accumulation', 'markup', 'distribution', or 'markdown'
        """
        if len(df) < lookback:
            return 'unknown'
        
        # Simple implementation using price and volume
        recent_df = df.iloc[-lookback:]
        
        price_change = (recent_df['close'].iloc[-1] / recent_df['close'].iloc[0]) - 1
        volume_trend = recent_df['volume'].iloc[-20:].mean() / recent_df['volume'].iloc[-lookback:-20].mean()
        
        if price_change > 0.02 and volume_trend > 1.1:
            return 'markup'
        elif price_change < -0.02 and volume_trend > 1.1:
            return 'markdown'
        elif abs(price_change) < 0.02 and volume_trend < 0.9:
            return 'accumulation'
        elif abs(price_change) < 0.02 and volume_trend > 1.1:
            return 'distribution'
        else:
            return 'transition'


if __name__ == "__main__":
    # Test processor
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='15min')
    data = {
        'open': np.random.uniform(50000, 51000, len(dates)),
        'high': np.random.uniform(50500, 51500, len(dates)),
        'low': np.random.uniform(49500, 50500, len(dates)),
        'close': np.random.uniform(50000, 51000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    
    # Process
    processor = DataProcessor()
    processed_df = processor.process_ohlcv(df, "TEST/USDT")
    
    # Quality metrics
    quality = processor.calculate_data_quality(processed_df)
    print("\nData Quality Metrics:")
    print(f"Total Rows: {quality['total_rows']}")
    print(f"Date Range: {quality['date_range']}")
    print(f"Duplicates: {quality['duplicates']}")
    print(f"Gaps: {len(quality['gaps'])}")
    
    print("\n✅ Data processor test complete")

