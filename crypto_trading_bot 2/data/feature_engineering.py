"""
Feature Engineering for the crypto trading bot.

This module creates 200+ features for machine learning models:
- Price-based features (returns, volatility)
- Volume features (VWAP, volume profile)
- Technical indicators (trend, momentum, mean reversion)
- Microstructure features (order book, spreads)
- Derivatives features (funding, open interest, liquidations)
- Market regime features
- Correlation features
- Sentiment features
- Time-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for trading strategies.
    
    Generates 200+ features from OHLCV and additional data.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_count = 0
        self.feature_names = []
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Create all features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            additional_data: Dict with additional data (funding, OI, etc.)
            
        Returns:
            DataFrame with all features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            elif 'time' in df.columns:
                df.set_index('time', inplace=True)
            else:
                # Try to convert index to datetime
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    logger.warning("Could not convert index to datetime, creating default datetime index")
                    df.index = pd.date_range(start='2022-01-01', periods=len(df), freq='15min')
        
        logger.info("Starting feature engineering...")
        
        # Price-based features
        df = self.add_price_features(df)
        
        # Volume features
        df = self.add_volume_features(df)
        
        # Technical indicators - Trend
        df = self.add_trend_indicators(df)
        
        # Technical indicators - Momentum
        df = self.add_momentum_indicators(df)
        
        # Technical indicators - Mean Reversion
        df = self.add_mean_reversion_indicators(df)
        
        # Technical indicators - Volume
        df = self.add_volume_indicators(df)
        
        # Volatility features
        df = self.add_volatility_features(df)
        
        # Microstructure features (if order book data available)
        if additional_data and 'orderbook' in additional_data:
            df = self.add_microstructure_features(df, additional_data['orderbook'])
        
        # Derivatives features
        if additional_data:
            if 'funding_rate' in additional_data:
                df = self.add_funding_features(df, additional_data['funding_rate'])
            if 'open_interest' in additional_data:
                df = self.add_open_interest_features(df, additional_data['open_interest'])
            if 'liquidations' in additional_data:
                df = self.add_liquidation_features(df, additional_data['liquidations'])
        
        # Market regime features
        df = self.add_regime_features(df)
        
        # Correlation features (if multiple symbols available)
        if additional_data and 'correlation_data' in additional_data:
            df = self.add_correlation_features(df, additional_data['correlation_data'])
        
        # Sentiment features
        if additional_data and 'sentiment' in additional_data:
            df = self.add_sentiment_features(df, additional_data['sentiment'])
        
        # Time-based features
        df = self.add_time_features(df)
        
        # Count features
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        self.feature_count = len(feature_cols)
        self.feature_names = feature_cols
        
        logger.info(f"Feature engineering complete: {self.feature_count} features created")
        
        # Drop any NaN rows (from indicator calculations)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns (various periods)
        for period in [1, 5, 15, 30, 60]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['returns_1']).cumprod() - 1
        
        # Price momentum
        df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_pct'] = (df['high'] - df['low']) / df['low']
        
        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap features
        df['gap'] = df['open'] / df['close'].shift(1) - 1
        df['gap_up'] = (df['gap'] > 0).astype(int)
        df['gap_down'] = (df['gap'] < 0).astype(int)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        # Volume ratio
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        df['volume_ratio_50'] = df['volume'] / df['volume_ma_50']
        
        # Volume rate of change
        df['volume_roc'] = df['volume'].pct_change(10)
        
        # Volume momentum
        df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
        
        # Volume * Price (dollar volume)
        df['dollar_volume'] = df['volume'] * df['close']
        df['dollar_volume_ma_20'] = df['dollar_volume'].rolling(20).mean()
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        )
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # EMAs
        for period in [8, 21, 50, 100, 200]:
            ema = EMAIndicator(df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
            df[f'ema_{period}_dist'] = (df['close'] / df[f'ema_{period}']) - 1
        
        # EMA crossovers
        df['ema_cross_8_21'] = (df['ema_8'] > df['ema_21']).astype(int)
        df['ema_cross_21_50'] = (df['ema_21'] > df['ema_50']).astype(int)
        df['ema_cross_50_200'] = (df['ema_50'] > df['ema_200']).astype(int)
        
        # MACD
        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # ADX (Trend Strength)
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx_14'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        df['adx_diff'] = df['adx_pos'] - df['adx_neg']
        
        # Supertrend
        df = self._add_supertrend(df, period=10, multiplier=3)
        
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [14, 21]:
            rsi = RSIIndicator(df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        
        # Stochastic RSI
        stoch_rsi = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_rsi'] = stoch_rsi.stoch()
        df['stoch_rsi_signal'] = stoch_rsi.stoch_signal()
        
        # CCI (Commodity Channel Index)
        df['cci_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # Williams %R
        williams_r = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14)
        df['williams_r'] = williams_r.williams_r()
        
        # MFI (Money Flow Index)
        df['mfi_14'] = ta.volume.money_flow_index(
            df['high'], df['low'], df['close'], df['volume'], window=14
        )
        
        # ROC (Rate of Change)
        for period in [10, 20]:
            roc = ROCIndicator(df['close'], window=period)
            df[f'roc_{period}'] = roc.roc()
        
        return df
    
    def add_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mean reversion indicators."""
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_upper_breach'] = (df['close'] > df['bb_upper']).astype(int)
        df['bb_lower_breach'] = (df['close'] < df['bb_lower']).astype(int)
        
        # Keltner Channels
        kc = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['kc_width'] = df['kc_upper'] - df['kc_lower']
        
        # Z-Score (multiple windows)
        for window in [20, 50, 100]:
            mean = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'zscore_{window}'] = (df['close'] - mean) / std
        
        # Distance from moving averages
        for period in [20, 50, 100, 200]:
            ma = df['close'].rolling(period).mean()
            df[f'distance_ma_{period}'] = (df['close'] / ma) - 1
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        # OBV (On Balance Volume)
        obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_ma_20'] = df['obv'].rolling(20).mean()
        
        # CMF (Chaikin Money Flow)
        cmf = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20)
        df['cmf'] = cmf.chaikin_money_flow()
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14)
        df['vwap'] = vwap.volume_weighted_average_price()
        df['vwap_distance'] = (df['close'] / df['vwap']) - 1
        
        # Accumulation/Distribution
        df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Force Index
        df['force_index'] = ta.volume.force_index(df['close'], df['volume'], window=13)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # ATR (Average True Range)
        for period in [14, 21, 50]:
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
            df[f'atr_{period}'] = atr.average_true_range()
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']
        
        # Realized Volatility (multiple windows)
        for window in [20, 50, 100]:
            df[f'realized_vol_{window}'] = df['returns_1'].rolling(window).std() * np.sqrt(window)
        
        # Parkinson Volatility
        df['parkinson_vol'] = self._parkinson_volatility(df, window=20)
        
        # Garman-Klass Volatility
        df['garman_klass_vol'] = self._garman_klass_volatility(df, window=20)
        
        # Rolling standard deviation
        for window in [10, 20, 50]:
            df[f'std_{window}'] = df['close'].rolling(window).std()
            df[f'std_{window}_pct'] = df[f'std_{window}'] / df['close']
        
        # Volatility ratio
        df['vol_ratio_short_long'] = df['std_10'] / df['std_50']
        
        return df
    
    def add_microstructure_features(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Add microstructure features."""
        if not orderbook_data:
            return df
        
        # Bid-Ask spread
        df['bid_ask_spread'] = (orderbook_data['ask'] - orderbook_data['bid']) / orderbook_data['bid']
        
        # Order book imbalance (top 5 levels)
        bid_volume = sum([level[1] for level in orderbook_data['bids'][:5]])
        ask_volume = sum([level[1] for level in orderbook_data['asks'][:5]])
        df['ob_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Depth pressure
        df['depth_pressure'] = bid_volume / ask_volume
        
        # Mid price
        df['mid_price'] = (orderbook_data['bid'] + orderbook_data['ask']) / 2
        df['mid_price_distance'] = (df['close'] / df['mid_price']) - 1
        
        return df
    
    def add_funding_features(
        self,
        df: pd.DataFrame,
        funding_data: pd.Series
    ) -> pd.DataFrame:
        """Add funding rate features."""
        # Current funding rate
        df['funding_rate'] = funding_data
        
        # Funding rate moving averages
        df['funding_rate_ma_8h'] = df['funding_rate'].rolling(8).mean()  # 8 periods = 24h
        df['funding_rate_ma_24h'] = df['funding_rate'].rolling(24).mean()  # 3 days
        
        # Funding rate momentum
        df['funding_momentum'] = df['funding_rate'] - df['funding_rate'].shift(1)
        
        # Funding rate extremes
        df['funding_extreme_positive'] = (df['funding_rate'] > 0.001).astype(int)
        df['funding_extreme_negative'] = (df['funding_rate'] < -0.001).astype(int)
        
        # Annual funding rate equivalent
        df['funding_annualized'] = df['funding_rate'] * 365 * 3  # 3x per day
        
        return df
    
    def add_open_interest_features(
        self,
        df: pd.DataFrame,
        oi_data: pd.Series
    ) -> pd.DataFrame:
        """Add open interest features."""
        df['open_interest'] = oi_data
        
        # OI change rate
        df['oi_change'] = df['open_interest'].pct_change()
        df['oi_change_ma'] = df['oi_change'].rolling(20).mean()
        
        # OI / Volume ratio
        df['oi_volume_ratio'] = df['open_interest'] / df['volume']
        
        # OI trend
        df['oi_trend'] = df['open_interest'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        )
        
        return df
    
    def add_liquidation_features(
        self,
        df: pd.DataFrame,
        liq_data: Dict
    ) -> pd.DataFrame:
        """Add liquidation features."""
        # Recent liquidation volume
        df['liq_volume_long'] = liq_data.get('long_liquidations', 0)
        df['liq_volume_short'] = liq_data.get('short_liquidations', 0)
        df['liq_volume_total'] = df['liq_volume_long'] + df['liq_volume_short']
        
        # Liquidation ratio
        df['liq_ratio'] = df['liq_volume_long'] / (df['liq_volume_short'] + 1)
        
        # Liquidation levels (distance to major liquidation clusters)
        if 'liquidation_levels' in liq_data:
            nearest_liq = liq_data['liquidation_levels']
            df['distance_to_liq'] = abs(df['close'] - nearest_liq) / df['close']
        
        return df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        # Volatility regime (one-hot encoded, no string column)
        vol_regime_cat = pd.cut(
            df['atr_14_pct'],
            bins=[0, 0.02, 0.05, 1.0],
            labels=['low_vol', 'med_vol', 'high_vol']
        )
        # Only numeric features for ML models
        df['vol_regime_low'] = (vol_regime_cat == 'low_vol').astype(int)
        df['vol_regime_med'] = (vol_regime_cat == 'med_vol').astype(int)
        df['vol_regime_high'] = (vol_regime_cat == 'high_vol').astype(int)
        
        # Trend regime (based on ADX)
        df['trend_regime'] = (df['adx_14'] > 25).astype(int)  # 1 = trending, 0 = ranging
        
        # Market direction
        df['direction_bullish'] = (df['ema_8'] > df['ema_21']).astype(int)
        df['direction_bearish'] = (df['ema_8'] < df['ema_21']).astype(int)
        
        # Volatility clustering
        df['vol_clustering'] = df['atr_14'].rolling(20).std() / df['atr_14'].rolling(20).mean()
        
        return df
    
    def add_correlation_features(
        self,
        df: pd.DataFrame,
        correlation_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Add correlation features."""
        # BTC correlation
        if 'BTC' in correlation_data:
            btc_returns = correlation_data['BTC']
            df['btc_correlation'] = df['returns_1'].rolling(50).corr(btc_returns)
        
        # Cross-asset correlation
        if 'ETH' in correlation_data:
            eth_returns = correlation_data['ETH']
            df['eth_correlation'] = df['returns_1'].rolling(50).corr(eth_returns)
        
        return df
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_data: Dict
    ) -> pd.DataFrame:
        """Add sentiment features."""
        # Fear & Greed Index
        if 'fear_greed' in sentiment_data:
            df['fear_greed_index'] = sentiment_data['fear_greed']
            df['fear_greed_extreme_fear'] = (df['fear_greed_index'] < 20).astype(int)
            df['fear_greed_extreme_greed'] = (df['fear_greed_index'] > 80).astype(int)
        
        # BTC Dominance
        if 'btc_dominance' in sentiment_data:
            df['btc_dominance'] = sentiment_data['btc_dominance']
            df['btc_dominance_change'] = df['btc_dominance'].pct_change()
        
        # Funding rate aggregate (sentiment proxy)
        if 'funding_aggregate' in sentiment_data:
            df['funding_aggregate'] = sentiment_data['funding_aggregate']
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Hour of day
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Is weekend
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Session indicators (Asian, European, US)
        df['session_asian'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['session_european'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['session_us'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)
        
        return df
    
    # Helper methods
    
    def _add_supertrend(
        self,
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0
    ) -> pd.DataFrame:
        """Add Supertrend indicator."""
        hl2 = (df['high'] + df['low']) / 2
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
        
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upperband.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upperband.iloc[i-1]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < lowerband.iloc[i-1]:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        df['price_above_supertrend'] = (df['close'] > df['supertrend']).astype(int)
        
        return df
    
    def _parkinson_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        hl_ratio = np.log(df['high'] / df['low'])
        parkinson = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(window).mean())
        return parkinson
    
    def _garman_klass_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        
        gk = np.sqrt(
            0.5 * log_hl.rolling(window).mean() -
            (2 * np.log(2) - 1) * log_co.rolling(window).mean()
        )
        
        return gk
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature column names
        """
        return self.feature_names
    
    def get_feature_count(self) -> int:
        """
        Get total number of features.
        
        Returns:
            Feature count
        """
        return self.feature_count


if __name__ == "__main__":
    # Test feature engineering
    from data.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    df = fetcher.fetch_ohlcv('BTC/USDT', '15m', limit=500)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"\n✅ Created {engineer.get_feature_count()} features")
    print(f"\nDataFrame shape: {df_features.shape}")
    print(f"\nFeature columns (first 20):")
    print(engineer.get_feature_names()[:20])
    print(f"\n... and {len(engineer.get_feature_names()) - 20} more features")

