"""
Advanced ML Feature Engineering for ETF Trading System.
Generates technical indicators, statistical features, and market sentiment features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import ta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngine:
    """Advanced feature engineering for ML-based trading strategies."""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50]):
        """
        Initialize feature engine.
        
        Args:
            lookback_periods: List of periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        self.scalers = {}
        self.feature_names = []
        
    def generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        logger.info("Generating technical indicators...")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Create copy to avoid modifying original
        features_df = df.copy()
        
        # Price-based indicators
        features_df = self._add_price_indicators(features_df)
        
        # Volume-based indicators
        features_df = self._add_volume_indicators(features_df)
        
        # Momentum indicators
        features_df = self._add_momentum_indicators(features_df)
        
        # Volatility indicators
        features_df = self._add_volatility_indicators(features_df)
        
        # Support/Resistance indicators
        features_df = self._add_support_resistance_indicators(features_df)
        
        # Pattern recognition features
        features_df = self._add_pattern_features(features_df)
        
        return features_df
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based technical indicators."""
        
        # Simple Moving Averages
        for period in self.lookback_periods:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            
            # Price relative to moving averages
            df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # MACD
        macd_line, macd_signal, macd_histogram = ta.trend.MACD(df['close']).macd(), \
                                                ta.trend.MACD(df['close']).macd_signal(), \
                                                ta.trend.MACD(df['close']).macd_diff()
        df['macd_line'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_high, bb_mid, bb_low = ta.volatility.bollinger_hband(df['close']), \
                                  ta.volatility.bollinger_mavg(df['close']), \
                                  ta.volatility.bollinger_lband(df['close'])
        df['bb_high'] = bb_high
        df['bb_mid'] = bb_mid
        df['bb_low'] = bb_low
        df['bb_width'] = (bb_high - bb_low) / bb_mid
        df['bb_position'] = (df['close'] - bb_low) / (bb_high - bb_low)
        
        # Parabolic SAR
        df['sar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        
        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        
        # Volume Moving Averages
        for period in self.lookback_periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # On Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume Price Trend
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        # Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        
        # Accumulation/Distribution Line
        df['ad_line'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        
        # RSI with different periods
        for period in [14, 21, 30]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        
        # Stochastic Oscillator
        stoch_k = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        stoch_d = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_diff'] = stoch_k - stoch_d
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Commodity Channel Index
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # Rate of Change
        for period in [10, 20]:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()
        
        # Awesome Oscillator
        df['ao'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        
        # True Range and Average True Range
        df['tr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).true_range()
        
        # Keltner Channels
        kc_high = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close']).keltner_channel_hband()
        kc_low = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close']).keltner_channel_lband()
        df['kc_high'] = kc_high
        df['kc_low'] = kc_low
        df['kc_width'] = (kc_high - kc_low) / df['close']
        
        # Donchian Channels
        dc_high = ta.volatility.DonchianChannel(df['high'], df['low'], df['close']).donchian_channel_hband()
        dc_low = ta.volatility.DonchianChannel(df['high'], df['low'], df['close']).donchian_channel_lband()
        df['dc_high'] = dc_high
        df['dc_low'] = dc_low
        
        # Historical Volatility
        for period in [10, 20, 30]:
            returns = df['close'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
        
        return df
    
    def _add_support_resistance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance level indicators."""
        
        # Pivot Points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        # Distance from pivot levels
        df['dist_from_pivot'] = (df['close'] - df['pivot']) / df['pivot']
        df['dist_from_r1'] = (df['close'] - df['r1']) / df['r1']
        df['dist_from_s1'] = (df['close'] - df['s1']) / df['s1']
        
        # Recent highs and lows
        for period in [20, 50]:
            df[f'high_{period}'] = df['high'].rolling(window=period).max()
            df[f'low_{period}'] = df['low'].rolling(window=period).min()
            df[f'position_in_range_{period}'] = (df['close'] - df[f'low_{period}']) / (df[f'high_{period}'] - df[f'low_{period}'])
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        
        # Candlestick patterns (simplified)
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Doji detection
        df['is_doji'] = (df['body_size'] < 0.01).astype(int)
        
        # Hammer detection
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                          (df['upper_shadow'] < df['body_size'])).astype(int)
        
        # Shooting star detection
        df['is_shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                 (df['lower_shadow'] < df['body_size'])).astype(int)
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical features from price data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with statistical features
        """
        logger.info("Generating statistical features...")
        
        features_df = df.copy()
        
        # Returns
        features_df['returns'] = df['close'].pct_change()
        features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling statistics
        for period in self.lookback_periods:
            returns = features_df['returns']
            
            # Central tendency
            features_df[f'returns_mean_{period}'] = returns.rolling(window=period).mean()
            features_df[f'returns_median_{period}'] = returns.rolling(window=period).median()
            
            # Dispersion
            features_df[f'returns_std_{period}'] = returns.rolling(window=period).std()
            features_df[f'returns_var_{period}'] = returns.rolling(window=period).var()
            features_df[f'returns_range_{period}'] = (returns.rolling(window=period).max() - 
                                                     returns.rolling(window=period).min())
            
            # Shape
            features_df[f'returns_skew_{period}'] = returns.rolling(window=period).skew()
            features_df[f'returns_kurt_{period}'] = returns.rolling(window=period).kurt()
            
            # Quantiles
            features_df[f'returns_q25_{period}'] = returns.rolling(window=period).quantile(0.25)
            features_df[f'returns_q75_{period}'] = returns.rolling(window=period).quantile(0.75)
            
            # Price statistics
            features_df[f'price_std_{period}'] = df['close'].rolling(window=period).std()
            features_df[f'price_cv_{period}'] = (features_df[f'price_std_{period}'] / 
                                                features_df[f'sma_{period}'])
        
        # Autocorrelation features
        for lag in [1, 5, 10]:
            features_df[f'returns_autocorr_lag_{lag}'] = features_df['returns'].rolling(window=50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        return features_df
    
    def generate_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with microstructure features
        """
        logger.info("Generating market microstructure features...")
        
        features_df = df.copy()
        
        # Spread indicators
        features_df['hl_spread'] = (df['high'] - df['low']) / df['close']
        features_df['oc_spread'] = abs(df['open'] - df['close']) / df['close']
        
        # Price impact
        features_df['price_impact'] = (df['close'] - df['open']) / df['volume']
        
        # VWAP (Volume Weighted Average Price)
        features_df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        features_df['price_vs_vwap'] = (df['close'] - features_df['vwap']) / features_df['vwap']
        
        # Relative Volume
        for period in [10, 20]:
            features_df[f'relative_volume_{period}'] = (df['volume'] / 
                                                       df['volume'].rolling(window=period).mean())
        
        # Order flow proxies
        features_df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features_df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        return features_df
    
    def apply_feature_scaling(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Apply feature scaling to numerical columns.
        
        Args:
            df: DataFrame with features
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Applying {method} scaling...")
        
        # Select numerical columns (exclude datetime and categorical)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove price columns that shouldn't be scaled
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in numerical_cols if col not in price_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform features
        scaled_features = scaler.fit_transform(df[feature_cols])
        
        # Create new DataFrame with scaled features
        scaled_df = df.copy()
        scaled_df[feature_cols] = scaled_features
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        return scaled_df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str, k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select best features using statistical tests.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            k: Number of best features to select
            
        Returns:
            Tuple of (DataFrame with selected features, list of selected feature names)
        """
        logger.info(f"Selecting {k} best features...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in [np.float64, np.int64]]
        X = df[feature_cols].dropna()
        y = df[target_col].loc[X.index]
        
        # Select k best features
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Create DataFrame with selected features
        selected_df = df[[target_col] + selected_features].copy()
        
        self.feature_names = selected_features
        
        return selected_df, selected_features
    
    def apply_pca(self, df: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            df: DataFrame with features
            n_components: Number of components or variance ratio to retain
            
        Returns:
            Tuple of (DataFrame with PCA components, fitted PCA object)
        """
        logger.info(f"Applying PCA with {n_components} variance retention...")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove any missing values
        X = df[numerical_cols].dropna()
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame with PCA components
        pca_cols = [f'pca_component_{i+1}' for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
        
        logger.info(f"PCA reduced {len(numerical_cols)} features to {X_pca.shape[1]} components")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        return pca_df, pca
    
    def create_target_variables(self, df: pd.DataFrame, horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Create target variables for ML models.
        
        Args:
            df: DataFrame with price data
            horizons: List of prediction horizons in days
            
        Returns:
            DataFrame with target variables
        """
        logger.info("Creating target variables...")
        
        targets_df = df.copy()
        
        for horizon in horizons:
            # Future returns
            targets_df[f'future_return_{horizon}d'] = df['close'].pct_change(periods=horizon).shift(-horizon)
            
            # Future direction (classification)
            targets_df[f'future_direction_{horizon}d'] = (targets_df[f'future_return_{horizon}d'] > 0).astype(int)
            
            # Future volatility
            returns = df['close'].pct_change()
            targets_df[f'future_volatility_{horizon}d'] = (returns.rolling(window=horizon).std().shift(-horizon) * 
                                                          np.sqrt(252))
            
            # Maximum favorable excursion (MFE) and Maximum adverse excursion (MAE)
            future_highs = df['high'].rolling(window=horizon).max().shift(-horizon)
            future_lows = df['low'].rolling(window=horizon).min().shift(-horizon)
            
            targets_df[f'mfe_{horizon}d'] = (future_highs - df['close']) / df['close']
            targets_df[f'mae_{horizon}d'] = (df['close'] - future_lows) / df['close']
        
        return targets_df
    
    def get_feature_importance_analysis(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV
        from sklearn.feature_selection import mutual_info_regression
        
        logger.info("Analyzing feature importance...")
        
        # Prepare data
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in [np.float64, np.int64]]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        importance_scores = pd.DataFrame(index=feature_cols)
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance_scores['random_forest'] = rf.feature_importances_
        
        # Lasso importance
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        importance_scores['lasso'] = np.abs(lasso.coef_)
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        importance_scores['mutual_info'] = mi_scores
        
        # Correlation with target
        importance_scores['correlation'] = np.abs(X.corrwith(y))
        
        # Average importance
        importance_scores['average'] = importance_scores.mean(axis=1)
        
        return importance_scores.sort_values('average', ascending=False)

def create_comprehensive_features(price_data: pd.DataFrame, 
                                 lookback_periods: List[int] = [5, 10, 20, 50],
                                 target_horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """
    Create comprehensive feature set for ML trading models.
    
    Args:
        price_data: DataFrame with OHLCV data
        lookback_periods: Periods for rolling calculations
        target_horizons: Horizons for target variables
        
    Returns:
        DataFrame with all features and targets
    """
    logger.info("Creating comprehensive feature set...")
    
    # Initialize feature engine
    engine = AdvancedFeatureEngine(lookback_periods=lookback_periods)
    
    # Generate all feature types
    df = engine.generate_technical_features(price_data)
    df = engine.generate_statistical_features(df)
    df = engine.generate_market_microstructure_features(df)
    df = engine.create_target_variables(df, horizons=target_horizons)
    
    # Remove infinite and too large values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Created {len(df.columns)} total features")
    
    return df
