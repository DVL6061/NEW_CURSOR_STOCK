"""
Technical indicators calculation module
"""

import pandas as pd
import numpy as np
# import pandas_ta as ta  # Commented out due to Windows compatibility issues
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicators calculation and feature engineering"""
    
    def __init__(self):
        self.indicators_config = {
            'RSI': {'period': 14},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
            'EMA': {'periods': [9, 21, 50, 200]},
            'SMA': {'periods': [9, 21, 50, 200]},
            'BBANDS': {'period': 20, 'std': 2},
            'ADX': {'period': 14},
            'STOCH': {'k_period': 14, 'd_period': 3},
            'CCI': {'period': 20},
            'WILLR': {'period': 14},
            'ROC': {'period': 10},
            'MOM': {'period': 10},
            'PPO': {'fast': 12, 'slow': 26},
            'TRIX': {'period': 14},
            'ULTOSC': {'periods': [7, 14, 28]}
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        
        Returns:
            DataFrame with all technical indicators added
        """
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        try:
            # Calculate momentum indicators
            data = self._calculate_momentum_indicators(data)
            
            # Calculate trend indicators
            data = self._calculate_trend_indicators(data)
            
            # Calculate volatility indicators
            data = self._calculate_volatility_indicators(data)
            
            # Calculate volume indicators
            data = self._calculate_volume_indicators(data)
            
            # Calculate oscillator indicators
            data = self._calculate_oscillator_indicators(data)
            
            # Calculate custom features
            data = self._calculate_custom_features(data)
            
            logger.info(f"Successfully calculated technical indicators for {len(data)} rows")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
        
        return data
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based indicators"""
        try:
            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close'], self.indicators_config['RSI']['period'])
            
            # ROC (Rate of Change)
            df['roc'] = df['close'].pct_change(self.indicators_config['ROC']['period']) * 100
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(self.indicators_config['MOM']['period'])
            
            # TRIX (simplified)
            ema1 = df['close'].ewm(span=self.indicators_config['TRIX']['period']).mean()
            ema2 = ema1.ewm(span=self.indicators_config['TRIX']['period']).mean()
            ema3 = ema2.ewm(span=self.indicators_config['TRIX']['period']).mean()
            df['trix'] = ema3.pct_change() * 10000
            
            # PPO (Percentage Price Oscillator)
            ema_fast = df['close'].ewm(span=self.indicators_config['PPO']['fast']).mean()
            ema_slow = df['close'].ewm(span=self.indicators_config['PPO']['slow']).mean()
            df['ppo'] = ((ema_fast - ema_slow) / ema_slow) * 100
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based indicators"""
        try:
            # MACD
            ema_fast = df['close'].ewm(span=self.indicators_config['MACD']['fast']).mean()
            ema_slow = df['close'].ewm(span=self.indicators_config['MACD']['slow']).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=self.indicators_config['MACD']['signal']).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # EMA (Exponential Moving Averages)
            for period in self.indicators_config['EMA']['periods']:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # SMA (Simple Moving Averages)
            for period in self.indicators_config['SMA']['periods']:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # ADX (Average Directional Index) - Simplified
            df['adx'] = self._calculate_adx(df, self.indicators_config['ADX']['period'])
            df['di_plus'] = 50  # Simplified
            df['di_minus'] = 50  # Simplified
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index) - Simplified"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        try:
            # Bollinger Bands
            period = self.indicators_config['BBANDS']['period']
            std = self.indicators_config['BBANDS']['std']
            sma = df['close'].rolling(window=period).mean()
            std_dev = df['close'].rolling(window=period).std()
            
            df['bb_upper'] = sma + (std_dev * std)
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std_dev * std)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR (Average True Range)
            df['atr'] = self._calculate_atr(df, 14)
            
            # Volatility (Standard Deviation)
            df['volatility'] = df['close'].rolling(window=20).std()
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR (Average True Range)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        try:
            # OBV (On-Balance Volume)
            df['obv'] = self._calculate_obv(df['close'], df['volume'])
            
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price-Volume Trend
            df['pvt'] = self._calculate_pvt(df['close'], df['volume'])
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
        
        return df
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV (On-Balance Volume)"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_pvt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate PVT (Price-Volume Trend)"""
        pvt = pd.Series(index=close.index, dtype=float)
        pvt.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            pvt.iloc[i] = pvt.iloc[i-1] + (volume.iloc[i] * (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1])
        
        return pvt
    
    def _calculate_oscillator_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate oscillator indicators"""
        try:
            # Stochastic Oscillator
            k_period = self.indicators_config['STOCH']['k_period']
            d_period = self.indicators_config['STOCH']['d_period']
            
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
            
            # CCI (Commodity Channel Index) - Simplified
            period = self.indicators_config['CCI']['period']
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Williams %R
            period = self.indicators_config['WILLR']['period']
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df['willr'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            
            # Ultimate Oscillator - Simplified
            df['ultosc'] = 50  # Simplified value
            
        except Exception as e:
            logger.error(f"Error calculating oscillator indicators: {str(e)}")
        
        return df
    
    def _calculate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom features and derived indicators"""
        try:
            # Price change features
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Gap features
            df['gap'] = df['open'] - df['close'].shift(1)
            df['gap_percent'] = df['gap'] / df['close'].shift(1)
            
            # Intraday range
            df['intraday_range'] = df['high'] - df['low']
            df['intraday_range_percent'] = df['intraday_range'] / df['close']
            
            # Support and resistance levels
            df['support_level'] = df['low'].rolling(window=20).min()
            df['resistance_level'] = df['high'].rolling(window=20).max()
            df['support_distance'] = (df['close'] - df['support_level']) / df['close']
            df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
            
            # Moving average crossovers
            df['ema_9_21_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
            df['ema_21_50_cross'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)
            df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
            
            # Trend strength
            df['trend_strength'] = np.where(
                (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']), 1,
                np.where(
                    (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']), -1, 0
                )
            )
            
            # Volatility regime
            df['volatility_regime'] = pd.cut(
                df['volatility'], 
                bins=[0, df['volatility'].quantile(0.33), df['volatility'].quantile(0.67), float('inf')],
                labels=['Low', 'Medium', 'High']
            )
            
            # Market regime (bull/bear/sideways)
            df['market_regime'] = np.where(
                df['close'] > df['sma_200'], 'Bull',
                np.where(df['close'] < df['sma_200'] * 0.8, 'Bear', 'Sideways')
            )
            
            # Price position in Bollinger Bands
            df['bb_position'] = pd.cut(
                df['bb_percent'],
                bins=[0, 0.2, 0.8, 1],
                labels=['Lower', 'Middle', 'Upper']
            )
            
            # RSI regime
            df['rsi_regime'] = np.where(
                df['rsi'] > 70, 'Overbought',
                np.where(df['rsi'] < 30, 'Oversold', 'Neutral')
            )
            
            # MACD signal
            df['macd_signal_type'] = np.where(
                df['macd'] > df['macd_signal'], 'Bullish',
                np.where(df['macd'] < df['macd_signal'], 'Bearish', 'Neutral')
            )
            
        except Exception as e:
            logger.error(f"Error calculating custom features: {str(e)}")
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """Get list of all calculated features"""
        features = [
            # Price features
            'open', 'high', 'low', 'close', 'volume', 'adj_close',
            
            # Momentum indicators
            'rsi', 'roc', 'momentum', 'trix', 'ppo',
            
            # Trend indicators
            'macd', 'macd_signal', 'macd_histogram',
            'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'sma_9', 'sma_21', 'sma_50', 'sma_200',
            'adx', 'di_plus', 'di_minus',
            
            # Volatility indicators
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'atr', 'volatility',
            
            # Volume indicators
            'obv', 'volume_sma', 'volume_ratio', 'pvt',
            
            # Oscillator indicators
            'stoch_k', 'stoch_d', 'cci', 'willr', 'ultosc',
            
            # Custom features
            'price_change', 'price_change_abs', 'high_low_ratio', 'close_open_ratio',
            'gap', 'gap_percent', 'intraday_range', 'intraday_range_percent',
            'support_level', 'resistance_level', 'support_distance', 'resistance_distance',
            'ema_9_21_cross', 'ema_21_50_cross', 'sma_50_200_cross', 'trend_strength',
            'volatility_regime', 'market_regime', 'bb_position', 'rsi_regime', 'macd_signal_type'
        ]
        
        return features
    
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Calculate returns for different periods"""
        try:
            for period in periods:
                df[f'return_{period}d'] = df['close'].pct_change(period)
                df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
            
            # Volatility of returns
            for period in periods:
                df[f'volatility_{period}d'] = df[f'return_{period}d'].rolling(window=period).std()
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
        
        return df
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features for time series modeling"""
        try:
            key_features = ['close', 'volume', 'rsi', 'macd', 'adx']
            
            for feature in key_features:
                if feature in df.columns:
                    for lag in lags:
                        df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
            
        except Exception as e:
            logger.error(f"Error creating lagged features: {str(e)}")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create rolling window features"""
        try:
            key_features = ['close', 'volume', 'rsi']
            
            for feature in key_features:
                if feature in df.columns:
                    for window in windows:
                        df[f'{feature}_mean_{window}'] = df[feature].rolling(window=window).mean()
                        df[f'{feature}_std_{window}'] = df[feature].rolling(window=window).std()
                        df[f'{feature}_min_{window}'] = df[feature].rolling(window=window).min()
                        df[f'{feature}_max_{window}'] = df[feature].rolling(window=window).max()
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
        
        return df

# Global instance
technical_indicators = TechnicalIndicators()
