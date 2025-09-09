"""
XGBoost model for stock price prediction using tabular features
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
from pathlib import Path
from datetime import datetime
import sqlite3
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from src.features.technical_indicators import technical_indicators

logger = logging.getLogger(__name__)

class XGBoostStockPredictor:
    """XGBoost model for stock price prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.feature_names = None
        self.model_path = settings.MODELS_DIR / "xgboost_model.pkl"
        self.scaler_path = settings.MODELS_DIR / "xgboost_scaler.pkl"
        self.encoders_path = settings.MODELS_DIR / "xgboost_encoders.pkl"
        
        # Model parameters
        self.params = settings.XGBOOST_PARAMS.copy()
        
        # Feature configuration
        self.target_column = 'future_return'
        self.prediction_horizon = 1  # days ahead
        
        # Feature categories
        self.price_features = ['open', 'high', 'low', 'close', 'volume']
        self.technical_features = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'sma_9', 'sma_21', 'sma_50', 'sma_200',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'adx', 'di_plus', 'di_minus', 'atr', 'volatility',
            'obv', 'volume_sma', 'volume_ratio', 'pvt',
            'stoch_k', 'stoch_d', 'cci', 'willr', 'ultosc',
            'roc', 'momentum', 'trix', 'ppo'
        ]
        self.fundamental_features = [
            'pe_ratio', 'pb_ratio', 'roe', 'roi', 'market_cap',
            'debt_to_equity', 'current_ratio', 'quick_ratio',
            'gross_margin', 'operating_margin', 'net_margin'
        ]
        self.sentiment_features = ['sentiment_score', 'sentiment_label']
        self.custom_features = [
            'price_change', 'price_change_abs', 'high_low_ratio', 'close_open_ratio',
            'gap', 'gap_percent', 'intraday_range', 'intraday_range_percent',
            'support_distance', 'resistance_distance',
            'ema_9_21_cross', 'ema_21_50_cross', 'sma_50_200_cross', 'trend_strength'
        ]
    
    def prepare_features(self, df: pd.DataFrame, fundamental_data: Dict = None, 
                        sentiment_data: Dict = None) -> pd.DataFrame:
        """
        Prepare features for XGBoost model
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            fundamental_data: Dictionary with fundamental metrics
            sentiment_data: Dictionary with sentiment metrics
        
        Returns:
            DataFrame with prepared features
        """
        try:
            # Start with a copy of the data
            features_df = df.copy()
            
            # Calculate technical indicators if not present
            if not any(col in features_df.columns for col in self.technical_features):
                logger.info("Calculating technical indicators...")
                features_df = technical_indicators.calculate_all_indicators(features_df)
            
            # Add fundamental features
            if fundamental_data:
                features_df = self._add_fundamental_features(features_df, fundamental_data)
            
            # Add sentiment features
            if sentiment_data:
                features_df = self._add_sentiment_features(features_df, sentiment_data)
            
            # Create target variable (future returns)
            features_df = self._create_target_variable(features_df)
            
            # Add lagged features
            features_df = technical_indicators.create_lagged_features(features_df)
            
            # Add rolling features
            features_df = technical_indicators.create_rolling_features(features_df)
            
            # Add returns for different periods
            features_df = technical_indicators.calculate_returns(features_df)
            
            # Handle categorical features
            features_df = self._handle_categorical_features(features_df)
            
            # Select final features
            features_df = self._select_features(features_df)
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            logger.info(f"Prepared {len(features_df.columns)} features for {len(features_df)} samples")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def _add_fundamental_features(self, df: pd.DataFrame, fundamental_data: Dict) -> pd.DataFrame:
        """Add fundamental features to DataFrame"""
        try:
            for feature in self.fundamental_features:
                if feature in fundamental_data:
                    df[feature] = fundamental_data[feature]
                else:
                    df[feature] = np.nan
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding fundamental features: {str(e)}")
            return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, sentiment_data: Dict) -> pd.DataFrame:
        """Add sentiment features to DataFrame"""
        try:
            for feature in self.sentiment_features:
                if feature in sentiment_data:
                    df[feature] = sentiment_data[feature]
                else:
                    df[feature] = 0.0 if feature == 'sentiment_score' else 'neutral'
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {str(e)}")
            return df
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable (future returns)"""
        try:
            # Calculate future returns
            df[self.target_column] = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
            
            # Create classification target (up/down)
            df['future_direction'] = np.where(df[self.target_column] > 0, 1, 0)
            
            # Create volatility target
            df['future_volatility'] = df[self.target_column].rolling(window=5).std().shift(-self.prediction_horizon)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
            return df
    
    def _handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical features by encoding them"""
        try:
            categorical_columns = [
                'volatility_regime', 'market_regime', 'bb_position', 
                'rsi_regime', 'macd_signal_type', 'sentiment_label'
            ]
            
            for col in categorical_columns:
                if col in df.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling categorical features: {str(e)}")
            return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for the model"""
        try:
            # Combine all feature categories
            all_features = (self.price_features + self.technical_features + 
                          self.fundamental_features + self.sentiment_features + 
                          self.custom_features)
            
            # Add lagged and rolling features
            lagged_features = [col for col in df.columns if '_lag_' in col]
            rolling_features = [col for col in df.columns if '_mean_' in col or '_std_' in col or '_min_' in col or '_max_' in col]
            return_features = [col for col in df.columns if 'return_' in col or 'volatility_' in col]
            
            all_features.extend(lagged_features + rolling_features + return_features)
            
            # Select only features that exist in the DataFrame
            available_features = [col for col in all_features if col in df.columns]
            
            # Add target columns
            target_columns = [self.target_column, 'future_direction', 'future_volatility']
            available_features.extend([col for col in target_columns if col in df.columns])
            
            # Remove duplicates and sort
            available_features = sorted(list(set(available_features)))
            
            # Filter out non-numeric columns (except target columns)
            numeric_features = []
            for col in available_features:
                if col in target_columns:
                    numeric_features.append(col)
                elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_features.append(col)
                else:
                    logger.warning(f"Skipping non-numeric feature: {col} (dtype: {df[col].dtype})")
            
            self.feature_names = numeric_features
            
            return df[numeric_features]
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # For numerical features, fill with median
            numerical_features = df.select_dtypes(include=[np.number]).columns
            for col in numerical_features:
                if col not in [self.target_column, 'future_direction', 'future_volatility']:
                    df[col] = df[col].fillna(df[col].median())
            
            # Skip categorical features since we only use numeric features
            
            # Drop rows where target is missing
            df = df.dropna(subset=[self.target_column])
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, 
              validation_size: float = 0.2) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            df: DataFrame with features and target
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
        
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Starting XGBoost model training...")
            
            # Prepare data
            X = df.drop(columns=[self.target_column, 'future_direction', 'future_volatility'], errors='ignore')
            y = df[self.target_column]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Split data chronologically (time series split)
            split_idx = int(len(df) * (1 - test_size - validation_size))
            val_idx = int(len(df) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_val = X.iloc[split_idx:val_idx]
            X_test = X.iloc[val_idx:]
            
            y_train = y.iloc[:split_idx]
            y_val = y.iloc[split_idx:val_idx]
            y_test = y.iloc[val_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize XGBoost model
            self.model = xgb.XGBRegressor(**self.params)
            
            # Train model
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                verbose=False
            )
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_val_pred = self.model.predict(X_val_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
            test_metrics = self._calculate_metrics(y_test, y_test_pred)
            
            # Get feature importance
            self.feature_importance = self._get_feature_importance()
            
            # Save model
            self.save_model()
            
            logger.info("XGBoost model training completed successfully")
            
            return {
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': self.feature_importance,
                'model_params': self.params,
                'feature_count': len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: DataFrame with features
        
        Returns:
            Array of predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Prepare features
            X_processed = self._prepare_prediction_features(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X_processed)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _prepare_prediction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Select only the features used in training
            if self.feature_names:
                X_processed = X[self.feature_names].copy()
            else:
                X_processed = X.copy()
            
            # Handle categorical features
            categorical_columns = [
                'volatility_regime', 'market_regime', 'bb_position', 
                'rsi_regime', 'macd_signal_type', 'sentiment_label'
            ]
            
            for col in categorical_columns:
                if col in X_processed.columns and col in self.label_encoders:
                    try:
                        X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        X_processed[col] = 0
            
            # Handle missing values
            X_processed = X_processed.fillna(0)
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if self.model is None or self.feature_names is None:
                return {}
            
            importance_scores = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance_scores))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def save_model(self):
        """Save trained model and preprocessing objects"""
        try:
            # Create models directory
            settings.MODELS_DIR.mkdir(exist_ok=True)
            
            # Save model
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
            
            # Save scaler
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save label encoders
            joblib.dump(self.label_encoders, self.encoders_path)
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load trained model and preprocessing objects"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info("XGBoost model loaded successfully")
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Scaler loaded successfully")
            
            if self.encoders_path.exists():
                self.label_encoders = joblib.load(self.encoders_path)
                logger.info("Label encoders loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def get_prediction_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction confidence (using prediction variance)"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # For XGBoost, we can use the prediction variance as confidence
            # This is a simplified approach - in practice, you might want to use
            # ensemble methods or Bayesian approaches for better uncertainty quantification
            
            predictions = self.predict(X)
            
            # Calculate confidence based on feature importance and prediction magnitude
            confidence = np.ones_like(predictions) * 0.8  # Base confidence
            
            # Adjust confidence based on prediction magnitude
            confidence = np.where(np.abs(predictions) > 0.1, confidence * 0.9, confidence)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting prediction confidence: {str(e)}")
            return np.ones(len(X)) * 0.5  # Default confidence

# Global instance
xgboost_predictor = XGBoostStockPredictor()
