"""
Ensemble model combining XGBoost, Informer, and FinBERT predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from src.models.xgboost_model import xgboost_predictor
from src.models.informer_model import informer_predictor
from src.models.sentiment_analyzer import sentiment_analyzer
from src.data.yahoo_finance import yahoo_data
from src.data.news_scraper import news_scraper

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Prediction result with confidence and metadata"""
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    confidence: float
    recommendation: str
    timeframe: str
    prediction_date: datetime
    model_predictions: Dict[str, float]
    model_confidences: Dict[str, float]
    sentiment_score: float
    technical_signals: Dict[str, Any]
    risk_level: str

class EnsembleStockPredictor:
    """Ensemble model combining multiple prediction approaches"""
    
    def __init__(self):
        self.model_weights = {
            'xgboost': 0.4,      # Tabular features
            'informer': 0.4,     # Time series patterns
            'sentiment': 0.2     # News sentiment
        }
        
        self.model_path = settings.MODELS_DIR / "ensemble_model.pkl"
        self.weights_path = settings.MODELS_DIR / "ensemble_weights.pkl"
        
        # Recommendation thresholds
        self.recommendation_thresholds = {
            'strong_buy': 0.05,    # >5% predicted increase
            'buy': 0.02,           # >2% predicted increase
            'hold': -0.02,         # -2% to 2% predicted change
            'sell': -0.05,         # <-5% predicted decrease
            'strong_sell': -0.1    # <-10% predicted decrease
        }
        
        # Risk level thresholds
        self.risk_thresholds = {
            'low': 0.3,      # <30% volatility
            'medium': 0.6,   # 30-60% volatility
            'high': 0.8      # >60% volatility
        }
    
    def predict(self, symbol: str, timeframe: str = 'short', 
                days_ahead: int = None) -> PredictionResult:
        """
        Make ensemble prediction for a stock
        
        Args:
            symbol: Stock symbol
            timeframe: Prediction timeframe
            days_ahead: Number of days to predict ahead
        
        Returns:
            PredictionResult object
        """
        try:
            logger.info(f"Making ensemble prediction for {symbol} ({timeframe})")
            
            # Get timeframe configuration
            timeframe_config = settings.TIMEFRAMES.get(timeframe, settings.TIMEFRAMES['short'])
            if days_ahead is None:
                days_ahead = timeframe_config['prediction_days']
            
            # Get current data
            current_data = self._get_current_data(symbol, timeframe_config)
            if current_data is None:
                raise ValueError(f"Could not retrieve data for {symbol}")
            
            # Get individual model predictions
            xgboost_pred = self._get_xgboost_prediction(symbol, current_data, days_ahead)
            informer_pred = self._get_informer_prediction(symbol, current_data, days_ahead)
            sentiment_pred = self._get_sentiment_prediction(symbol, current_data)
            
            # Combine predictions
            ensemble_prediction = self._combine_predictions(
                xgboost_pred, informer_pred, sentiment_pred
            )
            
            # Calculate confidence
            confidence = self._calculate_ensemble_confidence(
                xgboost_pred, informer_pred, sentiment_pred
            )
            
            # Get current price
            current_price = current_data['close'].iloc[-1]
            
            # Calculate predicted price and change
            predicted_price = current_price * (1 + ensemble_prediction)
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Generate recommendation
            recommendation = self._generate_recommendation(price_change_percent)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(current_data, confidence)
            
            # Get technical signals
            technical_signals = self._get_technical_signals(current_data)
            
            # Coerce model outputs to plain floats for API schema
            def _to_float(value, default=0.0):
                try:
                    import numpy as np
                    if hasattr(value, 'item'):
                        return float(value.item())
                    if isinstance(value, (list, tuple)):
                        return float(value[0]) if value else default
                    if 'numpy' in str(type(value)):
                        return float(np.array(value).flatten()[0])
                    return float(value)
                except Exception:
                    return default

            def _to_native_type(value):
                """Convert numpy types to native Python types"""
                try:
                    import numpy as np
                    if hasattr(value, 'item'):
                        return value.item()
                    if isinstance(value, np.ndarray):
                        return value.tolist()
                    if isinstance(value, (np.integer, np.floating)):
                        return float(value)
                    if isinstance(value, (np.bool_, bool)):
                        return bool(value)
                    return value
                except Exception:
                    return value

            mp_x = _to_float(xgboost_pred.get('prediction', 0.0))
            mp_i = _to_float(informer_pred.get('prediction', 0.0))
            mp_s = _to_float(sentiment_pred.get('prediction', 0.0))

            mc_x = _to_float(xgboost_pred.get('confidence', 0.5), 0.5)
            mc_i = _to_float(informer_pred.get('confidence', 0.5), 0.5)
            mc_s = _to_float(sentiment_pred.get('confidence', 0.5), 0.5)

            # Convert all values to native Python types
            current_price = _to_native_type(current_price)
            predicted_price = _to_native_type(predicted_price)
            price_change = _to_native_type(price_change)
            price_change_percent = _to_native_type(price_change_percent)
            confidence = _to_native_type(confidence)
            sentiment_score = _to_native_type(sentiment_pred.get('sentiment_score', 0.0))

            # Create result
            result = PredictionResult(
                symbol=symbol,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                confidence=confidence,
                recommendation=recommendation,
                timeframe=timeframe,
                prediction_date=datetime.now(),
                model_predictions={
                    'xgboost': mp_x,
                    'informer': mp_i,
                    'sentiment': mp_s
                },
                model_confidences={
                    'xgboost': mc_x,
                    'informer': mc_i,
                    'sentiment': mc_s
                },
                sentiment_score=sentiment_pred.get('sentiment_score', 0.0),
                technical_signals=technical_signals,
                risk_level=risk_level
            )
            
            logger.info(f"Ensemble prediction completed for {symbol}: {recommendation}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction for {symbol}: {str(e)}")
            raise
    
    def _get_current_data(self, symbol: str, timeframe_config: Dict) -> Optional[pd.DataFrame]:
        """Get current data for the symbol"""
        try:
            # Fetch OHLCV data
            ohlcv_data = yahoo_data.fetch_ohlcv_data(
                symbol, 
                period=timeframe_config['period'],
                interval=timeframe_config['interval']
            )
            
            if ohlcv_data.empty:
                return None
            
            # Calculate technical indicators
            from src.features.technical_indicators import technical_indicators
            data_with_indicators = technical_indicators.calculate_all_indicators(ohlcv_data)
            
            return data_with_indicators
            
        except Exception as e:
            logger.error(f"Error getting current data for {symbol}: {str(e)}")
            return None
    
    def _get_xgboost_prediction(self, symbol: str, data: pd.DataFrame, 
                               days_ahead: int) -> Dict[str, float]:
        """Get XGBoost prediction"""
        try:
            # Load XGBoost model if not loaded
            if xgboost_predictor.model is None:
                xgboost_predictor.load_model()
            
            # Fetch latest fundamentals
            try:
                from src.data.yahoo_finance import yahoo_data as _yahoo
                fundamentals = _yahoo.fetch_fundamental_data(symbol, use_cache=True)
            except Exception:
                fundamentals = None

            # Get latest features (without preparing all features, just get the raw data)
            latest_features = data.iloc[-1:]
            
            # Make prediction with fundamental data
            prediction = xgboost_predictor.predict(latest_features, fundamental_data=fundamentals)[0]
            confidence = xgboost_predictor.get_prediction_confidence(latest_features, fundamental_data=fundamentals)[0]
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting XGBoost prediction: {str(e)}")
            return {'prediction': 0.0, 'confidence': 0.5}
    
    def _get_informer_prediction(self, symbol: str, data: pd.DataFrame, 
                                days_ahead: int) -> Dict[str, float]:
        """Get Informer prediction"""
        try:
            # Load Informer model if not loaded
            if informer_predictor.model is None:
                informer_predictor.load_model()
            
            # Prepare data for Informer
            X, _ = informer_predictor.prepare_data(data, target_column='close')
            
            if len(X) == 0:
                return {'prediction': 0.0, 'confidence': 0.5}
            
            # Get latest sequence
            latest_sequence = X[-1:]
            
            # Make prediction
            prediction = informer_predictor.predict(latest_sequence)[0]
            confidence = informer_predictor.get_prediction_confidence(latest_sequence)[0]
            
            # Convert to percentage change
            current_price = data['close'].iloc[-1]
            predicted_price = prediction[0] if len(prediction) > 0 else current_price
            price_change_percent = (predicted_price - current_price) / current_price
            
            return {
                'prediction': price_change_percent,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting Informer prediction: {str(e)}")
            return {'prediction': 0.0, 'confidence': 0.5}
    
    def _get_sentiment_prediction(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Get sentiment-based prediction"""
        try:
            # Get recent news articles
            recent_articles = news_scraper.get_recent_articles(days=7, symbols=[symbol])
            
            if not recent_articles:
                return {'prediction': 0.0, 'confidence': 0.5, 'sentiment_score': 0.0}
            
            # Analyze sentiment for each article
            sentiment_scores = []
            for article in recent_articles:
                analyzed_article = sentiment_analyzer.analyze_article(article)
                if analyzed_article.sentiment_score is not None:
                    sentiment_scores.append(analyzed_article.sentiment_score)
            
            if not sentiment_scores:
                return {'prediction': 0.0, 'confidence': 0.5, 'sentiment_score': 0.0}
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiment_scores)
            
            # Convert sentiment to price prediction (simplified mapping)
            # Positive sentiment -> positive price change, negative sentiment -> negative price change
            sentiment_prediction = avg_sentiment * 0.02  # Scale sentiment to price change
            
            # Calculate confidence based on sentiment consistency
            sentiment_std = np.std(sentiment_scores)
            confidence = max(0.1, 1.0 - sentiment_std)
            
            return {
                'prediction': sentiment_prediction,
                'confidence': confidence,
                'sentiment_score': avg_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment prediction: {str(e)}")
            return {'prediction': 0.0, 'confidence': 0.5, 'sentiment_score': 0.0}
    
    def _combine_predictions(self, xgboost_pred: Dict, informer_pred: Dict, 
                           sentiment_pred: Dict) -> float:
        """Combine individual model predictions"""
        try:
            # Weighted average of predictions
            weighted_prediction = (
                self.model_weights['xgboost'] * xgboost_pred['prediction'] +
                self.model_weights['informer'] * informer_pred['prediction'] +
                self.model_weights['sentiment'] * sentiment_pred['prediction']
            )
            
            return weighted_prediction
            
        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return 0.0
    
    def _calculate_ensemble_confidence(self, xgboost_pred: Dict, informer_pred: Dict, 
                                     sentiment_pred: Dict) -> float:
        """Calculate ensemble confidence"""
        try:
            # Weighted average of individual confidences
            # Safely extract scalar confidences
            def _scalar(value, default=0.5):
                try:
                    import numpy as np
                    if hasattr(value, 'item'):
                        return float(value.item())
                    if isinstance(value, (list, tuple)):
                        return float(value[0]) if value else default
                    if 'numpy' in str(type(value)):
                        return float(np.array(value).flatten()[0])
                    return float(value)
                except Exception:
                    return default

            x_conf = _scalar(xgboost_pred.get('confidence', 0.5))
            i_conf = _scalar(informer_pred.get('confidence', 0.5))
            s_conf = _scalar(sentiment_pred.get('confidence', 0.5))

            weighted_confidence = (
                self.model_weights['xgboost'] * x_conf +
                self.model_weights['informer'] * i_conf +
                self.model_weights['sentiment'] * s_conf
            )
            
            # Adjust confidence based on prediction agreement
            import numpy as np
            def _scalar_pred(v, default=0.0):
                try:
                    if hasattr(v, 'item'):
                        return float(v.item())
                    if isinstance(v, (list, tuple)):
                        return float(v[0]) if v else default
                    if 'numpy' in str(type(v)):
                        return float(np.array(v).flatten()[0])
                    return float(v)
                except Exception:
                    return default

            predictions = [
                _scalar_pred(xgboost_pred.get('prediction', 0.0)),
                _scalar_pred(informer_pred.get('prediction', 0.0)),
                _scalar_pred(sentiment_pred.get('prediction', 0.0)),
            ]

            prediction_std = float(np.std(predictions))
            agreement_factor = max(0.1, 1.0 - prediction_std)
            
            final_confidence = weighted_confidence * agreement_factor
            
            return min(1.0, max(0.1, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {str(e)}")
            return 0.5
    
    def _generate_recommendation(self, price_change_percent: float) -> str:
        """Generate buy/sell/hold recommendation"""
        try:
            if price_change_percent >= self.recommendation_thresholds['strong_buy']:
                return 'Strong Buy'
            elif price_change_percent >= self.recommendation_thresholds['buy']:
                return 'Buy'
            elif price_change_percent <= self.recommendation_thresholds['strong_sell']:
                return 'Strong Sell'
            elif price_change_percent <= self.recommendation_thresholds['sell']:
                return 'Sell'
            else:
                return 'Hold'
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return 'Hold'
    
    def _calculate_risk_level(self, data: pd.DataFrame, confidence: float) -> str:
        """Calculate risk level based on volatility and confidence"""
        try:
            # Calculate recent volatility
            recent_returns = data['close'].pct_change().dropna()
            volatility = recent_returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Adjust risk based on confidence
            adjusted_volatility = volatility * (1.0 - confidence)
            
            if adjusted_volatility <= self.risk_thresholds['low']:
                return 'Low'
            elif adjusted_volatility <= self.risk_thresholds['medium']:
                return 'Medium'
            else:
                return 'High'
                
        except Exception as e:
            logger.error(f"Error calculating risk level: {str(e)}")
            return 'Medium'
    
    def _get_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get technical analysis signals"""
        try:
            latest = data.iloc[-1]
            
            def _to_native_type(value):
                """Convert numpy types to native Python types"""
                try:
                    import numpy as np
                    if hasattr(value, 'item'):
                        return value.item()
                    if isinstance(value, (np.integer, np.floating)):
                        return float(value)
                    if isinstance(value, (np.bool_, bool)):
                        return bool(value)
                    return value
                except Exception:
                    return value
            
            signals = {
                'rsi': _to_native_type(latest.get('rsi', 50)),
                'macd_signal': str(latest.get('macd_signal_type', 'Neutral')),
                'trend_strength': _to_native_type(latest.get('trend_strength', 0)),
                'bb_position': str(latest.get('bb_position', 'Middle')),
                'volume_ratio': _to_native_type(latest.get('volume_ratio', 1.0)),
                'adx': _to_native_type(latest.get('adx', 25))
            }
            
            # Generate signal interpretations
            signals['rsi_signal'] = (
                'Overbought' if signals['rsi'] > 70 else
                'Oversold' if signals['rsi'] < 30 else 'Neutral'
            )
            
            signals['trend_signal'] = (
                'Strong Bullish' if signals['trend_strength'] == 1 else
                'Strong Bearish' if signals['trend_strength'] == -1 else 'Sideways'
            )
            
            signals['volume_signal'] = (
                'High Volume' if signals['volume_ratio'] > 1.5 else
                'Low Volume' if signals['volume_ratio'] < 0.5 else 'Normal Volume'
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting technical signals: {str(e)}")
            return {}
    
    def predict_multiple_symbols(self, symbols: List[str], timeframe: str = 'short') -> List[PredictionResult]:
        """Predict multiple symbols"""
        results = []
        
        for symbol in symbols:
            try:
                result = self.predict(symbol, timeframe)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_top_picks(self, symbols: List[str], timeframe: str = 'short', 
                     top_n: int = 5) -> List[PredictionResult]:
        """Get top N stock picks based on predictions"""
        try:
            # Get predictions for all symbols
            predictions = self.predict_multiple_symbols(symbols, timeframe)
            
            # Sort by predicted return and confidence
            sorted_predictions = sorted(
                predictions,
                key=lambda x: x.price_change_percent * x.confidence,
                reverse=True
            )
            
            return sorted_predictions[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting top picks: {str(e)}")
            return []
    
    def save_model(self):
        """Save ensemble model configuration"""
        try:
            settings.MODELS_DIR.mkdir(exist_ok=True)
            
            # Save model weights
            joblib.dump(self.model_weights, self.weights_path)
            
            logger.info(f"Ensemble model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble model: {str(e)}")
    
    def load_model(self):
        """Load ensemble model configuration"""
        try:
            if self.weights_path.exists():
                self.model_weights = joblib.load(self.weights_path)
                logger.info("Ensemble model weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {str(e)}")

# Global instance
ensemble_predictor = EnsembleStockPredictor()
