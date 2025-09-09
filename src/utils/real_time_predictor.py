"""
Real-time prediction support module
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
import threading
from dataclasses import dataclass
import json
from pathlib import Path

from config.settings import settings
from src.models.ensemble_model import ensemble_predictor
from src.data.yahoo_finance import yahoo_data
from src.data.news_scraper import news_scraper
from src.models.sentiment_analyzer import sentiment_analyzer

logger = logging.getLogger(__name__)

@dataclass
class RealTimePrediction:
    """Real-time prediction data structure"""
    symbol: str
    timestamp: datetime
    prediction: Dict[str, Any]
    confidence: float
    alert_level: str
    price_change_threshold: float

class RealTimePredictor:
    """Real-time prediction manager"""
    
    def __init__(self):
        self.is_running = False
        self.predictions_cache = {}
        self.alerts = []
        self.callbacks = []
        self.update_interval = 300  # 5 minutes
        self.alert_thresholds = {
            'high_confidence': 0.8,
            'significant_change': 0.05,  # 5% price change
            'risk_alert': 0.7
        }
        
        # Real-time data sources
        self.monitored_symbols = settings.DEFAULT_SYMBOLS[:5]  # Monitor top 5 symbols
        self.last_update_times = {}
        
        # Cache directory
        self.cache_dir = settings.DATA_DIR / "real_time_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def start(self):
        """Start real-time prediction service"""
        if self.is_running:
            logger.warning("Real-time predictor is already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time prediction service...")
        
        # Start background thread
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        # Schedule periodic updates
        schedule.every(5).minutes.do(self._update_predictions)
        schedule.every(1).hour.do(self._update_news_sentiment)
        schedule.every(30).minutes.do(self._check_alerts)
        
        logger.info("Real-time prediction service started")
    
    def stop(self):
        """Stop real-time prediction service"""
        if not self.is_running:
            logger.warning("Real-time predictor is not running")
            return
        
        self.is_running = False
        schedule.clear()
        logger.info("Real-time prediction service stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in background thread"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def _update_predictions(self):
        """Update predictions for monitored symbols"""
        try:
            logger.info("Updating real-time predictions...")
            
            for symbol in self.monitored_symbols:
                try:
                    # Get latest prediction
                    prediction_result = ensemble_predictor.predict(
                        symbol=symbol,
                        timeframe='intraday'  # Use intraday for real-time
                    )
                    
                    # Store in cache
                    self.predictions_cache[symbol] = {
                        'prediction': prediction_result,
                        'timestamp': datetime.now(),
                        'confidence': prediction_result.confidence,
                        'recommendation': prediction_result.recommendation,
                        'risk_level': prediction_result.risk_level
                    }
                    
                    # Update last update time
                    self.last_update_times[symbol] = datetime.now()
                    
                    # Check for alerts
                    self._check_prediction_alerts(symbol, prediction_result)
                    
                    logger.info(f"Updated prediction for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error updating prediction for {symbol}: {str(e)}")
                    continue
            
            # Save cache to disk
            self._save_cache()
            
        except Exception as e:
            logger.error(f"Error in real-time prediction update: {str(e)}")
    
    def _update_news_sentiment(self):
        """Update news sentiment analysis"""
        try:
            logger.info("Updating news sentiment...")
            
            # Get recent news for monitored symbols
            articles = news_scraper.get_recent_articles(days=1, symbols=self.monitored_symbols)
            
            if articles:
                # Analyze sentiment
                sentiment_scores = []
                for article in articles:
                    analyzed_article = sentiment_analyzer.analyze_article(article)
                    if analyzed_article.sentiment_score is not None:
                        sentiment_scores.append(analyzed_article.sentiment_score)
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    
                    # Store sentiment data
                    sentiment_data = {
                        'timestamp': datetime.now(),
                        'average_sentiment': avg_sentiment,
                        'article_count': len(articles),
                        'sentiment_scores': sentiment_scores
                    }
                    
                    # Save sentiment data
                    sentiment_file = self.cache_dir / "sentiment_data.json"
                    with open(sentiment_file, 'w') as f:
                        json.dump(sentiment_data, f, default=str)
                    
                    logger.info(f"Updated sentiment: {avg_sentiment:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating news sentiment: {str(e)}")
    
    def _check_prediction_alerts(self, symbol: str, prediction_result):
        """Check for prediction alerts"""
        try:
            alerts = []
            
            # High confidence alert
            if prediction_result.confidence >= self.alert_thresholds['high_confidence']:
                alerts.append({
                    'type': 'high_confidence',
                    'symbol': symbol,
                    'message': f"High confidence prediction for {symbol}: {prediction_result.recommendation}",
                    'confidence': prediction_result.confidence,
                    'timestamp': datetime.now()
                })
            
            # Significant price change alert
            if abs(prediction_result.price_change_percent) >= self.alert_thresholds['significant_change'] * 100:
                alerts.append({
                    'type': 'significant_change',
                    'symbol': symbol,
                    'message': f"Significant price change predicted for {symbol}: {prediction_result.price_change_percent:.2f}%",
                    'change_percent': prediction_result.price_change_percent,
                    'timestamp': datetime.now()
                })
            
            # Risk alert
            if prediction_result.risk_level == 'High':
                alerts.append({
                    'type': 'high_risk',
                    'symbol': symbol,
                    'message': f"High risk detected for {symbol}",
                    'risk_level': prediction_result.risk_level,
                    'timestamp': datetime.now()
                })
            
            # Add alerts to list
            self.alerts.extend(alerts)
            
            # Trigger callbacks
            for callback in self.callbacks:
                try:
                    callback(alerts)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error checking prediction alerts: {str(e)}")
    
    def _check_alerts(self):
        """Check for various alerts"""
        try:
            # Check for stale predictions
            stale_symbols = []
            for symbol, last_update in self.last_update_times.items():
                if datetime.now() - last_update > timedelta(hours=1):
                    stale_symbols.append(symbol)
            
            if stale_symbols:
                alert = {
                    'type': 'stale_data',
                    'symbols': stale_symbols,
                    'message': f"Stale data detected for symbols: {', '.join(stale_symbols)}",
                    'timestamp': datetime.now()
                }
                self.alerts.append(alert)
            
            # Clean old alerts (keep last 100)
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    def _save_cache(self):
        """Save prediction cache to disk"""
        try:
            cache_data = {
                'predictions': {
                    symbol: {
                        'prediction': {
                            'symbol': pred['prediction'].symbol,
                            'current_price': pred['prediction'].current_price,
                            'predicted_price': pred['prediction'].predicted_price,
                            'price_change_percent': pred['prediction'].price_change_percent,
                            'recommendation': pred['prediction'].recommendation,
                            'confidence': pred['prediction'].confidence,
                            'risk_level': pred['prediction'].risk_level
                        },
                        'timestamp': pred['timestamp'].isoformat(),
                        'confidence': pred['confidence'],
                        'recommendation': pred['recommendation'],
                        'risk_level': pred['risk_level']
                    }
                    for symbol, pred in self.predictions_cache.items()
                },
                'last_update_times': {
                    symbol: timestamp.isoformat()
                    for symbol, timestamp in self.last_update_times.items()
                },
                'alerts': [
                    {
                        'type': alert['type'],
                        'symbol': alert.get('symbol', ''),
                        'message': alert['message'],
                        'timestamp': alert['timestamp'].isoformat()
                    }
                    for alert in self.alerts[-50:]  # Keep last 50 alerts
                ]
            }
            
            cache_file = self.cache_dir / "predictions_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def load_cache(self):
        """Load prediction cache from disk"""
        try:
            cache_file = self.cache_dir / "predictions_cache.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Load predictions
                for symbol, pred_data in cache_data.get('predictions', {}).items():
                    self.predictions_cache[symbol] = {
                        'prediction': pred_data['prediction'],
                        'timestamp': datetime.fromisoformat(pred_data['timestamp']),
                        'confidence': pred_data['confidence'],
                        'recommendation': pred_data['recommendation'],
                        'risk_level': pred_data['risk_level']
                    }
                
                # Load update times
                for symbol, timestamp_str in cache_data.get('last_update_times', {}).items():
                    self.last_update_times[symbol] = datetime.fromisoformat(timestamp_str)
                
                # Load alerts
                for alert_data in cache_data.get('alerts', []):
                    alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                    self.alerts.append(alert_data)
                
                logger.info("Loaded prediction cache from disk")
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
    
    def get_latest_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest prediction for a symbol"""
        return self.predictions_cache.get(symbol)
    
    def get_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached predictions"""
        return self.predictions_cache.copy()
    
    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:] if self.alerts else []
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remove alert callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_prediction_history(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get prediction history for a symbol"""
        try:
            history_file = self.cache_dir / f"{symbol}_history.json"
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # Filter by time range
                cutoff_time = datetime.now() - timedelta(hours=hours)
                filtered_history = [
                    entry for entry in history_data
                    if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
                ]
                
                return filtered_history
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting prediction history for {symbol}: {str(e)}")
            return []
    
    def save_prediction_history(self, symbol: str, prediction_data: Dict[str, Any]):
        """Save prediction to history"""
        try:
            history_file = self.cache_dir / f"{symbol}_history.json"
            
            # Load existing history
            history_data = []
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
            
            # Add new entry
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction_data
            }
            history_data.append(history_entry)
            
            # Keep only last 1000 entries
            if len(history_data) > 1000:
                history_data = history_data[-1000:]
            
            # Save back to file
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving prediction history for {symbol}: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get real-time system status"""
        try:
            status = {
                'is_running': self.is_running,
                'monitored_symbols': self.monitored_symbols,
                'cache_size': len(self.predictions_cache),
                'alert_count': len(self.alerts),
                'last_update_times': {
                    symbol: timestamp.isoformat()
                    for symbol, timestamp in self.last_update_times.items()
                },
                'update_interval': self.update_interval,
                'alert_thresholds': self.alert_thresholds
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {}

# Global instance
real_time_predictor = RealTimePredictor()
