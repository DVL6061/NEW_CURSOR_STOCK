"""
Comprehensive test suite for the Enterprise Stock Forecasting System
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Import modules to test
from src.data.yahoo_finance import YahooFinanceData
from src.data.news_scraper import NewsScraper, NewsArticle
from src.features.technical_indicators import TechnicalIndicators
from src.models.sentiment_analyzer import FinBERTSentimentAnalyzer
from src.models.xgboost_model import XGBoostStockPredictor
from src.models.informer_model import InformerStockPredictor
from src.models.ensemble_model import EnsembleStockPredictor
from src.models.shap_explainer import SHAPExplainer

class TestYahooFinanceData:
    """Test Yahoo Finance data acquisition"""
    
    @pytest.fixture
    def yahoo_data(self):
        return YahooFinanceData()
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)) * 5,
            'high': 105 + np.random.randn(len(dates)) * 5,
            'low': 95 + np.random.randn(len(dates)) * 5,
            'close': 100 + np.random.randn(len(dates)) * 5,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return data
    
    def test_init_database(self, yahoo_data):
        """Test database initialization"""
        assert yahoo_data.db_path.exists()
    
    @patch('yfinance.Ticker')
    def test_fetch_ohlcv_data(self, mock_ticker, yahoo_data, sample_ohlcv_data):
        """Test OHLCV data fetching"""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_ohlcv_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test data fetching
        result = yahoo_data.fetch_ohlcv_data('TATAMOTORS.NS', period='1y')
        
        assert not result.empty
        assert 'symbol' in result.columns
        assert 'interval' in result.columns
        assert len(result) == len(sample_ohlcv_data)
    
    @patch('yfinance.Ticker')
    def test_fetch_fundamental_data(self, mock_ticker, yahoo_data):
        """Test fundamental data fetching"""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'trailingPE': 15.5,
            'priceToBook': 2.1,
            'returnOnEquity': 0.12,
            'marketCap': 1000000000
        }
        mock_ticker.return_value = mock_ticker_instance
        
        # Test fundamental data fetching
        result = yahoo_data.fetch_fundamental_data('TATAMOTORS.NS')
        
        assert 'symbol' in result
        assert 'pe_ratio' in result
        assert 'pb_ratio' in result
        assert result['pe_ratio'] == 15.5
    
    def test_get_latest_price(self, yahoo_data, sample_ohlcv_data):
        """Test getting latest price"""
        with patch.object(yahoo_data, 'fetch_ohlcv_data', return_value=sample_ohlcv_data):
            price = yahoo_data.get_latest_price('TATAMOTORS.NS')
            assert price is not None
            assert isinstance(price, float)

class TestNewsScraper:
    """Test news scraping functionality"""
    
    @pytest.fixture
    def news_scraper(self):
        return NewsScraper()
    
    @pytest.fixture
    def sample_article(self):
        return NewsArticle(
            title="Tata Motors reports strong Q3 results",
            content="Tata Motors has reported strong quarterly results...",
            url="https://example.com/news/tata-motors-q3",
            source="Economic Times",
            published_date=datetime.now(),
            sentiment_score=0.2,
            sentiment_label="positive"
        )
    
    def test_init_database(self, news_scraper):
        """Test database initialization"""
        assert news_scraper.db_path.exists()
    
    @patch('requests.Session.get')
    def test_scrape_article_content(self, mock_get, news_scraper):
        """Test article content scraping"""
        # Mock HTML response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<html><body><h1>Test Article</h1><p>Test content</p></body></html>'
        mock_get.return_value = mock_response
        
        # Test article scraping
        article = news_scraper._scrape_article_content('https://example.com/test', 'Test Source')
        
        assert article is not None
        assert article.title == "Test Article"
        assert article.content == "Test content"
    
    def test_save_articles(self, news_scraper, sample_article):
        """Test saving articles to database"""
        articles = [sample_article]
        news_scraper.save_articles(articles)
        
        # Verify article was saved
        recent_articles = news_scraper.get_recent_articles(days=1)
        assert len(recent_articles) >= 1
    
    def test_get_recent_articles(self, news_scraper, sample_article):
        """Test retrieving recent articles"""
        articles = [sample_article]
        news_scraper.save_articles(articles)
        
        recent_articles = news_scraper.get_recent_articles(days=7)
        assert len(recent_articles) >= 1
        assert recent_articles[0].title == sample_article.title

class TestTechnicalIndicators:
    """Test technical indicators calculation"""
    
    @pytest.fixture
    def technical_indicators(self):
        return TechnicalIndicators()
    
    @pytest.fixture
    def sample_data(self):
        """Sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)) * 5,
            'high': 105 + np.random.randn(len(dates)) * 5,
            'low': 95 + np.random.randn(len(dates)) * 5,
            'close': 100 + np.random.randn(len(dates)) * 5,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return data
    
    def test_calculate_all_indicators(self, technical_indicators, sample_data):
        """Test calculating all technical indicators"""
        result = technical_indicators.calculate_all_indicators(sample_data)
        
        assert not result.empty
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'ema_9' in result.columns
        assert 'bb_upper' in result.columns
        assert 'adx' in result.columns
    
    def test_calculate_returns(self, technical_indicators, sample_data):
        """Test returns calculation"""
        result = technical_indicators.calculate_returns(sample_data, periods=[1, 5])
        
        assert 'return_1d' in result.columns
        assert 'return_5d' in result.columns
        assert 'volatility_1d' in result.columns
        assert 'volatility_5d' in result.columns
    
    def test_create_lagged_features(self, technical_indicators, sample_data):
        """Test creating lagged features"""
        result = technical_indicators.create_lagged_features(sample_data, lags=[1, 2])
        
        assert 'close_lag_1' in result.columns
        assert 'close_lag_2' in result.columns
        assert 'volume_lag_1' in result.columns
    
    def test_get_feature_list(self, technical_indicators):
        """Test getting feature list"""
        features = technical_indicators.get_feature_list()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'close' in features
        assert 'rsi' in features
        assert 'macd' in features

class TestSentimentAnalyzer:
    """Test sentiment analysis functionality"""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        return FinBERTSentimentAnalyzer()
    
    @pytest.fixture
    def sample_text(self):
        return "Tata Motors reported strong quarterly results with revenue growth of 15%"
    
    @pytest.fixture
    def sample_article(self):
        return NewsArticle(
            title="Tata Motors Q3 Results",
            content="Strong quarterly performance with revenue growth",
            url="https://example.com/news",
            source="Economic Times",
            published_date=datetime.now()
        )
    
    @patch('torch.device')
    def test_analyze_sentiment(self, mock_device, sentiment_analyzer, sample_text):
        """Test sentiment analysis"""
        # Mock model and tokenizer
        with patch.object(sentiment_analyzer, 'model') as mock_model, \
             patch.object(sentiment_analyzer, 'tokenizer') as mock_tokenizer:
            
            # Mock tokenizer output
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
            
            # Mock model output
            mock_outputs = Mock()
            mock_outputs.logits = torch.tensor([[0.1, 0.8, 0.1]])  # Neutral prediction
            mock_model.return_value = mock_outputs
            
            result = sentiment_analyzer.analyze_sentiment(sample_text)
            
            assert result.text == sample_text
            assert result.sentiment_label in ['positive', 'negative', 'neutral']
            assert isinstance(result.sentiment_score, float)
            assert isinstance(result.confidence, float)
    
    def test_analyze_article(self, sentiment_analyzer, sample_article):
        """Test article sentiment analysis"""
        with patch.object(sentiment_analyzer, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = Mock(
                sentiment_score=0.2,
                sentiment_label="positive"
            )
            
            result = sentiment_analyzer.analyze_article(sample_article)
            
            assert result.sentiment_score == 0.2
            assert result.sentiment_label == "positive"
    
    def test_get_sentiment_summary(self, sentiment_analyzer):
        """Test sentiment summary generation"""
        articles = [
            Mock(sentiment_score=0.2, sentiment_label="positive"),
            Mock(sentiment_score=-0.1, sentiment_label="negative"),
            Mock(sentiment_score=0.0, sentiment_label="neutral")
        ]
        
        summary = sentiment_analyzer.get_sentiment_summary(articles)
        
        assert summary['total_articles'] == 3
        assert summary['positive_count'] == 1
        assert summary['negative_count'] == 1
        assert summary['neutral_count'] == 1
        assert isinstance(summary['average_sentiment'], float)

class TestXGBoostModel:
    """Test XGBoost model functionality"""
    
    @pytest.fixture
    def xgboost_predictor(self):
        return XGBoostStockPredictor()
    
    @pytest.fixture
    def sample_features_data(self):
        """Sample features data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)) * 5,
            'high': 105 + np.random.randn(len(dates)) * 5,
            'low': 95 + np.random.randn(len(dates)) * 5,
            'close': 100 + np.random.randn(len(dates)) * 5,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'rsi': 50 + np.random.randn(len(dates)) * 20,
            'macd': np.random.randn(len(dates)) * 2,
            'ema_9': 100 + np.random.randn(len(dates)) * 5,
            'ema_21': 100 + np.random.randn(len(dates)) * 5
        }, index=dates)
        
        return data
    
    def test_prepare_features(self, xgboost_predictor, sample_features_data):
        """Test feature preparation"""
        fundamental_data = {
            'pe_ratio': 15.5,
            'pb_ratio': 2.1,
            'roe': 0.12
        }
        
        sentiment_data = {
            'sentiment_score': 0.2,
            'sentiment_label': 'positive'
        }
        
        result = xgboost_predictor.prepare_features(
            sample_features_data, fundamental_data, sentiment_data
        )
        
        assert not result.empty
        assert xgboost_predictor.target_column in result.columns
        assert 'future_direction' in result.columns
    
    @patch('xgboost.XGBRegressor')
    def test_train(self, mock_xgb, xgboost_predictor, sample_features_data):
        """Test model training"""
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.randn(100)
        mock_xgb.return_value = mock_model
        
        # Prepare features
        features_df = xgboost_predictor.prepare_features(sample_features_data)
        
        # Train model
        result = xgboost_predictor.train(features_df)
        
        assert 'train_metrics' in result
        assert 'validation_metrics' in result
        assert 'test_metrics' in result
        assert 'feature_importance' in result
    
    def test_predict(self, xgboost_predictor, sample_features_data):
        """Test model prediction"""
        # Mock trained model
        xgboost_predictor.model = Mock()
        xgboost_predictor.model.predict.return_value = np.array([0.05])
        xgboost_predictor.scaler = Mock()
        xgboost_predictor.scaler.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        
        # Test prediction
        X = sample_features_data.iloc[-1:].drop(columns=['close'], errors='ignore')
        predictions = xgboost_predictor.predict(X)
        
        assert len(predictions) == 1
        assert isinstance(predictions[0], float)

class TestInformerModel:
    """Test Informer model functionality"""
    
    @pytest.fixture
    def informer_predictor(self):
        return InformerStockPredictor()
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Sample time series data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        }, index=dates)
        
        return data
    
    def test_prepare_data(self, informer_predictor, sample_time_series_data):
        """Test data preparation for Informer"""
        X, y = informer_predictor.prepare_data(sample_time_series_data)
        
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[1] == informer_predictor.seq_len
        assert y.shape[1] == informer_predictor.pred_len
    
    @patch('torch.nn.Module')
    def test_train(self, mock_module, informer_predictor, sample_time_series_data):
        """Test Informer model training"""
        # Mock model
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[1.0, 1.1, 1.2]])
        mock_module.return_value = mock_model
        
        # Prepare data
        X, y = informer_predictor.prepare_data(sample_time_series_data)
        
        # Train model
        result = informer_predictor.train(sample_time_series_data)
        
        assert 'train_metrics' in result
        assert 'validation_metrics' in result
        assert 'train_losses' in result
        assert 'val_losses' in result
    
    def test_predict_future(self, informer_predictor, sample_time_series_data):
        """Test future prediction"""
        # Mock trained model
        informer_predictor.model = Mock()
        informer_predictor.model.return_value = torch.tensor([[105.0]])
        informer_predictor.scaler = Mock()
        informer_predictor.scaler.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        informer_predictor.scaler.inverse_transform.return_value = np.array([[105.0]])
        
        # Test future prediction
        predictions = informer_predictor.predict_future(sample_time_series_data, steps=5)
        
        assert len(predictions) == 5
        assert isinstance(predictions[0], float)

class TestEnsembleModel:
    """Test ensemble model functionality"""
    
    @pytest.fixture
    def ensemble_predictor(self):
        return EnsembleStockPredictor()
    
    @pytest.fixture
    def sample_prediction_result(self):
        from src.models.ensemble_model import PredictionResult
        
        return PredictionResult(
            symbol='TATAMOTORS.NS',
            current_price=100.0,
            predicted_price=105.0,
            price_change=5.0,
            price_change_percent=5.0,
            confidence=0.8,
            recommendation='Buy',
            timeframe='short',
            prediction_date=datetime.now(),
            model_predictions={'xgboost': 0.03, 'informer': 0.04, 'sentiment': 0.02},
            model_confidences={'xgboost': 0.8, 'informer': 0.7, 'sentiment': 0.6},
            sentiment_score=0.2,
            technical_signals={'rsi': 55, 'macd_signal': 'Bullish'},
            risk_level='Medium'
        )
    
    @patch('src.models.ensemble_model.yahoo_data')
    @patch('src.models.ensemble_model.ensemble_predictor._get_xgboost_prediction')
    @patch('src.models.ensemble_model.ensemble_predictor._get_informer_prediction')
    @patch('src.models.ensemble_model.ensemble_predictor._get_sentiment_prediction')
    def test_predict(self, mock_sentiment, mock_informer, mock_xgboost, mock_yahoo, 
                    ensemble_predictor, sample_prediction_result):
        """Test ensemble prediction"""
        # Mock dependencies
        mock_yahoo.fetch_ohlcv_data.return_value = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        mock_xgboost.return_value = {'prediction': 0.03, 'confidence': 0.8}
        mock_informer.return_value = {'prediction': 0.04, 'confidence': 0.7}
        mock_sentiment.return_value = {'prediction': 0.02, 'confidence': 0.6, 'sentiment_score': 0.2}
        
        # Test prediction
        result = ensemble_predictor.predict('TATAMOTORS.NS', 'short')
        
        assert result.symbol == 'TATAMOTORS.NS'
        assert isinstance(result.current_price, float)
        assert isinstance(result.predicted_price, float)
        assert isinstance(result.confidence, float)
        assert result.recommendation in ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
    
    def test_generate_recommendation(self, ensemble_predictor):
        """Test recommendation generation"""
        assert ensemble_predictor._generate_recommendation(6.0) == 'Strong Buy'
        assert ensemble_predictor._generate_recommendation(3.0) == 'Buy'
        assert ensemble_predictor._generate_recommendation(0.0) == 'Hold'
        assert ensemble_predictor._generate_recommendation(-3.0) == 'Sell'
        assert ensemble_predictor._generate_recommendation(-6.0) == 'Strong Sell'
    
    def test_calculate_risk_level(self, ensemble_predictor):
        """Test risk level calculation"""
        # Mock data with low volatility
        low_vol_data = pd.DataFrame({
            'close': [100, 100.1, 100.2, 100.1, 100.0]
        })
        
        risk_level = ensemble_predictor._calculate_risk_level(low_vol_data, 0.8)
        assert risk_level in ['Low', 'Medium', 'High']

class TestSHAPExplainer:
    """Test SHAP explainability functionality"""
    
    @pytest.fixture
    def shap_explainer(self):
        return SHAPExplainer()
    
    @pytest.fixture
    def sample_features(self):
        """Sample features for testing"""
        return pd.DataFrame({
            'close': [100.0],
            'rsi': [55.0],
            'macd': [0.5],
            'volume': [1000000],
            'pe_ratio': [15.5]
        })
    
    def test_explain_prediction(self, shap_explainer, sample_features):
        """Test prediction explanation"""
        # Mock explainer
        shap_explainer.explainer = Mock()
        shap_explainer.explainer.shap_values.return_value = np.array([[0.1, 0.2, -0.1, 0.05, 0.15]])
        shap_explainer.explainer.expected_value = 0.0
        shap_explainer.feature_names = ['close', 'rsi', 'macd', 'volume', 'pe_ratio']
        
        # Test explanation
        explanation = shap_explainer.explain_prediction(sample_features)
        
        assert 'base_value' in explanation
        assert 'prediction_confidence' in explanation
        assert 'top_features' in explanation
        assert 'natural_explanation' in explanation
    
    def test_make_feature_readable(self, shap_explainer):
        """Test feature name conversion"""
        assert shap_explainer._make_feature_readable('rsi') == 'RSI (Relative Strength Index)'
        assert shap_explainer._make_feature_readable('macd') == 'MACD'
        assert shap_explainer._make_feature_readable('pe_ratio') == 'P/E Ratio'
        assert shap_explainer._make_feature_readable('unknown_feature') == 'Unknown Feature'
    
    def test_get_fallback_explanation(self, shap_explainer, sample_features):
        """Test fallback explanation when SHAP is not available"""
        explanation = shap_explainer._get_fallback_explanation(sample_features)
        
        assert 'base_value' in explanation
        assert 'prediction_confidence' in explanation
        assert 'top_features' in explanation
        assert 'fallback' in explanation
        assert explanation['fallback'] is True

# Integration Tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_prediction(self, temp_dir):
        """Test end-to-end prediction pipeline"""
        # This would test the complete pipeline from data fetching to prediction
        # Implementation would depend on the specific integration requirements
        pass
    
    def test_model_persistence(self, temp_dir):
        """Test model saving and loading"""
        # Test that models can be saved and loaded correctly
        pass

# Performance Tests
class TestPerformance:
    """Performance tests"""
    
    def test_prediction_speed(self):
        """Test prediction speed"""
        # Test that predictions are generated within acceptable time limits
        pass
    
    def test_memory_usage(self):
        """Test memory usage"""
        # Test that the system doesn't use excessive memory
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
