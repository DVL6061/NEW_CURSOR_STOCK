"""
Model training script for the Enterprise Stock Forecasting System
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import settings
from src.data.yahoo_finance import yahoo_data
from src.data.news_scraper import news_scraper
from src.features.technical_indicators import technical_indicators
from src.models.xgboost_model import xgboost_predictor
from src.models.informer_model import informer_predictor
from src.models.ensemble_model import ensemble_predictor
from src.models.sentiment_analyzer import sentiment_analyzer
from src.models.shap_explainer import shap_explainer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training orchestrator"""
    
    def __init__(self):
        self.symbols = settings.DEFAULT_SYMBOLS
        self.training_data = {}
        self.training_results = {}
    
    def prepare_training_data(self):
        """Prepare training data for all models"""
        logger.info("Preparing training data...")
        
        for symbol in self.symbols:
            try:
                logger.info(f"Fetching data for {symbol}...")
                
                # Fetch OHLCV data
                ohlcv_data = yahoo_data.fetch_ohlcv_data(
                    symbol, 
                    period='5y', 
                    interval='1d'
                )
                
                # Fetch fundamental data
                fundamental_data = yahoo_data.fetch_fundamental_data(symbol)
                
                # Calculate technical indicators
                data_with_indicators = technical_indicators.calculate_all_indicators(ohlcv_data)
                
                # Store training data
                self.training_data[symbol] = {
                    'ohlcv': ohlcv_data,
                    'fundamental': fundamental_data,
                    'features': data_with_indicators
                }
                
                logger.info(f"Data prepared for {symbol}: {len(data_with_indicators)} samples")
                
            except Exception as e:
                logger.error(f"Error preparing data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Training data prepared for {len(self.training_data)} symbols")
    
    def train_xgboost_model(self):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        try:
            # Combine data from all symbols
            all_features = []
            
            for symbol, data in self.training_data.items():
                try:
                    # Prepare features
                    features_df = xgboost_predictor.prepare_features(
                        data['features'], 
                        data['fundamental']
                    )
                    
                    if not features_df.empty:
                        all_features.append(features_df)
                        logger.info(f"Prepared {len(features_df)} samples for {symbol}")
                
                except Exception as e:
                    logger.error(f"Error preparing features for {symbol}: {str(e)}")
                    continue
            
            if not all_features:
                raise ValueError("No training data available for XGBoost")
            
            # Combine all data
            combined_data = pd.concat(all_features, ignore_index=True)
            logger.info(f"Combined training data: {len(combined_data)} samples")
            
            # Train model
            training_results = xgboost_predictor.train(combined_data)
            self.training_results['xgboost'] = training_results
            
            logger.info("XGBoost model training completed")
            logger.info(f"Train MAE: {training_results['train_metrics']['mae']:.4f}")
            logger.info(f"Validation MAE: {training_results['validation_metrics']['mae']:.4f}")
            logger.info(f"Test MAE: {training_results['test_metrics']['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def train_informer_model(self):
        """Train Informer model"""
        logger.info("Training Informer model...")
        
        try:
            # Use the first symbol for training (can be extended to multiple symbols)
            symbol = self.symbols[0]
            data = self.training_data[symbol]['features']
            
            # Train model
            training_results = informer_predictor.train(data, target_column='close')
            self.training_results['informer'] = training_results
            
            logger.info("Informer model training completed")
            logger.info(f"Train MAE: {training_results['train_metrics']['mae']:.4f}")
            logger.info(f"Validation MAE: {training_results['validation_metrics']['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training Informer model: {str(e)}")
            raise
    
    def train_sentiment_model(self):
        """Train/validate sentiment analysis model"""
        logger.info("Training sentiment analysis model...")
        
        try:
            # Fetch news articles for sentiment analysis
            articles = news_scraper.scrape_all_sources(
                self.symbols[:3],  # Use first 3 symbols for news
                max_articles_per_source=20
            )
            
            if articles:
                # Analyze sentiment for all articles
                analyzed_articles = []
                for article in articles:
                    analyzed_article = sentiment_analyzer.analyze_article(article)
                    analyzed_articles.append(analyzed_article)
                
                # Save articles with sentiment
                news_scraper.save_articles(analyzed_articles)
                
                # Get sentiment summary
                sentiment_summary = sentiment_analyzer.get_sentiment_summary(analyzed_articles)
                
                self.training_results['sentiment'] = {
                    'articles_processed': len(analyzed_articles),
                    'sentiment_summary': sentiment_summary
                }
                
                logger.info(f"Sentiment analysis completed for {len(analyzed_articles)} articles")
                logger.info(f"Average sentiment: {sentiment_summary['average_sentiment']:.3f}")
                logger.info(f"Positive articles: {sentiment_summary['positive_count']}")
                logger.info(f"Negative articles: {sentiment_summary['negative_count']}")
            
            else:
                logger.warning("No articles found for sentiment analysis")
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {str(e)}")
            raise
    
    def setup_ensemble_model(self):
        """Setup ensemble model"""
        logger.info("Setting up ensemble model...")
        
        try:
            # Load individual models
            xgboost_predictor.load_model()
            informer_predictor.load_model()
            
            # Save ensemble configuration
            ensemble_predictor.save_model()
            
            logger.info("Ensemble model setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up ensemble model: {str(e)}")
            raise
    
    def setup_shap_explainer(self):
        """Setup SHAP explainer"""
        logger.info("Setting up SHAP explainer...")
        
        try:
            # Initialize SHAP explainer with trained XGBoost model
            shap_explainer._initialize_explainer()
            
            # Save explainer
            shap_explainer.save_explainer()
            
            logger.info("SHAP explainer setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {str(e)}")
            raise
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("Evaluating models...")
        
        try:
            # Test predictions on a sample symbol
            test_symbol = self.symbols[0]
            
            # Get ensemble prediction
            prediction_result = ensemble_predictor.predict(test_symbol, 'short')
            
            logger.info(f"Sample prediction for {test_symbol}:")
            logger.info(f"  Current Price: ₹{prediction_result.current_price:.2f}")
            logger.info(f"  Predicted Price: ₹{prediction_result.predicted_price:.2f}")
            logger.info(f"  Price Change: {prediction_result.price_change_percent:.2f}%")
            logger.info(f"  Recommendation: {prediction_result.recommendation}")
            logger.info(f"  Confidence: {prediction_result.confidence:.1%}")
            logger.info(f"  Risk Level: {prediction_result.risk_level}")
            
            # Test SHAP explanation
            try:
                explanation = shap_explainer.explain_prediction(
                    self.training_data[test_symbol]['features'].iloc[-1:],
                    prediction_result
                )
                
                logger.info("SHAP explanation generated successfully")
                logger.info(f"  Top feature: {explanation['top_features'][0]['feature']}")
                logger.info(f"  Natural explanation: {explanation['natural_explanation'][:100]}...")
                
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}")
            
            self.training_results['evaluation'] = {
                'test_symbol': test_symbol,
                'prediction': {
                    'current_price': prediction_result.current_price,
                    'predicted_price': prediction_result.predicted_price,
                    'price_change_percent': prediction_result.price_change_percent,
                    'recommendation': prediction_result.recommendation,
                    'confidence': prediction_result.confidence,
                    'risk_level': prediction_result.risk_level
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            raise
    
    def save_training_report(self):
        """Save training report"""
        logger.info("Saving training report...")
        
        try:
            report = {
                'training_date': datetime.now().isoformat(),
                'symbols_trained': list(self.training_data.keys()),
                'training_results': self.training_results,
                'model_configurations': {
                    'xgboost_params': settings.XGBOOST_PARAMS,
                    'informer_params': settings.INFORMER_PARAMS,
                    'ensemble_weights': ensemble_predictor.model_weights
                }
            }
            
            # Save report
            report_path = settings.MODELS_DIR / "training_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving training report: {str(e)}")
    
    def run_full_training(self):
        """Run complete training pipeline"""
        logger.info("Starting full model training pipeline...")
        
        try:
            # Step 1: Prepare training data
            self.prepare_training_data()
            
            # Step 2: Train individual models
            self.train_xgboost_model()
            self.train_informer_model()
            self.train_sentiment_model()
            
            # Step 3: Setup ensemble and explainability
            self.setup_ensemble_model()
            self.setup_shap_explainer()
            
            # Step 4: Evaluate models
            self.evaluate_models()
            
            # Step 5: Save training report
            self.save_training_report()
            
            logger.info("Full training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Enterprise Stock Forecasting Models")
    parser.add_argument("--model", choices=['xgboost', 'informer', 'sentiment', 'ensemble', 'all'], 
                       default='all', help="Model to train")
    parser.add_argument("--symbols", nargs='+', default=settings.DEFAULT_SYMBOLS[:3],
                       help="Stock symbols to train on")
    parser.add_argument("--verbose", action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update symbols if provided
    if args.symbols:
        settings.DEFAULT_SYMBOLS = args.symbols
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        if args.model == 'all':
            trainer.run_full_training()
        elif args.model == 'xgboost':
            trainer.prepare_training_data()
            trainer.train_xgboost_model()
        elif args.model == 'informer':
            trainer.prepare_training_data()
            trainer.train_informer_model()
        elif args.model == 'sentiment':
            trainer.train_sentiment_model()
        elif args.model == 'ensemble':
            trainer.setup_ensemble_model()
            trainer.setup_shap_explainer()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
