"""
FastAPI backend for the Enterprise Stock Forecasting System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from config.settings import settings
from src.models.ensemble_model import ensemble_predictor, PredictionResult
from src.models.shap_explainer import shap_explainer
from src.data.yahoo_finance import yahoo_data
from src.data.news_scraper import news_scraper
from src.models.sentiment_analyzer import sentiment_analyzer

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., 'TATAMOTORS.NS')")
    timeframe: str = Field(default="short", description="Prediction timeframe")
    days_ahead: Optional[int] = Field(default=None, description="Number of days to predict ahead")

class PredictionResponse(BaseModel):
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

class ExplanationRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field(default="short", description="Prediction timeframe")

class ExplanationResponse(BaseModel):
    base_value: float
    prediction_confidence: float
    top_features: List[Dict[str, Any]]
    feature_categories: Dict[str, List[Dict[str, Any]]]
    natural_explanation: str
    insights: List[str]

class NewsRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols")
    days: int = Field(default=7, description="Number of days to look back")

class NewsResponse(BaseModel):
    articles: List[Dict[str, Any]]
    sentiment_summary: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    data_sources: Dict[str, bool]

# Global variables for model state
models_loaded = {
    'xgboost': False,
    'informer': False,
    'ensemble': False,
    'shap': False
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Enterprise Stock Forecasting API...")
    
    # Load models
    await load_models()
    
    # Start background tasks
    asyncio.create_task(periodic_data_update())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enterprise Stock Forecasting API...")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Enterprise-grade stock forecasting API with XGBoost, Informer, and FinBERT",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_models():
    """Load all models on startup"""
    try:
        logger.info("Loading models...")
        
        # Load XGBoost model
        try:
            from src.models.xgboost_model import xgboost_predictor
            xgboost_predictor.load_model()
            models_loaded['xgboost'] = True
            logger.info("XGBoost model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {str(e)}")
        
        # Load Informer model
        try:
            from src.models.informer_model import informer_predictor
            informer_predictor.load_model()
            models_loaded['informer'] = True
            logger.info("Informer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Informer model: {str(e)}")
        
        # Load ensemble model
        try:
            ensemble_predictor.load_model()
            models_loaded['ensemble'] = True
            logger.info("Ensemble model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ensemble model: {str(e)}")
        
        # Load SHAP explainer
        try:
            shap_explainer.load_explainer()
            models_loaded['shap'] = True
            logger.info("SHAP explainer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SHAP explainer: {str(e)}")
        
        logger.info("Model loading completed")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

async def periodic_data_update():
    """Periodic data update task"""
    while True:
        try:
            logger.info("Performing periodic data update...")
            
            # Update data for default symbols
            for symbol in settings.DEFAULT_SYMBOLS[:3]:  # Update top 3 symbols
                try:
                    # Fetch fresh data
                    yahoo_data.fetch_ohlcv_data(symbol, use_cache=False)
                    yahoo_data.fetch_fundamental_data(symbol, use_cache=False)
                    
                    # Fetch fresh news
                    articles = news_scraper.scrape_all_sources([symbol], max_articles_per_source=10)
                    news_scraper.save_articles(articles)
                    
                    logger.info(f"Updated data for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error updating data for {symbol}: {str(e)}")
            
            # Wait for next update (1 hour)
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in periodic data update: {str(e)}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Enterprise Stock Forecasting API",
        "version": settings.API_VERSION,
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check data sources
        data_sources = {
            'yahoo_finance': True,  # Assume available
            'news_sources': True   # Assume available
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            models_loaded=models_loaded,
            data_sources=data_sources
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """Predict stock price using ensemble model"""
    try:
        logger.info(f"Prediction request for {request.symbol} ({request.timeframe})")
        
        # Validate timeframe
        if request.timeframe not in settings.TIMEFRAMES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timeframe. Available: {list(settings.TIMEFRAMES.keys())}"
            )
        
        # Make prediction
        prediction_result = ensemble_predictor.predict(
            symbol=request.symbol,
            timeframe=request.timeframe,
            days_ahead=request.days_ahead
        )
        
        # Convert to response model
        response = PredictionResponse(
            symbol=prediction_result.symbol,
            current_price=prediction_result.current_price,
            predicted_price=prediction_result.predicted_price,
            price_change=prediction_result.price_change,
            price_change_percent=prediction_result.price_change_percent,
            confidence=prediction_result.confidence,
            recommendation=prediction_result.recommendation,
            timeframe=prediction_result.timeframe,
            prediction_date=prediction_result.prediction_date,
            model_predictions=prediction_result.model_predictions,
            model_confidences=prediction_result.model_confidences,
            sentiment_score=prediction_result.sentiment_score,
            technical_signals=prediction_result.technical_signals,
            risk_level=prediction_result.risk_level
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_multiple_stocks(symbols: List[str], timeframe: str = "short"):
    """Predict multiple stocks"""
    try:
        logger.info(f"Batch prediction request for {len(symbols)} symbols")
        
        # Validate timeframe
        if timeframe not in settings.TIMEFRAMES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timeframe. Available: {list(settings.TIMEFRAMES.keys())}"
            )
        
        # Make predictions
        prediction_results = ensemble_predictor.predict_multiple_symbols(symbols, timeframe)
        
        # Convert to response models
        responses = []
        for result in prediction_results:
            response = PredictionResponse(
                symbol=result.symbol,
                current_price=result.current_price,
                predicted_price=result.predicted_price,
                price_change=result.price_change,
                price_change_percent=result.price_change_percent,
                confidence=result.confidence,
                recommendation=result.recommendation,
                timeframe=result.timeframe,
                prediction_date=result.prediction_date,
                model_predictions=result.model_predictions,
                model_confidences=result.model_confidences,
                sentiment_score=result.sentiment_score,
                technical_signals=result.technical_signals,
                risk_level=result.risk_level
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """Explain model prediction using SHAP"""
    try:
        logger.info(f"Explanation request for {request.symbol}")
        
        # Get prediction result
        prediction_result = ensemble_predictor.predict(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        # Get current data for explanation
        timeframe_config = settings.TIMEFRAMES.get(request.timeframe, settings.TIMEFRAMES['short'])
        current_data = yahoo_data.fetch_ohlcv_data(
            request.symbol,
            period=timeframe_config['period'],
            interval=timeframe_config['interval']
        )
        
        if current_data.empty:
            raise HTTPException(status_code=404, detail="No data available for explanation")
        
        # Prepare features for explanation
        from src.features.technical_indicators import technical_indicators
        features_df = technical_indicators.calculate_all_indicators(current_data)
        
        # Get explanation
        explanation = shap_explainer.explain_prediction(features_df.iloc[-1:], prediction_result)
        
        # Convert to response model
        response = ExplanationResponse(
            base_value=explanation.get('base_value', 0.0),
            prediction_confidence=explanation.get('prediction_confidence', 0.5),
            top_features=explanation.get('top_features', []),
            feature_categories=explanation.get('feature_categories', {}),
            natural_explanation=explanation.get('natural_explanation', ''),
            insights=explanation.get('insights', [])
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Explanation failed for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/news", response_model=NewsResponse)
async def get_news(request: NewsRequest):
    """Get financial news and sentiment analysis"""
    try:
        logger.info(f"News request for {len(request.symbols)} symbols")
        
        # Get recent articles
        articles = news_scraper.get_recent_articles(request.days, request.symbols)
        
        # Analyze sentiment for articles
        analyzed_articles = []
        for article in articles:
            analyzed_article = sentiment_analyzer.analyze_article(article)
            analyzed_articles.append({
                'title': analyzed_article.title,
                'content': analyzed_article.content[:500] + '...' if len(analyzed_article.content) > 500 else analyzed_article.content,
                'url': analyzed_article.url,
                'source': analyzed_article.source,
                'published_date': analyzed_article.published_date.isoformat(),
                'sentiment_score': analyzed_article.sentiment_score,
                'sentiment_label': analyzed_article.sentiment_label,
                'language': analyzed_article.language
            })
        
        # Get sentiment summary
        sentiment_summary = sentiment_analyzer.get_sentiment_summary(articles)
        
        response = NewsResponse(
            articles=analyzed_articles,
            sentiment_summary=sentiment_summary
        )
        
        return response
        
    except Exception as e:
        logger.error(f"News request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"News request failed: {str(e)}")

@app.get("/top-picks", response_model=List[PredictionResponse])
async def get_top_picks(timeframe: str = "short", top_n: int = 5):
    """Get top stock picks based on predictions"""
    try:
        logger.info(f"Top picks request ({timeframe}, top {top_n})")
        
        # Get top picks
        top_picks = ensemble_predictor.get_top_picks(
            symbols=settings.DEFAULT_SYMBOLS,
            timeframe=timeframe,
            top_n=top_n
        )
        
        # Convert to response models
        responses = []
        for result in top_picks:
            response = PredictionResponse(
                symbol=result.symbol,
                current_price=result.current_price,
                predicted_price=result.predicted_price,
                price_change=result.price_change,
                price_change_percent=result.price_change_percent,
                confidence=result.confidence,
                recommendation=result.recommendation,
                timeframe=result.timeframe,
                prediction_date=result.prediction_date,
                model_predictions=result.model_predictions,
                model_confidences=result.model_confidences,
                sentiment_score=result.sentiment_score,
                technical_signals=result.technical_signals,
                risk_level=result.risk_level
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Top picks request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Top picks request failed: {str(e)}")

@app.get("/symbols")
async def get_available_symbols():
    """Get list of available stock symbols"""
    return {
        "default_symbols": settings.DEFAULT_SYMBOLS,
        "timeframes": list(settings.TIMEFRAMES.keys()),
        "news_sources": list(settings.NEWS_SOURCES.keys())
    }

@app.get("/data/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y", interval: str = "1d"):
    """Get historical stock data"""
    try:
        # Fetch data
        data = yahoo_data.fetch_ohlcv_data(symbol, period=period, interval=interval)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert to JSON-serializable format
        data_dict = data.reset_index().to_dict('records')
        
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": data_dict,
            "count": len(data_dict)
        }
        
    except Exception as e:
        logger.error(f"Data request failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data request failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
