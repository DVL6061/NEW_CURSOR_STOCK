"""
Configuration settings for the Enterprise Stock Forecasting System
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    CONFIG_DIR: Path = PROJECT_ROOT / "config"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./stock_forecast.db"
    
    # Yahoo Finance settings
    DEFAULT_PERIOD: str = "5y"
    DEFAULT_INTERVAL: str = "1d"
    SUPPORTED_INTERVALS: List[str] = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    
    # Indian stock symbols
    DEFAULT_SYMBOLS: List[str] = [
        "TATAMOTORS.NS", "RELIANCE.NS", "TCS.NS"
    ]
    
    # News sources
    NEWS_SOURCES: Dict[str, str] = {
        "cnbc": "https://www.cnbc.com/search/?query=indian%20stocks",
        "moneycontrol": "https://www.moneycontrol.com/news/business/markets/",
        "mint": "https://www.livemint.com/market"
    }
    
    # Model settings
    XGBOOST_PARAMS: Dict = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    INFORMER_PARAMS: Dict = {
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 24,
        "factor": 5,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "dropout": 0.05,
        "attn": "prob",
        "embed": "timeF",
        "freq": "h"
    }
    
    # FinBERT settings
    FINBERT_MODEL_NAME: str = "yiyanghkust/finbert-tone"
    MAX_NEWS_LENGTH: int = 512
    
    # Technical indicators
    TECHNICAL_INDICATORS: List[str] = [
        "RSI", "MACD", "EMA", "SMA", "BBANDS", "ADX", "STOCH", 
        "CCI", "WILLR", "ROC", "MOM", "PPO", "TRIX", "ULTOSC"
    ]
    
    # Timeframes
    TIMEFRAMES: Dict[str, Dict] = {
        "scalping": {"interval": "1m", "period": "7d", "prediction_days": 1},
        "intraday": {"interval": "5m", "period": "30d", "prediction_days": 1},
        "short": {"interval": "1d", "period": "1y", "prediction_days": 7},
        "medium": {"interval": "1d", "period": "2y", "prediction_days": 90},
        "long": {"interval": "1wk", "period": "5y", "prediction_days": 365}
    }
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Stock Forecasting API"
    API_VERSION: str = "1.0.0"
    
    # Streamlit settings
    STREAMLIT_TITLE: str = "Enterprise Stock Forecasting System"
    STREAMLIT_PAGE_CONFIG: Dict = {
        "page_title": "Stock Forecast",
        "page_icon": "📈",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
