# Enterprise Stock Forecasting System

An enterprise-grade, production-ready Python system for forecasting Indian stocks using advanced machine learning models and real-time financial data.

## Features

- **Multi-Model Ensemble**: XGBoost, Informer Transformer, and FinBERT for comprehensive predictions
- **Real-time Data**: Yahoo Finance integration for OHLCV and fundamental data
- **News Sentiment Analysis**: Multi-source financial news aggregation with sentiment scoring
- **Technical Indicators**: Complete suite of technical analysis indicators
- **Explainable AI**: SHAP integration for model interpretability
- **Interactive Dashboard**: Streamlit-based frontend with multiple timeframe support
- **Multilingual Support**: English/Hindi news translation
- **Production Ready**: FastAPI backend with real-time prediction capabilities

## Architecture

```
├── src/
│   ├── data/           # Data acquisition and preprocessing
│   ├── models/         # ML models (XGBoost, Informer, FinBERT)
│   ├── features/       # Feature engineering and technical indicators
│   ├── backend/        # FastAPI backend
│   ├── frontend/       # Streamlit frontend
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── data/               # Data storage
├── models/             # Trained model artifacts
├── tests/              # Test suite
└── docs/               # Documentation
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run src/frontend/app.py
   ```

3. **Access Backend API**
   ```bash
   uvicorn src.backend.main:app --reload
   ```

## Supported Stocks

- TATAMOTORS.NS (Tata Motors)
- All NSE-listed stocks with Yahoo Finance data

## Timeframes

- **Scalping**: Intraday (minutes)
- **Intraday**: Daily
- **Short**: 1-7 days
- **Medium**: 15-90 days
- **Long**: 3-12 months

## Model Performance

- **XGBoost**: Baseline tabular model with technical indicators
- **Informer**: Transformer-based long-sequence forecasting
- **FinBERT**: Financial sentiment analysis
- **Ensemble**: Weighted combination for final predictions

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License.
