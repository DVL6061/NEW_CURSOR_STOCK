# Enterprise Stock Forecasting System - Usage Guide

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd NEW_CURSOR_STOCK

# Install dependencies
pip install -r requirements.txt

# Setup the system
python run_system.py setup
```

### 2. Training Models

```bash
# Train all models
python train_models.py --model all

# Train specific models
python train_models.py --model xgboost
python train_models.py --model informer
python train_models.py --model sentiment
```

### 3. Running the System

```bash
# Run complete system (backend + frontend)
python run_system.py full

# Run individual components
python run_system.py backend    # FastAPI backend only
python run_system.py frontend   # Streamlit frontend only
```

### 4. Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📊 System Architecture

### Components

1. **Data Acquisition**
   - Yahoo Finance integration for OHLCV and fundamental data
   - Multi-source news scraping (CNBC, Moneycontrol, Mint, ET, Business Standard)
   - Real-time data updates

2. **Feature Engineering**
   - Complete technical indicators suite (RSI, MACD, EMA/SMA, Bollinger Bands, ADX, Stochastic)
   - Custom features and lagged variables
   - Rolling window statistics

3. **Machine Learning Models**
   - **XGBoost**: Tabular features and technical indicators
   - **Informer**: Transformer-based time series forecasting
   - **FinBERT**: Financial news sentiment analysis
   - **Ensemble**: Weighted combination of all models

4. **Explainability**
   - SHAP integration for model interpretability
   - Natural language explanations
   - Feature importance analysis

5. **Real-time Support**
   - Continuous prediction updates
   - Alert system for significant changes
   - Caching and performance optimization

## 🔧 Configuration

### Settings (`config/settings.py`)

```python
# Stock symbols to monitor
DEFAULT_SYMBOLS = ["TATAMOTORS.NS", "RELIANCE.NS", "TCS.NS", ...]

# Model parameters
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    ...
}

# Timeframes
TIMEFRAMES = {
    "scalping": {"interval": "1m", "period": "7d", "prediction_days": 1},
    "intraday": {"interval": "5m", "period": "30d", "prediction_days": 1},
    "short": {"interval": "1d", "period": "1y", "prediction_days": 7},
    "medium": {"interval": "1d", "period": "2y", "prediction_days": 90},
    "long": {"interval": "1wk", "period": "5y", "prediction_days": 365}
}
```

## 📈 Using the Frontend

### Dashboard
- Overview of system status
- Quick predictions for top stocks
- Performance metrics

### Prediction Page
- Select stock symbol and timeframe
- Get detailed predictions with confidence intervals
- View technical signals and sentiment analysis

### Explanation Page
- Understand model decisions
- View feature importance
- Read natural language explanations

### News Page
- Financial news aggregation
- Sentiment analysis
- Source filtering

### Top Picks Page
- Best performing stocks
- Risk-return analysis
- Recommendation summaries

### Data Page
- Historical data visualization
- Technical analysis charts
- Price and volume analysis

## 🔌 API Usage

### Prediction Endpoint

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "symbol": "TATAMOTORS.NS",
    "timeframe": "short"
})

prediction = response.json()
print(f"Predicted price: ₹{prediction['predicted_price']:.2f}")
print(f"Recommendation: {prediction['recommendation']}")
```

### Batch Predictions

```python
# Multiple symbols
response = requests.post("http://localhost:8000/predict/batch", json={
    "symbols": ["TATAMOTORS.NS", "RELIANCE.NS", "TCS.NS"],
    "timeframe": "short"
})

predictions = response.json()
for pred in predictions:
    print(f"{pred['symbol']}: {pred['recommendation']}")
```

### Explanation Endpoint

```python
# Get model explanation
response = requests.post("http://localhost:8000/explain", json={
    "symbol": "TATAMOTORS.NS",
    "timeframe": "short"
})

explanation = response.json()
print(explanation['natural_explanation'])
```

### News Endpoint

```python
# Get financial news
response = requests.post("http://localhost:8000/news", json={
    "symbols": ["TATAMOTORS.NS"],
    "days": 7
})

news = response.json()
for article in news['articles']:
    print(f"{article['title']} - {article['sentiment_label']}")
```

## 🧪 Testing

```bash
# Run all tests
python run_system.py test

# Run specific test modules
pytest tests/test_models.py -v
pytest tests/test_data.py -v
pytest tests/test_api.py -v
```

## 📊 Model Performance

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

### Model Comparison
- **XGBoost**: Best for tabular features and technical indicators
- **Informer**: Excellent for long-sequence time series patterns
- **FinBERT**: Superior sentiment analysis for financial news
- **Ensemble**: Combines strengths of all models

## 🔄 Real-time Features

### Automatic Updates
- Data refreshed every 5 minutes
- News sentiment updated hourly
- Alerts for significant changes

### Alert System
- High confidence predictions (>80%)
- Significant price changes (>5%)
- High risk detection
- Stale data warnings

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Retrain models
   python train_models.py --model all
   ```

2. **API Connection Issues**
   ```bash
   # Check backend status
   curl http://localhost:8000/health
   ```

3. **Data Fetching Errors**
   ```bash
   # Check internet connection
   # Verify Yahoo Finance access
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch sizes in settings
   # Use fewer symbols for training
   ```

### Logs
- Application logs: `training.log`
- System logs: Console output
- Error logs: Check individual module logs

## 📚 Advanced Usage

### Custom Models
- Add new technical indicators in `src/features/technical_indicators.py`
- Implement new models in `src/models/`
- Update ensemble weights in `src/models/ensemble_model.py`

### Custom Data Sources
- Add new news sources in `config/settings.py`
- Implement new scrapers in `src/data/news_scraper.py`
- Add new data providers in `src/data/`

### Custom Timeframes
- Define new timeframes in `config/settings.py`
- Update prediction logic in `src/models/ensemble_model.py`

## 🔒 Security Considerations

- API endpoints should be secured in production
- Rate limiting for external API calls
- Input validation for all user inputs
- Secure storage of sensitive data

## 📈 Performance Optimization

- Use GPU acceleration for deep learning models
- Implement model caching
- Optimize database queries
- Use async processing for I/O operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review logs for error details
- Create an issue in the repository
- Contact the development team

---

**Note**: This system is for educational and research purposes. Always verify predictions with multiple sources before making investment decisions.

venv\Scripts\python.exe -c "import uvicorn; from src.backend.main import app; uvicorn.run(app, host='127.0.0.1', port=8001)"

& c:/Users/dhruv/OneDrive/Desktop/NEW_CURSOR_STOCK/venv/Scripts/Activate.ps1   

python.exe -m streamlit run src/frontend/app.py --server.port 8501 --server.address 127.0.0.1        