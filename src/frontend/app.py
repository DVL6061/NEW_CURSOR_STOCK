"""
Streamlit frontend for the Enterprise Stock Forecasting System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

from config.settings import settings

# Configure Streamlit page
st.set_page_config(
    page_title=settings.STREAMLIT_PAGE_CONFIG["page_title"],
    page_icon=settings.STREAMLIT_PAGE_CONFIG["page_icon"],
    layout=settings.STREAMLIT_PAGE_CONFIG["layout"],
    initial_sidebar_state=settings.STREAMLIT_PAGE_CONFIG["initial_sidebar_state"]
)

# API Configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"

class StockForecastingApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.session = requests.Session()
    
    def run(self):
        """Run the Streamlit application"""
        # Header
        st.title(settings.STREAMLIT_TITLE)
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        page = st.session_state.get('page', 'dashboard')
        
        if page == 'dashboard':
            self.render_dashboard()
        elif page == 'predict':
            self.render_prediction_page()
        elif page == 'explain':
            self.render_explanation_page()
        elif page == 'news':
            self.render_news_page()
        elif page == 'top_picks':
            self.render_top_picks_page()
        elif page == 'data':
            self.render_data_page()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("📈 Navigation")
        
        # Navigation buttons
        if st.sidebar.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
        
        if st.sidebar.button("🔮 Predict", use_container_width=True):
            st.session_state.page = 'predict'
        
        if st.sidebar.button("🧠 Explain", use_container_width=True):
            st.session_state.page = 'explain'
        
        if st.sidebar.button("📰 News", use_container_width=True):
            st.session_state.page = 'news'
        
        if st.sidebar.button("⭐ Top Picks", use_container_width=True):
            st.session_state.page = 'top_picks'
        
        if st.sidebar.button("📊 Data", use_container_width=True):
            st.session_state.page = 'data'
        
        st.sidebar.markdown("---")
        
        # API Status
        st.sidebar.subheader("🔗 API Status")
        if self.check_api_health():
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.error("❌ API Disconnected")
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        self.selected_timeframe = st.sidebar.selectbox(
            "Timeframe",
            options=list(settings.TIMEFRAMES.keys()),
            index=2,  # Default to 'short'
            help="Select prediction timeframe"
        )
        
        self.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        if self.auto_refresh:
            self.refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
    
    def check_api_health(self) -> bool:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.header("📊 Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Models", "3", "XGBoost, Informer, FinBERT")
        
        with col2:
            st.metric("Supported Stocks", len(settings.DEFAULT_SYMBOLS), "NSE Listed")
        
        with col3:
            st.metric("News Sources", len(settings.NEWS_SOURCES), "Financial")
        
        with col4:
            st.metric("API Status", "🟢 Online" if self.check_api_health() else "🔴 Offline")
        
        st.markdown("---")
        
        # Quick predictions for top stocks
        st.subheader("🚀 Quick Predictions")
        
        # Get top picks
        try:
            top_picks = self.get_top_picks(timeframe=self.selected_timeframe, top_n=5)
            
            if top_picks:
                # Create predictions dataframe
                predictions_df = pd.DataFrame([
                    {
                        'Symbol': pick['symbol'],
                        'Current Price': f"₹{pick['current_price']:.2f}",
                        'Predicted Price': f"₹{pick['predicted_price']:.2f}",
                        'Change %': f"{pick['price_change_percent']:.2f}%",
                        'Recommendation': pick['recommendation'],
                        'Confidence': f"{pick['confidence']:.1%}",
                        'Risk': pick['risk_level']
                    }
                    for pick in top_picks
                ])
                
                # Color code recommendations
                def color_recommendation(val):
                    if val in ['Strong Buy', 'Buy']:
                        return 'background-color: #d4edda'
                    elif val in ['Strong Sell', 'Sell']:
                        return 'background-color: #f8d7da'
                    else:
                        return 'background-color: #fff3cd'
                
                styled_df = predictions_df.style.applymap(color_recommendation, subset=['Recommendation'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price change chart
                    fig = px.bar(
                        predictions_df,
                        x='Symbol',
                        y='Change %',
                        color='Recommendation',
                        title="Predicted Price Changes",
                        color_discrete_map={
                            'Strong Buy': '#28a745',
                            'Buy': '#20c997',
                            'Hold': '#ffc107',
                            'Sell': '#fd7e14',
                            'Strong Sell': '#dc3545'
                        }
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Confidence vs Risk scatter
                    fig = px.scatter(
                        predictions_df,
                        x='Confidence',
                        y='Risk',
                        size='Change %',
                        hover_name='Symbol',
                        title="Confidence vs Risk Analysis",
                        color='Recommendation'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("Unable to fetch predictions. Please check API connection.")
        
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
        
        # Auto refresh
        if self.auto_refresh:
            time.sleep(self.refresh_interval)
            st.rerun()
    
    def render_prediction_page(self):
        """Render prediction page"""
        st.header("🔮 Stock Prediction")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.selectbox(
                    "Select Stock Symbol",
                    options=settings.DEFAULT_SYMBOLS,
                    help="Choose from available NSE stocks"
                )
            
            with col2:
                timeframe = st.selectbox(
                    "Prediction Timeframe",
                    options=list(settings.TIMEFRAMES.keys()),
                    index=2,  # Default to 'short'
                    help="Select prediction horizon"
                )
            
            submitted = st.form_submit_button("Predict", use_container_width=True)
        
        if submitted:
            with st.spinner("Making prediction..."):
                try:
                    prediction = self.get_prediction(symbol, timeframe)
                    
                    if prediction:
                        self.display_prediction_result(prediction)
                    else:
                        st.error("Failed to get prediction. Please try again.")
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    
    def display_prediction_result(self, prediction: Dict[str, Any]):
        """Display prediction result"""
        st.success("✅ Prediction completed!")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"₹{prediction['current_price']:.2f}"
            )
        
        with col2:
            st.metric(
                "Predicted Price",
                f"₹{prediction['predicted_price']:.2f}",
                f"{prediction['price_change_percent']:.2f}%"
            )
        
        with col3:
            st.metric(
                "Recommendation",
                prediction['recommendation'],
                f"{prediction['confidence']:.1%} confidence"
            )
        
        with col4:
            st.metric(
                "Risk Level",
                prediction['risk_level']
            )
        
        st.markdown("---")
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Predictions")
            
            # Model predictions chart
            model_data = pd.DataFrame([
                {'Model': 'XGBoost', 'Prediction': prediction['model_predictions']['xgboost']},
                {'Model': 'Informer', 'Prediction': prediction['model_predictions']['informer']},
                {'Model': 'Sentiment', 'Prediction': prediction['model_predictions']['sentiment']}
            ])
            
            fig = px.bar(
                model_data,
                x='Model',
                y='Prediction',
                title="Individual Model Predictions",
                color='Prediction',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Technical Signals")
            
            tech_signals = prediction['technical_signals']
            
            # RSI
            rsi_value = tech_signals.get('rsi', 50)
            rsi_color = 'red' if rsi_value > 70 else 'green' if rsi_value < 30 else 'orange'
            st.metric("RSI", f"{rsi_value:.1f}", help="Relative Strength Index")
            
            # MACD Signal
            macd_signal = tech_signals.get('macd_signal', 'Neutral')
            st.metric("MACD Signal", macd_signal)
            
            # Trend Strength
            trend_strength = tech_signals.get('trend_strength', 0)
            trend_text = 'Bullish' if trend_strength > 0 else 'Bearish' if trend_strength < 0 else 'Sideways'
            st.metric("Trend", trend_text)
            
            # Volume
            volume_ratio = tech_signals.get('volume_ratio', 1.0)
            volume_text = 'High' if volume_ratio > 1.5 else 'Low' if volume_ratio < 0.5 else 'Normal'
            st.metric("Volume", volume_text)
        
        # Sentiment analysis
        st.subheader("📰 Sentiment Analysis")
        
        sentiment_score = prediction['sentiment_score']
        sentiment_color = 'green' if sentiment_score > 0.1 else 'red' if sentiment_score < -0.1 else 'orange'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
        
        with col2:
            st.metric("Model Confidence", f"{prediction['confidence']:.1%}")
        
        with col3:
            st.metric("Prediction Date", prediction['prediction_date'][:10])
    
    def render_explanation_page(self):
        """Render explanation page"""
        st.header("🧠 Model Explanation")
        
        # Input form
        with st.form("explanation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.selectbox(
                    "Select Stock Symbol",
                    options=settings.DEFAULT_SYMBOLS,
                    key="explain_symbol"
                )
            
            with col2:
                timeframe = st.selectbox(
                    "Prediction Timeframe",
                    options=list(settings.TIMEFRAMES.keys()),
                    index=2,
                    key="explain_timeframe"
                )
            
            submitted = st.form_submit_button("Explain Prediction", use_container_width=True)
        
        if submitted:
            with st.spinner("Generating explanation..."):
                try:
                    explanation = self.get_explanation(symbol, timeframe)
                    
                    if explanation:
                        self.display_explanation(explanation)
                    else:
                        st.error("Failed to get explanation. Please try again.")
                
                except Exception as e:
                    st.error(f"Explanation failed: {str(e)}")
    
    def display_explanation(self, explanation: Dict[str, Any]):
        """Display model explanation"""
        st.success("✅ Explanation generated!")
        
        # Natural language explanation
        st.subheader("💬 Natural Language Explanation")
        st.info(explanation['natural_explanation'])
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        
        top_features = explanation['top_features']
        if top_features:
            features_df = pd.DataFrame(top_features)
            
            # Feature importance chart
            fig = px.bar(
                features_df,
                x='feature',
                y='abs_shap_value',
                title="Top Contributing Features",
                color='shap_value',
                color_continuous_scale='RdYlGn'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature details table
            st.subheader("🔍 Feature Details")
            st.dataframe(features_df, use_container_width=True)
        
        # Feature categories
        feature_categories = explanation.get('feature_categories', {})
        if feature_categories:
            st.subheader("📋 Feature Categories")
            
            for category, features in feature_categories.items():
                with st.expander(f"{category.title()} ({len(features)} features)"):
                    for feature in features:
                        st.write(f"**{feature['name']}**: {feature['value']:.4f} (Impact: {feature['shap_value']:.4f})")
        
        # Insights
        insights = explanation.get('insights', [])
        if insights:
            st.subheader("💡 Key Insights")
            for insight in insights:
                st.write(f"• {insight}")
    
    def render_news_page(self):
        """Render news page"""
        st.header("📰 Financial News & Sentiment")
        
        # Input form
        with st.form("news_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                symbols = st.multiselect(
                    "Select Stock Symbols",
                    options=settings.DEFAULT_SYMBOLS,
                    default=settings.DEFAULT_SYMBOLS[:3]
                )
            
            with col2:
                days = st.slider("Days to look back", 1, 30, 7)
            
            submitted = st.form_submit_button("Get News", use_container_width=True)
        
        if submitted:
            with st.spinner("Fetching news..."):
                try:
                    news_data = self.get_news(symbols, days)
                    
                    if news_data:
                        self.display_news(news_data)
                    else:
                        st.error("Failed to fetch news. Please try again.")
                
                except Exception as e:
                    st.error(f"News fetch failed: {str(e)}")
    
    def display_news(self, news_data: Dict[str, Any]):
        """Display news and sentiment analysis"""
        st.success("✅ News fetched successfully!")
        
        # Sentiment summary
        sentiment_summary = news_data['sentiment_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", sentiment_summary['total_articles'])
        
        with col2:
            st.metric("Positive", sentiment_summary['positive_count'], 
                     f"{sentiment_summary['sentiment_distribution'].get('positive', 0):.1f}%")
        
        with col3:
            st.metric("Negative", sentiment_summary['negative_count'],
                     f"{sentiment_summary['sentiment_distribution'].get('negative', 0):.1f}%")
        
        with col4:
            st.metric("Average Sentiment", f"{sentiment_summary['average_sentiment']:.3f}")
        
        st.markdown("---")
        
        # News articles
        articles = news_data['articles']
        
        if articles:
            st.subheader("📄 Recent Articles")
            
            for article in articles:
                with st.expander(f"{article['title']} - {article['source']}"):
                    st.write(f"**Published:** {article['published_date']}")
                    st.write(f"**Sentiment:** {article['sentiment_label']} ({article['sentiment_score']:.3f})")
                    st.write(f"**Language:** {article['language']}")
                    st.write(f"**Content:** {article['content']}")
                    st.write(f"**Link:** [Read more]({article['url']})")
        else:
            st.info("No recent articles found for the selected symbols.")
    
    def render_top_picks_page(self):
        """Render top picks page"""
        st.header("⭐ Top Stock Picks")
        
        # Input form
        with st.form("top_picks_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                timeframe = st.selectbox(
                    "Timeframe",
                    options=list(settings.TIMEFRAMES.keys()),
                    index=2,
                    key="top_picks_timeframe"
                )
            
            with col2:
                top_n = st.slider("Number of picks", 3, 10, 5)
            
            submitted = st.form_submit_button("Get Top Picks", use_container_width=True)
        
        if submitted:
            with st.spinner("Analyzing top picks..."):
                try:
                    top_picks = self.get_top_picks(timeframe, top_n)
                    
                    if top_picks:
                        self.display_top_picks(top_picks)
                    else:
                        st.error("Failed to get top picks. Please try again.")
                
                except Exception as e:
                    st.error(f"Top picks analysis failed: {str(e)}")
    
    def display_top_picks(self, top_picks: List[Dict[str, Any]]):
        """Display top picks"""
        st.success(f"✅ Found {len(top_picks)} top picks!")
        
        # Create dataframe
        picks_df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Symbol': pick['symbol'],
                'Current Price': f"₹{pick['current_price']:.2f}",
                'Predicted Price': f"₹{pick['predicted_price']:.2f}",
                'Expected Return': f"{pick['price_change_percent']:.2f}%",
                'Recommendation': pick['recommendation'],
                'Confidence': f"{pick['confidence']:.1%}",
                'Risk Level': pick['risk_level']
            }
            for i, pick in enumerate(top_picks)
        ])
        
        # Display table
        st.dataframe(picks_df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Expected returns chart
            fig = px.bar(
                picks_df,
                x='Symbol',
                y='Expected Return',
                color='Recommendation',
                title="Expected Returns by Stock",
                color_discrete_map={
                    'Strong Buy': '#28a745',
                    'Buy': '#20c997',
                    'Hold': '#ffc107',
                    'Sell': '#fd7e14',
                    'Strong Sell': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk vs Return scatter
            fig = px.scatter(
                picks_df,
                x='Expected Return',
                y='Risk Level',
                size='Confidence',
                hover_name='Symbol',
                title="Risk vs Return Analysis",
                color='Recommendation'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_data_page(self):
        """Render data visualization page"""
        st.header("📊 Historical Data Analysis")
        
        # Input form
        with st.form("data_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                symbol = st.selectbox(
                    "Select Stock Symbol",
                    options=settings.DEFAULT_SYMBOLS,
                    key="data_symbol"
                )
            
            with col2:
                period = st.selectbox(
                    "Period",
                    options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                    index=3
                )
            
            with col3:
                interval = st.selectbox(
                    "Interval",
                    options=['1d', '1wk', '1mo'],
                    index=0
                )
            
            submitted = st.form_submit_button("Load Data", use_container_width=True)
        
        if submitted:
            with st.spinner("Loading data..."):
                try:
                    data = self.get_stock_data(symbol, period, interval)
                    
                    if data:
                        self.display_data_analysis(data, symbol)
                    else:
                        st.error("Failed to load data. Please try again.")
                
                except Exception as e:
                    st.error(f"Data loading failed: {str(e)}")
    
    def display_data_analysis(self, data: Dict[str, Any], symbol: str):
        """Display data analysis"""
        st.success("✅ Data loaded successfully!")
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Price chart
        st.subheader(f"📈 {symbol} Price Chart")
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ))
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_20'],
            name="SMA 20",
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_50'],
            name="SMA 50",
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader("📊 Volume Analysis")
        
        fig = px.bar(
            df.reset_index(),
            x='date',
            y='volume',
            title="Trading Volume"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("📋 Data Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Data Points", len(df))
            st.metric("Date Range", f"{df.index.min().date()} to {df.index.max().date()}")
            st.metric("Current Price", f"₹{df['close'].iloc[-1]:.2f}")
        
        with col2:
            st.metric("Price Change", f"₹{df['close'].iloc[-1] - df['close'].iloc[0]:.2f}")
            st.metric("Price Change %", f"{((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
            st.metric("Average Volume", f"{df['volume'].mean():,.0f}")
    
    # API Methods
    def get_prediction(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get prediction from API"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/predict",
                json={"symbol": symbol, "timeframe": timeframe},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return None
    
    def get_explanation(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get explanation from API"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/explain",
                json={"symbol": symbol, "timeframe": timeframe},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return None
    
    def get_news(self, symbols: List[str], days: int) -> Optional[Dict[str, Any]]:
        """Get news from API"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/news",
                json={"symbols": symbols, "days": days},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return None
    
    def get_top_picks(self, timeframe: str, top_n: int) -> List[Dict[str, Any]]:
        """Get top picks from API"""
        try:
            response = self.session.get(
                f"{self.api_base_url}/top-picks",
                params={"timeframe": timeframe, "top_n": top_n},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return []
        
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return []
    
    def get_stock_data(self, symbol: str, period: str, interval: str) -> Optional[Dict[str, Any]]:
        """Get stock data from API"""
        try:
            response = self.session.get(
                f"{self.api_base_url}/data/{symbol}",
                params={"period": period, "interval": interval},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return None

# Run the application
if __name__ == "__main__":
    app = StockForecastingApp()
    app.run()
