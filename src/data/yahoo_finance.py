"""
Yahoo Finance data acquisition module for Indian stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
from contextlib import contextmanager

from config.settings import settings

logger = logging.getLogger(__name__)

class YahooFinanceData:
    """Yahoo Finance data acquisition and management"""
    
    def __init__(self):
        self.db_path = settings.DATA_DIR / "yahoo_data.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for data storage"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    interval TEXT,
                    PRIMARY KEY (symbol, date, interval)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    symbol TEXT,
                    date DATE,
                    pe_ratio REAL,
                    pb_ratio REAL,
                    roe REAL,
                    roi REAL,
                    market_cap REAL,
                    enterprise_value REAL,
                    debt_to_equity REAL,
                    current_ratio REAL,
                    quick_ratio REAL,
                    gross_margin REAL,
                    operating_margin REAL,
                    net_margin REAL,
                    revenue REAL,
                    net_income REAL,
                    total_debt REAL,
                    total_equity REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
    
    @contextmanager
    def get_db_connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def fetch_ohlcv_data(
        self, 
        symbol: str, 
        period: str = "5y", 
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'TATAMOTORS.NS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            if use_cache:
                cached_data = self._get_cached_ohlcv(symbol, interval)
                if cached_data is not None and not cached_data.empty:
                    logger.info(f"Using cached OHLCV data for {symbol}")
                    return cached_data
            
            # Fetch fresh data
            logger.info(f"Fetching OHLCV data for {symbol} ({period}, {interval})")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean and format data
            data = self._clean_ohlcv_data(data, symbol, interval)
            
            # Cache the data
            if use_cache:
                self._cache_ohlcv_data(data, symbol, interval)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            raise
    
    def fetch_fundamental_data(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        Fetch fundamental data for a symbol
        
        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data
        
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            # Check cache first
            if use_cache:
                cached_data = self._get_cached_fundamentals(symbol)
                if cached_data:
                    logger.info(f"Using cached fundamental data for {symbol}")
                    return cached_data
            
            # Fetch fresh data
            logger.info(f"Fetching fundamental data for {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamental_data = self._extract_fundamental_metrics(info, symbol)
            
            # Cache the data
            if use_cache:
                self._cache_fundamental_data(fundamental_data, symbol)
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        period: str = "5y", 
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data_dict[symbol] = self.fetch_ohlcv_data(symbol, period, interval)
                logger.info(f"Successfully fetched data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return data_dict
    
    def _clean_ohlcv_data(self, data: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """Clean and format OHLCV data"""
        # Reset index to make date a column
        data = data.reset_index()
        
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Keep only the columns we need for our database schema
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj close']
        available_columns = [col for col in required_columns if col in data.columns]
        data = data[available_columns]
        
        # Add symbol and interval columns
        data['symbol'] = symbol
        data['interval'] = interval
        
        # Handle missing values
        data = data.dropna()
        
        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'adj close']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _extract_fundamental_metrics(self, info: Dict, symbol: str) -> Dict:
        """Extract key fundamental metrics from Yahoo Finance info"""
        metrics = {
            'symbol': symbol,
            'date': datetime.now().date(),
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'roe': info.get('returnOnEquity'),
            'roi': info.get('returnOnInvestment'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'gross_margin': info.get('grossMargins'),
            'operating_margin': info.get('operatingMargins'),
            'net_margin': info.get('profitMargins'),
            'revenue': info.get('totalRevenue'),
            'net_income': info.get('netIncomeToCommon'),
            'total_debt': info.get('totalDebt'),
            'total_equity': info.get('totalStockholderEquity')
        }
        
        # Convert None values to NaN for consistency
        for key, value in metrics.items():
            if value is None:
                metrics[key] = np.nan
        
        return metrics
    
    def _cache_ohlcv_data(self, data: pd.DataFrame, symbol: str, interval: str):
        """Cache OHLCV data to database"""
        try:
            with self.get_db_connection() as conn:
                # Remove existing data for this symbol and interval
                conn.execute(
                    "DELETE FROM ohlcv_data WHERE symbol = ? AND interval = ?",
                    (symbol, interval)
                )
                
                # Insert new data
                data.to_sql('ohlcv_data', conn, if_exists='append', index=False)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching OHLCV data for {symbol}: {str(e)}")
    
    def _get_cached_ohlcv(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Retrieve cached OHLCV data"""
        try:
            with self.get_db_connection() as conn:
                query = """
                    SELECT * FROM ohlcv_data 
                    WHERE symbol = ? AND interval = ?
                    ORDER BY date DESC
                """
                data = pd.read_sql_query(query, conn, params=(symbol, interval))
                
                if not data.empty:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                    return data
                
        except Exception as e:
            logger.error(f"Error retrieving cached OHLCV data for {symbol}: {str(e)}")
        
        return None
    
    def _cache_fundamental_data(self, data: Dict, symbol: str):
        """Cache fundamental data to database"""
        try:
            with self.get_db_connection() as conn:
                # Remove existing data for this symbol
                conn.execute("DELETE FROM fundamental_data WHERE symbol = ?", (symbol,))
                
                # Insert new data
                df = pd.DataFrame([data])
                df.to_sql('fundamental_data', conn, if_exists='append', index=False)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching fundamental data for {symbol}: {str(e)}")
    
    def _get_cached_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Retrieve cached fundamental data"""
        try:
            with self.get_db_connection() as conn:
                query = """
                    SELECT * FROM fundamental_data 
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                """
                data = pd.read_sql_query(query, conn, params=(symbol,))
                
                if not data.empty:
                    return data.iloc[0].to_dict()
                
        except Exception as e:
            logger.error(f"Error retrieving cached fundamental data for {symbol}: {str(e)}")
        
        return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest closing price for a symbol"""
        try:
            data = self.fetch_ohlcv_data(symbol, period="1d", interval="1d")
            if not data.empty:
                return float(data['close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
        
        return None
    
    def get_price_history(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get price history for a specific date range"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol} in the specified date range")
            
            return self._clean_ohlcv_data(data, symbol, interval)
            
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {str(e)}")
            raise

# Global instance
yahoo_data = YahooFinanceData()
