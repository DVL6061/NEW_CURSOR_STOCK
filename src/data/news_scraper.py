"""
Financial news scraping module for multiple sources
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import re
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    language: str = "en"
    translated_title: Optional[str] = None
    translated_content: Optional[str] = None

class NewsScraper:
    """Multi-source financial news scraper"""
    
    def __init__(self):
        self.db_path = settings.DATA_DIR / "news_data.db"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for news storage"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT,
                    url TEXT UNIQUE,
                    source TEXT,
                    published_date DATETIME,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    language TEXT DEFAULT 'en',
                    translated_title TEXT,
                    translated_content TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
    
    def scrape_all_sources(self, symbols: List[str], max_articles_per_source: int = 50) -> List[NewsArticle]:
        """
        Scrape news from all configured sources
        
        Args:
            symbols: List of stock symbols to search for
            max_articles_per_source: Maximum articles to fetch per source
        
        Returns:
            List of NewsArticle objects
        """
        all_articles = []
        
        for source_name, base_url in settings.NEWS_SOURCES.items():
            try:
                logger.info(f"Scraping news from {source_name}")
                articles = self._scrape_source(source_name, base_url, symbols, max_articles_per_source)
                all_articles.extend(articles)
                logger.info(f"Scraped {len(articles)} articles from {source_name}")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {str(e)}")
                continue
        
        return all_articles
    
    def _scrape_source(self, source_name: str, base_url: str, symbols: List[str], max_articles: int) -> List[NewsArticle]:
        """Scrape news from a specific source"""
        articles = []
        
        try:
            if source_name == "cnbc":
                articles = self._scrape_cnbc(base_url, symbols, max_articles)
            elif source_name == "moneycontrol":
                articles = self._scrape_moneycontrol(base_url, symbols, max_articles)
            elif source_name == "mint":
                articles = self._scrape_mint(base_url, symbols, max_articles)
            elif source_name == "economic_times":
                articles = self._scrape_economic_times(base_url, symbols, max_articles)
            elif source_name == "business_standard":
                articles = self._scrape_business_standard(base_url, symbols, max_articles)
            
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {str(e)}")
        
        return articles
    
    def _scrape_cnbc(self, base_url: str, symbols: List[str], max_articles: int) -> List[NewsArticle]:
        """Scrape news from CNBC"""
        articles = []
        
        try:
            # CNBC search for Indian stocks
            search_url = f"https://www.cnbc.com/search/?query={'%20'.join(symbols)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:max_articles]:
                href = link.get('href')
                if href and '/news/' in href:
                    full_url = urljoin(base_url, href)
                    article = self._scrape_article_content(full_url, "CNBC")
                    if article:
                        articles.append(article)
                        time.sleep(1)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Error scraping CNBC: {str(e)}")
        
        return articles
    
    def _scrape_moneycontrol(self, base_url: str, symbols: List[str], max_articles: int) -> List[NewsArticle]:
        """Scrape news from Moneycontrol"""
        articles = []
        
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:max_articles]:
                href = link.get('href')
                if href and '/news/' in href:
                    full_url = urljoin(base_url, href)
                    article = self._scrape_article_content(full_url, "Moneycontrol")
                    if article:
                        articles.append(article)
                        time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error scraping Moneycontrol: {str(e)}")
        
        return articles
    
    def _scrape_mint(self, base_url: str, symbols: List[str], max_articles: int) -> List[NewsArticle]:
        """Scrape news from Mint"""
        articles = []
        
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:max_articles]:
                href = link.get('href')
                if href and '/market/' in href:
                    full_url = urljoin(base_url, href)
                    article = self._scrape_article_content(full_url, "Mint")
                    if article:
                        articles.append(article)
                        time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error scraping Mint: {str(e)}")
        
        return articles
    
    def _scrape_economic_times(self, base_url: str, symbols: List[str], max_articles: int) -> List[NewsArticle]:
        """Scrape news from Economic Times"""
        articles = []
        
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:max_articles]:
                href = link.get('href')
                if href and '/markets/' in href:
                    full_url = urljoin(base_url, href)
                    article = self._scrape_article_content(full_url, "Economic Times")
                    if article:
                        articles.append(article)
                        time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error scraping Economic Times: {str(e)}")
        
        return articles
    
    def _scrape_business_standard(self, base_url: str, symbols: List[str], max_articles: int) -> List[NewsArticle]:
        """Scrape news from Business Standard"""
        articles = []
        
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:max_articles]:
                href = link.get('href')
                if href and '/markets/' in href:
                    full_url = urljoin(base_url, href)
                    article = self._scrape_article_content(full_url, "Business Standard")
                    if article:
                        articles.append(article)
                        time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error scraping Business Standard: {str(e)}")
        
        return articles
    
    def _scrape_article_content(self, url: str, source: str) -> Optional[NewsArticle]:
        """Scrape content from a specific article URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, source)
            if not title:
                return None
            
            # Extract content
            content = self._extract_content(soup, source)
            if not content:
                return None
            
            # Extract published date
            published_date = self._extract_published_date(soup, source)
            if not published_date:
                published_date = datetime.now()
            
            # Detect language
            language = self._detect_language(title + " " + content)
            
            return NewsArticle(
                title=title,
                content=content,
                url=url,
                source=source,
                published_date=published_date,
                language=language
            )
        
        except Exception as e:
            logger.error(f"Error scraping article content from {url}: {str(e)}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, source: str) -> Optional[str]:
        """Extract article title based on source"""
        title_selectors = {
            "CNBC": ["h1", ".ArticleHeader-headline", ".headline"],
            "Moneycontrol": ["h1", ".article_title", ".title"],
            "Mint": ["h1", ".headline", ".title"],
            "Economic Times": ["h1", ".artTitle", ".title"],
            "Business Standard": ["h1", ".story-title", ".title"]
        }
        
        selectors = title_selectors.get(source, ["h1", ".title", ".headline"])
        
        for selector in selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                return title_elem.get_text().strip()
        
        return None
    
    def _extract_content(self, soup: BeautifulSoup, source: str) -> Optional[str]:
        """Extract article content based on source"""
        content_selectors = {
            "CNBC": [".ArticleBody-articleBody", ".story-content", ".content"],
            "Moneycontrol": [".content_wrapper", ".article_content", ".content"],
            "Mint": [".story-content", ".content", ".article-content"],
            "Economic Times": [".artText", ".content", ".article-content"],
            "Business Standard": [".story-content", ".content", ".article-content"]
        }
        
        selectors = content_selectors.get(source, [".content", ".article-content", ".story-content"])
        
        for selector in selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove script and style elements
                for script in content_elem(["script", "style"]):
                    script.decompose()
                
                content = content_elem.get_text().strip()
                if len(content) > 100:  # Ensure meaningful content
                    return content
        
        return None
    
    def _extract_published_date(self, soup: BeautifulSoup, source: str) -> Optional[datetime]:
        """Extract published date based on source"""
        date_selectors = {
            "CNBC": [".ArticleHeader-timestamp", ".timestamp", "time"],
            "Moneycontrol": [".article_date", ".date", "time"],
            "Mint": [".story-date", ".date", "time"],
            "Economic Times": [".artDate", ".date", "time"],
            "Business Standard": [".story-date", ".date", "time"]
        }
        
        selectors = date_selectors.get(source, [".date", "time"])
        
        for selector in selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_text = date_elem.get_text().strip()
                try:
                    # Try to parse various date formats
                    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y", "%d %B %Y"]:
                        try:
                            return datetime.strptime(date_text, fmt)
                        except ValueError:
                            continue
                except:
                    continue
        
        return None
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection for English/Hindi"""
        # Check for Hindi characters (Devanagari script)
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        if hindi_pattern.search(text):
            return "hi"
        return "en"
    
    def save_articles(self, articles: List[NewsArticle]):
        """Save articles to database"""
        try:
            with self.get_db_connection() as conn:
                for article in articles:
                    conn.execute("""
                        INSERT OR REPLACE INTO news_articles 
                        (title, content, url, source, published_date, language)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        article.title,
                        article.content,
                        article.url,
                        article.source,
                        article.published_date,
                        article.language
                    ))
                conn.commit()
                logger.info(f"Saved {len(articles)} articles to database")
        
        except Exception as e:
            logger.error(f"Error saving articles to database: {str(e)}")
    
    def get_recent_articles(self, days: int = 7, symbols: List[str] = None) -> List[NewsArticle]:
        """Get recent articles from database"""
        try:
            with self.get_db_connection() as conn:
                query = """
                    SELECT * FROM news_articles 
                    WHERE published_date >= datetime('now', '-{} days')
                    ORDER BY published_date DESC
                """.format(days)
                
                df = pd.read_sql_query(query, conn)
                
                articles = []
                for _, row in df.iterrows():
                    article = NewsArticle(
                        title=row['title'],
                        content=row['content'],
                        url=row['url'],
                        source=row['source'],
                        published_date=pd.to_datetime(row['published_date']),
                        sentiment_score=row['sentiment_score'],
                        sentiment_label=row['sentiment_label'],
                        language=row['language'],
                        translated_title=row['translated_title'],
                        translated_content=row['translated_content']
                    )
                    articles.append(article)
                
                return articles
        
        except Exception as e:
            logger.error(f"Error retrieving recent articles: {str(e)}")
            return []
    
    def translate_article(self, article: NewsArticle, target_language: str = "hi") -> NewsArticle:
        """Translate article content (placeholder for translation service)"""
        # This would integrate with Google Translate or similar service
        # For now, return the original article
        return article

# Global instance
news_scraper = NewsScraper()
