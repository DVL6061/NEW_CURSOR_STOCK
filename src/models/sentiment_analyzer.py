"""
FinBERT-based sentiment analysis for financial news
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import sqlite3
from contextlib import contextmanager
import re
from dataclasses import dataclass

from config.settings import settings
from src.data.news_scraper import NewsArticle

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    model_name: str

class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial news"""
    
    def __init__(self):
        self.model_name = settings.FINBERT_MODEL_NAME
        self.max_length = settings.MAX_NEWS_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Sentiment labels mapping
        self.label_mapping = {
            0: "negative",
            1: "neutral", 
            2: "positive"
        }
        
        # Financial keywords for enhanced analysis
        self.financial_keywords = {
            'positive': [
                'bullish', 'growth', 'profit', 'gain', 'rise', 'increase', 'up', 'positive',
                'strong', 'robust', 'outperform', 'beat', 'exceed', 'surge', 'rally',
                'breakthrough', 'milestone', 'record', 'high', 'peak', 'boom'
            ],
            'negative': [
                'bearish', 'decline', 'loss', 'fall', 'decrease', 'down', 'negative',
                'weak', 'poor', 'underperform', 'miss', 'disappoint', 'plunge', 'crash',
                'crisis', 'recession', 'low', 'bottom', 'slump', 'downturn'
            ]
        }
    
    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # negative, neutral, positive
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"FinBERT model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of financial text
        
        Args:
            text: Text to analyze
        
        Returns:
            SentimentResult object
        """
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get sentiment score (-1 to 1)
            sentiment_score = self._calculate_sentiment_score(probabilities[0])
            
            # Get sentiment label
            sentiment_label = self.label_mapping[predicted_class]
            
            # Apply financial keyword enhancement
            enhanced_result = self._enhance_with_financial_keywords(
                processed_text, sentiment_score, sentiment_label, confidence
            )
            
            return SentimentResult(
                text=text,
                sentiment_score=enhanced_result['sentiment_score'],
                sentiment_label=enhanced_result['sentiment_label'],
                confidence=enhanced_result['confidence'],
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            # Return neutral sentiment as fallback
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                sentiment_label="neutral",
                confidence=0.5,
                model_name=self.model_name
            )
    
    def analyze_article(self, article: NewsArticle) -> NewsArticle:
        """
        Analyze sentiment of a news article
        
        Args:
            article: NewsArticle object
        
        Returns:
            Updated NewsArticle with sentiment analysis
        """
        try:
            # Combine title and content for analysis
            if not article.content:
                logger.warning(f"Article content is empty for {article.url}, using title only.")
                full_text = article.title
            else:
                full_text = f"{article.title}. {article.content}"
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(full_text)
            
            # Update article with sentiment information
            article.sentiment_score = sentiment_result.sentiment_score
            article.sentiment_label = sentiment_result.sentiment_label
            
            return article
            
        except Exception as e:
            logger.error(f"Error analyzing article sentiment for {article.url}: {str(e)}")
            # Set neutral sentiment as fallback
            article.sentiment_score = 0.0
            article.sentiment_label = "neutral"
            return article
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        try:
            # Process texts in batches for efficiency
            batch_size = 8
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Preprocess batch
                processed_texts = [self._preprocess_text(text) for text in batch_texts]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    processed_texts,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                
                # Process each result in the batch
                for j, text in enumerate(batch_texts):
                    original_text = batch_texts[j]
                    predicted_class = torch.argmax(probabilities[j], dim=-1).item()
                    confidence = probabilities[j][predicted_class].item()
                    sentiment_score = self._calculate_sentiment_score(probabilities[j])
                    sentiment_label = self.label_mapping[predicted_class]
                    
                    # Apply financial keyword enhancement
                    enhanced_result = self._enhance_with_financial_keywords(
                        processed_texts[j], sentiment_score, sentiment_label, confidence
                    )
                    
                    results.append(SentimentResult(
                        text=original_text,
                        sentiment_score=enhanced_result['sentiment_score'],
                        sentiment_label=enhanced_result['sentiment_label'],
                        confidence=enhanced_result['confidence'],
                        model_name=self.model_name
                    ))
            
        except Exception as e:
            logger.error(f"Error analyzing batch sentiment: {str(e)}")
            # Return neutral sentiment for all texts as fallback
            results = [
                SentimentResult(
                    text=text,
                    sentiment_score=0.0,
                    sentiment_label="neutral",
                    confidence=0.5,
                    model_name=self.model_name
                ) for text in texts
            ]
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Remove URLs
            text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)

            # Remove special characters but keep financial symbols
            text = re.sub(r'[^\w\s$%.,!?]', ' ', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove very short texts
            if len(text.split()) < 3:
                return "neutral market sentiment"
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def _calculate_sentiment_score(self, probabilities: torch.Tensor) -> float:
        """Calculate sentiment score from probabilities"""
        try:
            # Convert probabilities to sentiment score (-1 to 1)
            # negative: -1, neutral: 0, positive: 1
            negative_prob = probabilities[0].item()
            neutral_prob = probabilities[1].item()
            positive_prob = probabilities[2].item()
            
            # Weighted average
            sentiment_score = (-1 * negative_prob) + (0 * neutral_prob) + (1 * positive_prob)
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 0.0
    
    def _enhance_with_financial_keywords(self, text: str, sentiment_score: float, 
                                       sentiment_label: str, confidence: float) -> Dict:
        """Enhance sentiment analysis with financial keyword detection"""
        try:
            enhanced_score = sentiment_score
            enhanced_label = sentiment_label
            enhanced_confidence = confidence
            
            # Count financial keywords
            positive_keywords = sum(1 for keyword in self.financial_keywords['positive'] 
                                  if keyword in text.lower())
            negative_keywords = sum(1 for keyword in self.financial_keywords['negative'] 
                                   if keyword in text.lower())
            
            # Adjust sentiment based on keyword presence
            if positive_keywords > negative_keywords:
                # Boost positive sentiment
                enhanced_score = min(1.0, enhanced_score + 0.1 * positive_keywords)
                if enhanced_score > 0.2:
                    enhanced_label = "positive"
                enhanced_confidence = min(1.0, enhanced_confidence + 0.1)
                
            elif negative_keywords > positive_keywords:
                # Boost negative sentiment
                enhanced_score = max(-1.0, enhanced_score - 0.1 * negative_keywords)
                if enhanced_score < -0.2:
                    enhanced_label = "negative"
                enhanced_confidence = min(1.0, enhanced_confidence + 0.1)
            
            return {
                'sentiment_score': enhanced_score,
                'sentiment_label': enhanced_label,
                'confidence': enhanced_confidence
            }
            
        except Exception as e:
            logger.error(f"Error enhancing with financial keywords: {str(e)}")
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence
            }
    
    def get_sentiment_summary(self, articles: List[NewsArticle]) -> Dict:
        """
        Get sentiment summary for a list of articles
        
        Args:
            articles: List of NewsArticle objects
        
        Returns:
            Dictionary with sentiment summary statistics
        """
        try:
            if not articles:
                return {
                    'total_articles': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'average_sentiment': 0.0,
                    'sentiment_distribution': {}
                }
            
            # Count sentiments
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            sentiment_scores = []
            
            for article in articles:
                if article.sentiment_label:
                    sentiment_counts[article.sentiment_label] += 1
                if article.sentiment_score is not None:
                    sentiment_scores.append(article.sentiment_score)
            
            # Calculate statistics
            total_articles = len(articles)
            average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Calculate distribution percentages
            sentiment_distribution = {
                label: (count / total_articles) * 100 if total_articles > 0 else 0
                for label, count in sentiment_counts.items()
            }
            
            return {
                'total_articles': total_articles,
                'positive_count': sentiment_counts['positive'],
                'negative_count': sentiment_counts['negative'],
                'neutral_count': sentiment_counts['neutral'],
                'average_sentiment': average_sentiment,
                'sentiment_distribution': sentiment_distribution
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {str(e)}")
            return {
                'total_articles': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'average_sentiment': 0.0,
                'sentiment_distribution': {}
            }
    
    def update_article_sentiment_in_db(self, article: NewsArticle):
        """Update article sentiment in database"""
        try:
            db_path = settings.DATA_DIR / "news_data.db"
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    UPDATE news_articles 
                    SET sentiment_score = ?, sentiment_label = ?
                    WHERE url = ?
                """, (article.sentiment_score, article.sentiment_label, article.url))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating article sentiment in database: {str(e)}")

# Global instance
sentiment_analyzer = FinBERTSentimentAnalyzer()
