"""
SHAP explainability module for model predictions
"""

import shap
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from src.models.xgboost_model import xgboost_predictor
from src.models.ensemble_model import ensemble_predictor

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """SHAP-based model explainability"""
    
    def __init__(self):
        self.explainer = None
        self.feature_names = None
        self.explainer_path = settings.MODELS_DIR / "shap_explainer.pkl"
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer"""
        try:
            # Load XGBoost model if available
            if xgboost_predictor.model is not None:
                self.explainer = shap.TreeExplainer(xgboost_predictor.model)
                self.feature_names = xgboost_predictor.feature_names
                logger.info("SHAP TreeExplainer initialized for XGBoost model")
            else:
                logger.warning("XGBoost model not available for SHAP explainer")
                
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
    
    def explain_prediction(self, X: pd.DataFrame, prediction_result: Any = None) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP
        
        Args:
            X: Input features
            prediction_result: Prediction result object
        
        Returns:
            Dictionary with SHAP explanations
        """
        try:
            if self.explainer is None or xgboost_predictor.model is None:
                logger.warning("SHAP explainer not initialized or model missing; using fallback explanation")
                return self._get_fallback_explanation(X, prediction_result)
            
            # Prepare features for SHAP
            X_processed = self._prepare_features_for_shap(X)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            # Get base value
            base_value = self.explainer.expected_value
            
            # Create explanation summary
            explanation = self._create_explanation_summary(
                X_processed, shap_values, base_value, prediction_result
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return self._get_fallback_explanation(X, prediction_result)
    
    def explain_multiple_predictions(self, X: pd.DataFrame, 
                                   prediction_results: List[Any] = None) -> Dict[str, Any]:
        """
        Explain multiple predictions
        
        Args:
            X: Input features for multiple samples
            prediction_results: List of prediction results
        
        Returns:
            Dictionary with SHAP explanations for multiple predictions
        """
        try:
            if self.explainer is None:
                logger.warning("SHAP explainer not initialized")
                return self._get_fallback_explanation(X, prediction_results)
            
            # Prepare features for SHAP
            X_processed = self._prepare_features_for_shap(X)
            
            # Calculate SHAP values for all samples
            shap_values = self.explainer.shap_values(X_processed)
            
            # Get base value
            base_value = self.explainer.expected_value
            
            # Create comprehensive explanation
            explanation = self._create_comprehensive_explanation(
                X_processed, shap_values, base_value, prediction_results
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining multiple predictions: {str(e)}")
            return self._get_fallback_explanation(X, prediction_results)
    
    def _prepare_features_for_shap(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for SHAP analysis"""
        try:
            # Select only the features used in the model
            if self.feature_names:
                cols = [c for c in self.feature_names if c in X.columns]
                if not cols:
                    raise ValueError("No overlapping features for SHAP explanation")
                X_selected = X[cols].copy()
            else:
                X_selected = X.select_dtypes(include=['number']).copy()
            
            # Handle missing values
            X_selected = X_selected.fillna(0)
            
            # Convert to numpy array
            X_array = X_selected.values
            
            return X_array
            
        except Exception as e:
            logger.error(f"Error preparing features for SHAP: {str(e)}")
            return X.values if hasattr(X, 'values') else np.array(X)
    
    def _create_explanation_summary(self, X: np.ndarray, shap_values: np.ndarray, 
                                  base_value: float, prediction_result: Any = None) -> Dict[str, Any]:
        """Create explanation summary for a single prediction"""
        try:
            # Get feature names
            feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(X.shape[1])]
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'value': X[0] if len(X.shape) > 1 else X,
                'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                'importance': np.abs(shap_values[0] if len(shap_values.shape) > 1 else shap_values)
            })
            
            # Sort by absolute SHAP value
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Get top contributing features
            top_features = feature_importance.head(10)
            
            # Categorize features
            feature_categories = self._categorize_features(top_features)
            
            # Generate natural language explanation
            natural_explanation = self._generate_natural_explanation(
                top_features, base_value, prediction_result
            )
            
            # Calculate prediction confidence based on SHAP values
            shap_confidence = self._calculate_shap_confidence(shap_values)
            
            return {
                'base_value': base_value,
                'prediction_confidence': shap_confidence,
                'top_features': top_features.to_dict('records'),
                'feature_categories': feature_categories,
                'natural_explanation': natural_explanation,
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'feature_names': feature_names
            }
            
        except Exception as e:
            logger.error(f"Error creating explanation summary: {str(e)}")
            return {}
    
    def _create_comprehensive_explanation(self, X: np.ndarray, shap_values: np.ndarray, 
                                         base_value: float, prediction_results: List[Any] = None) -> Dict[str, Any]:
        """Create comprehensive explanation for multiple predictions"""
        try:
            # Get feature names
            feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(X.shape[1])]
            
            # Calculate mean absolute SHAP values across all samples
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance summary
            feature_importance_summary = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap,
                'mean_shap': np.mean(shap_values, axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            
            # Get top contributing features
            top_features = feature_importance_summary.head(15)
            
            # Analyze feature interactions
            feature_interactions = self._analyze_feature_interactions(X, shap_values, feature_names)
            
            # Generate comprehensive insights
            insights = self._generate_comprehensive_insights(
                top_features, feature_interactions, prediction_results
            )
            
            return {
                'base_value': base_value,
                'feature_importance_summary': top_features.to_dict('records'),
                'feature_interactions': feature_interactions,
                'insights': insights,
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'feature_names': feature_names
            }
            
        except Exception as e:
            logger.error(f"Error creating comprehensive explanation: {str(e)}")
            return {}
    
    def _categorize_features(self, feature_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Categorize features by type"""
        try:
            categories = {
                'technical_indicators': [],
                'price_features': [],
                'fundamental_metrics': [],
                'sentiment_features': [],
                'volume_features': [],
                'other': []
            }
            
            for _, row in feature_df.iterrows():
                feature_name = row['feature']
                feature_info = {
                    'name': feature_name,
                    'value': row['value'],
                    'shap_value': row['shap_value'],
                    'importance': row['importance']
                }
                
                # Categorize based on feature name
                if any(indicator in feature_name.lower() for indicator in ['rsi', 'macd', 'ema', 'sma', 'bb', 'adx', 'stoch']):
                    categories['technical_indicators'].append(feature_info)
                elif any(price in feature_name.lower() for price in ['open', 'high', 'low', 'close', 'price']):
                    categories['price_features'].append(feature_info)
                elif any(fund in feature_name.lower() for fund in ['pe_ratio', 'pb_ratio', 'roe', 'roi', 'margin']):
                    categories['fundamental_metrics'].append(feature_info)
                elif 'sentiment' in feature_name.lower():
                    categories['sentiment_features'].append(feature_info)
                elif 'volume' in feature_name.lower():
                    categories['volume_features'].append(feature_info)
                else:
                    categories['other'].append(feature_info)
            
            # Remove empty categories
            categories = {k: v for k, v in categories.items() if v}
            
            return categories
            
        except Exception as e:
            logger.error(f"Error categorizing features: {str(e)}")
            return {}
    
    def _generate_natural_explanation(self, top_features: pd.DataFrame, 
                                     base_value: float, prediction_result: Any = None) -> str:
        """Generate natural language explanation"""
        try:
            explanation_parts = []
            
            # Start with base prediction
            explanation_parts.append(f"The model's base prediction is {base_value:.4f}.")
            
            # Explain top contributing factors
            for i, (_, feature) in enumerate(top_features.head(5).iterrows()):
                feature_name = feature['feature']
                feature_value = feature['value']
                shap_value = feature['shap_value']
                
                # Convert feature name to readable format
                readable_name = self._make_feature_readable(feature_name)
                
                if shap_value > 0:
                    explanation_parts.append(
                        f"{readable_name} ({feature_value:.4f}) contributes positively to the prediction (+{shap_value:.4f})."
                    )
                else:
                    explanation_parts.append(
                        f"{readable_name} ({feature_value:.4f}) contributes negatively to the prediction ({shap_value:.4f})."
                    )
            
            # Add recommendation context if available
            if prediction_result and hasattr(prediction_result, 'recommendation'):
                explanation_parts.append(f"This leads to a {prediction_result.recommendation} recommendation.")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating natural explanation: {str(e)}")
            return "Unable to generate explanation due to technical issues."
    
    def _make_feature_readable(self, feature_name: str) -> str:
        """Convert feature name to readable format"""
        try:
            # Common feature name mappings
            mappings = {
                'rsi': 'RSI (Relative Strength Index)',
                'macd': 'MACD',
                'ema_9': '9-day Exponential Moving Average',
                'ema_21': '21-day Exponential Moving Average',
                'ema_50': '50-day Exponential Moving Average',
                'sma_200': '200-day Simple Moving Average',
                'bb_percent': 'Bollinger Bands Position',
                'adx': 'ADX (Average Directional Index)',
                'volume_ratio': 'Volume Ratio',
                'sentiment_score': 'News Sentiment Score',
                'pe_ratio': 'P/E Ratio',
                'pb_ratio': 'P/B Ratio',
                'roe': 'Return on Equity',
                'price_change': 'Price Change',
                'trend_strength': 'Trend Strength'
            }
            
            return mappings.get(feature_name.lower(), feature_name.replace('_', ' ').title())
            
        except Exception as e:
            logger.error(f"Error making feature readable: {str(e)}")
            return feature_name
    
    def _analyze_feature_interactions(self, X: np.ndarray, shap_values: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature interactions"""
        try:
            # Calculate correlation between SHAP values
            shap_corr = np.corrcoef(shap_values.T)
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    corr_value = shap_corr[i, j]
                    if abs(corr_value) > 0.5:  # Strong correlation threshold
                        strong_correlations.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                        })
            
            # Sort by correlation strength
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'strong_correlations': strong_correlations[:10],  # Top 10 interactions
                'correlation_matrix': shap_corr.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {str(e)}")
            return {}
    
    def _generate_comprehensive_insights(self, top_features: pd.DataFrame, 
                                       feature_interactions: Dict, 
                                       prediction_results: List[Any] = None) -> List[str]:
        """Generate comprehensive insights"""
        try:
            insights = []
            
            # Feature importance insights
            top_feature = top_features.iloc[0]
            insights.append(
                f"The most important factor is {self._make_feature_readable(top_feature['feature'])} "
                f"with an average impact of {top_feature['mean_abs_shap']:.4f}."
            )
            
            # Technical indicators insights
            tech_features = [f for f in top_features['feature'] if any(
                indicator in f.lower() for indicator in ['rsi', 'macd', 'ema', 'sma', 'bb']
            )]
            if tech_features:
                insights.append(
                    f"Technical indicators ({len(tech_features)} features) are highly influential "
                    "in the model's predictions."
                )
            
            # Sentiment insights
            sentiment_features = [f for f in top_features['feature'] if 'sentiment' in f.lower()]
            if sentiment_features:
                insights.append(
                    "News sentiment analysis plays a significant role in the prediction model."
                )
            
            # Feature interaction insights
            if feature_interactions.get('strong_correlations'):
                insights.append(
                    f"Strong feature interactions detected between "
                    f"{len(feature_interactions['strong_correlations'])} feature pairs."
                )
            
            # Prediction consistency insights
            if prediction_results:
                recommendations = [r.recommendation for r in prediction_results if hasattr(r, 'recommendation')]
                if recommendations:
                    most_common_rec = max(set(recommendations), key=recommendations.count)
                    insights.append(
                        f"The most common recommendation across predictions is '{most_common_rec}'."
                    )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating comprehensive insights: {str(e)}")
            return ["Unable to generate insights due to technical issues."]
    
    def _calculate_shap_confidence(self, shap_values: np.ndarray) -> float:
        """Calculate prediction confidence based on SHAP values"""
        try:
            # Confidence based on SHAP value consistency
            if len(shap_values.shape) > 1:
                shap_std = np.std(shap_values, axis=0)
                mean_shap_std = np.mean(shap_std)
                confidence = max(0.1, 1.0 - mean_shap_std)
            else:
                confidence = 0.8  # Default confidence for single prediction
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating SHAP confidence: {str(e)}")
            return 0.5
    
    def _get_fallback_explanation(self, X: pd.DataFrame, prediction_result: Any = None) -> Dict[str, Any]:
        """Get fallback explanation when SHAP is not available"""
        try:
            # Simple feature importance based on feature values
            if hasattr(X, 'values'):
                feature_values = X.values[0] if len(X.shape) > 1 else X.values
                feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(len(feature_values))]
            else:
                feature_values = X
                feature_names = [f"feature_{i}" for i in range(len(feature_values))]
            
            # Create simple importance based on absolute values (numeric only)
            try:
                fv = np.array(feature_values, dtype=float)
                importance = np.abs(fv)
            except Exception:
                # If conversion fails, fall back to zeros to avoid crashes
                importance = np.zeros(len(feature_values))
            importance_indices = np.argsort(importance)[::-1]
            
            top_features = []
            for i in importance_indices[:10]:
                top_features.append({
                    'feature': feature_names[i],
                    'value': feature_values[i],
                    'importance': importance[i]
                })
            
            return {
                'base_value': 0.0,
                'prediction_confidence': 0.5,
                'top_features': top_features,
                'natural_explanation': "SHAP explainer not available. Showing basic feature importance.",
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"Error creating fallback explanation: {str(e)}")
            return {
                'base_value': 0.0,
                'prediction_confidence': 0.5,
                'top_features': [],
                'natural_explanation': "Unable to generate explanation.",
                'fallback': True
            }
    
    def save_explainer(self):
        """Save SHAP explainer"""
        try:
            settings.MODELS_DIR.mkdir(exist_ok=True)
            
            if self.explainer is not None:
                joblib.dump(self.explainer, self.explainer_path)
                logger.info(f"SHAP explainer saved to {self.explainer_path}")
            
        except Exception as e:
            logger.error(f"Error saving SHAP explainer: {str(e)}")
    
    def load_explainer(self):
        """Load SHAP explainer"""
        try:
            if self.explainer_path.exists():
                self.explainer = joblib.load(self.explainer_path)
                logger.info("SHAP explainer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SHAP explainer: {str(e)}")

# Global instance
shap_explainer = SHAPExplainer()
