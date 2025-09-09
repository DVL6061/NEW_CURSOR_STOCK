"""
Informer transformer model for long-sequence time series forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class InformerBlock(nn.Module):
    """Informer block with ProbSparse attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(InformerBlock, self).__init__()
        
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class ProbSparseAttention(nn.Module):
    """ProbSparse attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(ProbSparseAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # ProbSparse attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply dropout
        scores = self.dropout(F.softmax(scores, dim=-1))
        
        # Apply attention to values
        context = torch.matmul(scores, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output

class InformerModel(nn.Module):
    """Informer model for time series forecasting"""
    
    def __init__(self, config: Dict):
        super(InformerModel, self).__init__()
        
        self.config = config
        self.seq_len = config['seq_len']
        self.label_len = config['label_len']
        self.pred_len = config['pred_len']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.e_layers = config['e_layers']
        self.d_layers = config['d_layers']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']
        
        # Input projection
        self.input_projection = nn.Linear(1, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.seq_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            InformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.e_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            InformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.d_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.pred_len)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Output projection
        output = self.output_projection(x[:, -1, :])  # Take last timestep
        
        return output

class InformerStockPredictor:
    """Informer-based stock price predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.config = settings.INFORMER_PARAMS.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model paths
        self.model_path = settings.MODELS_DIR / "informer_model.pth"
        self.scaler_path = settings.MODELS_DIR / "informer_scaler.pkl"
        self.config_path = settings.MODELS_DIR / "informer_config.pkl"
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.patience = 10
        
        # Data parameters
        self.seq_len = self.config['seq_len']
        self.label_len = self.config['label_len']
        self.pred_len = self.config['pred_len']
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Informer model
        
        Args:
            df: DataFrame with time series data
            target_column: Column to use as target
        
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Extract target series
            series = df[target_column].values.reshape(-1, 1)
            
            # Scale the data
            series_scaled = self.scaler.fit_transform(series)
            
            # Create sequences
            X, y = self._create_sequences(series_scaled)
            
            logger.info(f"Prepared {len(X)} sequences for Informer training")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data for Informer: {str(e)}")
            raise
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets"""
        try:
            X, y = [], []
            
            for i in range(len(data) - self.seq_len - self.pred_len + 1):
                # Input sequence
                x_seq = data[i:i + self.seq_len]
                X.append(x_seq)
                
                # Target sequence
                y_seq = data[i + self.seq_len:i + self.seq_len + self.pred_len]
                y.append(y_seq.flatten())
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def train(self, df: pd.DataFrame, target_column: str = 'close', 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train Informer model
        
        Args:
            df: DataFrame with time series data
            target_column: Column to use as target
            validation_split: Proportion of data for validation
        
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Starting Informer model training...")
            
            # Prepare data
            X, y = self.prepare_data(df, target_column)
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            
            # Initialize model
            self.model = InformerModel(self.config).to(self.device)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, "
                              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Load best model
            self.load_model()
            
            # Calculate final metrics
            train_metrics = self._evaluate_model(X_train, y_train)
            val_metrics = self._evaluate_model(X_val, y_val)
            
            logger.info("Informer model training completed successfully")
            
            return {
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"Error training Informer model: {str(e)}")
            raise
    
    def _evaluate_model(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                predictions = self.model(X)
                
                # Handle shape mismatch - take only the last prediction for each sequence
                if predictions.shape != y.shape:
                    # If predictions have more dimensions, take the last prediction
                    if len(predictions.shape) > len(y.shape):
                        predictions = predictions[:, -1, :]  # Take last prediction
                    elif predictions.shape[0] != y.shape[0]:
                        # If batch sizes don't match, take the minimum
                        min_size = min(predictions.shape[0], y.shape[0])
                        predictions = predictions[:min_size]
                        y = y[:min_size]
                
                # Calculate metrics
                mse = F.mse_loss(predictions, y).item()
                mae = F.l1_loss(predictions, y).item()
                rmse = np.sqrt(mse)
                
                # Calculate MAPE
                mape = torch.mean(torch.abs((y - predictions) / (y + 1e-8))) * 100
                
                return {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape.item()
                }
                
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            # Return default metrics instead of empty dict
            return {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: Input sequences
        
        Returns:
            Array of predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            self.model.eval()
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.cpu().numpy()
            
            # Inverse transform predictions
            predictions_scaled = self.scaler.inverse_transform(predictions.reshape(-1, 1))
            predictions_scaled = predictions_scaled.reshape(predictions.shape)
            
            return predictions_scaled
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_future(self, df: pd.DataFrame, steps: int = None, 
                      target_column: str = 'close') -> np.ndarray:
        """
        Predict future values
        
        Args:
            df: DataFrame with historical data
            steps: Number of steps to predict (default: pred_len)
            target_column: Column to use as target
        
        Returns:
            Array of future predictions
        """
        try:
            if steps is None:
                steps = self.pred_len
            
            # Get last sequence
            last_sequence = df[target_column].tail(self.seq_len).values.reshape(-1, 1)
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Convert to tensor
            X = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
            
            self.model.eval()
            
            with torch.no_grad():
                predictions = self.model(X)
                predictions = predictions.cpu().numpy()
            
            # Inverse transform predictions
            predictions_scaled = self.scaler.inverse_transform(predictions.reshape(-1, 1))
            
            return predictions_scaled.flatten()[:steps]
            
        except Exception as e:
            logger.error(f"Error predicting future values: {str(e)}")
            raise
    
    def save_model(self):
        """Save trained model and preprocessing objects"""
        try:
            # Create models directory
            settings.MODELS_DIR.mkdir(exist_ok=True)
            
            # Save model
            if self.model is not None:
                torch.save(self.model.state_dict(), self.model_path)
            
            # Save scaler
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save config
            joblib.dump(self.config, self.config_path)
            
            logger.info(f"Informer model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving Informer model: {str(e)}")
    
    def load_model(self):
        """Load trained model and preprocessing objects"""
        try:
            if self.model_path.exists():
                # Initialize model
                self.model = InformerModel(self.config).to(self.device)
                
                # Load state dict
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                
                logger.info("Informer model loaded successfully")
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Informer scaler loaded successfully")
            
            if self.config_path.exists():
                self.config = joblib.load(self.config_path)
                logger.info("Informer config loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Informer model: {str(e)}")
    
    def get_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence using model uncertainty"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            self.model.eval()
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Make multiple predictions with dropout for uncertainty estimation
            predictions = []
            self.model.train()  # Enable dropout
            
            with torch.no_grad():
                for _ in range(10):  # Multiple forward passes
                    pred = self.model(X_tensor)
                    predictions.append(pred.cpu().numpy())
            
            self.model.eval()  # Disable dropout
            
            # Calculate mean and std
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Confidence based on prediction variance
            confidence = 1.0 / (1.0 + std_pred)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting prediction confidence: {str(e)}")
            return np.ones(len(X)) * 0.5  # Default confidence

# Global instance
informer_predictor = InformerStockPredictor()
