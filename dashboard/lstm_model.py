"""
LSTM model building and prediction
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from typing import Tuple, Optional


def build_lstm_model(
    lookback: int = 14,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    seed: int = 42
) -> Sequential:
    """
    Build LSTM model for time series forecasting
    
    Parameters:
    -----------
    lookback : int
        Number of previous timesteps
    lstm_units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate to prevent overfitting
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Sequential
        Compiled Keras model
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = Sequential([
        LSTM(lstm_units, input_shape=(lookback, 1), return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: Sequential,
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 8,
    verbose: int = 0
) -> Tuple[Sequential, dict]:
    """
    Train LSTM model with early stopping
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training input sequences
    y_train : np.ndarray
        Training target values
    X_val : np.ndarray
        Validation input sequences
    y_val : np.ndarray
        Validation target values
    model : Sequential
        Keras model
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size for training
    patience : int
        Early stopping patience
    verbose : int
        Verbosity level
        
    Returns:
    --------
    Tuple[Sequential, dict]
        Trained model and history dict
    """
    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=verbose
    )
    
    return model, history.history


def predict_future(
    model: Sequential,
    last_sequence: np.ndarray,
    horizon: int,
    scaler
) -> np.ndarray:
    """
    Generate multi-step forecast
    
    Parameters:
    -----------
    model : Sequential
        Trained LSTM model
    last_sequence : np.ndarray
        Last known sequence (shape: (lookback, 1))
    horizon : int
        Number of days to forecast
    scaler : MinMaxScaler
        Fitted scaler for inverse transformation
        
    Returns:
    --------
    np.ndarray
        Forecast values (inverse scaled)
    """
    lookback = last_sequence.shape[0]
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(horizon):
        # Predict next value
        next_pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Update sequence for next prediction
        current_seq = np.vstack([current_seq[1:], [[next_pred]]])
    
    # Inverse scale predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions)
    
    return predictions_original.flatten()


def save_model(model: Sequential, path: str):
    """Save model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load_model(path: str) -> Sequential:
    """Load model from disk"""
    return tf.keras.models.load_model(path)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Metrics: MAE, RMSE, MAPE, R²
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 2) if not np.isnan(mape) else None,
        'R²': round(r2, 4)
    }
