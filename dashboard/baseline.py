"""
Baseline Model for Time Series Forecasting

Used when there's insufficient data for XGBoost training.
Implements: Exponential Smoothing (Simple Exponential Smoothing)

This is the most comprehensive baseline that:
- Adapts to level changes in the series
- Works well with small datasets (10+ samples)
- Is standard in industry forecasting
- Balances simplicity with flexibility
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ExponentialSmoothing:
    """Simple Exponential Smoothing for baseline forecasting"""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize with smoothing parameter
        
        Args:
            alpha: Smoothing parameter (0-1). 
                   Higher = more weight on recent values.
                   Default 0.3 works well for most time series.
        """
        self.alpha = alpha
        self.level = None
    
    def fit(self, y: np.ndarray):
        """
        Fit on training data
        
        Initializes the level component with the first observation,
        then exponentially updates as new data arrives.
        """
        self.level = y[0]
        for val in y[1:]:
            self.level = self.alpha * val + (1 - self.alpha) * self.level
    
    def forecast(self, horizon: int) -> np.ndarray:
        """
        Generate h-step ahead forecast
        
        For exponential smoothing, the forecast is constant at all horizons
        (the estimated level).
        """
        return np.full(horizon, self.level)
    
    def predict(self, y_test: np.ndarray) -> np.ndarray:
        """
        Predict on test set with rolling fit
        
        Updates the level for each new observation to mimic
        online learning during test period.
        """
        predictions = []
        level = self.level if self.level is not None else y_test[0]
        
        for val in y_test:
            predictions.append(level)
            level = self.alpha * val + (1 - self.alpha) * level
        
        return np.array(predictions)


def build_baseline_fallback(df: pd.DataFrame,
                            target_col: str = 'CANTITATE',
                            test_size: float = 0.2) -> Tuple[Optional[ExponentialSmoothing], Dict]:
    """
    Build Exponential Smoothing baseline model when XGBoost fails
    
    Args:
        df: Raw dataframe with 'DATA' and target_col columns
        target_col: Target column name (e.g., 'CANTITATE')
        test_size: Fraction of data for testing (default 0.2)
        
    Returns:
        Tuple of (model, info_dict)
        
    Info dict contains:
        - model_name: 'Exponential Smoothing'
        - metrics: {'mae': ..., 'rmse': ..., 'r2': ...}
        - is_baseline: True
        - n_samples: Number of valid samples
        - warning: Message about insufficient data for XGBoost
    """
    
    # Sort by date
    df = df.sort_values('DATA').reset_index(drop=True)
    
    # Extract target values
    y = df[target_col].values
    
    # Remove NaN
    y = y[~np.isnan(y)]
    
    if len(y) < 5:
        return None, {'error': 'Prea puține date valide (<5 eșantioane)'}
    
    # For very small datasets, use all data as train/test on same data
    if len(y) < 10:
        # Train and test on same data
        model = ExponentialSmoothing(alpha=0.3)
        model.fit(y)
        pred = model.predict(y)
        
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
    else:
        # Standard train/test split
        split_idx = int(len(y) * (1 - test_size))
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        model = ExponentialSmoothing(alpha=0.3)
        model.fit(y_train)
        pred = model.predict(y_test)
        
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
    
    return model, {
        'model_name': 'Exponential Smoothing',
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'is_baseline': True,
        'n_samples': len(y),
        'warning': 'Baseline model - insufficient data for XGBoost (< 30 samples)'
    }

