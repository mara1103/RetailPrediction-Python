"""
XGBoost Model Training & Prediction Module

Feature Engineering:
- Temporal features (day of week, month, is_weekend, day of month)
- Lag features (1, 7, 14, 21 days)
- Rolling statistics (mean, std over 7 and 14 days)

Training:
- Train/test split (80/20)
- XGBoost with optimized hyperparameters
- Evaluation metrics (MAE, RMSE, R², MAPE)

Prediction:
- Multi-step forecasting with recursive predictions
- Automatic feature recalculation for future dates
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class XGBoostFeatureEngineer:
    """Feature engineering for XGBoost time series"""
    
    def __init__(self, target_col='CANTITATE', lag_features=[1, 7, 14, 21], 
                 rolling_windows=[7, 14]):
        """
        Initialize feature engineer
        
        Args:
            target_col: Target column name
            lag_features: List of lag days to include
            rolling_windows: List of rolling window sizes
        """
        self.target_col = target_col
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.feature_cols = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from dataframe
        
        Args:
            df: DataFrame with 'DATA' (datetime) and target column
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure DATA is datetime
        if df['DATA'].dtype != 'datetime64[ns]':
            df['DATA'] = pd.to_datetime(df['DATA'], errors='coerce')
        
        # Drop rows with missing target
        df = df.dropna(subset=[self.target_col])
        
        # ═════════════════════════════════════════════════════════════════
        # 1. TEMPORAL FEATURES
        # ═════════════════════════════════════════════════════════════════
        df['DOW'] = df['DATA'].dt.dayofweek                           # 0-6 (Mon-Sun)
        df['month'] = df['DATA'].dt.month                             # 1-12
        df['ESTE_WEEKEND'] = (df['DOW'] >= 5).astype(int)             # 1 if Sat/Sun
        df['ZI_DIN_LUNA'] = df['DATA'].dt.day                         # 1-31
        df['week'] = df['DATA'].dt.isocalendar().week                 # Week of year
        
        # ═════════════════════════════════════════════════════════════════
        # 2. LAG FEATURES (shifted target values)
        # ═════════════════════════════════════════════════════════════════
        for lag in self.lag_features:
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        
        # ═════════════════════════════════════════════════════════════════
        # 3. ROLLING STATISTICS (using shift(1) to avoid leakage)
        # ═════════════════════════════════════════════════════════════════
        for window in self.rolling_windows:
            df[f'roll_mean_{window}'] = df[self.target_col].shift(1).rolling(window).mean()
            df[f'roll_std_{window}'] = df[self.target_col].shift(1).rolling(window).std()
            df[f'roll_max_{window}'] = df[self.target_col].shift(1).rolling(window).max()
            df[f'roll_min_{window}'] = df[self.target_col].shift(1).rolling(window).min()
        
        # ═════════════════════════════════════════════════════════════════
        # 4. IDENTIFY FEATURE COLUMNS (everything except target and DATE)
        # ═════════════════════════════════════════════════════════════════
        drop_cols = ['DATA', self.target_col, 'ID_ARTICOL']
        
        # Include all numeric columns that are features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in numeric_cols if c not in drop_cols]
        
        # Drop NaN rows created by lag/rolling
        df = df.dropna(subset=self.feature_cols + [self.target_col])
        
        return df
    
    def get_feature_cols(self) -> list:
        """Get list of feature column names"""
        return self.feature_cols if self.feature_cols else []


class XGBoostModel:
    """XGBoost regression model for time series forecasting"""
    
    def __init__(self, 
                 n_estimators: int = 600,
                 max_depth: int = 6,
                 learning_rate: float = 0.03,
                 subsample: float = 0.9,
                 colsample_bytree: float = 0.9,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=0
        )
        self.feature_cols = None
        self.scaler = None
        self.target_mean = None
        self.target_std = None
    
    def train(self, df_train: pd.DataFrame, feature_cols: list, 
              target_col: str = 'CANTITATE', 
              remove_stockout: bool = True) -> Dict:
        """
        Train XGBoost model
        
        Args:
            df_train: Training dataframe with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            remove_stockout: If True, remove rows where STOC_INITIAL=0 and CANTITATE=0
            
        Returns:
            Dictionary with training info
        """
        df = df_train.copy()
        
        # Remove stockout periods if specified
        if remove_stockout and 'STOC_INITIAL' in df.columns:
            initial_len = len(df)
            df = df[~((df['STOC_INITIAL'] == 0) & (df[target_col] == 0))]
            removed = initial_len - len(df)
        else:
            removed = 0
        
        self.feature_cols = feature_cols
        X = df[feature_cols]
        y = df[target_col]
        
        # Store statistics for normalization (optional)
        self.target_mean = y.mean()
        self.target_std = y.std()
        
        # Train
        self.model.fit(X, y, verbose=0)
        
        return {
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'removed_stockout': removed
        }
    
    def predict_test(self, df_test: pd.DataFrame, 
                     target_col: str = 'CANTITATE') -> Tuple[np.ndarray, Dict]:
        """
        Predict on test set and calculate metrics
        
        Args:
            df_test: Test dataframe with features
            target_col: Name of target column
            
        Returns:
            Tuple of (predictions array, metrics dictionary)
        """
        X_test = df_test[self.feature_cols]
        y_true = df_test[target_col].values
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) if (y_true != 0).all() else np.nan,
            'n_samples': len(y_true)
        }
        
        return y_pred, metrics
    
    def predict_future(self, df_historical: pd.DataFrame, 
                      feature_engineer: XGBoostFeatureEngineer,
                      horizon: int = 30,
                      target_col: str = 'CANTITATE') -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Multi-step recursive forecasting
        
        Args:
            df_historical: Historical dataframe (for last values and feature recalculation)
            feature_engineer: Feature engineer object for feature creation
            horizon: Number of days to forecast
            target_col: Name of target column
            
        Returns:
            Tuple of (predictions array, dataframe with historical + future)
        """
        df = df_historical.copy()
        df = df.sort_values('DATA').reset_index(drop=True)
        
        # Historical data
        last_date = pd.to_datetime(df['DATA'].iloc[-1])
        
        # Future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        predictions = []
        
        for future_date in future_dates:
            # Create features for future date
            future_row = {
                'DATA': future_date,
                target_col: np.nan,  # Will be filled by prediction
                'DOW': future_date.dayofweek,
                'month': future_date.month,
                'ESTE_WEEKEND': 1 if future_date.dayofweek >= 5 else 0,
                'ZI_DIN_LUNA': future_date.day,
                'week': future_date.isocalendar()[1]
            }
            
            # Add lag features (from recent historical + previous predictions)
            for lag in feature_engineer.lag_features:
                if lag <= len(df):
                    future_row[f'lag_{lag}'] = df[target_col].iloc[-lag] if lag <= len(df) else np.nan
            
            # Add rolling features (from recent historical)
            for window in feature_engineer.rolling_windows:
                if len(df) >= window:
                    window_data = df[target_col].tail(window).values
                    future_row[f'roll_mean_{window}'] = np.mean(window_data)
                    future_row[f'roll_std_{window}'] = np.std(window_data)
                    future_row[f'roll_max_{window}'] = np.max(window_data)
                    future_row[f'roll_min_{window}'] = np.min(window_data)
            
            # Predict
            future_df = pd.DataFrame([future_row])
            
            # Fill missing lag features with forward fill
            for col in self.feature_cols:
                if col not in future_df.columns:
                    future_df[col] = np.nan
                if pd.isna(future_df[col]).any():
                    future_df[col].fillna(df[col].iloc[-1] if col in df.columns else 0, inplace=True)
            
            X_future = future_df[self.feature_cols]
            pred = self.model.predict(X_future)[0]
            pred = max(0, pred)  # Ensure non-negative
            
            predictions.append(pred)
            
            # Add to historical for next iteration
            future_row[target_col] = pred
            df = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)
        
        return np.array(predictions), df
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance scores"""
        importance = self.model.get_booster().get_score(importance_type='weight')
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


def build_and_train_xgboost(df: pd.DataFrame,
                            target_col: str = 'CANTITATE',
                            test_size: float = 0.2,
                            min_samples: int = 30) -> Tuple[Optional[XGBoostModel], 
                                                              Optional[XGBoostFeatureEngineer],
                                                              Dict]:
    """
    Complete pipeline: feature engineering + training
    
    Args:
        df: Raw dataframe
        target_col: Target column name
        test_size: Fraction of data for testing
        min_samples: Minimum samples required for training
        
    Returns:
        Tuple of (model, feature_engineer, info_dict)
    """
    # Check minimum data
    if len(df) < min_samples:
        return None, None, {
            'error': f'Prea puține date: {len(df)} < {min_samples}',
            'n_samples': len(df)
        }
    
    # Feature engineering
    fe = XGBoostFeatureEngineer(target_col=target_col)
    df_feat = fe.engineer_features(df)
    
    if len(df_feat) < min_samples:
        return None, None, {
            'error': f'După feature engineering: {len(df_feat)} < {min_samples}',
            'n_samples': len(df_feat)
        }
    
    # Train/test split
    split_idx = int(len(df_feat) * (1 - test_size))
    df_train = df_feat.iloc[:split_idx].copy()
    df_test = df_feat.iloc[split_idx:].copy()
    
    # Train model
    model = XGBoostModel()
    train_info = model.train(df_train, fe.get_feature_cols(), target_col=target_col)
    
    # Evaluate
    y_pred, metrics = model.predict_test(df_test, target_col=target_col)
    
    return model, fe, {
        **train_info,
        'metrics': metrics,
        'df_test': df_test,
        'y_pred': y_pred
    }
