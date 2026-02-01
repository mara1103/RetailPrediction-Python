"""
XGBoost Model Training & Prediction Module

Feature Engineering:
- Temporal features (day of week, month, is_weekend, day of month)
- Lag features (1, 7, 14 days)
- Rolling statistics (mean, std over 7 and 14 days)
- Extra lag/rolling for VAL_IESIRE_FARA_TVA

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
    
    def __init__(self, target_col='CANTITATE', lag_features=[1, 7, 14], 
                 rolling_windows=[7, 14], extra_lag_cols=None):
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
        self.extra_lag_cols = extra_lag_cols or ['VAL_IESIRE_FARA_TVA']
        
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

        # Extra lag/rolling for VAL_IESIRE_FARA_TVA (if present)
        # drop raw VAL_IESIRE_FARA_TVA to avoid same-day leakage
        if 'VAL_IESIRE_FARA_TVA' in df.columns:
            df = df.drop(columns=['VAL_IESIRE_FARA_TVA'])

        for col in self.extra_lag_cols:
            if col in df.columns:
                for lag in self.lag_features:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
                for window in self.rolling_windows:
                    shifted = df[col].shift(1)
                    df[f'{col}_roll_mean_{window}'] = shifted.rolling(window).mean()
                    df[f'{col}_roll_std_{window}'] = shifted.rolling(window).std()
        
        # ═════════════════════════════════════════════════════════════════
        # 4. IDENTIFY FEATURE COLUMNS (everything except target and DATE)
        # ═════════════════════════════════════════════════════════════════
        # Use only engineered features to avoid leaking or freezing exogenous columns
        base_features = ['DOW', 'month', 'ESTE_WEEKEND', 'ZI_DIN_LUNA', 'week']
        lag_features = [f'lag_{lag}' for lag in self.lag_features]
        roll_features = []
        for window in self.rolling_windows:
            roll_features.extend([
                f'roll_mean_{window}',
                f'roll_std_{window}',
                f'roll_max_{window}',
                f'roll_min_{window}'
            ])
        extra_features = []
        for col in self.extra_lag_cols:
            if col in df.columns:
                extra_features.extend([f'{col}_lag{lag}' for lag in self.lag_features])
                for window in self.rolling_windows:
                    extra_features.extend([
                        f'{col}_roll_mean_{window}',
                        f'{col}_roll_std_{window}'
                    ])
        self.feature_cols = base_features + lag_features + roll_features + extra_features
        
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
                 reg_alpha: float = 0.0,
                 min_child_weight: float = 1.0,
                 random_state: int = 42,
                 use_log1p: bool = True):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            reg_lambda: L2 regularization
            reg_alpha: L1 regularization
            min_child_weight: Minimum sum of instance weight needed in a child
            random_state: Random seed
            use_log1p: If True, train on log1p(target) and invert on predict
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_weight=min_child_weight,
            random_state=random_state,
            verbose=0
        )
        self.feature_cols = None
        self.scaler = None
        self.target_mean = None
        self.target_std = None
        self.use_log1p = use_log1p
    
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
        y_train = np.log1p(y) if self.use_log1p else y
        
        # Store statistics for normalization (optional)
        self.target_mean = y.mean()
        self.target_std = y.std()
        
        # Train
        self.model.fit(X, y_train, verbose=0)
        
        return {
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'removed_stockout': removed
        }
    
    def _build_feature_row(
        self,
        df_history: pd.DataFrame,
        feature_engineer: XGBoostFeatureEngineer,
        target_col: str,
        current_date: pd.Timestamp,
        row_known: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        future_row = {
            'DATA': current_date,
            target_col: np.nan,
            'DOW': current_date.dayofweek,
            'month': current_date.month,
            'ESTE_WEEKEND': 1 if current_date.dayofweek >= 5 else 0,
            'ZI_DIN_LUNA': current_date.day,
            'week': current_date.isocalendar()[1]
        }

        for lag in feature_engineer.lag_features:
            if lag <= len(df_history):
                future_row[f'lag_{lag}'] = df_history[target_col].iloc[-lag]

        for window in feature_engineer.rolling_windows:
            if len(df_history) >= window:
                window_data = df_history[target_col].tail(window).values
                future_row[f'roll_mean_{window}'] = np.mean(window_data)
                future_row[f'roll_std_{window}'] = np.std(window_data)
                future_row[f'roll_max_{window}'] = np.max(window_data)
                future_row[f'roll_min_{window}'] = np.min(window_data)

        # VAL_IESIRE_FARA_TVA lags/rolling (use known history)
        for col in feature_engineer.extra_lag_cols:
            if col in df_history.columns:
                for lag in feature_engineer.lag_features:
                    if lag <= len(df_history):
                        future_row[f'{col}_lag{lag}'] = df_history[col].iloc[-lag]
                for window in feature_engineer.rolling_windows:
                    if len(df_history) >= window:
                        window_data = df_history[col].tail(window).values
                        future_row[f'{col}_roll_mean_{window}'] = np.mean(window_data)
                        future_row[f'{col}_roll_std_{window}'] = np.std(window_data)

        future_df = pd.DataFrame([future_row])

        for col in self.feature_cols:
            if col in future_df.columns:
                continue
            if row_known is not None and col in row_known.index:
                future_df[col] = row_known[col]
            elif col in df_history.columns:
                future_df[col] = df_history[col].iloc[-1]
            else:
                future_df[col] = 0

        return future_df

    def predict_test_walk_forward(
        self,
        df_all: pd.DataFrame,
        feature_engineer: XGBoostFeatureEngineer,
        last_train_date: pd.Timestamp,
        test_dates: pd.Series,
        target_col: str = 'CANTITATE'
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        df_sorted = df_all.sort_values('DATA').reset_index(drop=True)
        df_lookup = df_sorted.set_index('DATA')

        df_history = df_sorted[df_sorted['DATA'] <= last_train_date].copy()

        predictions = []
        y_true = []
        dates = []

        for current_date in test_dates:
            if current_date not in df_lookup.index:
                continue

            row_known = df_lookup.loc[current_date]
            if isinstance(row_known, pd.DataFrame):
                row_known = row_known.iloc[0]

            feature_row = self._build_feature_row(
                df_history,
                feature_engineer,
                target_col,
                current_date,
                row_known=row_known
            )

            X_test = feature_row[self.feature_cols]
            pred = self.model.predict(X_test)[0]
            if self.use_log1p:
                pred = np.expm1(pred)
            pred = max(0, pred)

            predictions.append(pred)
            y_true.append(row_known[target_col])
            dates.append(current_date)

            df_history = pd.concat([df_history, row_known.to_frame().T], ignore_index=True)

        y_pred = np.array(predictions)
        y_true_arr = np.array(y_true)

        metrics = {
            'mae': mean_absolute_error(y_true_arr, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true_arr, y_pred)),
            'r2': r2_score(y_true_arr, y_pred),
            'mape': mean_absolute_percentage_error(y_true_arr, y_pred) if (y_true_arr != 0).all() else np.nan,
            'n_samples': len(y_true_arr)
        }

        df_test = pd.DataFrame({'DATA': dates, target_col: y_true_arr})

        return y_pred, df_test, metrics
    
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
            if self.use_log1p:
                pred = np.expm1(pred)
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
                            min_samples: int = 30,
                            horizon: int = 14,
                            tune: bool = True) -> Tuple[Optional[XGBoostModel], 
                                                         Optional[XGBoostFeatureEngineer],
                                                         Dict]:
    """
    Complete pipeline: feature engineering + training
    
    Args:
        df: Raw dataframe
        target_col: Target column name
        test_size: Fraction of data for testing (legacy; ignored for horizon split)
        min_samples: Minimum samples required for training
        horizon: Holdout horizon (last N days)
        tune: If True, run a small hyperparameter search
        
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
    
    # Train/test split (last N days as test)
    if len(df_feat) <= horizon:
        return None, None, {
            'error': f'Prea puține date pentru split pe {horizon} zile',
            'n_samples': len(df_feat)
        }
    df_feat = df_feat.sort_values('DATA').reset_index(drop=True)
    df_train = df_feat.iloc[:-horizon].copy()
    df_test = df_feat.iloc[-horizon:].copy()
    
    # Optional tuning on a small validation split
    best_params = None
    if tune and len(df_train) >= (min_samples + max(7, horizon)):
        val_horizon = min(horizon, max(7, int(0.1 * len(df_train))))
        df_train_fit = df_train.iloc[:-val_horizon].copy()
        df_val = df_train.iloc[-val_horizon:].copy()

        param_grid = [
            {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'reg_alpha': 0.0},
            {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'reg_alpha': 0.0},
            {'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'reg_alpha': 0.0},
            {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.8, 'min_child_weight': 5.0, 'reg_lambda': 1.0, 'reg_alpha': 0.1},
            {'n_estimators': 900, 'max_depth': 5, 'learning_rate': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 3.0, 'reg_lambda': 1.2, 'reg_alpha': 0.2},
            {'n_estimators': 1200, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5.0, 'reg_lambda': 1.5, 'reg_alpha': 0.4},
        ]

        best_rmse = None
        for params in param_grid:
            model_tmp = XGBoostModel(**params, use_log1p=True)
            model_tmp.train(df_train_fit, fe.get_feature_cols(), target_col=target_col)

            _, _, metrics = model_tmp.predict_test_walk_forward(
                df,
                fe,
                last_train_date=df_train_fit['DATA'].iloc[-1],
                test_dates=df_val['DATA'],
                target_col=target_col
            )

            rmse = metrics.get('rmse')
            if rmse is None:
                continue
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_params = params

        # if tuning was skipped or failed, fall back to defaults
        if best_params is None:
            best_params = {}
    else:
        best_params = {}

    # Train final model
    model = XGBoostModel(**best_params, use_log1p=True)
    train_info = model.train(df_train, fe.get_feature_cols(), target_col=target_col)
    
    # Evaluate with walk-forward to avoid leakage in test predictions
    last_train_date = df_train['DATA'].iloc[-1]
    y_pred, df_test, metrics = model.predict_test_walk_forward(
        df,
        fe,
        last_train_date=last_train_date,
        test_dates=df_test['DATA'],
        target_col=target_col
    )
    
    return model, fe, {
        **train_info,
        'metrics': metrics,
        'df_test': df_test,
        'y_pred': y_pred,
        'best_params': best_params,
        'horizon': horizon
    }
