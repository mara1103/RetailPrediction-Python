"""
Data preprocessing and utilities for inventory optimization dashboard
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and perform initial preprocessing
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    df = pd.read_csv(csv_path)
    
    # Detect and convert date column
    date_col = None
    for col in df.columns:
        if 'data' in col.lower() or 'date' in col.lower():
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.rename(columns={date_col: 'DATA'})
    else:
        raise ValueError("Nu am găsit o coloană de dată în CSV")
    
    # Detect target column (quantity/sales)
    target_col = None
    for col in df.columns:
        if 'cantitate' in col.lower() or 'qty' in col.lower() or 'vânzări' in col.lower():
            target_col = col
            break
    
    if not target_col:
        # Default to first numeric column after common identifiers
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col not in ['ID_ARTICOL', 'RUPTURA_STOC']:
                target_col = col
                break
    
    # Sort by date
    df = df.sort_values('DATA').reset_index(drop=True)
    
    # Fill missing values in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col != 'ID_ARTICOL':
            df[col] = df[col].fillna(0)
    
    return df, target_col if target_col else 'CANTITATE'


def filter_by_article(df: pd.DataFrame, article_id, article_col: str = None) -> pd.DataFrame:
    """
    Filter dataframe for specific article ID
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe
    article_id : int or str
        Article ID to filter (will auto-convert to match column dtype)
    article_col : str, optional
        Column name with article IDs (auto-detect if None)
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    # Auto-detect article column if not provided
    if article_col is None:
        article_col = None
        for col in df.columns:
            if 'articol' in col.lower():
                article_col = col
                break
    
    if article_col is None:
        raise ValueError("Nu am găsit o coloană ID_ARTICOL")
    
    # Convert article_id to match column dtype
    col_dtype = df[article_col].dtype
    try:
        if 'int' in str(col_dtype):
            article_id = int(article_id)
        elif 'float' in str(col_dtype):
            article_id = float(article_id)
        else:
            article_id = str(article_id)
    except (ValueError, TypeError):
        pass  # Use article_id as-is
    
    df_article = df[df[article_col] == article_id].copy()
    if len(df_article) == 0:
        raise ValueError(f"No data found for article {article_id} in column {article_col}")
    
    return df_article.sort_values('DATA').reset_index(drop=True)


def handle_missing_dates(df: pd.DataFrame, fill_method: str = 'forward') -> pd.DataFrame:
    """
    Fill missing dates in time series
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with DATE column
    fill_method : str
        'forward' for forward fill, 'zero' for zero fill
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with continuous dates
    """
    df = df.sort_values('DATA').reset_index(drop=True)
    
    date_range = pd.date_range(
        start=df['DATA'].min(),
        end=df['DATA'].max(),
        freq='D'
    )
    
    df_full = pd.DataFrame({'DATA': date_range})
    df = df_full.merge(df, on='DATA', how='left')
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col != 'ID_ARTICOL':
            if fill_method == 'forward':
                df[col] = df[col].fillna(method='ffill').fillna(0)
            else:
                df[col] = df[col].fillna(0)
    
    # Fill non-numeric columns
    non_numeric = df.select_dtypes(exclude=['number']).columns
    for col in non_numeric:
        if col != 'DATA':
            df[col] = df[col].fillna(method='ffill').fillna('')
    
    return df.reset_index(drop=True)


class TimeSeriesScaler:
    """
    MinMax scaler wrapper for time series
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fit = False
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        self.fit = True
        return self.scaler.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data"""
        if not self.fit:
            raise ValueError("Scaler not fitted yet")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform"""
        if not self.fit:
            raise ValueError("Scaler not fitted yet")
        return self.scaler.inverse_transform(data)


def make_sequences(
    data_scaled: np.ndarray,
    lookback: int = 14
) -> tuple:
    """
    Create sequences for time series forecasting
    
    Parameters:
    -----------
    data_scaled : np.ndarray
        Scaled 1D array of values
    lookback : int
        Number of previous timesteps to use for prediction
        
    Returns:
    --------
    tuple
        (X, y) where X shape is (samples, lookback, 1) and y shape is (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, 0])
        y.append(data_scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    return X[..., np.newaxis], y


def get_stockout_metrics(df: pd.DataFrame, target_col: str) -> dict:
    """
    Calculate stockout risk metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with time series data
    target_col : str
        Column name of target variable (quantity/sales)
        
    Returns:
    --------
    dict
        Metrics including avg daily sales, low stock days, etc.
    """
    metrics = {}
    
    # Average daily quantity
    metrics['avg_daily'] = df[target_col].mean()
    metrics['std_daily'] = df[target_col].std()
    metrics['max_daily'] = df[target_col].max()
    metrics['min_daily'] = df[target_col].min()
    
    # Find zero-sale days (potential stockouts)
    zero_days = (df[target_col] == 0).sum()
    metrics['zero_sale_days'] = zero_days
    metrics['zero_sale_percent'] = round(100 * zero_days / len(df), 2)
    
    # Coefficient of variation (demand volatility)
    if metrics['avg_daily'] > 0:
        metrics['cv'] = metrics['std_daily'] / metrics['avg_daily']
    else:
        metrics['cv'] = 0
    
    return metrics


def forecast_to_dataframe(
    forecast_values: np.ndarray,
    last_date: datetime,
    horizon: int
) -> pd.DataFrame:
    """
    Convert forecast array to DataFrame
    
    Parameters:
    -----------
    forecast_values : np.ndarray
        Predicted values
    last_date : datetime
        Last date in historical data
    horizon : int
        Number of days to forecast
        
    Returns:
    --------
    pd.DataFrame
        Forecast dataframe with dates and values
    """
    dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=horizon,
        freq='D'
    )
    
    return pd.DataFrame({
        'DATA': dates,
        'PREDICTIE': forecast_values[:horizon]
    })
