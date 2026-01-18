"""
Visualization and reporting utilities
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime


def plot_forecast_comparison(
    df_historical: pd.DataFrame,
    df_forecast: pd.DataFrame,
    df_test_pred: pd.DataFrame,
    target_col: str,
    title: str = "Actual vs Predicted"
) -> go.Figure:
    """
    Create interactive Plotly chart: historical + forecast + test predictions
    
    Parameters:
    -----------
    df_historical : pd.DataFrame
        Historical data with DATE and target_col
    df_forecast : pd.DataFrame
        Forecast with DATE and PREDICTIE
    df_test_pred : pd.DataFrame
        Test set predictions with DATE and PREDICTIE
    target_col : str
        Name of actual values column
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure
        Interactive Plotly figure
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df_historical['DATA'],
        y=df_historical[target_col],
        mode='lines',
        name='Istoric',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Data:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Test predictions (if available)
    if df_test_pred is not None and len(df_test_pred) > 0:
        fig.add_trace(go.Scatter(
            x=df_test_pred['DATA'],
            y=df_test_pred['PREDICTIE'],
            mode='lines',
            name='Predicție Test',
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate='<b>Data:</b> %{x|%Y-%m-%d}<br><b>Predicție:</b> %{y:.2f}<extra></extra>'
        ))
    
    # Future forecast
    if df_forecast is not None and len(df_forecast) > 0:
        fig.add_trace(go.Scatter(
            x=df_forecast['DATA'],
            y=df_forecast['PREDICTIE'],
            mode='lines+markers',
            name='Prognoză',
            line=dict(color='red', width=2),
            marker=dict(size=5),
            hovertemplate='<b>Data:</b> %{x|%Y-%m-%d}<br><b>Prognoză:</b> %{y:.2f}<extra></extra>'
        ))
    
    # Add vertical line to separate history from forecast
    if df_forecast is not None and len(df_forecast) > 0:
        last_hist_date = df_historical['DATA'].max()
        fig.add_vline(
            x=last_hist_date,
            line_dash='dash',
            line_color='gray',
            annotation_text='Prezent',
            annotation_position='top right'
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Data',
        yaxis_title='Valoare',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        font=dict(size=11)
    )
    
    return fig


def plot_distribution(
    data: pd.Series,
    title: str = "Distribuție",
    nbins: int = 30
) -> go.Figure:
    """
    Create histogram of data distribution
    """
    fig = px.histogram(
        x=data,
        nbins=nbins,
        title=title,
        labels={'x': 'Valoare'},
        opacity=0.7
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series = None
) -> go.Figure:
    """
    Plot residuals (actual - predicted)
    """
    residuals = y_true - y_pred
    
    if dates is not None:
        fig = px.scatter(
            x=dates,
            y=residuals,
            title='Reziduuri (Actual - Predicție)',
            labels={'x': 'Data', 'y': 'Reziduu'},
            opacity=0.6
        )
    else:
        fig = px.scatter(
            x=np.arange(len(residuals)),
            y=residuals,
            title='Reziduuri (Actual - Predicție)',
            labels={'x': 'Index', 'y': 'Reziduu'},
            opacity=0.6
        )
    
    # Add zero line
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    
    fig.update_layout(
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_metrics_table(metrics: dict) -> pd.DataFrame:
    """
    Format metrics as DataFrame for display
    """
    data = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                data.append({'Metrica': key, 'Valoare': f"{value:.4f}"})
            else:
                data.append({'Metrica': key, 'Valoare': str(value)})
        elif value is not None:
            data.append({'Metrica': key, 'Valoare': str(value)})
    
    return pd.DataFrame(data)


def create_kpi_card(
    label: str,
    value: float,
    unit: str = "",
    delta: float = None
) -> dict:
    """
    Create KPI card data for display
    """
    formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    
    return {
        'label': label,
        'value': formatted_value,
        'unit': unit,
        'delta': delta
    }


def plot_monthly_aggregation(
    df: pd.DataFrame,
    target_col: str,
    agg_func: str = 'sum'
) -> go.Figure:
    """
    Plot monthly aggregated values
    """
    df['LUNA'] = df['DATA'].dt.to_period('M').astype(str)
    
    if agg_func == 'sum':
        monthly_data = df.groupby('LUNA')[target_col].sum()
    elif agg_func == 'mean':
        monthly_data = df.groupby('LUNA')[target_col].mean()
    else:
        monthly_data = df.groupby('LUNA')[target_col].agg(agg_func)
    
    fig = px.bar(
        x=monthly_data.index,
        y=monthly_data.values,
        title=f'Agregare lunară – {target_col} ({agg_func})',
        labels={'x': 'Lună', 'y': 'Valoare'},
        opacity=0.7
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white'
    )
    
    return fig
