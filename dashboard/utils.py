"""
Utility functions for advanced features and batch processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple


class ForecastBatchProcessor:
    """
    Process forecasts for multiple articles at once
    """
    
    def __init__(self, csv_path: str):
        """Initialize with CSV path"""
        self.csv_path = csv_path
        from preprocessing import load_and_prepare_data
        self.df, self.target_col = load_and_prepare_data(csv_path)
    
    def get_all_articles(self) -> List[int]:
        """Get all unique article IDs"""
        for col in self.df.columns:
            if 'articol' in col.lower():
                return sorted(self.df[col].unique())
        return []
    
    def forecast_single_article(
        self,
        article_id: int,
        target_col: str = 'CANTITATE',
        test_size: float = 0.2,
        min_samples: int = 30
    ) -> Dict:
        """
        Generate forecast for single article using XGBoost
        
        Returns:
        --------
        dict with keys: 'article_id', 'metrics', 'forecast', 'error'
        """
        try:
            from preprocessing import filter_by_article
            from xgboost_model import build_and_train_xgboost
            from baseline import build_baseline_fallback
            
            # Filter article data
            df_article = filter_by_article(self.df, article_id, article_col='ID_ARTICOL')
            
            # Try XGBoost
            model, fe, info = build_and_train_xgboost(
                df_article,
                target_col=target_col,
                test_size=test_size,
                min_samples=min_samples
            )
            
            if model is None:
                # Fallback to baseline
                baseline_model, baseline_info = build_baseline_fallback(
                    df_article,
                    target_col=target_col
                )
                
                if baseline_model is None:
                    return {
                        'article_id': article_id,
                        'error': 'Insufficient data for both XGBoost and baseline'
                    }
                
                # Use baseline
                y = df_article[target_col].values
                y = y[~np.isnan(y)]
                baseline_model.fit(y)
                forecast = baseline_model.forecast(horizon=30)
                
                return {
                    'article_id': article_id,
                    'metrics': baseline_info['metrics'],
                    'forecast': forecast,
                    'model_type': 'baseline',
                    'error': None
                }
            
            # XGBoost success
            future_pred, df_extended = model.predict_future(
                df_article,
                fe,
                horizon=30,
                target_col=target_col
            )
            
            return {
                'article_id': article_id,
                'metrics': info['metrics'],
                'forecast': future_pred,
                'model_type': 'xgboost',
                'error': None
            }
            
        except Exception as e:
            return {
                'article_id': article_id,
                'error': str(e)
            }


def export_results_to_excel(
    results: Dict,
    output_path: str = 'forecast_results.xlsx'
):
    """
    Export forecast results to Excel with multiple sheets
    
    Parameters:
    -----------
    results : dict
        Results from forecast (containing forecast_results from Streamlit session)
    output_path : str
        Output Excel file path
    """
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
    except ImportError:
        print("openpyxl not installed. Install with: pip install openpyxl")
        return
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Historical Data
        results['df_article'].to_excel(
            writer,
            sheet_name='Historic',
            index=False
        )
        
        # Sheet 2: Forecast
        results['df_forecast'].to_excel(
            writer,
            sheet_name='Forecast',
            index=False
        )
        
        # Sheet 3: Test Predictions
        results['df_test_pred'].to_excel(
            writer,
            sheet_name='Test_Predictions',
            index=False
        )
        
        # Sheet 4: Metrics
        metrics_df = pd.DataFrame(
            list(results['metrics'].items()),
            columns=['Metric', 'Value']
        )
        metrics_df.to_excel(
            writer,
            sheet_name='Metrics',
            index=False
        )
    
    print(f"✅ Results exported to {output_path}")


def generate_forecast_report(
    results: Dict,
    article_name: str = "Unknown",
    output_html: str = 'forecast_report.html'
):
    """
    Generate an HTML report of forecast results
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Forecast Report - {article_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }}
            h1 {{ color: #333; border-bottom: 3px solid #FF6B6B; padding-bottom: 10px; }}
            h2 {{ color: #666; margin-top: 30px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #FF6B6B; }}
            .metric-card strong {{ color: #FF6B6B; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            table th {{ background: #f0f0f0; padding: 10px; text-align: left; border-bottom: 2px solid #ddd; }}
            table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Raport Prognoză - {article_name}</h1>
            
            <h2>📈 Metrice Evaluare</h2>
            <div class="metrics">
    """
    
    for metric_name, metric_value in results['metrics'].items():
        html_content += f"""
                <div class="metric-card">
                    <strong>{metric_name}</strong><br/>
                    {metric_value if metric_value is not None else 'N/A'}
                </div>
        """
    
    html_content += """
            </div>
            
            <h2>🗓️ Prognoză pentru Zilele Viitoare</h2>
            <table>
                <tr>
                    <th>Data</th>
                    <th>Predicție</th>
                </tr>
    """
    
    for _, row in results['df_forecast'].iterrows():
        html_content += f"""
                <tr>
                    <td>{row['DATA'].strftime('%Y-%m-%d')}</td>
                    <td>{row['PREDICTIE']:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <div class="footer">
                <p>Raport generat pe: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p>Dashboard: Optimizare Stocuri - Metode Predictive (XGBoost)</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Report exported to {output_html}")


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalies using Z-score method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with time series
    column : str
        Column to analyze
    threshold : float
        Z-score threshold (2.0 = 95% confidence)
        
    Returns:
    --------
    pd.DataFrame
        Rows with anomalies
    """
    from scipy import stats
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    anomalies = df[np.abs(stats.zscore(df[column])) > threshold]
    
    return anomalies


def calculate_safety_stock(
    avg_demand: float,
    std_demand: float,
    lead_time: int,
    service_level: float = 0.95
) -> float:
    """
    Calculate safety stock using service level
    
    Parameters:
    -----------
    avg_demand : float
        Average daily demand
    std_demand : float
        Standard deviation of demand
    lead_time : int
        Lead time in days
    service_level : float
        Service level (0.95 = 95% availability)
        
    Returns:
    --------
    float
        Recommended safety stock
    """
    from scipy.stats import norm
    
    z_score = norm.ppf(service_level)
    safety_stock = z_score * std_demand * np.sqrt(lead_time)
    
    return safety_stock


def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: float
) -> float:
    """
    Calculate Economic Order Quantity (EOQ)
    
    Parameters:
    -----------
    annual_demand : float
        Total annual demand
    ordering_cost : float
        Cost per order
    holding_cost : float
        Annual holding cost per unit
        
    Returns:
    --------
    float
        Optimal order quantity
    """
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return eoq


if __name__ == "__main__":
    print("🔧 Utility functions module loaded")
    print("Available functions:")
    print("  - ForecastBatchProcessor: Process multiple articles")
    print("  - export_results_to_excel: Export to Excel")
    print("  - generate_forecast_report: Create HTML report")
    print("  - detect_anomalies: Find outliers")
    print("  - calculate_safety_stock: Inventory safety stock")
    print("  - calculate_eoq: Economic order quantity")
