"""
Main Streamlit Dashboard Application - XGBoost Version
Inventory Optimization - Predictive Methods using XGBoost

Features:
- XGBoost regression for demand forecasting
- Automatic fallback to baseline models when data is insufficient
- Per-article model training
- Feature engineering (temporal, lags, rolling)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import io

# Import custom modules
from preprocessing import load_and_prepare_data, filter_by_article, get_stockout_metrics
from xgboost_model import build_and_train_xgboost, XGBoostModel, XGBoostFeatureEngineer
from baseline import build_baseline_fallback
from visualization import (
    plot_forecast_comparison,
    plot_distribution,
    create_metrics_table,
    create_kpi_card,
    plot_monthly_aggregation
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Optimizare Stocuri - XGBoost Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force refresh on code changes
if 'last_reload' not in st.session_state:
    st.session_state.last_reload = 0

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================
@st.cache_data
def load_csv_data(csv_path: str):
    """Load and prepare CSV data once"""
    return load_and_prepare_data(csv_path)


@st.cache_resource
def get_article_ids(df: pd.DataFrame):
    """Get unique article IDs (prefer ID_ARTICOL for GDPR compliance)"""
    if 'ID_ARTICOL' in df.columns:
        return sorted(df['ID_ARTICOL'].unique())
    
    # Fallback: look for other ID columns
    # for col in df.columns:
    #     if 'id' in col.lower() and 'articol' in col.lower():
    #         return sorted(df[col].unique())
    
    return []


@st.cache_resource
def get_column_options(df: pd.DataFrame):
    """Get numeric columns for target selection"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    exclude = ['RUPTURA_STOC', 'ID_ARTICOL', 'STOC_INITIAL', 'STOC_FINAL']
    return [c for c in numeric_cols if c not in exclude]


# ============================================================================
# HEADER
# ============================================================================
st.title("📊 Dashboard Optimizare Stocuri - XGBoost")
st.markdown("""
Aplicație pentru **predicția cererii și managementul stocurilor** folosind **XGBoost**.
Modelul se antrenează per articol cu feature engineering temporal și statistic.
""")

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================
st.sidebar.header("⚙️ Configurare")

# CSV file path
csv_path = st.sidebar.text_input(
    "📁 Calea fișierului CSV",
    value="../data_raw/date_24_art_loreal_20%.csv",
    help="Introdu calea relativă sau absolută la fișierul CSV"
)

# Load data
try:
    df_full, default_target = load_csv_data(csv_path)
    article_ids = get_article_ids(df_full)
    column_options = get_column_options(df_full)
    st.sidebar.success("✅ CSV încărcat cu succes")
except Exception as e:
    st.error(f"❌ Eroare la încărcare CSV: {str(e)}")
    st.stop()

# Always use ID_ARTICOL for filtering (GDPR friendly - numeric IDs only)
article_col = 'ID_ARTICOL'

# Article selection
st.sidebar.subheader("📦 Articol")
selected_article_id = st.sidebar.selectbox(
    "Selectează articolul",
    options=article_ids,
    help="Selectează articolul pentru care dorești prognoză"
)

# Display article ID only (GDPR friendly)
st.sidebar.info(f"📦 ID: {selected_article_id}")

# Target column selection
st.sidebar.subheader("Variabilă Țintă")
target_col = st.sidebar.selectbox(
    "Coloană țintă pentru prognoză",
    options=column_options,
    index=column_options.index(default_target) if default_target in column_options else 0,
    help="Selectează coloana cu valorile de antrenat"
)

# Model parameters
st.sidebar.subheader("⚙️ Parametrii Model")

col1, col2 = st.sidebar.columns(2)
with col1:
    lookback = st.number_input("Lookback (zile)", min_value=7, max_value=60, value=28)
with col2:
    horizon = st.number_input("Orizont (zile)", min_value=1, max_value=90, value=30)

col1, col2 = st.sidebar.columns(2)
with col1:
    test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5) / 100
with col2:
    min_samples = st.number_input("Min. eșantioane", min_value=10, max_value=100, value=30)

# Forecast button
st.sidebar.divider()
forecast_button = st.sidebar.button("▶️ Generează Prognoză", use_container_width=True)

# ============================================================================
# MAIN LOGIC
# ============================================================================
if forecast_button:
    st.divider()
    
    # Filter data for selected article
    df_article = filter_by_article(df_full, selected_article_id, article_col)
    
    if df_article is None or len(df_article) == 0:
        st.error(f"❌ Nu s-au găsit date pentru articolul {selected_article_id}")
        st.stop()
    
    st.info(f"📈 ID Articol: {selected_article_id} | Date: {len(df_article)} rânduri")
    
    # ════════════════════════════════════════════════════════════════════════
    # ATTEMPT XGBOOST
    # ════════════════════════════════════════════════════════════════════════
    model, fe, info = build_and_train_xgboost(
        df_article,
        target_col=target_col,
        test_size=test_size,
        min_samples=min_samples
    )
    
    if model is None:
        # ════════════════════════════════════════════════════════════════════
        # FALLBACK TO BASELINE
        # ════════════════════════════════════════════════════════════════════
        st.warning(f"⚠️ Insuficiență date pentru XGBoost: {info.get('error', 'Motiv necunoscut')}")
        st.info("📊 Se folosește model baseline (Exponential Smoothing / Moving Average)")
        
        baseline_model, baseline_info = build_baseline_fallback(df_article, target_col=target_col)
        
        if baseline_model is None:
            st.error("❌ Insuficiență date chiar și pentru baseline!")
            st.stop()
        
        # Display baseline metrics
        st.subheader("📊 Model Baseline")
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = baseline_info['metrics']['metrics']
        with col1:
            create_kpi_card("Model", baseline_info['metrics']['model_name'], 'info')
        with col2:
            create_kpi_card("MAE", f"{metrics['mae']:.2f}", 'metric')
        with col3:
            create_kpi_card("RMSE", f"{metrics['rmse']:.2f}", 'metric')
        with col4:
            create_kpi_card("R²", f"{metrics['r2']:.3f}", 'metric')
        
        # Baseline forecast
        y_baseline = df_article[target_col].values
        y_baseline = y_baseline[~np.isnan(y_baseline)]
        
        baseline_model.fit(y_baseline)
        future_pred = baseline_model.forecast(horizon)
        
        # Prepare output dataframe
        df_plot = pd.DataFrame({
            'DATA': pd.date_range(
                start=df_article['DATA'].max() + timedelta(days=1),
                periods=horizon,
                freq='D'
            ),
            'Predicție': future_pred,
            'Tip': 'Prognoză (Baseline)'
        })
        
        st.success(f"✅ Prognoză {horizon} zile generated (baseline model)")
        
        # Display forecast
        st.subheader("📈 Prognoză Baseline")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_plot.round(2), use_container_width=True, hide_index=True)
        
        with col2:
            csv = df_plot.to_csv(index=False).encode()
            st.download_button(
                "📥 Descarcă CSV",
                csv,
                file_name=f"prognoza_baseline_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        # ════════════════════════════════════════════════════════════════════
        # XGBOOST SUCCESS
        # ════════════════════════════════════════════════════════════════════
        st.success(f"✅ XGBoost Model antrenat cu succes")
        
        # Display training info
        st.subheader("📊 Metrice Evaluare")
        
        metrics = info['metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_kpi_card("MAE", f"{metrics['mae']:.2f}", 'metric')
        with col2:
            create_kpi_card("RMSE", f"{metrics['rmse']:.2f}", 'metric')
        with col3:
            create_kpi_card("R²", f"{metrics['r2']:.3f}", 'metric')
        with col4:
            n_features = info['n_features']
            create_kpi_card("Features", f"{n_features}", 'info')
        
        # ════════════════════════════════════════════════════════════════════
        # GENERATE FORECAST
        # ════════════════════════════════════════════════════════════════════
        future_pred, df_extended = model.predict_future(
            df_article,
            fe,
            horizon=horizon,
            target_col=target_col
        )
        
        # Prepare plot data
        last_date = df_article['DATA'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
        
        df_plot = pd.DataFrame({
            'DATA': forecast_dates,
            'Predicție': future_pred,
            'Tip': 'Prognoză XGBoost'
        })
        
        st.success(f"✅ Prognoză {horizon} zile generată")
        
        # ════════════════════════════════════════════════════════════════════
        # TABS: CHARTS, METRICS, TABLE, RISK
        # ════════════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Grafic", "📊 Metrici", "📋 Tabel", "⚠️ Risc"])
        
        with tab1:
            st.subheader("Comparație Actual vs Predicție vs Prognoză")
            
            # Prepare data for comparison
            test_df = info['df_test']
            y_pred_test = info['y_pred']
            
            df_comparison = test_df.copy()
            df_comparison['PREDICTIE'] = y_pred_test
            
            # Plot
            fig = plot_forecast_comparison(
                df_article[['DATA', target_col]],
                df_comparison[['DATA', target_col, 'PREDICTIE']],
                df_plot,
                target_col=target_col
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Metrici Detaliate")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f} unități")
                st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f} unități")
            
            with col2:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
                st.metric("Test Samples", f"{metrics['n_samples']}")
            
            st.divider()
            
            # Feature importance
            st.subheader("🎯 Top Features")
            try:
                importance_df = model.get_feature_importance(top_n=10)
                fig_importance = plot_monthly_aggregation(importance_df)
                st.plotly_chart(fig_importance, use_container_width=True)
            except:
                st.info("Feature importance nu poate fi calculată pentru acest model")
        
        with tab3:
            st.subheader("📋 Tabel Prognoză")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                df_display = df_plot.copy()
                df_display['Predicție'] = df_display['Predicție'].round(2)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            with col2:
                csv = df_plot.to_csv(index=False).encode()
                st.download_button(
                    "📥 Descarcă CSV",
                    csv,
                    file_name=f"prognoza_xgboost_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with tab4:
            st.subheader("⚠️ Analiză Risc Ruptură")
            
            metrics_stockout = get_stockout_metrics(df_article, target_col=target_col)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_kpi_card(
                    "Volatilitate",
                    f"{metrics_stockout['std']:.2f}",
                    'metric'
                )
            
            with col2:
                zero_pct = (metrics_stockout['zero_sale_days'] / len(df_article)) * 100
                create_kpi_card(
                    "Zile Zero Vânzare",
                    f"{zero_pct:.1f}%",
                    'warning' if zero_pct > 20 else 'metric'
                )
            
            with col3:
                create_kpi_card(
                    "Coef. Variație",
                    f"{metrics_stockout['cv']:.2f}",
                    'metric'
                )
            
            st.divider()
            
            # Risk assessment
            avg_demand = df_article[target_col].mean()
            std_demand = df_article[target_col].std()
            forecast_mean = future_pred.mean()
            forecast_std = future_pred.std()
            
            st.markdown("#### 📌 Recomandări")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("""
                **Metrici Istoric:**
                - Cerere medie: {:.1f} unități/zi
                - Deviere standard: {:.1f}
                - Coef. Variație: {:.2f}
                """.format(avg_demand, std_demand, metrics_stockout['cv']))
            
            with col2:
                st.write("""
                **Prognoză {}d:**
                - Cerere medie: {:.1f} unități/zi
                - Deviere standard: {:.1f}
                - Min/Max: {:.0f} / {:.0f}
                """.format(horizon, forecast_mean, forecast_std, 
                          future_pred.min(), future_pred.max()))
            
            # Safety stock recommendation
            service_level = 0.95
            z_score = 1.645  # 95% service level
            safety_stock = z_score * std_demand
            
            st.success(f"""
            **Stock de Siguranță (SL 95%):** {safety_stock:.0f} unități
            
            Reaprovizionează când stocul ajunge sub {safety_stock:.0f} unități
            pentru a evita ruptura cu o probabilitate de 95%.
            """)


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
**Dashboard Optimizare Stocuri | XGBoost Forecasting**  
Disertație - Metode Predictive pentru Optimizarea Stocurilor
""")
