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
import plotly.express as px
from datetime import datetime, timedelta

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
    page_icon="",
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
def load_csv_data(csv_path: str, bf_handling: str):
    """Load and prepare CSV data once"""
    return load_and_prepare_data(
        csv_path,
        bf_handling=bf_handling
    )


@st.cache_resource
def get_article_ids(df: pd.DataFrame):
    """Get unique article IDs (prefer ID_ARTICOL for GDPR compliance)"""
    if 'ID_ARTICOL' in df.columns:
        return sorted(df['ID_ARTICOL'].unique())

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
st.title("Dashboard Optimizare Stocuri - XGBoost")
st.markdown("""
Aplicatie pentru **predictia cererii si managementul stocurilor** folosind **XGBoost**.
Modelul se antreneaza per articol cu feature engineering temporal si statistic.
""")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.4rem; padding-bottom: 1.0rem;}
    [data-testid="stVerticalBlock"] {gap: 0.5rem;}
    .stAlert {padding-top: 0.4rem; padding-bottom: 0.4rem;}
    h1 {margin-top: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Configurare")
    st.caption("Calea fisierului CSV")
    csv_path = st.text_input(
        "Cale CSV",
        value="../data_raw/date_24_art_loreal_20%.csv",
        help="Introdu calea relativa sau absoluta la fisierul CSV",
        label_visibility="collapsed"
    )

    col_bf, col_article, col_target = st.columns(3)
    with col_bf:
        st.caption("Black Friday")
        bf_handling = st.selectbox(
            "Black Friday",
            options=["none", "exclude"],
            format_func=lambda x: {
                "none": "Nu modifica",
                "exclude": "Exclude zile BF (2024: 15-20 nov, 2025: 14-18 nov)"
            }[x],
            help="Exclude zilele Black Friday",
            label_visibility="collapsed"
        )

    # Load data
    try:
        df_full, default_target = load_csv_data(csv_path, bf_handling)
        article_ids = get_article_ids(df_full)
        column_options = get_column_options(df_full)
        st.success("CSV incarcat cu succes")
    except Exception as e:
        st.error(f"Eroare la incarcare CSV: {str(e)}")
        st.stop()

    article_col = "ID_ARTICOL"

    with col_article:
        st.caption("Articol")
        selected_article_id = st.selectbox(
            "Articol",
            options=article_ids,
            help="Selecteaza articolul pentru care doresti prognoza",
            label_visibility="collapsed"
        )

    with col_target:
        st.caption("Variabila tinta")
        target_col = st.selectbox(
            "Variabila tinta",
            options=column_options,
            index=column_options.index(default_target) if default_target in column_options else 0,
            help="Selecteaza coloana cu valorile de antrenat",
            label_visibility="collapsed"
        )

    st.caption(f"ID articol: {selected_article_id}")

    with st.expander("Parametrii Model", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            lookback = st.number_input("Lookback (zile)", min_value=7, max_value=60, value=28)
        with col2:
            horizon = st.number_input("Orizont (zile)", min_value=1, max_value=90, value=30)

        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5) / 100
        with col2:
            min_samples = st.number_input("Min. esantioane", min_value=10, max_value=100, value=30)

    st.divider()
    forecast_button = st.button("Genereaza Prognoza", use_container_width=True)

with right_col:
    # ============================================================================
    # MAIN LOGIC
    # ============================================================================
    df_article = filter_by_article(df_full, selected_article_id, article_col)

    if not forecast_button:
        st.subheader("Vizualizare initiala")

        fig_hist = px.line(
            df_article,
            x="DATA",
            y=target_col,
            title="Evolutie istorica",
            labels={"DATA": "Data", target_col: "Valoare"}
        )
        fig_hist.update_xaxes(rangeslider_visible=True)
        fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_dist = plot_distribution(df_article[target_col], title="Distributie valori")
        fig_dist.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.divider()

        # Filter data for selected article
        if df_article is None or len(df_article) == 0:
            st.error(f"Nu s-au gasit date pentru articolul {selected_article_id}")
            st.stop()

        st.info(f"ID Articol: {selected_article_id} | Date: {len(df_article)} randuri")

        # ATTEMPT XGBOOST
        model, fe, info = build_and_train_xgboost(
            df_article,
            target_col=target_col,
            test_size=test_size,
            min_samples=min_samples
        )

        if model is None:
            # FALLBACK TO BASELINE
            st.warning(f"Insuficienta date pentru XGBoost: {info.get('error', 'Motiv necunoscut')}")
            st.info("Se foloseste model baseline (Exponential Smoothing / Moving Average)")

            baseline_model, baseline_info = build_baseline_fallback(df_article, target_col=target_col)

            if baseline_model is None:
                st.error("Insuficienta date chiar si pentru baseline!")
                st.stop()

            # Display baseline metrics
            st.subheader("Model Baseline")
            col1, col2, col3, col4 = st.columns(4)

            metrics = baseline_info['metrics']['metrics']
            with col1:
                create_kpi_card("Model", baseline_info['metrics']['model_name'], 'info')
            with col2:
                create_kpi_card("MAE", f"{metrics['mae']:.2f}", 'metric')
            with col3:
                create_kpi_card("RMSE", f"{metrics['rmse']:.2f}", 'metric')
            with col4:
                create_kpi_card("R2", f"{metrics['r2']:.3f}", 'metric')

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
                'PREDICTIE': future_pred,
                'Tip': 'Prognoza (Baseline)'
            })

            st.success(f"Prognoza {horizon} zile generata (baseline model)")

            # Display forecast
            st.subheader("Prognoza Baseline")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.dataframe(df_plot.round(2), use_container_width=True, hide_index=True)

            with col2:
                csv = df_plot.to_csv(index=False).encode()
                st.download_button(
                    "Descarca CSV",
                    csv,
                    file_name=f"prognoza_baseline_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        else:
            # XGBOOST SUCCESS
            st.success("XGBoost Model antrenat cu succes")

            # Display training info
            st.subheader("Metrice Evaluare")

            metrics = info['metrics']
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                create_kpi_card("MAE", f"{metrics['mae']:.2f}", 'metric')
            with col2:
                create_kpi_card("RMSE", f"{metrics['rmse']:.2f}", 'metric')
            with col3:
                create_kpi_card("R2", f"{metrics['r2']:.3f}", 'metric')
            with col4:
                n_features = info['n_features']
                create_kpi_card("Features", f"{n_features}", 'info')

            # GENERATE FORECAST
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
                'PREDICTIE': future_pred,
                'Tip': 'Prognoza XGBoost'
            })

            st.success(f"Prognoza {horizon} zile generata")

            tab1, tab2, tab3, tab4 = st.tabs(["Grafic", "Metrici", "Tabel", "Risc"])

            with tab1:
                st.subheader("Comparatie Actual vs Predictie vs Prognoza")

                test_df = info['df_test']
                y_pred_test = info['y_pred']

                df_comparison = test_df.copy()
                df_comparison['PREDICTIE'] = y_pred_test

                fig = plot_forecast_comparison(
                    df_article[['DATA', target_col]],
                    df_plot,
                    df_comparison[['DATA', target_col, 'PREDICTIE']],
                    target_col=target_col
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Metrici Detaliate")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f} unitati")
                    st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f} unitati")

                with col2:
                    st.metric("R2 Score", f"{metrics['r2']:.4f}")
                    st.metric("Test Samples", f"{metrics['n_samples']}")

                st.divider()

                st.subheader("Top Features")
                try:
                    importance_df = model.get_feature_importance(top_n=10)
                    fig_importance = plot_monthly_aggregation(importance_df)
                    st.plotly_chart(fig_importance, use_container_width=True)
                except Exception:
                    st.info("Feature importance nu poate fi calculata pentru acest model")

            with tab3:
                st.subheader("Tabel Prognoza")

                col1, col2 = st.columns([2, 1])

                with col1:
                    df_display = df_plot.copy()
                    df_display['PREDICTIE'] = df_display['PREDICTIE'].round(2)
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

                with col2:
                    csv = df_plot.to_csv(index=False).encode()
                    st.download_button(
                        "Descarca CSV",
                        csv,
                        file_name=f"prognoza_xgboost_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

            with tab4:
                st.subheader("Analiza Risc Ruptura")

                metrics_stockout = get_stockout_metrics(df_article, target_col=target_col)

                col1, col2, col3 = st.columns(3)

                with col1:
                    create_kpi_card(
                        "Volatilitate",
                        f"{metrics_stockout['std_daily']:.2f}",
                        'metric'
                    )

                with col2:
                    zero_pct = (metrics_stockout['zero_sale_days'] / len(df_article)) * 100
                    create_kpi_card(
                        "Zile Zero Vanzare",
                        f"{zero_pct:.1f}%",
                        'warning' if zero_pct > 20 else 'metric'
                    )

                with col3:
                    create_kpi_card(
                        "Coef. Variatie",
                        f"{metrics_stockout['cv']:.2f}",
                        'metric'
                    )

                st.divider()

                avg_demand = df_article[target_col].mean()
                std_demand = df_article[target_col].std()
                forecast_mean = future_pred.mean()
                forecast_std = future_pred.std()

                st.markdown("#### Recomandari")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        """
                        **Metrici Istoric:**
                        - Cerere medie: {:.1f} unitati/zi
                        - Deviere standard: {:.1f}
                        - Coef. Variatie: {:.2f}
                        """.format(avg_demand, std_demand, metrics_stockout['cv'])
                    )

                with col2:
                    st.write(
                        """
                        **Prognoza {}d:**
                        - Cerere medie: {:.1f} unitati/zi
                        - Deviere standard: {:.1f}
                        - Min/Max: {:.0f} / {:.0f}
                        """.format(horizon, forecast_mean, forecast_std,
                                   future_pred.min(), future_pred.max())
                    )

                service_level = 0.95
                z_score = 1.645
                safety_stock = z_score * std_demand

                st.success(
                    """
                    **Stock de Siguranta (SL 95%):** {stock:.0f} unitati

                    Reaprovizioneaza cand stocul ajunge sub {stock:.0f} unitati
                    pentru a evita ruptura cu o probabilitate de 95%.
                    """.format(stock=safety_stock)
                )

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
**Dashboard Optimizare Stocuri | XGBoost Forecasting**  
Disertatie - Metode Predictive pentru Optimizarea Stocurilor
""")
