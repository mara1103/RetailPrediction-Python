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
st.markdown(
    """
    <style>
    html {font-size: 93%;}
    body {background: #0c0f14;}
    .block-container {padding-top: 1.8rem; padding-bottom: 0.8rem;}
    [data-testid="stVerticalBlock"] {gap: 0.45rem;}
    .stAlert {padding: 0.25rem 0.6rem; font-size: 0.9rem;}
    .stTabs {margin-top: -0.35rem;}
    h1, h2, h3 {letter-spacing: 0.2px;}
    h1 {margin: 0; padding-top: 0.2rem; font-size: 2.25rem; line-height: 1.2;}
    h2 {margin-top: 0.4rem; font-size: 1.45rem;}
    h3 {margin-top: 0.35rem; font-size: 1.15rem;}
    .hero {
        padding: 1.1rem 1.2rem 1.0rem 1.2rem;
        border-radius: 14px;
        background: radial-gradient(1200px 280px at 10% -10%, #1b2a3a 0%, #10151e 45%, #0c0f14 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
    }
    .hero-title {color: #e9edf3; font-weight: 700;}
    .hero-subtitle {
        color: #c9d2dd;
        margin-top: 0.4rem;
        font-size: 0.98rem;
        line-height: 1.45;
    }
    .hero-accent {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        background: linear-gradient(135deg, rgba(55, 170, 255, 0.35), rgba(55, 170, 255, 0.12));
        color: #d7ecff;
        border: 1px solid rgba(140, 199, 255, 0.6);
        box-shadow: 0 6px 14px rgba(20, 110, 200, 0.25);
        margin-top: 0.45rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">Dashboard Optimizare Stocuri - XGBoost</h1>
        <div class="hero-accent">Inventory Forecasting</div>
        <div class="hero-subtitle">
            Aplicatie pentru <strong>predictia cererii</strong> si managementul stocurilor
            folosind <strong>XGBoost</strong>. Modelul se antreneaza per articol cu feature
            engineering temporal si statistic.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if "model_result" not in st.session_state:
    st.session_state.model_result = None

page = st.sidebar.radio("Navigare", ["EDA initiala", "Model", "Risc"], index=0)

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
    forecast_button = st.button(
        "Genereaza Prognoza",
        use_container_width=True,
        disabled=page != "Model"
    )

with right_col:
    # ============================================================================
    # MAIN LOGIC
    # ============================================================================
    df_article = filter_by_article(df_full, selected_article_id, article_col)
    selection_key = (csv_path, bf_handling, selected_article_id, target_col, horizon, test_size, min_samples)
    saved = st.session_state.model_result
    saved_is_valid = saved is not None and saved.get("selection_key") == selection_key

    if page == "EDA initiala":
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

        col1, col2 = st.columns(2)

        with col1:
            fig_dist = plot_distribution(df_article[target_col], title="Distributie valori")
            fig_dist.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            monthly_df = (
                df_article
                .assign(LUNA=df_article["DATA"].dt.to_period("M").dt.to_timestamp())
                .groupby("LUNA", as_index=False)[target_col]
                .mean()
            )
            fig_month = px.bar(
                monthly_df,
                x="LUNA",
                y=target_col,
                title="Medie lunara",
                labels={"LUNA": "Luna", target_col: "Valoare"}
            )
            fig_month.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_month, use_container_width=True)

        st.divider()
        st.caption("Tip: treci la pagina Model pentru antrenare si prognoza.")

    elif page == "Model":
        if df_article is None or len(df_article) == 0:
            st.error(f"Nu s-au gasit date pentru articolul {selected_article_id}")
            st.stop()

        if forecast_button:
            st.divider()
            st.info(f"ID Articol: {selected_article_id} | Date: {len(df_article)} randuri")

            model, fe, info = build_and_train_xgboost(
                df_article,
                target_col=target_col,
                test_size=test_size,
                min_samples=min_samples
            )

            if model is None:
                st.warning(f"Insuficienta date pentru XGBoost: {info.get('error', 'Motiv necunoscut')}")
                st.info("Se foloseste model baseline (Exponential Smoothing / Moving Average)")

                baseline_model, baseline_info = build_baseline_fallback(df_article, target_col=target_col)

                if baseline_model is None:
                    st.error("Insuficienta date chiar si pentru baseline!")
                    st.stop()

                y_baseline = df_article[target_col].values
                y_baseline = y_baseline[~np.isnan(y_baseline)]

                baseline_model.fit(y_baseline)
                future_pred = baseline_model.forecast(horizon)

                df_plot = pd.DataFrame({
                    'DATA': pd.date_range(
                        start=df_article['DATA'].max() + timedelta(days=1),
                        periods=horizon,
                        freq='D'
                    ),
                    'PREDICTIE': future_pred,
                    'Tip': 'Prognoza (Baseline)'
                })

                st.session_state.model_result = {
                    "selection_key": selection_key,
                    "model_type": "baseline",
                    "metrics": baseline_info['metrics']['metrics'],
                    "model_name": baseline_info['metrics']['model_name'],
                    "df_plot": df_plot,
                    "df_article": df_article,
                    "target_col": target_col,
                    "horizon": horizon,
                    "future_pred": future_pred
                }
            else:
                metrics = info['metrics']

                future_pred, df_extended = model.predict_future(
                    df_article,
                    fe,
                    horizon=horizon,
                    target_col=target_col
                )

                last_date = df_article['DATA'].max()
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

                df_plot = pd.DataFrame({
                    'DATA': forecast_dates,
                    'PREDICTIE': future_pred,
                    'Tip': 'Prognoza XGBoost'
                })

                st.session_state.model_result = {
                    "selection_key": selection_key,
                    "model_type": "xgboost",
                    "metrics": metrics,
                    "n_features": info['n_features'],
                    "df_plot": df_plot,
                    "df_article": df_article,
                    "target_col": target_col,
                    "horizon": horizon,
                    "future_pred": future_pred,
                    "df_test": info['df_test'],
                    "y_pred": info['y_pred'],
                    "model": model
                }

        if not saved_is_valid:
            st.info("Apasa «Genereaza Prognoza» pentru a rula modelul.")
        else:
            result = st.session_state.model_result
            if result["model_type"] == "baseline":
                st.success(f"Prognoza {result['horizon']} zile generata (baseline model)")

                st.subheader("Model Baseline")
                col1, col2, col3, col4 = st.columns(4)

                metrics = result["metrics"]
                with col1:
                    create_kpi_card("Model", result["model_name"], 'info')
                with col2:
                    create_kpi_card("MAE", f"{metrics['mae']:.2f}", 'metric')
                with col3:
                    create_kpi_card("RMSE", f"{metrics['rmse']:.2f}", 'metric')
                with col4:
                    create_kpi_card("R2", f"{metrics['r2']:.3f}", 'metric')

                st.subheader("Prognoza Baseline")
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.dataframe(result["df_plot"].round(2), use_container_width=True, hide_index=True)

                with col2:
                    csv = result["df_plot"].to_csv(index=False).encode()
                    st.download_button(
                        "Descarca CSV",
                        csv,
                        file_name=f"prognoza_baseline_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.success(f"Prognoza {result['horizon']} zile generata")

                tab1, tab2, tab3 = st.tabs(["Grafic", "Metrici", "Tabel"])

                with tab1:
                    st.subheader("Comparatie Actual vs Predictie vs Prognoza")

                    df_comparison = result["df_test"].copy()
                    df_comparison['PREDICTIE'] = result["y_pred"]

                    fig = plot_forecast_comparison(
                        result["df_article"][['DATA', result["target_col"]]],
                        result["df_plot"],
                        df_comparison[['DATA', result["target_col"], 'PREDICTIE']],
                        target_col=result["target_col"]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    col1, col2, col3, col4 = st.columns(4)

                    metrics = result["metrics"]
                    with col1:
                        create_kpi_card("MAE", f"{metrics['mae']:.2f}", 'metric')
                    with col2:
                        create_kpi_card("RMSE", f"{metrics['rmse']:.2f}", 'metric')
                    with col3:
                        create_kpi_card("R2", f"{metrics['r2']:.3f}", 'metric')
                    with col4:
                        create_kpi_card("Features", f"{result['n_features']}", 'info')

                    st.divider()
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
                        importance_df = result["model"].get_feature_importance(top_n=10)
                        fig_importance = plot_monthly_aggregation(importance_df)
                        st.plotly_chart(fig_importance, use_container_width=True)
                    except Exception:
                        st.info("Feature importance nu poate fi calculata pentru acest model")

                with tab3:
                    st.subheader("Tabel Prognoza")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        df_display = result["df_plot"].copy()
                        df_display['PREDICTIE'] = df_display['PREDICTIE'].round(2)
                        st.dataframe(df_display, use_container_width=True, hide_index=True)

                    with col2:
                        csv = result["df_plot"].to_csv(index=False).encode()
                        st.download_button(
                            "Descarca CSV",
                            csv,
                            file_name=f"prognoza_xgboost_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

    else:
        if not saved_is_valid:
            st.info("Rulaeaza modelul in pagina «Model» pentru a vedea analiza de risc.")
        else:
            result = st.session_state.model_result
            st.subheader("Analiza Risc Ruptura")

            metrics_stockout = get_stockout_metrics(result["df_article"], target_col=result["target_col"])
            col1, col2, col3 = st.columns(3)

            with col1:
                create_kpi_card(
                    "Volatilitate",
                    f"{metrics_stockout['std_daily']:.2f}",
                    'metric'
                )

            with col2:
                zero_pct = (metrics_stockout['zero_sale_days'] / len(result["df_article"])) * 100
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

            avg_demand = result["df_article"][result["target_col"]].mean()
            std_demand = result["df_article"][result["target_col"]].std()
            forecast_mean = result["future_pred"].mean()
            forecast_std = result["future_pred"].std()

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
                    """.format(result["horizon"], forecast_mean, forecast_std,
                               result["future_pred"].min(), result["future_pred"].max())
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
