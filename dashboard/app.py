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
import textwrap
import plotly.io as pio
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
    page_title="Inventory Optimization - XGBoost Dashboard",
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


def find_column(df: pd.DataFrame, keywords: list) -> str:
    for col in df.columns:
        col_lower = col.lower()
        if any(key in col_lower for key in keywords):
            return col
    return ""


def pick_numeric_column(
    df: pd.DataFrame,
    include_keywords: list,
    exclude_columns: list
) -> str:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        col_lower = col.lower()
        if col in exclude_columns:
            continue
        if any(key in col_lower for key in include_keywords):
            return col

    fallback_cols = [
        c for c in numeric_cols
        if c not in exclude_columns
        and "id" not in c.lower()
        and "stoc" not in c.lower()
    ]
    if not fallback_cols:
        return ""

    totals = df[fallback_cols].abs().sum().sort_values(ascending=False)
    return totals.index[0] if not totals.empty else ""


def get_article_name_col(df: pd.DataFrame) -> str:
    return find_column(df, ["articol", "denumire", "nume", "product", "article", "item"])


def gdpr_friendly_label(name: str) -> str:
    name_lower = str(name).lower()
    rules = [
        ("balm", ["balsam", "balm", "conditioner"]),
        ("shampoo", ["sampon", "shampoo"]),
        ("mask", ["masca", "mask"]),
        ("serum", ["ser", "serum"]),
        ("cream", ["crema", "cream"]),
        ("gel", ["gel"]),
        ("lotion", ["lotiune", "lotion"]),
        ("spray", ["spray"]),
        ("perfume", ["parfum", "perfume"]),
        ("deodorant", ["deodorant"]),
        ("soap", ["sapun", "soap"]),
        ("dye", ["vopsea", "colorant", "dye"]),
    ]
    for label, keys in rules:
        if any(k in name_lower for k in keys):
            return label
    return "article"


def build_gdpr_name_map(df: pd.DataFrame, article_col: str) -> dict:
    if not article_col:
        return {}
    name_col = get_article_name_col(df)
    label_map = {}
    for idx, article_id in enumerate(sorted(df[article_col].unique()), start=1):
        if name_col:
            raw_name = df.loc[df[article_col] == article_id, name_col].dropna()
            label = gdpr_friendly_label(raw_name.iloc[0]) if not raw_name.empty else "article"
        else:
            label = "article"
        label_map[article_id] = f"{label.capitalize()} {idx:02d}"
    return label_map


def format_compact(value, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}"
    return str(value)


DISPLAY_COLUMN_MAP = {
    "DATA": "DATE",
    "ID_ARTICOL": "ARTICLE_ID",
    "ARTICOL": "ARTICLE",
    "CANTITATE": "QUANTITY",
    "VAL_IESIRE_FARA_TVA": "NET_SALES_NO_VAT",
    "VAL_IESIRE_CU_TVA": "GROSS_SALES_WITH_VAT",
    "VAL_INTRARE_FARA_TVA": "NET_COST",
    "RATA_DISCOUNT": "DISCOUNT_RATE",
    "ADAOS": "MARKUP",
    "STOC_INITIAL": "OPENING_STOCK",
    "STOC_FINAL": "CLOSING_STOCK",
    "RUPTURA_STOC": "STOCKOUT",
    "PREDICTIE": "PREDICTION",
    "Tip": "TYPE",
}


ARTICLE_VALUE_MAP = {
    "gel purifiant ten": "purifying face gel",
    "gel curatare fata": "facial cleansing gel",
    "gel spalare piele sensibila": "sensitive skin cleansing gel",
    "gel anti-imperfectiuni": "anti-blemish gel",
    "balsam hidratant corp": "body moisturizing balm",
    "balsam reparator piele": "skin repair balm",
    "crema hidratanta fata spf": "face moisturizer SPF",
    "crema corectoare fata": "corrective face cream",
    "lotiune hidratanta fata si corp": "face and body hydrating lotion",
    "sampon antimatreata": "anti-dandruff shampoo",
    "fluid protectie solara fata": "face sunscreen fluid",
    "spray protectie solara": "sunscreen spray",
    "spray protectie solara copii": "kids sunscreen spray",
}


def to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ARTICOL" in out.columns:
        out["ARTICOL"] = out["ARTICOL"].astype(str).map(
            lambda x: ARTICLE_VALUE_MAP.get(x.lower(), x)
        )
    if "ARTICLE" in out.columns:
        out["ARTICLE"] = out["ARTICLE"].astype(str).map(
            lambda x: ARTICLE_VALUE_MAP.get(x.lower(), x)
        )
    if "DATA" in out.columns:
        out["DATA"] = pd.to_datetime(out["DATA"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "DATE" in out.columns:
        out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.rename(columns=DISPLAY_COLUMN_MAP)


def metric_display_name(metric_col: str, qty_col: str, val_net_col: str, val_gross_col: str) -> str:
    if metric_col == qty_col:
        return "Quantity"
    if metric_col == val_net_col:
        return "Value excl. VAT"
    if metric_col == val_gross_col:
        return "Value incl. VAT"
    return metric_col or "Value"


def render_kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return textwrap.dedent(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """
    ).strip()


# ============================================================================
# HEADER
# ============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

is_dark = st.session_state.dark_mode
bg_color = "#0c0f14" if is_dark else "#f6f4ef"
hero_bg = "radial-gradient(1200px 280px at 10% -10%, #1b2a3a 0%, #10151e 45%, #0c0f14 100%)" if is_dark else \
    "radial-gradient(1200px 280px at 10% -10%, #d7dde8 0%, #f2f4f7 45%, #f6f4ef 100%)"
text_color = "#e9edf3" if is_dark else "#1f2633"
muted_text = "#c9d2dd" if is_dark else "#5a6472"
card_bg = "linear-gradient(145deg, rgba(20, 27, 38, 0.95), rgba(12, 16, 23, 0.95))" if is_dark else \
    "linear-gradient(145deg, rgba(250, 250, 252, 0.98), rgba(235, 238, 244, 0.98))"
border_color = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(30, 36, 45, 0.12)"
shadow_color = "rgba(0, 0, 0, 0.35)" if is_dark else "rgba(18, 24, 32, 0.14)"
kpi_label_color = "#9aa7b4" if is_dark else "#4c5666"
kpi_value_color = "#e9edf3" if is_dark else "#1f2633"
kpi_sub_color = "#9aa7b4" if is_dark else "#6a7482"
select_bg = "#0f141c" if is_dark else "#ffffff"
select_text = "#e9edf3" if is_dark else "#1f2633"
select_border = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(31, 38, 51, 0.18)"
table_bg = "#0f141c" if is_dark else "#ffffff"
table_header_bg = "#141a24" if is_dark else "#f5f6f8"
table_header_text = "#e9edf3" if is_dark else "#1f2633"
table_border = "rgba(255,255,255,0.16)" if is_dark else "rgba(31,38,51,0.18)"
plotly_template = "plotly_dark" if is_dark else "plotly_white"
plot_bg = "#0c0f14" if is_dark else "#f6f4ef"
grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.12)"

pio.templates.default = plotly_template
px.defaults.template = plotly_template

def apply_plot_theme(fig):
    fig.update_layout(
        template=plotly_template,
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        title_font=dict(color=text_color),
        legend=dict(font=dict(color=text_color))
    )
    fig.update_xaxes(
        gridcolor=grid_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color)
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color)
    )
    return fig

st.markdown(
    """
    <style>
    html {font-size: 93%;}
    body {background: """ + bg_color + """; color: """ + text_color + """;}
    .stApp {background: """ + bg_color + """; color: """ + text_color + """;}
    .stMarkdown, .stText, .stCaption, .stHeader, .stSubheader, .stTitle {
        color: """ + text_color + """;
    }
    .stCaption {color: """ + muted_text + """;}
    .stButton>button {

    .stExpander > details {
        background: """ + ("#121722" if is_dark else "#ffffff") + """ !important;
        border: 1px solid """ + ("rgba(255,255,255,0.12)" if is_dark else "rgba(31,38,51,0.18)") + """ !important;
        border-radius: 10px;
    }
    .stExpander > details > summary {
        color: """ + text_color + """ !important;
        background: """ + ("#121722" if is_dark else "#ffffff") + """ !important;
    }

        color: """ + text_color + """ !important;
        background: """ + ("#171c26" if is_dark else "#ffffff") + """ !important;
        border: 1px solid """ + ("rgba(255,255,255,0.12)" if is_dark else "rgba(31,38,51,0.18)") + """ !important;
    }
    .block-container {padding-top: 1.8rem; padding-bottom: 0.8rem;}

    section[data-testid="stSidebar"] {
        background: """ + ("#0e131b" if is_dark else "#f3f1ec") + """ !important;
        color: """ + text_color + """ !important;
        border-right: 1px solid """ + border_color + """;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stText,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: """ + text_color + """ !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: """ + text_color + """ !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background: """ + select_bg + """ !important;
        color: """ + select_text + """ !important;
        border-color: """ + select_border + """ !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label span {
        color: """ + text_color + """ !important;
    }
    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: """ + text_color + """ !important;
    }
    section[data-testid="stSidebar"] .stButton>button {

    .stExpander > details {
        background: """ + ("#121722" if is_dark else "#ffffff") + """ !important;
        border: 1px solid """ + ("rgba(255,255,255,0.12)" if is_dark else "rgba(31,38,51,0.18)") + """ !important;
        border-radius: 10px;
    }
    .stExpander > details > summary {
        color: """ + text_color + """ !important;
        background: """ + ("#121722" if is_dark else "#ffffff") + """ !important;
    }

        background: """ + ( "#171c26" if is_dark else "#ffffff" ) + """ !important;
        color: """ + text_color + """ !important;
        border: 1px solid """ + select_border + """ !important;
    }
    header[data-testid="stHeader"] {
        background: """ + ( "#0c0f14" if is_dark else "#f6f4ef" ) + """ !important;
    }


    div[data-baseweb="select"] > div {
        background: """ + select_bg + """ !important;
        color: """ + select_text + """ !important;
        border-color: """ + select_border + """ !important;
    }
    div[data-baseweb="select"] * {color: """ + select_text + """ !important;}
    div[data-baseweb="select"] svg {fill: """ + select_text + """ !important;}
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        background: """ + table_bg + """ !important;
        border: 1px solid """ + table_border + """ !important;
        border-radius: 10px;
    }
    div[data-testid="stDataFrame"] table,
    div[data-testid="stTable"] table {
        background: """ + table_bg + """ !important;
        color: """ + text_color + """ !important;
    }
    div[data-testid="stDataFrame"] th,
    div[data-testid="stTable"] th {
        background: """ + table_header_bg + """ !important;
        color: """ + table_header_text + """ !important;
    }
    div[data-testid="stDataFrame"] td,
    div[data-testid="stTable"] td {
        background: """ + table_bg + """ !important;
        color: """ + text_color + """ !important;
        border-bottom: 1px solid """ + table_border + """ !important;
    }
    div[data-testid="stDataFrame"] [role="grid"],
    div[data-testid="stDataFrame"] [role="rowgroup"],
    div[data-testid="stDataFrame"] [role="row"],
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] .ag-root,
    div[data-testid="stDataFrame"] .ag-root-wrapper,
    div[data-testid="stDataFrame"] .ag-body-viewport,
    div[data-testid="stDataFrame"] .ag-center-cols-container {
        background: """ + table_bg + """ !important;
        color: """ + text_color + """ !important;
    }
    div[data-testid="stDataFrame"] .ag-header {
        background: """ + table_header_bg + """ !important;
        color: """ + table_header_text + """ !important;
        border-bottom: 1px solid """ + table_border + """ !important;
    }
    div[data-testid="stDataFrame"] .ag-header-cell,
    div[data-testid="stDataFrame"] .ag-header-cell-text {
        color: """ + table_header_text + """ !important;
    }
    div[data-testid="stDataFrame"] .ag-cell {
        color: """ + text_color + """ !important;
        background: """ + table_bg + """ !important;
    }
    div[data-testid="stDataFrame"] .ag-row:nth-child(even) {
        background: """ + table_header_bg + """ !important;
    }
    div[data-testid="stDataFrame"] .ag-row {
        border-bottom: 1px solid """ + table_border + """ !important;
    }
    div[data-testid="stDataFrame"] tr:nth-child(even) td,
    div[data-testid="stTable"] tr:nth-child(even) td {
        background: """ + table_header_bg + """ !important;
    }
    div[data-baseweb="select"]:hover > div {
        box-shadow: 0 0 0 2px """ + select_border + """ !important;
    }
    div[data-baseweb="select"] > div:focus,
    div[data-baseweb="select"] > div:focus-within {
        box-shadow: 0 0 0 2px """ + select_border + """ !important;
    }
    [data-testid="stVerticalBlock"] {gap: 0.45rem;}
    .stAlert {padding: 0.25rem 0.6rem; font-size: 0.9rem;}
    .stTabs {margin-top: -0.35rem;}
    h1, h2, h3 {letter-spacing: 0.2px;}
    h1 {margin: 0; padding-top: 0.2rem; font-size: 2.25rem; line-height: 1.2;}
    h2 {margin-top: 0.4rem; font-size: 1.45rem;}
    h3 {margin-top: 0.35rem; font-size: 1.15rem;}
        color: """ + text_color + """;
    }
    .hero {
        padding: 1.1rem 1.2rem 1.0rem 1.2rem;
        border-radius: 14px;
        background: """ + hero_bg + """;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 14px 30px """ + shadow_color + """;
    }
    .hero-title {color: """ + text_color + """; font-weight: 700;}
    .hero-subtitle {
        color: """ + muted_text + """;
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
        background: """ + ("linear-gradient(135deg, rgba(55, 170, 255, 0.35), rgba(55, 170, 255, 0.12))" if is_dark else "linear-gradient(135deg, rgba(41, 98, 255, 0.22), rgba(41, 98, 255, 0.08))") + """;
        color: """ + ("#d7ecff" if is_dark else "#1f4fbf") + """;
        border: 1px solid """ + ("rgba(140, 199, 255, 0.6)" if is_dark else "rgba(31, 79, 191, 0.35)") + """;
        box-shadow: 0 6px 14px """ + ("rgba(20, 110, 200, 0.25)" if is_dark else "rgba(31, 79, 191, 0.12)") + """;
        margin-top: 0.45rem;
        margin-bottom: 0.5rem;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .kpi-card {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: """ + card_bg + """;
        border: 1px solid """ + border_color + """;
        box-shadow: 0 10px 24px """ + shadow_color + """;
    }
    .kpi-label {
        color: """ + kpi_label_color + """;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-value {
        color: """ + kpi_value_color + """;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .kpi-sub {
        color: """ + kpi_sub_color + """;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">Inventory Optimization Dashboard - XGBoost</h1>
        <div class="hero-accent">Inventory Forecasting</div>
    </div>
    """,
    unsafe_allow_html=True
)


if "model_result" not in st.session_state:
    st.session_state.model_result = None

# Theme Toggle
st.sidebar.markdown("### Theme")
mode_label = "Switch to Day" if is_dark else "Switch to Night"
st.sidebar.button(mode_label, on_click=toggle_theme, use_container_width=True)

with st.container():
    col1, col2 = st.columns([6, 1])
    with col2:
        st.button(mode_label, on_click=toggle_theme, use_container_width=True)


page = st.sidebar.radio("Navigation", ["Data Explorer", "Model", "Risk"], index=0)

st.sidebar.header("Dataset")
st.sidebar.caption("CSV file path")
csv_path = st.sidebar.text_input(
    "CSV Path",
    value="../data_raw/date_24_art_portfolio_20.csv",
    help="Enter relative or absolute path to the CSV file",
    label_visibility="collapsed"
)

st.sidebar.caption("Black Friday")
bf_handling = st.sidebar.selectbox(
    "Black Friday",
    options=["none", "exclude"],
    format_func=lambda x: {
        "none": "No change",
        "exclude": "Exclude BF days (2024: Nov 15-20, 2025: Nov 14-18)"
    }[x],
    help="Exclude Black Friday dates",
    label_visibility="collapsed"
)

try:
    df_full, default_target = load_csv_data(csv_path, bf_handling)
    article_ids = get_article_ids(df_full)
    column_options = get_column_options(df_full)
    st.sidebar.success("CSV loaded successfully")
except Exception as e:
    st.sidebar.error(f"CSV loading error: {str(e)}")
    st.stop()

article_col = "ID_ARTICOL" if "ID_ARTICOL" in df_full.columns else find_column(
    df_full,
    ["id_articol", "articol", "sku", "article"]
)
if article_col:
    article_ids = sorted(df_full[article_col].unique())
    gdpr_name_map = build_gdpr_name_map(df_full, article_col)
    gdpr_name_list = [gdpr_name_map.get(a, f"Article {i+1:02d}") for i, a in enumerate(article_ids)]
    id_by_name = {gdpr_name_list[i]: article_ids[i] for i in range(len(article_ids))}
    name_by_id = {article_ids[i]: gdpr_name_list[i] for i in range(len(article_ids))}
else:
    gdpr_name_map = {}
    gdpr_name_list = []
    id_by_name = {}
    name_by_id = {}

lookback = 28
horizon = 30
test_size = 0.2
min_samples = 30
selected_article_id = None
target_col = default_target
forecast_button = False

if page != "Data Explorer":
    st.sidebar.header("Model settings")
    if article_col:
        if "model_article_id" not in st.session_state:
            st.session_state.model_article_id = article_ids[0]
        if "model_article_name" not in st.session_state:
            st.session_state.model_article_name = name_by_id.get(article_ids[0], gdpr_name_list[0])

        def _sync_model_name():
            st.session_state.model_article_name = name_by_id.get(
                st.session_state.model_article_id,
                gdpr_name_list[0]
            )

        def _sync_model_id():
            st.session_state.model_article_id = id_by_name.get(
                st.session_state.model_article_name,
                article_ids[0]
            )

        st.sidebar.caption("Article (ID)")
        st.sidebar.selectbox(
            "Article ID",
            options=article_ids,
            key="model_article_id",
            on_change=_sync_model_name,
            help="Select the article ID",
            label_visibility="collapsed"
        )

        st.sidebar.caption("Article (GDPR-friendly name)")
        st.sidebar.selectbox(
            "Article name",
            options=gdpr_name_list,
            key="model_article_name",
            on_change=_sync_model_id,
            help="Select the generic article label",
            label_visibility="collapsed"
        )

        selected_article_id = st.session_state.model_article_id
    else:
        st.sidebar.warning("No ARTICLE_ID column found for article selection.")

    st.sidebar.caption("Target variable")
    target_col = st.sidebar.selectbox(
        "Target variable",
        options=column_options,
        index=column_options.index(default_target) if default_target in column_options else 0,
        help="Select the column to train on",
        label_visibility="collapsed"
    )

    with st.sidebar.expander("Model parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            lookback = st.number_input("Lookback (days)", min_value=7, max_value=60, value=28)
        with col2:
            horizon = st.number_input("Horizon (days)", min_value=1, max_value=90, value=30)

        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5) / 100
        with col2:
            min_samples = st.number_input("Min. samples", min_value=10, max_value=100, value=30)

    st.sidebar.divider()
    forecast_button = st.sidebar.button(
        "Generate Forecast",
        use_container_width=True,
        disabled=page != "Model"
    )

df_article = None
if page != "Data Explorer" and selected_article_id is not None:
    df_article = filter_by_article(df_full, selected_article_id, article_col)

selection_key = (csv_path, bf_handling, selected_article_id, target_col, horizon, test_size, min_samples)
saved = st.session_state.model_result
saved_is_valid = saved is not None and saved.get("selection_key") == selection_key

# ============================================================================
# MAIN LOGIC
# ============================================================================
if page == "Data Explorer":
    st.subheader("Dataset exploration")
    st.caption("General data analysis for inventory optimization (without model configuration).")

    date_min = df_full["DATA"].min()
    date_max = df_full["DATA"].max()
    total_rows = len(df_full)
    total_cols = df_full.shape[1]
    total_articles = df_full[article_col].nunique() if article_col else 0
    total_days = (date_max - date_min).days + 1 if pd.notnull(date_min) and pd.notnull(date_max) else 0
    date_min_str = date_min.strftime("%Y-%m-%d") if pd.notnull(date_min) else "N/A"
    date_max_str = date_max.strftime("%Y-%m-%d") if pd.notnull(date_max) else "N/A"

    st.markdown(
        f"""
        **Dataset summary**  
        Date range: **{date_min_str}** - **{date_max_str}** ({total_days} days)  
        Unique articles: **{total_articles}**
        """
    )

    st.divider()

    df_focus = None
    if article_col:
        if "eda_article_id" not in st.session_state:
            st.session_state.eda_article_id = article_ids[0]
        if "eda_article_name" not in st.session_state:
            st.session_state.eda_article_name = name_by_id.get(article_ids[0], gdpr_name_list[0])

        def _sync_eda_name():
            st.session_state.eda_article_name = name_by_id.get(
                st.session_state.eda_article_id,
                gdpr_name_list[0]
            )

        def _sync_eda_id():
            st.session_state.eda_article_id = id_by_name.get(
                st.session_state.eda_article_name,
                article_ids[0]
            )

        col_sel1, col_sel2 = st.columns([1, 1])
        with col_sel1:
            st.caption("Article selection (ID)")
            st.selectbox(
                "Select ID",
                options=article_ids,
                key="eda_article_id",
                on_change=_sync_eda_name,
                label_visibility="collapsed"
            )
        with col_sel2:
            st.caption("Article selection (GDPR-friendly name)")
            st.selectbox(
                "Select name",
                options=gdpr_name_list,
                key="eda_article_name",
                on_change=_sync_eda_id,
                label_visibility="collapsed"
            )

        selected_eda_id = st.session_state.eda_article_id
        selected_eda_name = name_by_id.get(selected_eda_id, "Article")
        st.caption(f"Selected article: **{selected_eda_name}**")
        df_focus = filter_by_article(df_full, selected_eda_id, article_col)
    else:
        st.warning("No ARTICLE_ID column found for article selection. Showing global analysis.")
        df_focus = df_full.copy()

    df_focus_all = df_focus

    if "eda_filter_active" not in st.session_state:
        st.session_state.eda_filter_active = False
    if "eda_filter_year" not in st.session_state:
        st.session_state.eda_filter_year = ["All"]
    if "eda_filter_month" not in st.session_state:
        st.session_state.eda_filter_month = ["All"]

    with st.expander("Time filter", expanded=False):
        years = sorted(df_focus_all["DATA"].dt.year.dropna().unique())
        year_options = ["All"] + years
        st.multiselect("Years", options=year_options, key="eda_filter_year", default=["All"])

        month_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        months = sorted(df_focus_all["DATA"].dt.month.dropna().unique())
        month_labels = [f"{m:02d} - {month_map.get(m, m)}" for m in months]
        st.multiselect("Months", options=["All"] + month_labels, key="eda_filter_month", default=["All"])

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Apply filter", use_container_width=True):
                st.session_state.eda_filter_active = True
        with col_b:
            if st.button("Reset filter", use_container_width=True):
                st.session_state.eda_filter_active = False
                st.session_state.eda_filter_year = ["All"]
                st.session_state.eda_filter_month = ["All"]

    if st.session_state.eda_filter_active:
        df_focus = df_focus_all.copy()
        if st.session_state.eda_filter_year and "All" not in st.session_state.eda_filter_year:
            df_focus = df_focus[df_focus["DATA"].dt.year.isin([int(y) for y in st.session_state.eda_filter_year])]
        if st.session_state.eda_filter_month and "All" not in st.session_state.eda_filter_month:
            month_nums = [int(str(m).split("-")[0].strip()) for m in st.session_state.eda_filter_month]
            df_focus = df_focus[df_focus["DATA"].dt.month.isin(month_nums)]

        if df_focus.empty:
            st.warning("No data available for selected filter.")
            st.stop()

    exclude_cols = ['RUPTURA_STOC', article_col, 'STOC_INITIAL', 'STOC_FINAL']
    qty_col = pick_numeric_column(df_focus, ["cantitate", "qty", "quantity", "sales"], exclude_cols)
    val_net_col = pick_numeric_column(df_focus, ["val_iesire_fara_tva", "fara_tva", "net", "valoare"], exclude_cols)
    val_gross_col = pick_numeric_column(df_focus, ["val_iesire_cu_tva", "cu_tva", "gross", "valoare"], exclude_cols)

    if val_net_col == qty_col:
        val_net_col = ""
    if val_gross_col in (qty_col, val_net_col):
        val_gross_col = ""

    qty_total = df_focus[qty_col].sum() if qty_col else None
    val_net_total = df_focus[val_net_col].sum() if val_net_col else None
    val_gross_total = df_focus[val_gross_col].sum() if val_gross_col else None

    cards_html = textwrap.dedent(
        """
        <div class="kpi-grid">
            {c1}{c2}{c3}{c4}
        </div>
        """
    ).format(
        c1=render_kpi_card("Total quantity", format_compact(qty_total, 0)),
        c2=render_kpi_card("Value excl. VAT", format_compact(val_net_total, 2)),
        c3=render_kpi_card("Value incl. VAT", format_compact(val_gross_total, 2)),
        c4=render_kpi_card("Article structure", f"{len(df_focus):,} rows", f"{total_cols} columns")
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    if qty_col:
        daily_qty = df_focus.groupby("DATA", as_index=False)[qty_col].sum()
        fig_qty = px.line(
            daily_qty,
            x="DATA",
            y=qty_col,
            title="Daily quantity trend (total)",
            labels={"DATA": "Date", qty_col: "Quantity"}
        )
        fig_qty.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        apply_plot_theme(fig_qty)
        st.plotly_chart(fig_qty, use_container_width=True)

    main_metric = qty_col or val_net_col or val_gross_col
    if main_metric:
        main_label = metric_display_name(main_metric, qty_col, val_net_col, val_gross_col)
        col1, col2 = st.columns(2)
        with col1:
            fig_dist = plot_distribution(df_focus[main_metric], title=f"Distribution - {main_label}")
            fig_dist.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            fig_dist.update_xaxes(title_text=main_label)
            apply_plot_theme(fig_dist)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            monthly_avg = (
                df_focus
                .assign(LUNA=df_focus["DATA"].dt.to_period("M").dt.to_timestamp())
                .groupby("LUNA", as_index=False)[main_metric]
                .mean()
            )
            fig_month = px.bar(
                monthly_avg,
                x="LUNA",
                y=main_metric,
                title=f"Monthly average - {main_label}",
                labels={"LUNA": "Month", main_metric: f"{main_label} (avg)"}
            )
            fig_month.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            apply_plot_theme(fig_month)
            st.plotly_chart(fig_month, use_container_width=True)

    if article_col and df_focus[article_col].nunique() > 1:
        col1, col2 = st.columns(2)
        with col1:
            if val_net_col or qty_col:
                top_metric = val_net_col or qty_col
                top_label = metric_display_name(top_metric, qty_col, val_net_col, val_gross_col)
                top_df = (
                    df_focus.groupby(article_col, as_index=False)[top_metric]
                    .sum()
                    .sort_values(top_metric, ascending=False)
                    .head(10)
                )
                fig_top = px.bar(
                    top_df,
                    x=article_col,
                    y=top_metric,
                    title=f"Top 10 articles by {top_label} (sum)",
                    labels={article_col: "Article", top_metric: f"{top_label} (sum)"}
                )
                fig_top.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                apply_plot_theme(fig_top)
                st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            if qty_col:
                zero_share = (
                    df_focus.groupby(article_col)[qty_col]
                    .apply(lambda s: (s == 0).mean() * 100)
                    .reset_index(name="PCT_ZERO")
                    .sort_values("PCT_ZERO", ascending=False)
                    .head(12)
                )
                fig_zero = px.bar(
                    zero_share,
                    x=article_col,
                    y="PCT_ZERO",
                    title="Top articles by zero-sale days (%)",
                    labels={article_col: "Article", "PCT_ZERO": "% zero days"}
                )
                fig_zero.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                apply_plot_theme(fig_zero)
                st.plotly_chart(fig_zero, use_container_width=True)

    st.subheader("Data preview")
    st.dataframe(to_display_df(df_focus.head(20)), use_container_width=True, hide_index=True)

elif page == "Model":
    if df_article is None or len(df_article) == 0:
        st.error(f"No data found for article {selected_article_id}")
        st.stop()

    if forecast_button:
        st.divider()
        st.info(f"Article ID: {selected_article_id} | Data: {len(df_article)} rows")

        model, fe, info = build_and_train_xgboost(
            df_article,
            target_col=target_col,
            test_size=test_size,
            min_samples=min_samples,
            horizon=horizon,
            tune=True
        )

        if model is None:
            st.warning(f"Insufficient data for XGBoost: {info.get('error', 'Unknown reason')}")
            st.info("Using baseline model (Exponential Smoothing / Moving Average)")

            baseline_model, baseline_info = build_baseline_fallback(df_article, target_col=target_col)

            if baseline_model is None:
                st.error("Insufficient data even for baseline model!")
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
                'Tip': 'Baseline forecast'
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
                'Tip': 'XGBoost forecast'
            })

            st.session_state.model_result = {
                "selection_key": selection_key,
                "model_type": "xgboost",
                "metrics": metrics,
                "n_features": info['n_features'],
                "feature_cols": fe.get_feature_cols(),
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
        st.info('Click "Generate Forecast" to run the model.')
    else:
        result = st.session_state.model_result
        if result["model_type"] == "baseline":
            st.success(f"{result['horizon']}-day forecast generated (baseline model)")

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

            st.subheader("Baseline Forecast")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.dataframe(to_display_df(result["df_plot"].round(2)), use_container_width=True, hide_index=True)

            with col2:
                csv = result["df_plot"].to_csv(index=False).encode()
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"baseline_forecast_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.success(f"{result['horizon']}-day forecast generated")

            tab1, tab2, tab3 = st.tabs(["Chart", "Metrics", "Table"])

            with tab1:
                st.subheader("Actual vs Predicted vs Forecast")

                df_comparison = result["df_test"].copy()
                df_comparison['PREDICTIE'] = result["y_pred"]

                fig = plot_forecast_comparison(
                    result["df_article"][['DATA', result["target_col"]]],
                    result["df_plot"],
                    df_comparison[['DATA', result["target_col"], 'PREDICTIE']],
                    target_col=result["target_col"]
                )
                apply_plot_theme(fig)
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
                st.subheader("Detailed Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f} units")
                    st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f} units")

                with col2:
                    st.metric("R2 Score", f"{metrics['r2']:.4f}")
                    st.metric("Test Samples", f"{metrics['n_samples']}")

                st.divider()
                st.subheader("Top Features")
                try:
                        importance_df = result["model"].get_feature_importance(top_n=10)
                        fig_importance = plot_monthly_aggregation(importance_df)
                        apply_plot_theme(fig_importance)
                        st.plotly_chart(fig_importance, use_container_width=True)
                except Exception:
                    st.info("Feature importance cannot be computed for this model")

                with st.expander("Full feature list"):
                    st.write(result.get("feature_cols", []))

            with tab3:
                st.subheader("Forecast Table")

                col1, col2 = st.columns([2, 1])

                with col1:
                    df_display = result["df_plot"].copy()
                    df_display['PREDICTIE'] = df_display['PREDICTIE'].round(2)
                    st.dataframe(to_display_df(df_display), use_container_width=True, hide_index=True)

                with col2:
                    csv = result["df_plot"].to_csv(index=False).encode()
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name=f"xgboost_forecast_{selected_article_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

else:
    if not saved_is_valid:
        st.info('Run the model on the "Model" page to see risk analysis.')
    else:
        result = st.session_state.model_result
        st.subheader("Stockout Risk Analysis")

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
                "Zero-Sale Days",
                f"{zero_pct:.1f}%",
                'warning' if zero_pct > 20 else 'metric'
            )

        with col3:
            create_kpi_card(
                "Coeff. of Variation",
                f"{metrics_stockout['cv']:.2f}",
                'metric'
            )

        st.divider()

        avg_demand = result["df_article"][result["target_col"]].mean()
        std_demand = result["df_article"][result["target_col"]].std()
        forecast_mean = result["future_pred"].mean()
        forecast_std = result["future_pred"].std()

        st.markdown("#### Recommendations")
        col1, col2 = st.columns(2)

        with col1:
            st.write(
                """
                **Historical Metrics:**
                - Average demand: {:.1f} units/day
                - Standard deviation: {:.1f}
                - Coeff. of variation: {:.2f}
                """.format(avg_demand, std_demand, metrics_stockout['cv'])
            )

        with col2:
            st.write(
                """
                **{}-day Forecast:**
                - Average demand: {:.1f} units/day
                - Standard deviation: {:.1f}
                - Min/Max: {:.0f} / {:.0f}
                """.format(result["horizon"], forecast_mean, forecast_std,
                           result["future_pred"].min(), result["future_pred"].max())
            )

        service_level = 0.95
        z_score = 1.645
        safety_stock = z_score * std_demand

        st.success(
            """
            **Safety Stock (SL 95%):** {stock:.0f} units

                Reorder when stock drops below {stock:.0f} units
                to avoid stockout with 95% probability.
                """.format(stock=safety_stock)
            )

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
**Inventory Optimization Dashboard | XGBoost Forecasting**  
Thesis - Predictive Methods for Inventory Optimization
""")


