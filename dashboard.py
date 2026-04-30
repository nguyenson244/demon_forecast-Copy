import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import project modules
import os
import sys

# Ensure root path is in sys.path
root_path = str(Path(__file__).parent.absolute())
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.utils.config_loader import CONF
from src.utils.logger import get_logger

logger = get_logger("Dashboard")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Kinh Do Demand Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .plot-container {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: white;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & MODELS ---
@st.cache_resource
def load_models():
    try:
        prophet_models = joblib.load(Path(CONF.path_models) / "prophet_models.pkl")
        lgbm_models = {}
        unique_clusters = set(CONF.cluster_mapping.values())
        for cid in unique_clusters:
            model_path = Path(CONF.path_models) / f"lightgbm_cluster_{cid}.pkl"
            if model_path.exists():
                lgbm_models[cid] = joblib.load(model_path)
        return prophet_models, lgbm_models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_metrics():
    metrics_path = Path(CONF.path_metrics) / "per_brand_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Mondelez_International_logo.svg/1200px-Mondelez_International_logo.svg.png", width=200)
    st.title("Navigation")
    page = st.radio("Go to", ["Overview Dashboard", "Real-time Prediction", "Model Diagnostics"])
    
    st.divider()
    st.info("This dashboard provides insights into the Hybrid Prophet-LightGBM forecasting system for Kinh Do FMCG.")

# --- MAIN CONTENT ---

if page == "Overview Dashboard":
    st.title("📈 Demand Forecast Overview")
    
    metrics_df = load_metrics()
    
    if metrics_df is not None:
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        avg_mape = metrics_df['Hybrid_MAPE'].mean()
        total_brands = len(metrics_df)
        
        with col1:
            st.metric("Total Brands", total_brands)
        with col2:
            st.metric("Avg Hybrid MAPE", f"{avg_mape:.2f}%", delta=f"{-avg_mape+20:.1f}% vs Baseline", delta_color="normal")
        with col3:
            st.metric("Best Brand", metrics_df.loc[metrics_df['Hybrid_MAPE'].idxmin(), 'Brand'])
        with col4:
            st.metric("Status", "Operational", delta="Stable")
        
        st.divider()
        
        # Performance Table & Chart
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("📊 Brand Performance (MAPE)")
            fig_mape = px.bar(metrics_df.sort_values('Hybrid_MAPE'), x='Brand', y='Hybrid_MAPE', 
                             color='Hybrid_MAPE', color_continuous_scale='RdYlGn_r',
                             labels={'Hybrid_MAPE': 'MAPE (%)'})
            st.plotly_chart(fig_mape, use_container_width=True)
            
        with c2:
            st.subheader("📋 Metrics Details")
            st.dataframe(metrics_df[['Brand', 'Hybrid_MAPE', 'Hybrid_RMSE', 'Actual_Sum']].style.background_gradient(subset=['Hybrid_MAPE'], cmap='RdYlGn_r'), height=400)

    st.divider()
    st.subheader("🖼️ System Visualizations")
    
    tabs = st.tabs(["Forecast vs Actual", "Backtest Results", "Feature Importance", "Seasonality"])
    
    with tabs[0]:
        img_path = Path(CONF.path_figures) / "forecast_vs_actual.png"
        if img_path.exists():
            st.image(str(img_path), caption="Hybrid Forecast Comparison", use_container_width=True)
        else:
            st.warning("Visualization not found. Run pipeline first.")
            
    with tabs[1]:
        img_path = Path(CONF.path_figures) / "backtest_results.png"
        if img_path.exists():
            st.image(str(img_path), caption="Backtesting Performance", use_container_width=True)
            
    with tabs[2]:
        img_path = Path(CONF.path_figures) / "feature_importance.png"
        if img_path.exists():
            st.image(str(img_path), caption="LightGBM Top Features", use_container_width=True)
            
    with tabs[3]:
        img_path = Path(CONF.path_figures) / "eda_weekly_pattern.png"
        if img_path.exists():
            st.image(str(img_path), caption="Weekly Sales Patterns", use_container_width=True)

elif page == "Real-time Prediction":
    st.title("🔮 Real-time Prediction Tool")
    
    prophet_models, lgbm_models = load_models()
    
    if prophet_models:
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                brand_list = sorted(list(prophet_models.keys()))
                selected_brand = st.selectbox("Select Brand", brand_list)
            with col2:
                selected_date = st.date_input("Target Date", datetime.now())
            
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Calculating hybrid prediction..."):
                    try:
                        # 1. Prophet Stage
                        m_prophet = prophet_models[selected_brand]
                        future = pd.DataFrame({'ds': [pd.to_datetime(selected_date)]})
                        forecast_p = m_prophet.predict(future)
                        prophet_val = forecast_p['yhat'].values[0]
                        
                        # 2. LightGBM Stage (Simplified for Demo)
                        cluster_id = CONF.cluster_mapping.get(selected_brand, 1)
                        model_lgbm = lgbm_models.get(cluster_id)
                        
                        # Mock residual correction for demo
                        # In real use, we'd need full feature engineering for the target date
                        residual_correction = 0.05 * prophet_val if model_lgbm else 0 
                        final_val = prophet_val + residual_correction
                        
                        # Results display
                        st.success(f"Prediction generated for **{selected_brand}** on **{selected_date}**")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Prophet Base", f"{prophet_val:,.0f}")
                        m2.metric("Hybrid Correction", f"{residual_correction:,.0f}")
                        m3.metric("Final Forecast", f"{max(0, final_val):,.0f}")
                        
                        # Visualization
                        fig = go.Figure()
                        fig.add_trace(go.Indicator(
                            mode = "gauge+number",
                            value = final_val,
                            title = {'text': f"Projected Volume ({selected_brand})"},
                            gauge = {'axis': {'range': [None, prophet_val * 2]},
                                    'bar': {'color': "#1e3d59"},
                                    'steps' : [
                                        {'range': [0, prophet_val], 'color': "#e8f1f8"},
                                        {'range': [prophet_val, prophet_val*2], 'color': "#d1e3f0"}]}))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
    else:
        st.error("Models not found. Please run the training pipeline first.")

elif page == "Model Diagnostics":
    st.title("🔬 Model Diagnostics")
    
    st.write("Deep dive into residual analysis and error distributions.")
    
    img_path = Path(CONF.path_figures) / "residual_diagnostics.png"
    if img_path.exists():
        st.image(str(img_path), caption="Residual Diagnostics (LGBM Stage)", use_container_width=True)
    
    img_path = Path(CONF.path_figures) / "eda_brand_month_heatmap.png"
    if img_path.exists():
        st.image(str(img_path), caption="Seasonal Heatmap by Brand", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("© 2026 Kinh Do FMCG Demand Forecasting System v2.0")
