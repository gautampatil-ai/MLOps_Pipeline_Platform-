import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MLOps Platform | Inference Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "model.joblib"

# ---------------------------------------------------------
# Modern Custom CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main App Background & Typography */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-top: 0.4rem;
    }

    /* Metric Result Box */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #38BDF8;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.15);
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #38BDF8;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    /* Primary Predict Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #DC2626 0%, #B91C1C 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #1E293B;
        border-radius: 8px 8px 0px 0px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loading & Automated Training
# ---------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        st.info("ℹ️ Artifact absent. Initializing automated model training pipeline...")
        iris = load_iris()
        X, y = iris.data, iris.target
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline, iris.target_names, iris.feature_names
    else:
        pipeline = joblib.load(MODEL_PATH)
        iris = load_iris()
        return pipeline, iris.target_names, iris.feature_names

pipeline, target_names, feature_names = load_or_train_model()

# ---------------------------------------------------------
# Sidebar - Platform Status
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/server.png", width=60)
    st.title("System Status")
    
    st.success("🟢 Model Engine: Operational")
    st.write(f"**Model Type:** `RandomForest`")
    st.write(f"**Scaler:** `StandardScaler`")
    st.write(f"**Framework:** `scikit-learn`")
    st.divider()
    
    st.markdown("### 🎛️ Input Preset")
    if st.button("Set Default Iris Sample"):
        st.session_state["sl"] = 5.50
        st.session_state["sw"] = 4.90
        st.session_state["pl"] = 0.80
        st.session_state["pw"] = 0.20

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown("""
    <div class="header-card">
        <h1 class="header-title">⚡ MLOps Inference & Analytics Portal</h1>
        <p class="header-subtitle">Interactive Real-time Model Prediction, Batch Processing & Pipeline Diagnostics</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# UI Tabs Setup
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Processing", "📊 Pipeline Info"])

# =========================================================
# TAB 1: Single Prediction
# =========================================================
with tab1:
    st.subheader("Interactive Feature Inputs")
    st.caption("Adjust sliders or values to trigger custom model predictions.")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0, max_value=10.0,
            value=st.session_state.get("sl", 5.50),
            step=0.1, format="%.2f"
        )
        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0, max_value=10.0,
            value=st.session_state.get("sw", 4.90),
            step=0.1, format="%.2f"
        )
        
    with col2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.0, max_value=10.0,
            value=st.session_state.get("pl", 0.80),
            step=0.1, format="%.2f"
        )
        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.0, max_value=10.0,
            value=st.session_state.get("pw", 0.20),
            step=0.1, format="%.2f"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Class")
    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        prediction_idx = pipeline.predict(input_data)[0]
        prediction_probs = pipeline.predict_proba(input_data)[0]
        predicted_class = target_names[prediction_idx].capitalize()
        confidence = prediction_probs[prediction_idx] * 100

        # Display High-impact Metric Cards
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-value">{predicted_class}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Confidence</div>
                    <div class="metric-value">{confidence:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

        # Visual Probability Breakdown
        st.write("### 📊 Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Class": [c.capitalize() for c in target_names],
            "Probability": prediction_probs
        })
        st.bar_chart(prob_df, x="Class", y="Probability", color="#38BDF8")

# =========================================================
# TAB 2: Batch Prediction
# =========================================================
with tab2:
    st.subheader("Batch Dataset Inference")
    st.write("Upload a CSV file containing feature values to run model predictions in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("#### Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("⚡ Run Batch Predictions"):
            try:
                preds = pipeline.predict(df)
                df["Predicted_Class"] = [target_names[i].capitalize() for i in preds]
                
                st.success("Batch predictions complete!")
                st.dataframe(df, use_container_width=True)
                
                # Download Button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Result CSV",
                    data=csv,
                    file_name="inference_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error processing dataset: {e}")

# =========================================================
# TAB 3: Pipeline Info
# =========================================================
with tab3:
    st.subheader("ML Architecture Diagnostics")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Pipeline Version", "v1.6.1")
    m_col2.metric("Preprocessing", "StandardScaler")
    m_col3.metric("Estimator", "RandomForest (100 trees)")
    
    st.divider()
    st.json({
        "pipeline_steps": [
            {"step": 1, "name": "StandardScaler", "fitted": True},
            {"step": 2, "name": "RandomForestClassifier", "n_estimators": 100, "criterion": "gini"}
        ],
        "feature_names": list(feature_names),
        "target_classes": list(target_names)
    })
