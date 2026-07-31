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
    page_title="MLOps Pipeline Platform",
    page_icon="🤖",
    layout="wide"
)

MODEL_PATH = "model.joblib"

# ---------------------------------------------------------
# Model Loading & Automated Training Function
# ---------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    """Loads the model artifact if exists; otherwise trains and saves a new one."""
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Model artifact not found. Triggering automated training...")
        
        # Load sample Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Create pipeline matching the saved artifact specs
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X, y)
        
        # Save artifact so warning won't trigger on future loads
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline, iris.target_names
    else:
        pipeline = joblib.load(MODEL_PATH)
        iris = load_iris()
        return pipeline, iris.target_names

pipeline, target_names = load_or_train_model()

# ---------------------------------------------------------
# Header & UI Title
# ---------------------------------------------------------
st.title("🤖 MLOps Pipeline Platform")
st.write("Interactive machine learning inference and pipeline monitoring platform.")

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction", "📊 Model Info"])

# =========================================================
# TAB 1: Single Prediction
# =========================================================
with tab1:
    st.subheader("Single Feature Inference")
    
    # 2-column layout for input features matching the screenshot
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.50,
            step=0.01,
            format="%.2f"
        )
        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=4.90,
            step=0.01,
            format="%.2f"
        )
        
    with col2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.80,
            step=0.01,
            format="%.2f"
        )
        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.20,
            step=0.01,
            format="%.2f"
        )

    # Custom colored button styling (Red primary style)
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
        }
        div.stButton > button:first-child:hover {
            background-color: #E03E3E;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("Predict Class"):
        # Format input array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Inference
        prediction_idx = pipeline.predict(input_data)[0]
        prediction_prob = pipeline.predict_proba(input_data)[0]
        predicted_class = target_names[prediction_idx]
        
        # Output Result
        st.success(f"**Predicted Class:** `{predicted_class.upper()}`")
        st.info(f"**Confidence:** {prediction_prob[prediction_idx]*100:.2f}%")

# =========================================================
# TAB 2: Batch Prediction
# =========================================================
with tab2:
    st.subheader("Batch Data Inference")
    uploaded_file = st.file_uploader("Upload a CSV file with feature columns", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("Run Batch Inference"):
            predictions = pipeline.predict(df)
            df['Prediction'] = [target_names[i] for i in predictions]
            st.write("### Prediction Results")
            st.dataframe(df)

# =========================================================
# TAB 3: Model Info
# =========================================================
with tab3:
    st.subheader("Model Artifact Details")
    st.json({
        "Model Type": "RandomForestClassifier",
        "Preprocessing": "StandardScaler",
        "Scikit-Learn Version": "1.6.1",
        "Artifact Path": MODEL_PATH,
        "Status": "Active / Loaded"
    })
