import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.title("⚙️ MLOps Pipeline: Upload Data & Export Pickle Artifacts")

uploaded_file = st.file_uploader("Upload Any CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Uploaded Data Preview", df.head())

    # 1. Select Target Column
    target_col = st.selectbox("Select Target Column to Predict:", df.columns)

    if st.button("⚡ Train & Generate Pickle Files"):
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # Drop ID / High Cardinality Columns
        cols_to_drop = [c for c in X.columns if X[c].dtype == 'object' and X[c].nunique() > 0.8 * len(X)]
        X = X.drop(columns=cols_to_drop)

        # 2. Fit Preprocessors & Save Encoders
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Train Model
        is_classification = y.nunique() < 20 or y.dtype == 'object'
        if is_classification:
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y.astype(str))
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_scaled, y)

        # --------------------------------------------------
        # 4. EXPORT ALL ARTIFACTS TO PICKLE (.pkl)
        # --------------------------------------------------
        
        # A. Save Trained Model
        model_bytes = io.BytesIO()
        pickle.dump(model, model_bytes)
        model_bytes.seek(0)

        # B. Save Preprocessor (Scaler + Encoders)
        preprocessor_dict = {
            "scaler": scaler,
            "encoders": label_encoders,
            "feature_names": list(X.columns)
        }
        preprocessor_bytes = io.BytesIO()
        pickle.dump(preprocessor_dict, preprocessor_bytes)
        preprocessor_bytes.seek(0)

        st.success("✅ Training Complete! Pickle files successfully generated.")
        st.write("### 📥 Download Generated MLOps Artifacts")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📦 Download model.pkl",
                data=model_bytes,
                file_name="model.pkl",
                mime="application/octet-stream"
            )
        with col2:
            st.download_button(
                label="⚙️ Download preprocessor.pkl",
                data=preprocessor_bytes,
                file_name="preprocessor.pkl",
                mime="application/octet-stream"
            )
