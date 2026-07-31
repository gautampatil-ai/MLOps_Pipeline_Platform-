import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("⚙️ Lightweight MLOps Pipeline (< 25 MB Pickle)")

uploaded_file = st.file_uploader("Upload Any CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column to Predict:", df.columns)

    if st.button("⚡ Train & Package Small Pickle Files"):
        with st.spinner("Training model with memory optimization..."):
            X = df.drop(columns=[target_col]).copy()
            y = df[target_col].copy()

            # Clean high-cardinality text columns
            cols_to_drop = [c for c in X.columns if X[c].dtype == 'object' and X[c].nunique() > 0.8 * len(X)]
            X = X.drop(columns=cols_to_drop)

            # Fit Preprocessors
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # --- KEY FIX FOR SIZE REDUCTION (<25 MB) ---
            # Max trees = 50, Max depth = 10, Min samples leaf = 5
            is_classification = y.nunique() < 20 or y.dtype == 'object'
            
            if is_classification:
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y.astype(str))
                model = RandomForestClassifier(
                    n_estimators=50,       # Reduced from 100 to 50
                    max_depth=10,          # Prunes tree depth (drastically cuts size)
                    min_samples_leaf=5,    # Prevents huge overfitted leaf nodes
                    random_state=42
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42
                )

            model.fit(X_scaled, y)

            # Convert to Pickle Bytes with Compression
            model_buffer = io.BytesIO()
            pickle.dump(model, model_buffer, protocol=pickle.HIGHEST_PROTOCOL)
            model_bytes = model_buffer.getvalue()

            preprocessor_dict = {
                "scaler": scaler,
                "encoders": label_encoders,
                "features": list(X.columns)
            }
            prep_buffer = io.BytesIO()
            pickle.dump(preprocessor_dict, prep_buffer, protocol=pickle.HIGHEST_PROTOCOL)
            prep_bytes = prep_buffer.getvalue()

            # Calculate Size in MB
            model_size_mb = len(model_bytes) / (1024 * 1024)
            prep_size_mb = len(prep_bytes) / (1024 * 1024)

            # Store in Session State
            st.session_state['model_pkl'] = model_bytes
            st.session_state['prep_pkl'] = prep_bytes
            st.session_state['model_size'] = model_size_mb
            st.session_state['prep_size'] = prep_size_mb
            st.session_state['is_trained'] = True

    # Render Buttons and Size Warnings
    if st.session_state.get('is_trained', False):
        m_size = st.session_state['model_size']
        
        if m_size < 25:
            st.success(f"✅ Success! Generated `model.pkl` size is **{m_size:.2f} MB** (Under 25 MB target).")
        else:
            st.warning(f"⚠️ Warning: Model size is **{m_size:.2f} MB**. Try setting `max_depth=8` or `n_estimators=30` in code.")

        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label=f"📥 Download model.pkl ({st.session_state['model_size']:.2f} MB)",
                data=st.session_state['model_pkl'],
                file_name="model.pkl",
                mime="application/octet-stream",
                key="dl_model"
            )
            
        with col2:
            st.download_button(
                label=f"⚙️ Download preprocessor.pkl ({st.session_state['prep_size']:.2f} MB)",
                data=st.session_state['prep_pkl'],
                file_name="preprocessor.pkl",
                mime="application/octet-stream",
                key="dl_prep"
            )
