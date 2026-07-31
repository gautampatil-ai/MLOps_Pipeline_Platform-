import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("⚙️ MLOps Pipeline Artifact Generator")

uploaded_file = st.file_uploader("Upload Any CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column to Predict:", df.columns)

    # 1. Train and Store in Session State
    if st.button("⚡ Train & Prepare Downloads"):
        with st.spinner("Training model and packaging pickle files..."):
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

            # Fit Model
            is_classification = y.nunique() < 20 or y.dtype == 'object'
            if is_classification:
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y.astype(str))
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_scaled, y)

            # --- CONVERT TO PICKLE BYTES ---
            # Model Pickle
            model_buffer = io.BytesIO()
            pickle.dump(model, model_buffer)
            model_buffer.seek(0) # IMPORTANT: reset buffer pointer to start

            # Preprocessor Pickle
            preprocessor_dict = {
                "scaler": scaler,
                "encoders": label_encoders,
                "features": list(X.columns)
            }
            prep_buffer = io.BytesIO()
            pickle.dump(preprocessor_dict, prep_buffer)
            prep_buffer.seek(0) # IMPORTANT: reset buffer pointer to start

            # Save binary data in Session State so it persists
            st.session_state['model_pkl'] = model_buffer.getvalue()
            st.session_state['prep_pkl'] = prep_buffer.getvalue()
            st.session_state['is_trained'] = True

    # 2. Render Working Download Buttons
    if st.session_state.get('is_trained', False):
        st.success("✅ Training Complete! Click below to download files directly to your computer.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download model.pkl",
                data=st.session_state['model_pkl'],
                file_name="model.pkl",
                mime="application/octet-stream",
                key="dl_model"
            )
            
        with col2:
            st.download_button(
                label="⚙️ Download preprocessor.pkl",
                data=st.session_state['prep_pkl'],
                file_name="preprocessor.pkl",
                mime="application/octet-stream",
                key="dl_prep"
            )
