import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Universal MLOps Platform", page_icon="⚙️", layout="wide")

st.title("⚙️ Universal MLOps Prediction & Training Platform")
st.write("Upload **any** CSV dataset below. The platform will automatically preprocess features and run machine learning models.")

# --- Step 1: File Upload ---
uploaded_file = st.file_uploader("Upload Any CSV File", type=["csv"], key="mlops_upload")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### 📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # --- Step 2: Automated Preprocessing ---
    st.write("### ⚙️ Automated Preprocessing Setup")
    
    # 1. Ask user to choose target column to predict
    all_columns = list(df.columns)
    target_col = st.selectbox("Select Target Column to Predict:", all_columns, index=len(all_columns)-1)
    
    if target_col:
        # Separate features and target
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        # Auto-drop High-Cardinality text (Names, IDs, Timestamps)
        cols_to_drop = []
        for col in X.columns:
            # If a column has almost unique values per row (like Name or ID), drop it
            if X[col].dtype == 'object' and X[col].nunique() > 0.8 * len(X):
                cols_to_drop.append(col)
                
        if cols_to_drop:
            st.warning(f"⚠️ Auto-detected and dropped non-predictive ID/Name columns: `{cols_to_drop}`")
            X = X.drop(columns=cols_to_drop)
            
        # Automatically encode categorical text columns (e.g. Gender, Blood Type)
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            
        st.success(f"✅ Data processed successfully! **{X.shape[1]} features** remaining for training.")
        
        # --- Step 3: Train & Predict On-The-Fly ---
        if st.button("🚀 Train & Run MLOps Pipeline"):
            with st.spinner("Training model on your uploaded dataset..."):
                # Determine task type (Classification vs Regression)
                is_classification = y.nunique() < 20 or y.dtype == 'object'
                
                if is_classification:
                    if y.dtype == 'object':
                        y = LabelEncoder().fit_transform(y.astype(str))
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Split and fit
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                # Attach predictions to original preview
                predictions = model.predict(X)
                df['Model_Predictions'] = predictions
                
                st.balloons()
                st.success(f"🎉 Pipeline Executed! Model Accuracy/R² Score: **{score:.2%}**")
                
                st.write("### 📊 Dataset with Generated Predictions")
                st.dataframe(df, use_container_width=True)
