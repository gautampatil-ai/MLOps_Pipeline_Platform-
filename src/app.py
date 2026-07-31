import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Healthcare Prediction Portal", page_icon="🏥", layout="wide")

MODEL_PATH = "healthcare_model.joblib"

# ---------------------------------------------------------
# Load or Train Model on Healthcare Data
# ---------------------------------------------------------
@st.cache_resource
def load_healthcare_model():
    # If model exists, load it
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_healthcare_model()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🏥 Healthcare Batch Analytics & Prediction Portal")
st.write("Upload your healthcare dataset to make predictions.")

uploaded_file = st.file_uploader("Upload Healthcare CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Uploaded Data Preview")
    st.dataframe(df.head())

    # Filter out columns that are just names/IDs (not useful for math predictions)
    ignore_cols = ['Name', 'Doctor', 'Hospital', 'Date of Admission', 'Room Number']
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    st.write(f"**Selected Features for Model:** {', '.join(feature_cols)}")

    if st.button("⚡ Process Data"):
        st.success("Dataset successfully loaded and ready for modeling!")
