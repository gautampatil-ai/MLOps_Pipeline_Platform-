import os
import pandas as pd
import joblib
import streamlit as st
from train import run_pipeline

# Configuration & Paths
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "iris_pipeline.joblib"
)
CLASSES = ["setosa", "versicolor", "virginica"]

st.set_page_config(
    page_title="MLOps Pipeline Platform",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load model artifact into memory with caching."""
    if not os.path.exists(MODEL_PATH):
        st.info("⚠️ Model artifact not found. Triggering automated training...")
        return run_pipeline()
    return joblib.load(MODEL_PATH)


# Initialize session state for model loading
model = load_model()

# Header Section
st.title("🤖 MLOps Pipeline Platform")
st.markdown("Interactive machine learning inference and pipeline monitoring platform.")

# Sidebar Controls
st.sidebar.header("⚙️ MLOps Controls")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Retrain Model"):
    with st.spinner("Training model pipeline..."):
        model = run_pipeline()
        st.cache_resource.clear()
        st.sidebar.success("Model retrained successfully!")

# Tabbed Interface
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📂 Batch Prediction", "📊 Model Info"])

# Tab 1: Single Prediction
with tab1:
    st.subheader("Single Feature Inference")
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

    with col2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    if st.button("Predict Class", type="primary"):
        input_data = pd.DataFrame([{
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }])

        pred_idx = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        st.success(f"**Predicted Species:** {CLASSES[pred_idx].capitalize()} (Class {pred_idx})")

        # Class Probability breakdown
        prob_df = pd.DataFrame({
            "Species": [c.capitalize() for c in CLASSES],
            "Probability": probabilities
        })
        st.bar_chart(prob_df.set_index("Species"))

# Tab 2: Batch Prediction
with tab2:
    st.subheader("Batch CSV Prediction")
    st.markdown("Upload a CSV file containing columns: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`.")

    uploaded_file = st.file_uploader("Upload Features CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview Input Data:", batch_df.head())

            if st.button("Run Batch Inference"):
                predictions = model.predict(batch_df)
                batch_df["Prediction_Class"] = predictions
                batch_df["Species_Name"] = [CLASSES[int(p)] for p in predictions]

                st.success("Batch Prediction Complete!")
                st.dataframe(batch_df)

                # Download Results
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

# Tab 3: Model Pipeline Architecture
with tab3:
    st.subheader("Pipeline Details")
    st.json({
        "Model Type": "RandomForestClassifier",
        "Preprocessing": "StandardScaler",
        "Target Classes": CLASSES,
        "Status": "Loaded in memory"
    })
