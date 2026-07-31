import os
import joblib
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Define paths
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "iris_pipeline.joblib"
)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Pipeline Platform API",
    description="Production API serving Machine Learning Predictions",
    version="1.0.0"
)

# Global model container
model_pipeline = None


@app.on_event("startup")
def load_model():
    """Load model artifact into memory during app startup."""
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully on startup.")
        print(f"Loaded from path: {MODEL_PATH}")
    else:
        print("⚠️ Warning: Model artifact not found! Run train.py first.")


# Input Schema Definition
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Sepal length in cm")
    sepal_width: float = Field(..., example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal length in cm")
    petal_width: float = Field(..., example=0.2, description="Petal width in cm")


class BatchPredictionInput(BaseModel):
    inputs: List[IrisFeatures]


# Response Schema
class PredictionOutput(BaseModel):
    prediction_class: int
    class_name: str


CLASSES = ["setosa", "versicolor", "virginica"]


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    """Root endpoint."""
    return {
        "service": "MLOps Pipeline Platform API",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Render/Kubernetes monitoring."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded"
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict(features: IrisFeatures):
    """Predict flower class for a single feature vector."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available for inference."
        )

    # Convert request schema to pandas DataFrame matching feature names
    data = pd.DataFrame([{
        "sepal length (cm)": features.sepal_length,
        "sepal width (cm)": features.sepal_width,
        "petal length (cm)": features.petal_length,
        "petal width (cm)": features.petal_width,
    }])

    prediction = int(model_pipeline.predict(data)[0])
    return PredictionOutput(
        prediction_class=prediction,
        class_name=CLASSES[prediction]
    )


@app.post("/predict-batch", response_model=List[PredictionOutput])
def predict_batch(batch: BatchPredictionInput):
    """Batch prediction endpoint for multi-sample requests."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available for inference."
        )

    df_data = [
        {
            "sepal length (cm)": item.sepal_length,
            "sepal width (cm)": item.sepal_width,
            "petal length (cm)": item.petal_length,
            "petal width (cm)": item.petal_width,
        }
        for item in batch.inputs
    ]
    data = pd.DataFrame(df_data)
    predictions = model_pipeline.predict(data)

    return [
        PredictionOutput(
            prediction_class=int(pred),
            class_name=CLASSES[int(pred)]
        )
        for pred in predictions
    ]
