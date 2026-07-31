import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "iris_pipeline.joblib")


def run_pipeline():
    print("🚀 Starting MLOps Training Pipeline...")

    # 1. Load Data
    print("📦 Loading dataset...")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Define ML Pipeline (Preprocessing + Estimator)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Train Model
    print("🏋️ Training model...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Training completed. Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model pipeline artifact saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    run_pipeline()
