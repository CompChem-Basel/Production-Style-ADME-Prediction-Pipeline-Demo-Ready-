"""
Production ADME Prediction Pipeline
Author: Basel Mansour

"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import mlflow.sklearn
from scipy.stats import ks_2samp


# Logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Neuro.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


# Config

MODEL_DIR = "models"
DRIFT_THRESHOLD = 0.1
RSME_THRESHOLD = 1.0  # quality gate for CI/CD

os.makedirs(MODEL_DIR, exist_ok=True)


# Load dataset

def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset {file_path} is empty.")

    logging.info(f"Loaded dataset: {file_path} with shape {df.shape}")
    return df



# Train models

def train_models(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ["SMILES", "Absorption", "Distribution", "Metabolism", "Elimination"]]
    X = df[feature_cols]

    y_dict = {
        "Absorption": df["Absorption"],
        "Distribution": df["Distribution"],
        "Metabolism": df["Metabolism"],
        "Elimination": df["Elimination"],
    }

    models = {}
    for endpoint, y in y_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with mlflow.start_run(run_name=f"{endpoint}_model"):
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            logging.info(f"{endpoint} RMSE = {rmse:.3f}")

            # Save locally
            model_path = os.path.join(MODEL_DIR, f"{endpoint}_model.pkl")
            joblib.dump(model, model_path)

            # Log to MLflow
            mlflow.log_param("n_estimators", 200)
			mlflow.log_metric("RMSE", rmse)
            mlflow.sklearn.log_model(model, artifact_path=f"{endpoint}_model")

            if rmse > RSME_THRESHOLD:
                logging.warning(f"{endpoint} failed CI/CD quality gate (RMSE {rmse:.2f} > {RSME_THRESHOLD})")

            models[endpoint] = model

    return models, feature_cols



# Predict ADME

def predict_adme(input_file: str, feature_cols: list) -> pd.DataFrame:
    df = load_dataset(input_file)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X_new = df[feature_cols]
    results = df.copy()

    for endpoint in ["Absorption", "Distribution", "Metabolism", "Elimination"]:
        model_file = os.path.join(MODEL_DIR, f"{endpoint}_model.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found for {endpoint}. Train models first.")
        model = joblib.load(model_file)
        results[endpoint] = model.predict(X_new)

    return results

# Drift detection

def detect_drift(train_df: pd.DataFrame, new_df: pd.DataFrame, feature_cols: list, threshold=DRIFT_THRESHOLD):
    drift_report = {}
    for col in feature_cols:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        stat, p_value = ks_2samp(train_df[col], new_df[col])
        drift_report[col] = {"statistic": stat, "p_value": p_value, "drift": p_value < threshold}

    drifted = [col for col, rep in drift_report.items() if rep["drift"]]
    if drifted:
        logging.warning(f"⚠ Drift detected in: {drifted}")
    else:
        logging.info("✅ No significant drift detected")

    return drift_report

# Main

if __name__ == "__main__":
    train_file = os.getenv("TRAIN_FILE", "descriptor_dataset_train.csv")
    test_file = os.getenv("TEST_FILE", "descriptor_dataset_test.csv")

    # Train
    df_train = load_dataset(train_file)
    models, feature_cols = train_models(df_train)

    # Predict
    df_pred = predict_adme(test_file, feature_cols)
    df_pred.to_csv("adme_predictions.csv", index=False)
    logging.info("Predictions saved to adme_predictions.csv")

    # Drift check
    df_test = load_dataset(test_file)
    detect_drift(df_train, df_test, feature_cols)
