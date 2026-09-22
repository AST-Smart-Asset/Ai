"""Inference helpers for the RUL model."""

import json

import joblib
import pandas as pd

from src.config import RUL_FEATURES_PATH, RUL_MODEL_PATH, RUL_RISK_BANDS


def load_model(model_path=RUL_MODEL_PATH, features_path=RUL_FEATURES_PATH):
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        features = json.load(f)
    return model, features


def rul_to_risk(rul: float) -> tuple:
    """Map a predicted RUL value to a (risk_label, reason) pair."""
    for threshold, label, reason in RUL_RISK_BANDS:
        if rul <= threshold:
            return label, reason
    return RUL_RISK_BANDS[-1][1], RUL_RISK_BANDS[-1][2]


def predict_asset(model, features, sensor_data: dict) -> dict:
    """
    Predict RUL and maintenance risk for a single asset.

    `sensor_data` must contain every key listed in `features`
    (sensor1 .. sensor21).
    """
    data = pd.DataFrame([sensor_data])

    missing = [f for f in features if f not in data.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    predicted_rul = float(model.predict(data[features])[0])
    risk, reason = rul_to_risk(predicted_rul)

    return {
        "predicted_rul": round(predicted_rul, 2),
        "risk": risk,
        "reason": reason,
    }
