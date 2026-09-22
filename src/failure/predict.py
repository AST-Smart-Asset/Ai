"""Inference helpers for the 90-day failure classifier, with a deterministic fallback."""

import json

import joblib
import pandas as pd

from src.config import (
    FAILURE_FEATURES,
    FAILURE_FEATURES_PATH,
    FAILURE_MODEL_PATH,
    FAILURE_RISK_BANDS,
    FAILURE_THRESHOLD,
)


def load_model(model_path=FAILURE_MODEL_PATH, features_path=FAILURE_FEATURES_PATH):
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        features = json.load(f)
    return model, features


def probability_to_risk(probability: float) -> str:
    for threshold, label in FAILURE_RISK_BANDS:
        if probability >= threshold:
            return label
    return FAILURE_RISK_BANDS[-1][1]


def _rule_based_reasons(asset_features: dict) -> list:
    """
    Human-readable reasons layered on top of the ML probability.

    These are simple threshold rules (not SHAP/feature-contribution values)
    so the explanation stays legible for non-technical users even when the
    ML model is unavailable and the deterministic fallback kicks in.
    """
    reasons = []
    if asset_features["previous_total_cost"] > 1000:
        reasons.append("High historical maintenance cost")
    if asset_features["previous_downtime_minutes"] > 300:
        reasons.append("High historical downtime")
    if asset_features["asset_age_days"] > 365:
        reasons.append("Older asset")
    if asset_features["work_orders_last_90d"] > 0:
        reasons.append("Recent maintenance activity")
    if asset_features["days_to_maintenance"] <= 90:
        reasons.append("Maintenance due within 90 days")
    if not reasons:
        reasons.append("No strong historical risk indicator")
    return reasons


def predict_failure(model, features, asset_features: dict, threshold: float = FAILURE_THRESHOLD) -> dict:
    """Predict 90-day failure probability for a single asset using the ML model."""
    missing = [f for f in features if f not in asset_features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    input_df = pd.DataFrame([{f: asset_features[f] for f in features}])
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "model_version": "failure-classifier-90d-v1",
        "failure_probability": round(probability, 4),
        "predicted_failure_90d": prediction,
        "risk": probability_to_risk(probability),
        "reasons": _rule_based_reasons(asset_features),
        "fallback_used": False,
    }


def predict_failure_with_fallback(model, features, asset_features: dict, threshold: float = FAILURE_THRESHOLD) -> dict:
    """
    Same as `predict_failure`, but falls back to a deterministic
    "maintenance due within 90 days" rule if the ML model raises an error
    (e.g. missing/incompatible features). Used by the API layer.
    """
    try:
        return predict_failure(model, features, asset_features, threshold)
    except Exception:
        days = asset_features.get("days_to_maintenance", 9999)
        fallback_prediction = int(0 <= days <= 90)
        return {
            "model_version": "deterministic-due-date-fallback-v1",
            "failure_probability": None,
            "predicted_failure_90d": fallback_prediction,
            "risk": "Medium" if fallback_prediction else "Low",
            "reasons": ["ML model unavailable; using deterministic maintenance-due fallback"],
            "fallback_used": True,
        }
