import json
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent


def load_json(filename: str):
    with open(BASE_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Load trained models
# -----------------------------
rul_model = joblib.load(BASE_DIR / "rul_random_forest.pkl")
rul_features = load_json("features.json")

failure_model = joblib.load(BASE_DIR / "failure_classifier_90d.pkl")
failure_features = load_json("failure_classifier_features.json")


app = FastAPI(
    title="Predictive Maintenance AI API",
    version="1.0.0",
)


# -----------------------------
# Request schemas
# -----------------------------
class SensorInput(BaseModel):
    sensor1: float
    sensor2: float
    sensor3: float
    sensor4: float
    sensor5: float
    sensor6: float
    sensor7: float
    sensor8: float
    sensor9: float
    sensor10: float
    sensor11: float
    sensor12: float
    sensor13: float
    sensor14: float
    sensor15: float
    sensor16: float
    sensor17: float
    sensor18: float
    sensor19: float
    sensor20: float
    sensor21: float


class FailureInput(BaseModel):
    asset_age_days: float = Field(..., ge=0)
    days_to_maintenance: float
    previous_work_orders: float = Field(..., ge=0)
    previous_corrective_jobs: float = Field(..., ge=0)
    previous_total_cost: float = Field(..., ge=0)
    previous_downtime_minutes: float = Field(..., ge=0)
    work_orders_last_90d: float = Field(..., ge=0)
    corrective_jobs_last_90d: float = Field(..., ge=0)


# -----------------------------
# Helpers
# -----------------------------
def make_rul_row(sensor_data: dict):
    return np.array(
        [[float(sensor_data[name]) for name in rul_features]],
        dtype=float,
    )


def make_failure_row(data: dict):
    return np.array(
        [[float(data[name]) for name in failure_features]],
        dtype=float,
    )


def failure_fallback(data: dict):
    """Deterministic fallback based only on maintenance due-date proximity."""
    days = float(data["days_to_maintenance"])

    if 0 <= days <= 30:
        risk = "High"
        predicted = 1
        reason = "Maintenance is due within 30 days"
    elif 31 <= days <= 90:
        risk = "Medium"
        predicted = 1
        reason = "Maintenance is due within 90 days"
    else:
        risk = "Low"
        predicted = 0
        reason = "No near-term maintenance due date"

    return {
        "model_version": "deterministic-due-date-fallback-v1",
        "failure_probability": None,
        "predicted_failure_90d": predicted,
        "risk": risk,
        "reasons": [reason],
        "fallback_used": True,
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Predictive Maintenance AI",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "Predictive Maintenance AI",
        "models": {
            "rul": "loaded",
            "failure_90d": "loaded",
        },
    }


@app.post("/predict-rul")
def predict_rul(data: SensorInput):
    sensor_data = data.model_dump()
    row = make_rul_row(sensor_data)

    predicted_rul = float(rul_model.predict(row)[0])

    # Experimental C-MAPSS prototype bands.
    if predicted_rul <= 20:
        risk = "High"
        reason = "Very low estimated remaining useful life"
    elif predicted_rul <= 50:
        risk = "Medium"
        reason = "Moderately low estimated remaining useful life"
    else:
        risk = "Low"
        reason = "Estimated remaining useful life is above the experimental medium-risk band"

    return {
        "model_version": "rul-random-forest-v1",
        "predicted_rul": round(predicted_rul, 2),
        "risk": risk,
        "reason": reason,
    }


@app.post("/predict-failure-90d")
def predict_failure_90d(data: FailureInput):
    values = data.model_dump()

    try:
        row = make_failure_row(values)
        probability = float(failure_model.predict_proba(row)[0][1])
        predicted = int(probability >= 0.5)

        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.30:
            risk = "Medium"
        else:
            risk = "Low"

        reasons = []

        # Human-readable reasons are deliberately conservative:
        # they describe observed feature values, not causal claims.
        if values["days_to_maintenance"] <= 30:
            reasons.append("Maintenance is due within 30 days")
        if values["previous_corrective_jobs"] > 0:
            reasons.append("Previous corrective maintenance has been recorded")
        if values["corrective_jobs_last_90d"] > 0:
            reasons.append("A corrective job was recorded in the last 90 days")
        if values["work_orders_last_90d"] >= 3:
            reasons.append("Several work orders were recorded in the last 90 days")
        if values["previous_downtime_minutes"] > 0:
            reasons.append("Historical downtime has been recorded")
        if values["previous_total_cost"] > 0:
            reasons.append("Historical maintenance cost has been recorded")

        if not reasons:
            reasons.append("No strong historical risk indicator")

        return {
            "model_version": "failure-classifier-90d-v1",
            "failure_probability": round(probability, 4),
            "predicted_failure_90d": predicted,
            "risk": risk,
            "reasons": reasons,
            "fallback_used": False,
        }

    except Exception:
        # Required deterministic fallback.
        return failure_fallback(values)


@app.get("/model-info")
def model_info():
    return {
        "rul_model": {
            "name": "Random Forest Regressor",
            "version": "rul-random-forest-v1",
            "purpose": "Experimental remaining useful life prediction",
            "note": "C-MAPSS prototype; RUL bands are experimental cycles, not days.",
        },
        "failure_model": {
            "name": "Logistic Regression",
            "version": "failure-classifier-90d-v1",
            "purpose": "Experimental probability of failure within 90 days",
            "note": "Uses synthetic project data and proxy failure labels.",
        },
        "fallback": "Deterministic due-date rule for the 90-day failure endpoint.",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
