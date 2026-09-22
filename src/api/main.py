"""
FastAPI service exposing:
    GET  /health                -- readiness check
    POST /predict-rul           -- Remaining Useful Life + risk band
    POST /predict-failure-90d   -- 90-day failure probability + risk band

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Both models are loaded once at startup from `src/config.py`'s MODELS_DIR
(models/ by default). Train them first with the scripts in scripts/.
"""

from fastapi import FastAPI

from src.api.schemas import FailureInput, SensorInput
from src.failure.predict import load_model as load_failure_model
from src.failure.predict import predict_failure_with_fallback
from src.rul.predict import load_model as load_rul_model
from src.rul.predict import predict_asset

app = FastAPI(title="AST Smart Asset - Predictive Maintenance AI API", version="1.0.0")

rul_model, rul_features = load_rul_model()
failure_model, failure_features = load_failure_model()


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "AST Smart Asset - Predictive Maintenance AI",
        "models": {"rul": "loaded", "failure_90d": "loaded"},
    }


@app.post("/predict-rul")
def predict_rul(data: SensorInput):
    return predict_asset(rul_model, rul_features, data.model_dump())


@app.post("/predict-failure-90d")
def predict_failure_90d(data: FailureInput):
    return predict_failure_with_fallback(failure_model, failure_features, data.model_dump())
