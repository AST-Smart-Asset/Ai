import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import SERVICE_NAME, MODEL_VERSION, METRICS_FILE
from src.schemas import (
    AssetTelemetryInput,
    RiskPredictionOutput,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from src.predictor import MaintenanceRiskPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ast-ai.api")

predictor: MaintenanceRiskPredictor = None

def get_predictor() -> MaintenanceRiskPredictor:
    global predictor
    if predictor is None:
        predictor = MaintenanceRiskPredictor()
    return predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Initializing Predictive Maintenance AI service...")
    predictor = MaintenanceRiskPredictor()
    logger.info(f"Service ready. Model loaded: {predictor.is_model_loaded}")
    yield

app = FastAPI(
    title="AST Predictive Maintenance AI Service",
    description=(
        "Microservice for Project 3: Smart Asset Inventory & Predictive Maintenance (AST). "
        "Provides probabilistic failure risk scoring, ISO-10816 mechanical degradation analysis, "
        "and human-in-the-loop advisory maintenance recommendations."
    ),
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def root():
    p = get_predictor()
    return {
        "service": SERVICE_NAME,
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "model_loaded": p.is_model_loaded,
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    p = get_predictor()
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=MODEL_VERSION,
        model_loaded=p.is_model_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@app.post(
    "/predict/maintenance-risk",
    response_model=RiskPredictionOutput,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Predict failure risk for a single equipment asset",
)
def predict_maintenance_risk(asset: AssetTelemetryInput):
    p = get_predictor()
    try:
        result = p.predict(asset)
        return result
    except Exception as e:
        logger.error(f"Inference error for {asset.asset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(e)}",
        )

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Batch evaluate failure risks across multiple assets",
)
def predict_batch(request: BatchPredictionRequest):
    p = get_predictor()
    predictions = p.predict_batch(request.assets)

    critical = sum(1 for pred in predictions if pred.risk_band == "critical")
    high = sum(1 for pred in predictions if pred.risk_band == "high")
    medium = sum(1 for pred in predictions if pred.risk_band == "medium")
    low = sum(1 for pred in predictions if pred.risk_band == "low")

    return BatchPredictionResponse(
        total_evaluated=len(predictions),
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        low_count=low,
        predictions=predictions,
    )

@app.get("/model/metrics", tags=["Model Governance"])
def get_model_metrics():
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {
        "model_version": MODEL_VERSION,
        "status": "No serialized metrics file found. Run training script to generate.",
        "baseline_mode": "Deterministic rule fallback active",
    }
