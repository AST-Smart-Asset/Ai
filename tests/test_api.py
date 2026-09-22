import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data

def test_single_asset_prediction_api():
    payload = {
        "asset_id": "AST-TEST-001",
        "asset_type": "HVAC",
        "age_days": 365,
        "days_since_last_pm": 40,
        "cumulative_downtime_hours": 2.5,
        "historical_work_order_count": 2,
        "vibration_rms": 1.6,
        "operating_temperature_c": 54.0,
        "power_consumption_kwh": 18.0,
        "ambient_humidity": 50.0,
    }
    response = client.post("/predict/maintenance-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["asset_id"] == "AST-TEST-001"
    assert "failure_probability" in data
    assert "risk_band" in data
    assert "recommended_action" in data
    assert "Advisory prediction only" in data["advisory_notice"]
    assert "human technician" in data["advisory_notice"].lower()

def test_batch_prediction_api():
    payload = {
        "assets": [
            {
                "asset_id": "AST-BATCH-001",
                "asset_type": "Generator",
                "age_days": 100,
                "days_since_last_pm": 15,
                "vibration_rms": 1.1,
                "operating_temperature_c": 45.0,
            },
            {
                "asset_id": "AST-BATCH-002",
                "asset_type": "HVAC",
                "age_days": 1500,
                "days_since_last_pm": 180,
                "vibration_rms": 7.8,
                "operating_temperature_c": 92.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 2
    assert len(data["predictions"]) == 2

def test_model_metrics_endpoint():
    response = client.get("/model/metrics")
    assert response.status_code == 200
