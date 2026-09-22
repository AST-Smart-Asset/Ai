import pytest
from src.predictor import MaintenanceRiskPredictor
from src.schemas import AssetTelemetryInput

@pytest.fixture
def predictor():
    # Instantiate predictor (fallback or trained)
    return MaintenanceRiskPredictor()

def test_healthy_asset_prediction(predictor):
    asset = AssetTelemetryInput(
        asset_id="AST-HVAC-NORMAL",
        asset_type="HVAC",
        age_days=180,
        days_since_last_pm=20,
        cumulative_downtime_hours=0.0,
        historical_work_order_count=1,
        vibration_rms=1.2,
        operating_temperature_c=48.0,
        power_consumption_kwh=14.0,
        ambient_humidity=45.0,
    )
    result = predictor.predict(asset)
    assert result.asset_id == "AST-HVAC-NORMAL"
    assert result.failure_probability < 0.40
    assert result.risk_band in ["low", "medium"]
    assert result.predicted_failure_window_days >= 30
    assert "Advisory prediction only" in result.advisory_notice

def test_critical_asset_prediction(predictor):
    asset = AssetTelemetryInput(
        asset_id="AST-GEN-CRITICAL",
        asset_type="Generator",
        age_days=1800,
        days_since_last_pm=210,
        cumulative_downtime_hours=85.0,
        historical_work_order_count=9,
        vibration_rms=8.4,  # Far above ISO critical threshold 7.1
        operating_temperature_c=94.0, # Dangerous thermal overheat
        power_consumption_kwh=48.0,
        ambient_humidity=65.0,
    )
    result = predictor.predict(asset)
    assert result.asset_id == "AST-GEN-CRITICAL"
    assert result.failure_probability >= 0.70
    assert result.risk_band in ["high", "critical"]
    assert result.predicted_failure_window_days <= 14
    assert len(result.top_contributing_factors) >= 1

def test_deterministic_fallback_when_model_is_none():
    predictor = MaintenanceRiskPredictor()
    predictor.model = None  # Force fallback
    assert predictor.is_model_loaded is False

    asset = AssetTelemetryInput(
        asset_id="AST-FALLBACK-TEST",
        asset_type="Water Pump",
        age_days=400,
        days_since_last_pm=45,
        vibration_rms=1.5,
        operating_temperature_c=50.0,
    )
    result = predictor.predict(asset)
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.risk_band in ["low", "medium", "high", "critical"]
