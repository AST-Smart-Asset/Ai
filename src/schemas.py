from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class AssetTelemetryInput(BaseModel):
    asset_id: str = Field(..., description="Unique asset identifier, e.g. AST-HVAC-001")
    asset_type: str = Field(default="HVAC", description="Equipment classification / category")
    age_days: int = Field(..., ge=0, description="Total asset age in operating days")
    days_since_last_pm: int = Field(..., ge=0, description="Days elapsed since last completed maintenance")
    cumulative_downtime_hours: float = Field(default=0.0, ge=0.0, description="Total historical unplanned downtime hours")
    historical_work_order_count: int = Field(default=0, ge=0, description="Total previous work orders logged")
    vibration_rms: float = Field(..., ge=0.0, description="Vibration velocity RMS in mm/s (ISO 10816 baseline)")
    operating_temperature_c: float = Field(..., description="Operating component temperature in Celsius")
    power_consumption_kwh: float = Field(default=15.0, ge=0.0, description="Current power draw in kWh")
    ambient_humidity: float = Field(default=45.0, ge=0.0, le=100.0, description="Surrounding relative humidity percentage")

class ContributingFactor(BaseModel):
    feature: str
    impact_weight: float
    description: str

class RiskPredictionOutput(BaseModel):
    model_config = {"protected_namespaces": ()}
    asset_id: str
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Calculated failure probability in the next 30 days")
    risk_band: Literal["low", "medium", "high", "critical"] = Field(..., description="Categorical risk tier")
    predicted_failure_window_days: int = Field(..., description="Estimated days until severe failure / breakdown")
    top_contributing_factors: List[ContributingFactor]
    recommended_action: str
    advisory_notice: str = "Advisory prediction only. Human technician verification required prior to work order issuance."
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    assets: List[AssetTelemetryInput]

class BatchPredictionResponse(BaseModel):
    total_evaluated: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    predictions: List[RiskPredictionOutput]

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    service: str
    version: str
    model_loaded: bool
    timestamp: str
