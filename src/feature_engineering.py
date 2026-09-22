import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .schemas import AssetTelemetryInput

FEATURE_COLUMNS = [
    "age_days",
    "days_since_last_pm",
    "cumulative_downtime_hours",
    "historical_work_order_count",
    "vibration_rms",
    "operating_temperature_c",
    "power_consumption_kwh",
    "ambient_humidity",
    "thermal_vibration_index",
    "pm_overdue_ratio",
    "downtime_intensity",
]

ASSET_TYPE_MAP = {
    "HVAC": 0,
    "Generator": 1,
    "Medical": 2,
    "Lab Equipment": 3,
    "IT Server": 4,
    "Elevator": 5,
    "Water Pump": 6,
    "Other": 7,
}

def extract_features(asset: AssetTelemetryInput) -> Dict[str, float]:
    """Extract raw and derived engineering features from telemetry input."""
    tv_index = (asset.operating_temperature_c / 60.0) * (asset.vibration_rms / 2.5)
    pm_ratio = float(asset.days_since_last_pm) / 90.0
    dt_intensity = float(asset.cumulative_downtime_hours) / max(1.0, float(asset.age_days) / 30.0)

    return {
        "age_days": float(asset.age_days),
        "days_since_last_pm": float(asset.days_since_last_pm),
        "cumulative_downtime_hours": float(asset.cumulative_downtime_hours),
        "historical_work_order_count": float(asset.historical_work_order_count),
        "vibration_rms": float(asset.vibration_rms),
        "operating_temperature_c": float(asset.operating_temperature_c),
        "power_consumption_kwh": float(asset.power_consumption_kwh),
        "ambient_humidity": float(asset.ambient_humidity),
        "thermal_vibration_index": float(tv_index),
        "pm_overdue_ratio": float(pm_ratio),
        "downtime_intensity": float(dt_intensity),
    }

def to_dataframe(assets: List[AssetTelemetryInput]) -> pd.DataFrame:
    """Convert list of assets to model feature DataFrame."""
    records = [extract_features(a) for a in assets]
    return pd.DataFrame(records)[FEATURE_COLUMNS]
