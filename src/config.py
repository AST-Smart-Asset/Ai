import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_FILE = MODELS_DIR / "trained_model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

# Risk Banding Thresholds
CRITICAL_THRESHOLD = 0.75
HIGH_THRESHOLD = 0.50
MEDIUM_THRESHOLD = 0.25

# Engineering Sensor Guardrails (ISO-10816 mechanical standard)
VIBRATION_NORMAL_MAX = 2.8     # mm/s RMS
VIBRATION_WARNING_MAX = 4.5    # mm/s RMS
VIBRATION_CRITICAL_MIN = 7.1   # mm/s RMS

TEMPERATURE_NORMAL_MAX = 65.0  # °C
TEMPERATURE_WARNING_MAX = 80.0 # °C
TEMPERATURE_CRITICAL_MIN = 88.0# °C

PM_CYCLE_DAYS_WARNING = 90
PM_CYCLE_DAYS_CRITICAL = 150

MODEL_VERSION = "v1.2.0-rf-ensemble"
SERVICE_NAME = "AST Predictive Maintenance AI"
