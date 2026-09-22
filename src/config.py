"""
Central configuration for the AST Smart Asset AI engine.

Keeping feature lists, risk thresholds and default paths in one place
avoids the copy/paste drift that existed in the original notebook
(the `columns` / `features` lists were redefined 5+ times).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# RUL model (NASA C-MAPSS FD001)
# ---------------------------------------------------------------------------

CMAPSS_COLUMNS = [
    "engine_id",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
] + [f"sensor{i}" for i in range(1, 22)]

RUL_FEATURES = [f"sensor{i}" for i in range(1, 22)]

RUL_MODEL_PATH = MODELS_DIR / "rul_random_forest.pkl"
RUL_FEATURES_PATH = MODELS_DIR / "features.json"

# ---------------------------------------------------------------------------
# 90-day failure classifier (work orders / assets tables)
# ---------------------------------------------------------------------------

FAILURE_FEATURES = [
    "asset_age_days",
    "days_to_maintenance",
    "previous_work_orders",
    "previous_corrective_jobs",
    "previous_total_cost",
    "previous_downtime_minutes",
    "work_orders_last_90d",
    "corrective_jobs_last_90d",
]

CORRECTIVE_NOTES_PATTERN = r"corrective job|fault report"

FAILURE_MODEL_PATH = MODELS_DIR / "failure_classifier_90d.pkl"
FAILURE_FEATURES_PATH = MODELS_DIR / "failure_classifier_features.json"
FAILURE_METADATA_PATH = MODELS_DIR / "failure_classifier_90d_metadata.json"

FAILURE_HORIZON_DAYS = 90
FAILURE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Risk bands (shared "reason" wording used by both models / the API)
# ---------------------------------------------------------------------------

RUL_RISK_BANDS = (
    (20, "High", "Very low estimated remaining useful life"),
    (50, "Medium", "Moderate estimated remaining useful life"),
    (float("inf"), "Low", "Sufficient estimated remaining useful life"),
)

FAILURE_RISK_BANDS = (
    (0.7, "High"),
    (0.3, "Medium"),
    (0.0, "Low"),
)
