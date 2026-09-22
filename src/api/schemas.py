"""Pydantic request models for the FastAPI service."""

from pydantic import BaseModel


class SensorInput(BaseModel):
    """One reading per C-MAPSS sensor, used by /predict-rul."""

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
    """Asset/maintenance-history features used by /predict-failure-90d."""

    asset_age_days: float
    days_to_maintenance: float
    previous_work_orders: float
    previous_corrective_jobs: float
    previous_total_cost: float
    previous_downtime_minutes: float
    work_orders_last_90d: float
    corrective_jobs_last_90d: float
