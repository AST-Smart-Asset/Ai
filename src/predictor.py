import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import joblib
import numpy as np

from .config import (
    MODEL_FILE,
    CRITICAL_THRESHOLD,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    VIBRATION_NORMAL_MAX,
    VIBRATION_WARNING_MAX,
    VIBRATION_CRITICAL_MIN,
    TEMPERATURE_NORMAL_MAX,
    TEMPERATURE_WARNING_MAX,
    TEMPERATURE_CRITICAL_MIN,
    PM_CYCLE_DAYS_WARNING,
    PM_CYCLE_DAYS_CRITICAL,
    MODEL_VERSION,
)
from .schemas import (
    AssetTelemetryInput,
    RiskPredictionOutput,
    ContributingFactor,
)
from .feature_engineering import extract_features, to_dataframe, FEATURE_COLUMNS

logger = logging.getLogger("ast-ai.predictor")

class MaintenanceRiskPredictor:
    def __init__(self, model_path=MODEL_FILE):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded ML model from {self.model_path}")
            else:
                logger.warning(f"Model file {self.model_path} not found. Running in deterministic fallback mode.")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}. Fallback enabled.")
            self.model = None

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    def _fallback_predict(self, asset: AssetTelemetryInput) -> float:
        """Deterministic physics-grounded engineering baseline rule evaluation."""
        vib_weight = 0.35
        temp_weight = 0.25
        pm_weight = 0.20
        downtime_weight = 0.12
        age_weight = 0.08

        # Normalized sensor distress metrics [0.0, 1.0]
        vib_score = min(1.0, max(0.0, (asset.vibration_rms - 1.0) / (VIBRATION_CRITICAL_MIN - 1.0)))
        temp_score = min(1.0, max(0.0, (asset.operating_temperature_c - 40.0) / (TEMPERATURE_CRITICAL_MIN - 40.0)))
        pm_score = min(1.0, max(0.0, float(asset.days_since_last_pm) / float(PM_CYCLE_DAYS_CRITICAL)))
        dt_score = min(1.0, max(0.0, float(asset.cumulative_downtime_hours) / 120.0))
        age_score = min(1.0, max(0.0, float(asset.age_days) / 2000.0))

        # Composite distress index
        distress = (
            vib_score * vib_weight
            + temp_score * temp_weight
            + pm_score * pm_weight
            + dt_score * downtime_weight
            + age_score * age_weight
        )

        # Critical triggers (ISO-10816 override)
        if asset.vibration_rms >= VIBRATION_CRITICAL_MIN or asset.operating_temperature_c >= TEMPERATURE_CRITICAL_MIN:
            distress = max(distress, 0.85)

        return float(np.clip(distress, 0.01, 0.99))

    def _evaluate_contributing_factors(self, asset: AssetTelemetryInput, prob: float) -> List[ContributingFactor]:
        factors = []

        if asset.vibration_rms >= VIBRATION_CRITICAL_MIN:
            factors.append(ContributingFactor(
                feature="vibration_rms",
                impact_weight=0.45,
                description=f"Critical vibration: {asset.vibration_rms:.2f} mm/s (exceeds ISO-10816 limit {VIBRATION_CRITICAL_MIN} mm/s - severe mechanical misalignment or bearing fault)"
            ))
        elif asset.vibration_rms >= VIBRATION_WARNING_MAX:
            factors.append(ContributingFactor(
                feature="vibration_rms",
                impact_weight=0.25,
                description=f"Elevated vibration: {asset.vibration_rms:.2f} mm/s (nearing ISO-10816 warning threshold)"
            ))

        if asset.operating_temperature_c >= TEMPERATURE_CRITICAL_MIN:
            factors.append(ContributingFactor(
                feature="operating_temperature_c",
                impact_weight=0.35,
                description=f"Critical temperature: {asset.operating_temperature_c:.1f}°C (exceeds safe limit {TEMPERATURE_CRITICAL_MIN}°C - thermal dissipation failure)"
            ))
        elif asset.operating_temperature_c >= TEMPERATURE_WARNING_MAX:
            factors.append(ContributingFactor(
                feature="operating_temperature_c",
                impact_weight=0.20,
                description=f"High temperature: {asset.operating_temperature_c:.1f}°C (exceeds nominal operating range)"
            ))

        if asset.days_since_last_pm >= PM_CYCLE_DAYS_CRITICAL:
            factors.append(ContributingFactor(
                feature="days_since_last_pm",
                impact_weight=0.25,
                description=f"Preventive maintenance overdue: {asset.days_since_last_pm} days since last inspection (cycle is {PM_CYCLE_DAYS_WARNING} days)"
            ))
        elif asset.days_since_last_pm >= PM_CYCLE_DAYS_WARNING:
            factors.append(ContributingFactor(
                feature="days_since_last_pm",
                impact_weight=0.15,
                description=f"Maintenance cycle approaching due date: {asset.days_since_last_pm} days elapsed"
            ))

        if asset.cumulative_downtime_hours >= 50.0:
            factors.append(ContributingFactor(
                feature="cumulative_downtime_hours",
                impact_weight=0.15,
                description=f"Repeated unreliability: {asset.cumulative_downtime_hours:.1f} hours historical unplanned downtime"
            ))

        if not factors:
            factors.append(ContributingFactor(
                feature="nominal_baseline",
                impact_weight=0.05,
                description="All telemetry signals within nominal manufacturer tolerances."
            ))

        return factors

    def _determine_action_and_window(self, risk_band: str, asset: AssetTelemetryInput) -> tuple[str, int]:
        if risk_band == "critical":
            window = 5
            action = "URGENT: Dispatch senior technician for immediate physical vibration/thermal inspection within 24-48 hours. Consider derating asset load."
        elif risk_band == "high":
            window = 14
            action = "HIGH PRIORITY: Schedule preventive servicing, bearing lubrication, and electrical check within 7 days."
        elif risk_band == "medium":
            window = 30
            action = "MONITOR: Increase telemetry sampling frequency and verify upcoming scheduled routine PM."
        else:
            window = 90
            action = "NORMAL: Equipment healthy. Maintain standard manufacturer inspection schedule."

        return action, window

    def predict(self, asset: AssetTelemetryInput) -> RiskPredictionOutput:
        """Predict maintenance risk for a single asset with model and deterministic fallback."""
        prob: float = 0.0

        if self.model is not None:
            try:
                df = to_dataframe([asset])
                # Check if model has predict_proba
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(df)
                    prob = float(probs[0][1])
                else:
                    prob = float(self.model.predict(df)[0])
            except Exception as e:
                logger.warning(f"Model prediction failed for asset {asset.asset_id}: {e}. Falling back to rules.")
                prob = self._fallback_predict(asset)
        else:
            prob = self._fallback_predict(asset)

        # Risk band determination
        if prob >= CRITICAL_THRESHOLD:
            risk_band = "critical"
        elif prob >= HIGH_THRESHOLD:
            risk_band = "high"
        elif prob >= MEDIUM_THRESHOLD:
            risk_band = "medium"
        else:
            risk_band = "low"

        action, window = self._determine_action_and_window(risk_band, asset)
        factors = self._evaluate_contributing_factors(asset, prob)

        return RiskPredictionOutput(
            asset_id=asset.asset_id,
            failure_probability=round(prob, 4),
            risk_band=risk_band,
            predicted_failure_window_days=window,
            top_contributing_factors=factors,
            recommended_action=action,
            advisory_notice="Advisory prediction only. Automated work order issuance or asset decommissioning without human technician review is strictly prohibited.",
            model_version=MODEL_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def predict_batch(self, assets: List[AssetTelemetryInput]) -> List[RiskPredictionOutput]:
        return [self.predict(a) for a in assets]
