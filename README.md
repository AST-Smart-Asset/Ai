# AST Predictive Maintenance AI Microservice

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![Project 3](https://img.shields.io/badge/BUA-DevHub%20Field%20Phase-navy)](https://github.com/AST-Smart-Asset)

Production-grade AI predictive maintenance microservice for **Project 3: Smart Asset Inventory & Predictive Maintenance (AST)** at Badr University in Assiut (BUA DevHub Field Phase, Squad 3).

---

## 🎯 Purpose & Scope

This service implements **AST-FR-09** (Predictive Maintenance Risk Evaluation) and satisfies all **Common Pack AI Release Gate** documentation and testing criteria:
1. **Probabilistic Failure Prediction:** Calculates continuous 30-day failure risk probability $[0.0, 1.0]$.
2. **ISO-10816 Mechanical Standard Grounding:** Classifies vibration RMS and operational temperatures into standard industrial severity bands (`low`, `medium`, `high`, `critical`).
3. **Lead Time Forecasting:** Estimates remaining failure window days (e.g. 5 days for critical vs. 90+ days for healthy assets).
4. **Deterministic Heuristic Fallback:** Guarantees 100% service uptime even if model weights are unavailable or corrupted.
5. **Human-in-the-Loop Protocol:** Embeds advisory notices on all predictions, prohibiting automated unreviewed work orders or asset disposals.

---

## 📂 Project Architecture

```
ast-ai/
├── data/
│   ├── generate_dataset.py       # Mechanical physics synthetic dataset generator
│   └── asset_telemetry.csv       # 12,000+ asset operating records
├── models/
│   ├── trained_model.pkl         # Serialized Random Forest model artifact
│   └── metrics.json              # Evaluated holdout performance metrics
├── src/
│   ├── __init__.py
│   ├── config.py                 # Thresholds, ISO limits, and file paths
│   ├── feature_engineering.py    # Raw-to-feature transform pipeline
│   ├── predictor.py              # ML inference + deterministic fallback engine
│   ├── schemas.py                # Pydantic v2 schemas for request & response
│   └── train.py                  # Model training and validation script
├── tests/
│   ├── __init__.py
│   ├── test_api.py               # FastAPI endpoint integration tests
│   └── test_predictor.py         # Unit tests on deterministic fallback & safety limits
├── app.py                        # FastAPI application entry point
├── Dockerfile                    # Containerization manifest
├── MODEL_CARD.md                 # Formal Common Pack AI release documentation
├── requirements.txt              # Pinned Python package dependencies
├── train.py                      # Root training runner
└── README.md
```

---

## ⚡ Quick Start

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/AST-Smart-Asset/Ai.git
cd Ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train and Evaluate the Model

```bash
python train.py
```
This script generates the synthetic telemetry dataset, trains the Random Forest classifier, validates against holdout data (ROC-AUC $\ge 0.85$), and serializes the model to `models/trained_model.pkl`.

### 3. Run the Test Suite

```bash
pytest tests/ -v
```

### 4. Start the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger Documentation is accessible at `http://localhost:8000/docs`.

---

## 🔌 API Endpoints Reference

### 1. Single Asset Prediction
- **`POST /predict/maintenance-risk`**
- **Request Body:**
  ```json
  {
    "asset_id": "AST-HVAC-042",
    "asset_type": "HVAC",
    "age_days": 730,
    "days_since_last_pm": 135,
    "cumulative_downtime_hours": 18.5,
    "historical_work_order_count": 3,
    "vibration_rms": 5.4,
    "operating_temperature_c": 82.0,
    "power_consumption_kwh": 26.5,
    "ambient_humidity": 52.0
  }
  ```
- **Response:**
  ```json
  {
    "asset_id": "AST-HVAC-042",
    "failure_probability": 0.8142,
    "risk_band": "critical",
    "predicted_failure_window_days": 5,
    "top_contributing_factors": [
      {
        "feature": "vibration_rms",
        "impact_weight": 0.25,
        "description": "Elevated vibration: 5.40 mm/s (nearing ISO-10816 warning threshold)"
      },
      {
        "feature": "operating_temperature_c",
        "impact_weight": 0.20,
        "description": "High temperature: 82.0°C (exceeds nominal operating range)"
      }
    ],
    "recommended_action": "URGENT: Dispatch senior technician for immediate physical vibration/thermal inspection within 24-48 hours.",
    "advisory_notice": "Advisory prediction only. Automated work order issuance or asset decommissioning without human technician review is strictly prohibited.",
    "model_version": "v1.2.0-rf-ensemble",
    "timestamp": "2026-09-22T06:40:00Z"
  }
  ```

### 2. Batch Evaluation
- **`POST /predict/batch`**
- Evaluates an array of assets and returns aggregate risk distributions (`critical_count`, `high_count`, `medium_count`, `low_count`).

### 3. Health & Liveness Probe
- **`GET /health`**
- Returns `{ "status": "healthy", "model_loaded": true, "version": "v1.2.0-rf-ensemble" }`.

---

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t ast-ai-service .

# Run the container
docker run -d -p 8000:8000 --name ast-ai ast-ai-service

# Test the healthcheck
curl http://localhost:8000/health
```

---

## 🔗 Integration with Backend and Flutter

- **Backend API Gateway:** The Node.js Backend routes requests through `POST /api/v1/predictions/evaluate/:assetId` by querying asset telemetry from the PostgreSQL database and forwarding the feature vector to this service at `http://localhost:8000/predict/maintenance-risk`.
- **Flutter Mobile App:** Technicians tap **"Evaluate AI Failure Risk"** on the asset detail screen, receiving real-time risk gauges, failure countdown windows, and recommended maintenance interventions.