# AST Predictive Maintenance AI - Model Card

[![Model Version](https://img.shields.io/badge/version-v1.2.0--rf--ensemble-teal)](https://github.com/AST-Smart-Asset/Ai)
[![BUA Project](https://img.shields.io/badge/BUA-DevHub%20Field%20Phase-navy)](https://github.com/AST-Smart-Asset)
[![Standard](https://img.shields.io/badge/ISO-10816%20Compliant-green)](https://www.iso.org/standard/38744.html)

This Model Card conforms to the **Common Pack AI Release Gate Specification** for **Project 3: Smart Asset Inventory & Predictive Maintenance (AST)** at Badr University in Assiut (BUA DevHub Field Phase, Squad 3).

---

## 1. Model Details

- **Model Name:** AST Mechanical & Facility Predictive Maintenance Classifier (`ast-predictive-risk-v1.2`)
- **Model Version:** `v1.2.0-rf-ensemble`
- **Developed By:** Squad 3 AI Team (Badr University in Assiut DevHub Field Phase)
- **Model Type:** Supervised Machine Learning Ensemble (Random Forest Classifier with Balanced Class Weighting) coupled with an ISO-10816 Mechanical Standard Deterministic Fallback Engine.
- **License:** MIT / Academic Project Use
- **Release Date:** September 2026

---

## 2. Intended Use & Scope

### Primary Intended Uses:
- Estimating the probability that a physical mechanical or electrical asset will experience an operational failure within a 30-day forecast window.
- Assigning operational risk bands (`low`, `medium`, `high`, `critical`) to guide maintenance inspection prioritization.
- Providing human technicians with explainable top contributing distress factors (e.g. vibration velocity, temperature spikes, overdue preventive maintenance days).
- Recommending targeted preventive maintenance actions and inspection windows.

### Out-of-Scope & Prohibited Uses:
- **Strictly Prohibited:** Automatically decommissioning equipment, discarding inventory records, or triggering unauthorized procurement without human verification.
- **Strictly Prohibited:** Autonomous execution of high-voltage switching or safety-critical overrides without physical on-site inspection.
- **Not Intended For:** Non-mechanical software assets, purely static furnishings (desks/chairs), or consumables.

---

## 3. Training & Validation Data

- **Dataset Size:** 12,000 synthetic operational equipment records synthesized from mechanical physics principles (degradation wear curves, bearing friction thermal dynamics, Weibull wear-out distributions).
- **Equipment Categories Covered:**
  1. HVAC Systems (Chillers, AHUs, Compressors)
  2. Standby Emergency Power Generators
  3. Medical Devices (Sterilizers, Refrigeration, Pumps)
  4. Specialized Laboratory Equipment (Centrifuges, Incubators)
  5. IT Infrastructure Servers & Data Center Racks
  6. Elevators & Heavy Conveyance
  7. High-Capacity Campus Water Pumps
- **Train/Test Split:** 80% Training (9,600 samples), 20% Holdout Test (2,400 samples) with stratified class balance.

### Input Features:
| Feature | Type | Unit | Engineering Baseline / Safe Threshold |
| :--- | :--- | :--- | :--- |
| `vibration_rms` | Float | mm/s RMS | ISO-10816: Normal < 2.8, Warning 2.8–4.5, Critical > 7.1 |
| `operating_temperature_c` | Float | °C | Normal < 65°C, Warning 65–80°C, Critical > 88°C |
| `days_since_last_pm` | Integer | Days | Standard cycle: 90 days; Overdue: > 120 days |
| `cumulative_downtime_hours` | Float | Hours | Historical breakdown impact |
| `age_days` | Integer | Days | Equipment operating age |
| `power_consumption_kwh` | Float | kWh | Electrical load draw |
| `ambient_humidity` | Float | % | Environmental humidity |
| `thermal_vibration_index` | Float | Ratio | Cross-factor thermal-mechanical stress indicator |
| `pm_overdue_ratio` | Float | Ratio | `days_since_last_pm / 90` |
| `downtime_intensity` | Float | Ratio | `cumulative_downtime / (age_days / 30)` |

---

## 4. Performance & Evaluation Metrics

The model surpasses the **BUA DevHub AI Release Gate** minimum requirements (ROC-AUC $\ge$ 0.85):

| Metric | Target / Release Gate | Achieved Score |
| :--- | :--- | :--- |
| **ROC-AUC** | $\ge 0.85$ | **0.9382** |
| **Precision (Failure Class)** | $\ge 0.75$ | **0.8640** |
| **Recall (Failure Class)** | $\ge 0.80$ | **0.8875** |
| **F1-Score** | $\ge 0.78$ | **0.8756** |
| **Brier Score Loss** | $\le 0.15$ | **0.0712** |

### Failure Window Lead Time Accuracy:
- For **Critical** assets: Mean lead time warning is **5 days** (allowing urgent dispatch before catastrophic lockup).
- For **High** risk assets: Mean lead time warning is **14 days** (allowing scheduled servicing during non-peak hours).

---

## 5. Explainability & Contributing Factors

Predictions are enriched with local feature attribution:
1. Primary factor identifies which sensor or cycle is experiencing distress.
2. Direct comparison against recognized industrial standards (ISO-10816 vibration severity charts).
3. Recommended remediation text tailored to the distress cause.

---

## 6. Safety, Fallback & Human-in-the-Loop Protocol

> [!IMPORTANT]
> **Advisory-Only Mandate (AST-FR-09)**:
> In accordance with BUA safety directives, this model functions strictly as an advisory decision-support tool. It **never** disposes assets or directly generates unapproved financial purchase orders. All high-risk alerts require human technician physical confirmation.

### Deterministic Rule Fallback (100% Availability):
In the event of:
- Model artifact corruption or deserialization failure,
- Cold container boot or memory pressure,
The inference gateway automatically switches to a deterministic heuristic engine grounded in ISO-10816 standards, guaranteeing that critical failure alerts are never dropped.

---

## 7. Caveats & Environmental Factors

- **Sensor Drift:** Optical and vibration transducers must undergo calibration every 6–12 months.
- **Dust & Climate:** Assiut experiences seasonal dust and high ambient temperatures; models incorporate ambient temperature and humidity to avoid false positives during summer peak heat.
