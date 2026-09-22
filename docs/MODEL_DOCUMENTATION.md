# AI Model Documentation — AST Smart Asset

This document describes the two machine-learning models in this repo, how
they were built, how well they perform, and their known limitations.

---

## 1. Remaining Useful Life (RUL) Model

| | |
|---|---|
| **Algorithm** | Random Forest Regressor (scikit-learn) |
| **Dataset** | NASA C-MAPSS FD001 (turbofan engine degradation simulation) |
| **Target** | Remaining Useful Life (RUL), in cycles |
| **Code** | `src/rul/` |

### Input features
21 sensor readings per cycle: `sensor1` … `sensor21`.

### Training
- 100 trees, `random_state=42`.
- Engines are split 80/20 **by engine ID** (not by row) so that no engine's
  cycles appear in both train and test — this avoids leaking
  near-identical neighboring cycles across the split.
- Final model is retrained on the **full** training set and scored against
  the official `RUL_FD001.txt` ground truth (last cycle per test engine).

### Evaluation (from the original experiment)
| Metric | Model | Naive baseline (predict mean RUL) |
|---|---|---|
| MAE  | 23.33 cycles | 64.09 cycles |
| RMSE | 31.30 cycles | — |

### Risk mapping
| Predicted RUL | Risk | Reason |
|---|---|---|
| ≤ 20 | High | Very low estimated remaining useful life |
| ≤ 50 | Medium | Moderate estimated remaining useful life |
| > 50 | Low | Sufficient estimated remaining useful life |

> These thresholds are **experimental**, chosen for this ML experiment —
> they are not official engineering/maintenance thresholds.

### Explainability
Feature importance (top 5, from the original experiment):

| Feature | Importance |
|---|---|
| sensor11 | 0.454 |
| sensor9  | 0.129 |
| sensor4  | 0.083 |
| sensor12 | 0.048 |
| sensor14 | 0.039 |

### Output shape (`/predict-rul`)
```json
{
  "predicted_rul": 87.42,
  "risk": "Low",
  "reason": "Sufficient estimated remaining useful life"
}
```

---

## 2. 90-Day Asset Failure Classifier

| | |
|---|---|
| **Algorithm** | Logistic Regression (`class_weight="balanced"`) inside a scikit-learn Pipeline (median imputer → standard scaler → classifier) |
| **Dataset** | Internal `assets.csv`, `asset_events.csv`, `work_orders.csv` |
| **Target** | `failure_90d` — will this asset have a corrective/fault-related work order in the next 90 days? |
| **Code** | `src/failure/` |

### Label definition (proxy label)
A work order counts as a "failure event" when its `completion_notes`
contain **"corrective job"** or **"fault report"** (case-insensitive).
This is a **proxy**, not a confirmed hardware failure — see Limitations.

### Feature engineering
For every historical work-order date ("prediction point"), features are
computed using **only data available before that date** (point-in-time
correctness, no leakage from the future):

| Feature | Description |
|---|---|
| `asset_age_days` | Days since the asset's purchase date |
| `days_to_maintenance` | Days until the asset's next scheduled maintenance |
| `previous_work_orders` | Count of all prior work orders |
| `previous_corrective_jobs` | Count of prior corrective/fault-related work orders |
| `previous_total_cost` | Sum of prior parts + labor cost |
| `previous_downtime_minutes` | Sum of prior downtime |
| `work_orders_last_90d` | Work orders in the last 90 days |
| `corrective_jobs_last_90d` | Corrective work orders in the last 90 days |

Rows are dropped when their 90-day future window would run past the end
of the observed data (can't be labeled reliably).

### Splitting
Chronological (time-based) 80/20 split — never random — so the model is
always evaluated on data that comes **after** its training period.

### Evaluation vs. deterministic baselines
Two deterministic baselines were used for comparison:
1. **Recent-corrective-job baseline** — predict failure if the asset had a
   corrective job in the last 90 days.
2. **Due-date baseline** — predict failure if scheduled maintenance is due
   within the horizon window.

The ML model's precision/recall/F1 for each horizon are computed the same
way and printed by `scripts/train_failure_model.py`.

### Risk mapping (from predicted probability)
| Probability | Risk |
|---|---|
| ≥ 0.70 | High |
| ≥ 0.30 | Medium |
| < 0.30 | Low |

### Output shape (`/predict-failure-90d`)
```json
{
  "model_version": "failure-classifier-90d-v1",
  "failure_probability": 0.4231,
  "predicted_failure_90d": 0,
  "risk": "Medium",
  "reasons": ["Maintenance due within 90 days"],
  "fallback_used": false
}
```

### Deterministic fallback
If the ML model call fails for any reason (e.g. an incompatible feature
set after a schema change), the API automatically falls back to a
**deterministic rule**: flag failure if `days_to_maintenance` is between 0
and 90. The response marks `"fallback_used": true` and
`"model_version": "deterministic-due-date-fallback-v1"` so callers can
always tell which path produced the answer. This fallback is a
**production safety net** — the ML model should never be the single point
of failure for a risk signal.

---

## 3. Known Limitations

- The 90-day failure label is a **text-matching proxy**, not a confirmed
  failure event — completion notes could be inconsistent or mislabeled.
- The number of observed positive (failure) cases is small, which limits
  how confidently precision/recall can be estimated.
- The underlying tables (`assets`, `asset_events`, `work_orders`) may be
  synthetic/limited in size, which limits generalization to a larger
  real-world fleet.
- Neither model should be used to **automatically** change an asset's
  state or auto-close/auto-create work orders — both are decision-support
  signals for a human to review.

## 4. Retraining

Both models can be retrained at any time with fresh data using the
scripts in `scripts/` (see the main [README](../README.md) for exact
commands). Re-run training whenever:
- A new batch of work orders / asset events becomes available.
- The C-MAPSS-equivalent sensor schema changes.
- Evaluation metrics on new data drift meaningfully from the numbers
  above.
