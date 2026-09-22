# AST Smart Asset — AI (Predictive Maintenance Engine)

AI component of the **AST Smart Asset** project. This service trains and
serves two machine-learning models that turn raw sensor/maintenance data
into a maintenance **risk signal** an application can act on:

| Model | Question it answers | Endpoint |
|---|---|---|
| **RUL Regressor** | "How many cycles does this asset have left?" | `POST /predict-rul` |
| **90-Day Failure Classifier** | "How likely is this asset to fail in the next 90 days?" | `POST /predict-failure-90d` |

Full model details, metrics, and limitations: **[docs/MODEL_DOCUMENTATION.md](docs/MODEL_DOCUMENTATION.md)**

> This repo was refactored from an original Colab notebook
> (`copy_of_asset_risk_engine.py`) into a normal installable Python
> package, so it can be version-controlled, tested, and deployed outside
> of Colab.

---

## 1. Project structure

```
AI/
├── README.md
├── requirements.txt
├── docs/
│   └── MODEL_DOCUMENTATION.md   # metrics, features, limitations
├── data/                        # NOT committed — put your data here
│   ├── cmapss/                  # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   └── tables/                  # assets.csv, asset_events.csv, work_orders.csv
├── models/                      # trained artifacts land here (.pkl / .json)
├── scripts/
│   ├── train_rul_model.py       # CLI: train the RUL model
│   └── train_failure_model.py   # CLI: train the failure classifier
└── src/
    ├── config.py                 # shared feature lists, paths, risk thresholds
    ├── rul/
    │   ├── data_loader.py        # load + label the C-MAPSS dataset
    │   ├── train.py               # train/evaluate/save the RUL model
    │   └── predict.py             # inference + risk mapping
    ├── failure/
    │   ├── feature_engineering.py # point-in-time-correct feature builder
    │   ├── train.py               # train/evaluate/save the classifier
    │   └── predict.py             # inference + deterministic fallback
    └── api/
        ├── schemas.py             # Pydantic request models
        └── main.py                # FastAPI app (/health, /predict-*)
```

---

## 2. Setup

```bash
git clone https://github.com/AST-Smart-Asset/Ai.git
cd Ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Get the data

Neither dataset is committed to the repo (see `.gitignore`) — place it
locally:

- **RUL model** — download the NASA C-MAPSS dataset and place
  `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` under `data/cmapss/`.
- **Failure classifier** — place `assets.csv`, `asset_events.csv`,
  `work_orders.csv` (exported from the main AST Smart Asset backend)
  under `data/tables/`.

## 4. Train the models

```bash
# RUL model
python -m scripts.train_rul_model \
    --train-path data/cmapss/train_FD001.txt \
    --test-path  data/cmapss/test_FD001.txt \
    --rul-path   data/cmapss/RUL_FD001.txt

# 90-day failure classifier
python -m scripts.train_failure_model \
    --assets      data/tables/assets.csv \
    --events      data/tables/asset_events.csv \
    --work-orders data/tables/work_orders.csv
```

Each script prints its evaluation metrics and saves the trained model +
feature list (+ metadata, for the failure model) into `models/`.

## 5. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API loads both trained models from `models/` at startup, so training
must be run at least once first.

### `GET /health`
```json
{
  "status": "online",
  "service": "AST Smart Asset - Predictive Maintenance AI",
  "models": { "rul": "loaded", "failure_90d": "loaded" }
}
```

### `POST /predict-rul`
```bash
curl -X POST http://127.0.0.1:8000/predict-rul \
  -H "Content-Type: application/json" \
  -d '{
    "sensor1": 518.67, "sensor2": 642.15, "sensor3": 1589.7,
    "sensor4": 1400.6, "sensor5": 14.62,  "sensor6": 21.61,
    "sensor7": 554.36, "sensor8": 2388.06,"sensor9": 9046.19,
    "sensor10": 1.3,   "sensor11": 47.47, "sensor12": 521.66,
    "sensor13": 2388.02,"sensor14": 8138.62,"sensor15": 8.4195,
    "sensor16": 0.03,  "sensor17": 392,   "sensor18": 2388,
    "sensor19": 100,   "sensor20": 39.06, "sensor21": 23.419
  }'
```
```json
{ "predicted_rul": 112.4, "risk": "Low", "reason": "Sufficient estimated remaining useful life" }
```

### `POST /predict-failure-90d`
```bash
curl -X POST http://127.0.0.1:8000/predict-failure-90d \
  -H "Content-Type: application/json" \
  -d '{
    "asset_age_days": 420,
    "days_to_maintenance": 15,
    "previous_work_orders": 6,
    "previous_corrective_jobs": 2,
    "previous_total_cost": 1450.0,
    "previous_downtime_minutes": 340,
    "work_orders_last_90d": 1,
    "corrective_jobs_last_90d": 1
  }'
```
```json
{
  "model_version": "failure-classifier-90d-v1",
  "failure_probability": 0.61,
  "predicted_failure_90d": 1,
  "risk": "Medium",
  "reasons": ["High historical maintenance cost", "High historical downtime", "Maintenance due within 90 days"],
  "fallback_used": false
}
```

Interactive docs are also available once the server is running, at
`http://127.0.0.1:8000/docs` (Swagger UI, generated automatically by
FastAPI).

---

## 6. Notes on this refactor

The original notebook (`copy_of_asset_risk_engine.py`, kept in the repo
for reference / audit trail) was written and iterated on inside Google
Colab: it re-defines the same column lists multiple times, retrains
models more than once, and mixes exploration (`print`, `display`) with
the actual pipeline logic. This refactor:

- Moves every reusable piece of logic into `src/`, with no duplicated
  feature lists (see `src/config.py`).
- Separates **data loading**, **training**, and **inference** into their
  own modules per model.
- Keeps the FastAPI app (`src/api/main.py`) as a thin layer that only
  loads models and wires up requests — no training/feature-engineering
  logic lives there.
- Adds a deterministic fallback for the failure classifier explicitly, as
  a safety net for production use (see `src/failure/predict.py`).
- Drops Colab-only code (`google.colab.files.upload/download`,
  `nest_asyncio` + in-notebook `uvicorn` thread) in favor of a normal
  `uvicorn` CLI run.

## 7. Roadmap / suggested next steps

- [ ] Add automated tests (`pytest`) around `src/rul` and `src/failure`.
- [ ] Add a `Dockerfile` for containerized deployment.
- [ ] Add CI (GitHub Actions) to run tests + lint on every push.
- [ ] Track experiments/metrics over time (e.g. MLflow) instead of
      print statements.
- [ ] Version the trained model artifacts (e.g. via GitHub Releases or a
      model registry) instead of `.gitignore`-ing them.

## 8. Repository

https://github.com/AST-Smart-Asset/Ai
