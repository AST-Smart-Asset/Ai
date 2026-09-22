# Predictive Maintenance AI Service

FastAPI service for the project predictive-maintenance models.

## Files required

Keep these trained-model files beside `main.py`:

- `rul_random_forest.pkl`
- `features.json`
- `failure_classifier_90d.pkl`
- `failure_classifier_features.json`

## API

### Health

`GET /health`

### RUL

`POST /predict-rul`

Body:

```json
{
  "sensor1": 0,
  "sensor2": 0,
  "sensor3": 0,
  "sensor4": 0,
  "sensor5": 0,
  "sensor6": 0,
  "sensor7": 0,
  "sensor8": 0,
  "sensor9": 0,
  "sensor10": 0,
  "sensor11": 0,
  "sensor12": 0,
  "sensor13": 0,
  "sensor14": 0,
  "sensor15": 0,
  "sensor16": 0,
  "sensor17": 0,
  "sensor18": 0,
  "sensor19": 0,
  "sensor20": 0,
  "sensor21": 0
}
```

### 90-day failure classifier

`POST /predict-failure-90d`

Body:

```json
{
  "asset_age_days": 500,
  "days_to_maintenance": 30,
  "previous_work_orders": 4,
  "previous_corrective_jobs": 1,
  "previous_total_cost": 5000,
  "previous_downtime_minutes": 120,
  "work_orders_last_90d": 2,
  "corrective_jobs_last_90d": 1
}
```

The endpoint returns the model probability/risk when the model runs. If the model fails, it uses the deterministic due-date fallback.

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `/docs`.

## Render

Create a Render Web Service from this folder/repository.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Do not put API keys in the repository.

## Important project note

The failure classifier is experimental. Its labels are proxy labels derived from project work-order text, the dataset is synthetic, and the test set has few positive examples. The API therefore keeps a deterministic fallback and does not change asset/work-order state automatically.
