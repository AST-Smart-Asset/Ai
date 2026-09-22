"""
Feature engineering for the N-day asset-failure classifier.

Builds a labeled, point-in-time-correct dataset from three raw tables:
    assets.csv        -- one row per physical asset
    asset_events.csv  -- event log per asset (unused for features directly,
                          kept here for completeness / future use)
    work_orders.csv   -- maintenance work order history

A "failure" is a proxy label: a work order whose completion notes mention
a corrective job or fault report (see CORRECTIVE_NOTES_PATTERN in
src/config.py). This is NOT a confirmed hardware failure — see the
limitations section in docs/MODEL_DOCUMENTATION.md.
"""

import pandas as pd

from src.config import CORRECTIVE_NOTES_PATTERN

DATE_COLUMNS = ["scheduled_at", "started_at", "completed_at", "created_at", "updated_at"]


def load_tables(assets_path: str, events_path: str, work_orders_path: str):
    assets = pd.read_csv(assets_path)
    events = pd.read_csv(events_path)
    work_orders = pd.read_csv(work_orders_path)
    return assets, events, work_orders


def mark_corrective_orders(work_orders: pd.DataFrame) -> pd.DataFrame:
    """Flag work orders whose notes indicate a corrective/fault-related job."""
    wo = work_orders.copy()

    for col in DATE_COLUMNS:
        if col in wo.columns:
            wo[col] = pd.to_datetime(wo[col], utc=True, errors="coerce")

    wo["is_corrective"] = (
        wo["completion_notes"]
        .fillna("")
        .str.contains(CORRECTIVE_NOTES_PATTERN, case=False, regex=True)
        .astype(int)
    )

    for col in ["downtime_minutes", "parts_cost", "labor_cost"]:
        wo[col] = pd.to_numeric(wo[col], errors="coerce").fillna(0)

    wo["total_cost"] = wo["parts_cost"] + wo["labor_cost"]
    return wo.sort_values("scheduled_at")


def build_failure_events(wo: pd.DataFrame) -> pd.DataFrame:
    """Reduce corrective work orders to a clean (asset_id, scheduled_at) timeline."""
    corrective = wo[wo["is_corrective"] == 1]
    return (
        corrective[["asset_id", "scheduled_at"]]
        .dropna()
        .sort_values(["asset_id", "scheduled_at"])
        .reset_index(drop=True)
    )


def _has_future_failure(failure_events, asset_id, prediction_date, days) -> int:
    window = failure_events[
        (failure_events["asset_id"] == asset_id)
        & (failure_events["scheduled_at"] > prediction_date)
        & (failure_events["scheduled_at"] <= prediction_date + pd.Timedelta(days=days))
    ]
    return int(len(window) > 0)


def _asset_row_features(asset_row, prediction_date):
    if asset_row is None:
        return 0, 9999

    purchase_date = asset_row.get("purchase_date")
    next_due = asset_row.get("next_maintenance_due_at")

    if pd.notna(purchase_date):
        asset_age_days = max((prediction_date - purchase_date).total_seconds() / 86400, 0)
    else:
        asset_age_days = 0

    if pd.notna(next_due):
        days_to_maintenance = (next_due - prediction_date).total_seconds() / 86400
    else:
        days_to_maintenance = 9999

    return asset_age_days, days_to_maintenance


def build_labeled_dataset(
    assets: pd.DataFrame,
    work_orders: pd.DataFrame,
    horizons=(30, 60, 90),
    max_horizon: int = 90,
) -> pd.DataFrame:
    """
    Build the full point-in-time feature + label dataset.

    One row is created per historical work-order date ("prediction point").
    For each prediction point we compute:
      * static/historical features using ONLY data available before that date
      * a binary failure_<h>d label for each horizon in `horizons`, using
        ONLY corrective work orders that occur AFTER that date.

    Rows too close to the end of the observed history are dropped, since we
    cannot know whether a failure would have occurred in a window that runs
    past the end of the dataset.
    """
    wo = mark_corrective_orders(work_orders)
    ast = assets.copy()
    ast["purchase_date"] = pd.to_datetime(ast["purchase_date"], utc=True, errors="coerce")
    ast["next_maintenance_due_at"] = pd.to_datetime(
        ast["next_maintenance_due_at"], utc=True, errors="coerce"
    )
    ast_by_id = ast.set_index("id")

    failure_events = build_failure_events(wo)

    all_dates = pd.to_datetime(work_orders["scheduled_at"], utc=True, errors="coerce").dropna()
    dataset_end = all_dates.max()

    prediction_points = (
        work_orders[["asset_id", "scheduled_at"]]
        .assign(scheduled_at=lambda d: pd.to_datetime(d["scheduled_at"], utc=True, errors="coerce"))
        .dropna(subset=["asset_id", "scheduled_at"])
        .drop_duplicates()
    )
    prediction_points = prediction_points[
        prediction_points["scheduled_at"] <= dataset_end - pd.Timedelta(days=max_horizon)
    ].copy()
    prediction_points = prediction_points.rename(columns={"scheduled_at": "prediction_date"})

    for horizon in horizons:
        prediction_points[f"failure_{horizon}d"] = prediction_points.apply(
            lambda row: _has_future_failure(
                failure_events, row["asset_id"], row["prediction_date"], horizon
            ),
            axis=1,
        )

    def make_features(row):
        asset_id = row["asset_id"]
        prediction_date = row["prediction_date"]

        history = wo[(wo["asset_id"] == asset_id) & (wo["scheduled_at"] < prediction_date)]
        asset_row = ast_by_id.loc[asset_id] if asset_id in ast_by_id.index else None
        asset_age_days, days_to_maintenance = _asset_row_features(asset_row, prediction_date)

        recent = history["scheduled_at"] >= prediction_date - pd.Timedelta(days=90)

        return pd.Series(
            {
                "asset_age_days": asset_age_days,
                "days_to_maintenance": days_to_maintenance,
                "previous_work_orders": len(history),
                "previous_corrective_jobs": history["is_corrective"].sum(),
                "previous_total_cost": history["total_cost"].sum(),
                "previous_downtime_minutes": history["downtime_minutes"].sum(),
                "work_orders_last_90d": recent.sum(),
                "corrective_jobs_last_90d": (recent & (history["is_corrective"] == 1)).sum(),
            }
        )

    features_df = prediction_points.apply(make_features, axis=1)

    model_data = pd.concat(
        [prediction_points.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )
    return model_data.sort_values("prediction_date").reset_index(drop=True)


def valid_window(model_data: pd.DataFrame, failure_events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Restrict to prediction points for which a *full* future window of
    `horizon` days exists inside the observed failure history, so that a
    negative label isn't just an artifact of the data running out.
    """
    last_failure_date = failure_events["scheduled_at"].max()
    evaluation_end = last_failure_date - pd.Timedelta(days=horizon)
    valid = model_data[model_data["prediction_date"] <= evaluation_end].copy()
    return valid.sort_values("prediction_date").reset_index(drop=True)
