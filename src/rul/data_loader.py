"""
Loading and labeling of the NASA C-MAPSS FD001 turbofan-degradation dataset.

Dataset files (not included in this repo — download separately):
    train_FD001.txt
    test_FD001.txt
    RUL_FD001.txt
"""

import pandas as pd

from src.config import CMAPSS_COLUMNS


def load_train_data(train_path: str) -> pd.DataFrame:
    """Load the raw training file and attach column names + RUL labels."""
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    train_df = train_df.dropna(axis=1, how="all")
    train_df.columns = CMAPSS_COLUMNS
    return add_rul_labels(train_df)


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the RUL (Remaining Useful Life) target column.

    For the training set, an engine's RUL at a given cycle is simply the
    number of cycles remaining until that engine's last recorded cycle
    (i.e. simulated failure).
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df.apply(lambda row: max_cycle[row["engine_id"]] - row["cycle"], axis=1)
    return df


def load_test_data(test_path: str) -> pd.DataFrame:
    """Load the raw (unlabeled) test file and attach column names."""
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    test_df = test_df.dropna(axis=1, how="all")
    test_df.columns = CMAPSS_COLUMNS
    return test_df


def load_true_rul(rul_path: str) -> pd.DataFrame:
    """Load the official ground-truth RUL values for the test engines."""
    true_rul = pd.read_csv(rul_path, header=None)
    true_rul.columns = ["RUL"]
    return true_rul


def last_cycle_per_engine(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the test set to one row per engine: its last recorded cycle.

    This is the row the official C-MAPSS evaluation is scored against
    (RUL_FD001.txt gives one ground-truth value per engine, corresponding
    to its final observed cycle).
    """
    return (
        test_df.sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )
