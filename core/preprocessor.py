import numpy as np
import pandas as pd
from core.schema import SpecLimits


def remove_missing(df: pd.DataFrame, measurement_cols: list[str]) -> pd.DataFrame:
    """Drop rows with missing measurement values."""
    before = len(df)
    df = df.dropna(subset=measurement_cols)
    dropped = before - len(df)
    if dropped:
        print(f"[preprocessor] Dropped {dropped} rows with missing values.")
    return df


def remove_sensor_outliers(
    df: pd.DataFrame,
    measurement_cols: list[str],
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Remove extreme sensor noise using Z-score (conservative threshold)."""
    before = len(df)
    for col in measurement_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[df[col].between(mean - z_threshold * std, mean + z_threshold * std)]
    dropped = before - len(df)
    if dropped:
        print(f"[preprocessor] Removed {dropped} sensor noise outliers.")
    return df.reset_index(drop=True)


def flag_out_of_spec(
    df: pd.DataFrame,
    spec_limits: dict[str, SpecLimits],
) -> pd.DataFrame:
    """Add boolean columns flagging out-of-spec measurements."""
    for name, spec in spec_limits.items():
        if name in df.columns:
            df[f"{name}_oos"] = ~df[name].between(spec.lsl, spec.usl)
    oos_count = df[[c for c in df.columns if c.endswith("_oos")]].any(axis=1).sum()
    print(f"[preprocessor] {oos_count} batches flagged out-of-spec.")
    return df


def preprocess(
    df: pd.DataFrame,
    measurement_cols: list[str],
    spec_limits: dict[str, SpecLimits],
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = remove_missing(df, measurement_cols)
    df = remove_sensor_outliers(df, measurement_cols, z_threshold)
    df = flag_out_of_spec(df, spec_limits)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df