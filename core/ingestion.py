import pandas as pd
from core.schema import BatchRecord, BatchDataset, SpecLimits


REQUIRED_COLUMNS = {"batch_id", "timestamp"}


def load_csv(filepath: str) -> pd.DataFrame:
    """Load raw CSV and do basic column validation."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print(f"[ingestion] Loaded {len(df)} rows from {filepath}")
    return df


def dataframe_to_dataset(
    df: pd.DataFrame,
    measurement_cols: list[str],
    process_param_cols: list[str],
    spec_limits: dict[str, SpecLimits],
) -> BatchDataset:
    """Convert a raw DataFrame into a typed BatchDataset."""
    records = []
    for _, row in df.iterrows():
        record = BatchRecord(
            batch_id=row["batch_id"],
            timestamp=row["timestamp"],
            measurements={col: row[col] for col in measurement_cols},
            process_params={col: row[col] for col in process_param_cols},
            operator_id=row.get("operator_id"),
            line_id=row.get("line_id"),
        )
        records.append(record)

    return BatchDataset(records=records, spec_limits=spec_limits)