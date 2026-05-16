from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass
class SpecLimits:
    """Specification limits for a single measurement."""
    measurement_name: str
    usl: float
    lsl: float
    target: Optional[float] = None

    def __post_init__(self):
        if self.lsl >= self.usl:
            raise ValueError(f"LSL must be less than USL for {self.measurement_name}")
        if self.target is None:
            self.target = (self.usl + self.lsl) / 2

    @property
    def spec_range(self) -> float:
        return self.usl - self.lsl


@dataclass
class BatchRecord:
    """Single validated batch record."""
    batch_id: str
    timestamp: pd.Timestamp
    measurements: dict
    process_params: dict
    operator_id: Optional[str] = None
    line_id: Optional[str] = None


@dataclass
class BatchDataset:
    """Collection of batch records + their spec limits."""
    records: list[BatchRecord]
    spec_limits: dict[str, SpecLimits]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            row = {
                "batch_id": r.batch_id,
                "timestamp": r.timestamp,
                **r.measurements,
                **r.process_params,
                "operator_id": r.operator_id,
                "line_id": r.line_id,
            }
            rows.append(row)
        return pd.DataFrame(rows)