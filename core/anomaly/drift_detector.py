import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DriftResult:
    """Results from drift detection analysis."""
    measurement_name: str
    changepoints: list[int]         # Indices where change-points detected
    rolling_mean: np.ndarray        # Rolling mean values
    rolling_std: np.ndarray         # Rolling std values
    drift_flags: list[int]          # Indices where rolling mean drifted
    drift_threshold: float          # Threshold used for flagging

    @property
    def has_drift(self) -> bool:
        return len(self.drift_flags) > 0

    @property
    def has_changepoints(self) -> bool:
        return len(self.changepoints) > 0

    def summary(self):
        print(f"\n  Drift Detection — {self.measurement_name}")
        print(f"  Change-points detected : {len(self.changepoints)} at indices {self.changepoints}")
        print(f"  Drift flags            : {len(self.drift_flags)} points")
        print(f"  Drift threshold        : ±{self.drift_threshold:.4f}")


def detect_changepoints(
    values: np.ndarray,
    model: str = "rbf",
    penalty: float = 10.0,
) -> list[int]:
    """
    Detect change-points using PELT algorithm via ruptures library.
    Returns list of change-point indices.
    """
    try:
        import ruptures as rpt
        values = np.asarray(values, dtype=float)
        algo = rpt.Pelt(model=model).fit(values)
        breakpoints = algo.predict(pen=penalty)
        # Remove last point (ruptures always adds n as final breakpoint)
        return [bp for bp in breakpoints if bp < len(values)]
    except ImportError:
        print("[drift_detector] ruptures not installed — skipping change-point detection.")
        return []
    except Exception as e:
        print(f"[drift_detector] Change-point detection failed: {e}")
        return []


def detect_rolling_drift(
    values: np.ndarray,
    window: int = 10,
    threshold_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Detect drift using rolling mean.
    Flags points where rolling mean deviates beyond threshold_sigma
    from the overall mean.
    """
    values = np.asarray(values, dtype=float)
    series = pd.Series(values)

    rolling_mean = series.rolling(window=window, center=True).mean().values
    rolling_std  = series.rolling(window=window, center=True).std().values

    overall_mean = np.nanmean(values)
    overall_std  = np.nanstd(values)
    threshold    = threshold_sigma * overall_std

    drift_flags = [
        i for i, rm in enumerate(rolling_mean)
        if not np.isnan(rm) and abs(rm - overall_mean) > threshold
    ]

    return rolling_mean, rolling_std, drift_flags, threshold


def run_drift_detection(
    values: np.ndarray,
    measurement_name: str = "Measurement",
    window: int = 10,
    threshold_sigma: float = 1.5,
    penalty: float = 10.0,
) -> DriftResult:
    """Full drift detection pipeline."""
    values = np.asarray(values, dtype=float)

    changepoints = detect_changepoints(values, penalty=penalty)
    rolling_mean, rolling_std, drift_flags, threshold = detect_rolling_drift(
        values, window=window, threshold_sigma=threshold_sigma
    )

    print(f"[drift_detector] {measurement_name}: "
          f"{len(changepoints)} changepoints, "
          f"{len(drift_flags)} drift flags")

    return DriftResult(
        measurement_name=measurement_name,
        changepoints=changepoints,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        drift_flags=drift_flags,
        drift_threshold=threshold,
    )