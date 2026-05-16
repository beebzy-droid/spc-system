import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


@dataclass
class MLAnomalyResult:
    """Results from ML-based anomaly detection."""
    measurement_name: str
    isolation_forest_flags: list[int]   # Anomaly indices from IF
    one_class_svm_flags: list[int]      # Anomaly indices from OC-SVM
    consensus_flags: list[int]          # Flagged by BOTH models
    anomaly_scores: np.ndarray          # IF anomaly scores (lower = more anomalous)
    contamination: float                # % of data treated as anomalies

    @property
    def total_anomalies(self) -> int:
        return len(self.consensus_flags)

    def summary(self):
        print(f"\n  ML Anomaly Detection — {self.measurement_name}")
        print(f"  Isolation Forest flags : {len(self.isolation_forest_flags)}")
        print(f"  One-Class SVM flags    : {len(self.one_class_svm_flags)}")
        print(f"  Consensus anomalies    : {len(self.consensus_flags)} "
              f"(flagged by both models)")
        print(f"  Contamination rate     : {self.contamination:.1%}")


def build_features(
    df: pd.DataFrame,
    measurement_cols: list[str],
    process_param_cols: list[str] = None,
    window: int = 5,
) -> np.ndarray:
    """
    Build feature matrix for ML models.
    Features: raw values + rolling stats + process params
    """
    features = {}

    for col in measurement_cols:
        series = df[col]
        features[col] = series.values
        features[f"{col}_rolling_mean"] = series.rolling(window, min_periods=1).mean().values
        features[f"{col}_rolling_std"]  = series.rolling(window, min_periods=1).std().fillna(0).values
        features[f"{col}_diff"]         = series.diff().fillna(0).values

    if process_param_cols:
        for col in process_param_cols:
            if col in df.columns:
                features[col] = df[col].values

    return np.column_stack(list(features.values()))


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[list[int], np.ndarray]:
    """
    Run Isolation Forest anomaly detection.
    Returns (anomaly_indices, anomaly_scores)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    predictions = model.fit_predict(X_scaled)
    scores      = model.score_samples(X_scaled)

    anomaly_indices = [i for i, p in enumerate(predictions) if p == -1]
    return anomaly_indices, scores


def run_one_class_svm(
    X: np.ndarray,
    nu: float = 0.05,
) -> list[int]:
    """
    Run One-Class SVM anomaly detection.
    nu = upper bound on fraction of anomalies.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(kernel="rbf", nu=nu)
    predictions = model.fit_predict(X_scaled)

    return [i for i, p in enumerate(predictions) if p == -1]


def run_ml_detection(
    df: pd.DataFrame,
    measurement_cols: list[str],
    process_param_cols: list[str] = None,
    contamination: float = 0.05,
) -> MLAnomalyResult:
    """Full ML anomaly detection pipeline."""

    print(f"[ml_detector] Building features from "
          f"{len(measurement_cols)} measurement(s)...")

    X = build_features(df, measurement_cols, process_param_cols)

    # Run both models
    if_flags, if_scores = run_isolation_forest(X, contamination=contamination)
    svm_flags           = run_one_class_svm(X, nu=contamination)

    # Consensus — flagged by BOTH models
    consensus = sorted(set(if_flags) & set(svm_flags))

    print(f"[ml_detector] IF={len(if_flags)}, SVM={len(svm_flags)}, "
          f"Consensus={len(consensus)}")

    return MLAnomalyResult(
        measurement_name=", ".join(measurement_cols),
        isolation_forest_flags=if_flags,
        one_class_svm_flags=svm_flags,
        consensus_flags=consensus,
        anomaly_scores=if_scores,
        contamination=contamination,
    )