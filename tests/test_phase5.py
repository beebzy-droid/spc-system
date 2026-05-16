import sys
sys.path.insert(0, ".")

from core.ingestion import load_csv
from core.preprocessor import preprocess
from core.schema import SpecLimits
from core.anomaly.drift_detector import run_drift_detection
from core.anomaly.variance_monitor import run_variance_monitoring
from core.anomaly.ml_detector import run_ml_detection
from core.anomaly.anomaly_reporter import print_anomaly_report, plot_anomaly_dashboard

# --- Config ---
MEASUREMENT_COLS   = ["measurement_1", "measurement_2"]
PROCESS_PARAM_COLS = ["temperature", "pressure"]
SPEC_LIMITS = {
    "measurement_1": SpecLimits("measurement_1", usl=53.0, lsl=47.0, target=50.0),
    "measurement_2": SpecLimits("measurement_2", usl=106.0, lsl=94.0, target=100.0),
}

# --- Load & Preprocess ---
df = load_csv("sample_data/batches.csv")
df = preprocess(df, MEASUREMENT_COLS, SPEC_LIMITS)
batch_ids = df["batch_id"].tolist()

# --- Run Per Measurement ---
for col in MEASUREMENT_COLS:
    values = df[col].values
    print(f"\n{'='*60}")
    print(f"  Running Anomaly Detection — {col}")
    print(f"{'='*60}")

    # Drift detection
    drift = run_drift_detection(
        values=values,
        measurement_name=col,
        window=10,
        threshold_sigma=1.5,
        penalty=10.0,
    )

    # Variance monitoring
    variance = run_variance_monitoring(
        values=values,
        measurement_name=col,
        window=15,
        variance_threshold=2.0,
    )

    # ML detection (multivariate — uses all measurements + params)
    ml = run_ml_detection(
        df=df,
        measurement_cols=MEASUREMENT_COLS,
        process_param_cols=PROCESS_PARAM_COLS,
        contamination=0.05,
    )

    # Console report
    print_anomaly_report(drift, variance, ml)

    # Dashboard plot
    plot_anomaly_dashboard(
        values=values,
        batch_ids=batch_ids,
        drift=drift,
        variance=variance,
        ml=ml,
        measurement_name=col,
        save_path=f"data/processed/anomaly_{col}.png",
    )

# --- Final Summary ---
print("\n" + "="*60)
print("ANOMALY DETECTION — FINAL SUMMARY")
print("="*60)
for col in MEASUREMENT_COLS:
    values   = df[col].values
    drift    = run_drift_detection(values, col, penalty=10.0)
    variance = run_variance_monitoring(values, col)
    ml       = run_ml_detection(df, MEASUREMENT_COLS, PROCESS_PARAM_COLS)
    print(f"\n  {col}")
    print(f"    Change-points  : {len(drift.changepoints)}")
    print(f"    Drift flags    : {len(drift.drift_flags)}")
    print(f"    Variance spikes: {len(variance.variance_flags)}")
    print(f"    ML consensus   : {len(ml.consensus_flags)}")
print("="*60)
print("\n🎉 Phase 5 Complete — Dashboards saved to data/processed/")