import sys
sys.path.insert(0, ".")

from core.schema import SpecLimits
from core.reporting.spc_pipeline import run_full_spc_pipeline

# --- Config ---
MEASUREMENT_COLS   = ["measurement_1", "measurement_2"]
PROCESS_PARAM_COLS = ["temperature", "pressure"]
SPEC_LIMITS = {
    "measurement_1": SpecLimits("measurement_1", usl=53.0, lsl=47.0, target=50.0),
    "measurement_2": SpecLimits("measurement_2", usl=106.0, lsl=94.0, target=100.0),
}

# --- Run Full Pipeline ---
alert_summary, report_file = run_full_spc_pipeline(
    csv_path="sample_data/batches.csv",
    measurement_cols=MEASUREMENT_COLS,
    process_param_cols=PROCESS_PARAM_COLS,
    spec_limits=SPEC_LIMITS,
    chart_dir="data/processed",
    report_path="data/processed/spc_report.html",
    audit_path="data/processed/audit_trail.json",
    show_charts=True,
)

print(f"\n✅ Open your report here:")
print(f"   data/processed/spc_report.html")