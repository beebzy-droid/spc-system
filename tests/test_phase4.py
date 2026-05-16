import sys
sys.path.insert(0, ".")

from core.ingestion import load_csv
from core.preprocessor import preprocess
from core.schema import SpecLimits
from core.charts.imr_chart import compute_imr_limits
from core.rules.western_electric import run_all_rules
from core.rules.rule_reporter import print_rule_report, plot_rule_violations

# --- Config ---
MEASUREMENT_COLS = ["measurement_1", "measurement_2"]
SPEC_LIMITS = {
    "measurement_1": SpecLimits("measurement_1", usl=53.0, lsl=47.0, target=50.0),
    "measurement_2": SpecLimits("measurement_2", usl=106.0, lsl=94.0, target=100.0),
}

# --- Load & Preprocess ---
df = load_csv("sample_data/batches.csv")
df = preprocess(df, MEASUREMENT_COLS, SPEC_LIMITS)

# --- Run Rules on Each Measurement ---
for col in MEASUREMENT_COLS:
    values    = df[col].values
    batch_ids = df["batch_id"].tolist()

    # Get control limits from I-MR chart
    i_limits, _ = compute_imr_limits(values)

    print(f"\n🔍 Running Western Electric Rules on {col}...")
    violations = run_all_rules(
        values=values,
        center=i_limits.center_line,
        sigma=(i_limits.ucl - i_limits.center_line) / 3,
    )

    # Console report
    print_rule_report(violations, measurement_name=col)

    # Plot
    plot_rule_violations(
        values=values,
        batch_ids=batch_ids,
        violations=violations,
        measurement_name=col,
        center=i_limits.center_line,
        sigma=(i_limits.ucl - i_limits.center_line) / 3,
        save_path=f"data/processed/we_rules_{col}.png",
    )

# --- Final Summary ---
print("\n" + "="*60)
print("WESTERN ELECTRIC RULES — FINAL SUMMARY")
print("="*60)
for col in MEASUREMENT_COLS:
    values = df[col].values
    i_limits, _ = compute_imr_limits(values)
    violations = run_all_rules(
        values=values,
        center=i_limits.center_line,
        sigma=(i_limits.ucl - i_limits.center_line) / 3,
    )
    triggered = len(violations)
    total_pts  = sum(v.count for v in violations)
    print(f"  {col:20s} Rules triggered: {triggered}  |  Violation points: {total_pts}")
print("="*60)
print("\n🎉 Phase 4 Complete — Charts saved to data/processed/")