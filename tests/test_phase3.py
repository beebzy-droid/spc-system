import sys
sys.path.insert(0, ".")

from core.ingestion import load_csv
from core.preprocessor import preprocess
from core.schema import SpecLimits
from core.capability.indices import compute_all_capabilities
from core.capability.dashboard import plot_capability_dashboard

# --- Config ---
MEASUREMENT_COLS = ["measurement_1", "measurement_2"]
SPEC_LIMITS = {
    "measurement_1": SpecLimits("measurement_1", usl=53.0, lsl=47.0, target=50.0),
    "measurement_2": SpecLimits("measurement_2", usl=106.0, lsl=94.0, target=100.0),
}

# --- Load & Preprocess ---
df = load_csv("sample_data/batches.csv")
df = preprocess(df, MEASUREMENT_COLS, SPEC_LIMITS)

# --- Compute All Capabilities ---
print("\n📊 Computing capability indices...")
results = compute_all_capabilities(df, SPEC_LIMITS)

# --- Print Summaries ---
for name, result in results.items():
    result.summary()

# --- Plot Dashboards ---
for name, result in results.items():
    print(f"\n📊 Plotting dashboard for {name}...")
    plot_capability_dashboard(
        result=result,
        values=df[name].values,
        save_path=f"data/processed/capability_{name}.png",
    )

# --- Final Status ---
print("\n" + "="*50)
print("CAPABILITY SUMMARY")
print("="*50)
for name, result in results.items():
    print(f"  {name:20s} Cpk={result.cpk:.3f}  {result.status}")
print("="*50)
print("\n🎉 Phase 3 Complete — Dashboards saved to data/processed/")