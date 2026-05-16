import sys
sys.path.insert(0, ".")

from core.ingestion import load_csv, dataframe_to_dataset
from core.preprocessor import preprocess
from core.schema import SpecLimits

# --- Config ---
MEASUREMENT_COLS = ["measurement_1", "measurement_2"]
PROCESS_PARAM_COLS = ["temperature", "pressure"]

SPEC_LIMITS = {
    "measurement_1": SpecLimits("measurement_1", usl=53.0, lsl=47.0, target=50.0),
    "measurement_2": SpecLimits("measurement_2", usl=106.0, lsl=94.0, target=100.0),
}

# --- Run ---
df_raw = load_csv("sample_data/batches.csv")
df_clean = preprocess(df_raw, MEASUREMENT_COLS, SPEC_LIMITS)
dataset = dataframe_to_dataset(df_clean, MEASUREMENT_COLS, PROCESS_PARAM_COLS, SPEC_LIMITS)

# --- Verify ---
print(f"\n✅ Clean records   : {len(df_clean)}")
print(f"✅ BatchRecord count: {len(dataset.records)}")
print(f"✅ OOS batches      : {df_clean['measurement_1_oos'].sum()}")
print(f"✅ Spec range (m1)  : {SPEC_LIMITS['measurement_1'].spec_range}")
print(f"\nSample output:\n{df_clean.head(3).to_string()}")