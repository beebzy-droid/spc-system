import sys
sys.path.insert(0, ".")

import numpy as np
from core.ingestion import load_csv
from core.preprocessor import preprocess
from core.schema import SpecLimits
from core.charts.imr_chart import plot_imr_chart, compute_imr_limits
from core.charts.xbar_r_chart import plot_xbar_r_chart
from core.charts.cusum_ewma import plot_cusum_ewma

# --- Config ---
MEASUREMENT_COLS = ["measurement_1", "measurement_2"]
SPEC_LIMITS = {
    "measurement_1": SpecLimits("measurement_1", usl=53.0, lsl=47.0, target=50.0),
    "measurement_2": SpecLimits("measurement_2", usl=106.0, lsl=94.0, target=100.0),
}

# --- Load Data ---
df = load_csv("sample_data/batches.csv")
df = preprocess(df, MEASUREMENT_COLS, SPEC_LIMITS)

values    = df["measurement_1"].values
batch_ids = df["batch_id"].tolist()

# --- I-MR Chart ---
print("\n📊 Plotting I-MR Chart...")
i_limits, mr_limits = plot_imr_chart(
    values=values,
    batch_ids=batch_ids,
    measurement_name="Measurement 1",
    usl=53.0,
    lsl=47.0,
    save_path="data/processed/imr_chart.png",
)
print("\nI Chart Limits:")
i_limits.summary()
print("\nMR Chart Limits:")
mr_limits.summary()

# --- X̄-R Chart (subgroups of 5) ---
print("\n📊 Plotting X̄-R Chart...")
subgroup_size = 5
n = len(values) - (len(values) % subgroup_size)
subgroups = values[:n].reshape(-1, subgroup_size).tolist()
labels    = batch_ids[:n:subgroup_size]
plot_xbar_r_chart(
    subgroups=subgroups,
    subgroup_labels=labels,
    measurement_name="Measurement 1",
    save_path="data/processed/xbar_r_chart.png",
)

# --- CUSUM & EWMA ---
print("\n📊 Plotting CUSUM & EWMA Charts...")
cusum, ewma = plot_cusum_ewma(
    values=values,
    batch_ids=batch_ids,
    measurement_name="Measurement 1",
    target=50.0,
    save_path="data/processed/cusum_ewma_chart.png",
)

print(f"\n✅ CUSUM signals (upward drift)  : {len(cusum['signals_pos'])} detected")
print(f"✅ CUSUM signals (downward drift) : {len(cusum['signals_neg'])} detected")
print(f"✅ EWMA signals                   : {len(ewma['signals'])} detected")
print(f"\n🎉 Phase 2 Complete — Charts saved to data/processed/")