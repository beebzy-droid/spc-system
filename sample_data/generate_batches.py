import pandas as pd
import numpy as np

np.random.seed(42)
n_batches = 200

batch_ids = [f"BATCH-{str(i).zfill(4)}" for i in range(1, n_batches + 1)]
timestamps = pd.date_range(start="2024-01-01", periods=n_batches, freq="2h")

measurements = []
for i in range(n_batches):
    base = 50.0
    if 80 <= i < 100:
        base += (i - 80) * 0.15
    if 140 <= i < 160:
        noise = np.random.normal(0, 3.0)
    else:
        noise = np.random.normal(0, 1.0)
    measurements.append(round(base + noise, 4))

df = pd.DataFrame({
    "batch_id": batch_ids,
    "timestamp": timestamps,
    "measurement_1": measurements,
    "measurement_2": [round(np.random.normal(100, 2), 4) for _ in range(n_batches)],
    "temperature":   [round(np.random.normal(75, 0.5), 2) for _ in range(n_batches)],
    "pressure":      [round(np.random.normal(14.7, 0.3), 2) for _ in range(n_batches)],
    "operator_id":   np.random.choice(["OP-01", "OP-02", "OP-03"], n_batches),
    "line_id":       np.random.choice(["LINE-A", "LINE-B"], n_batches),
})

df.to_csv("sample_data/batches.csv", index=False)
print(f"Generated {n_batches} batch records.")