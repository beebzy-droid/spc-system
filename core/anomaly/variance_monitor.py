import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class VarianceResult:
    """Results from variance monitoring."""
    measurement_name: str
    window_stds: np.ndarray         # Std dev per sliding window
    window_indices: list[int]       # Center index of each window
    variance_flags: list[int]       # Windows with abnormal variance
    baseline_std: float             # Overall process std dev
    levene_pvalue: float            # Levene's test p-value
    levene_significant: bool        # True if variance shift detected

    @property
    def has_variance_shift(self) -> bool:
        return self.levene_significant or len(self.variance_flags) > 0

    def summary(self):
        print(f"\n  Variance Monitor — {self.measurement_name}")
        print(f"  Baseline σ          : {self.baseline_std:.4f}")
        print(f"  Levene p-value      : {self.levene_pvalue:.4f} "
              f"{'⚠️ SIGNIFICANT' if self.levene_significant else '✅ OK'}")
        print(f"  Variance spike flags: {len(self.variance_flags)} windows")


def run_variance_monitoring(
    values: np.ndarray,
    measurement_name: str = "Measurement",
    window: int = 15,
    variance_threshold: float = 2.0,
) -> VarianceResult:
    """
    Monitor process variance using:
    1. Sliding window standard deviation
    2. Levene's test comparing first half vs second half
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    baseline_std = np.std(values, ddof=1)

    # ── Sliding window variance ────────────────────────────────────
    window_stds = []
    window_indices = []

    for i in range(0, n - window, window // 2):
        segment = values[i: i + window]
        window_stds.append(np.std(segment, ddof=1))
        window_indices.append(i + window // 2)

    window_stds = np.array(window_stds)

    # Flag windows where std exceeds threshold * baseline
    variance_flags = [
        window_indices[i]
        for i, s in enumerate(window_stds)
        if s > variance_threshold * baseline_std
    ]

    # ── Levene's test (first half vs second half) ──────────────────
    mid = n // 2
    first_half  = values[:mid]
    second_half = values[mid:]
    levene_stat, levene_pvalue = stats.levene(first_half, second_half)
    levene_significant = levene_pvalue < 0.05

    print(f"[variance_monitor] {measurement_name}: "
          f"Levene p={levene_pvalue:.4f}, "
          f"{len(variance_flags)} variance spike window(s)")

    return VarianceResult(
        measurement_name=measurement_name,
        window_stds=window_stds,
        window_indices=window_indices,
        variance_flags=variance_flags,
        baseline_std=baseline_std,
        levene_pvalue=levene_pvalue,
        levene_significant=levene_significant,
    )