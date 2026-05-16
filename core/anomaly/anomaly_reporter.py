import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from core.anomaly.drift_detector import DriftResult
from core.anomaly.variance_monitor import VarianceResult
from core.anomaly.ml_detector import MLAnomalyResult


def print_anomaly_report(
    drift: DriftResult,
    variance: VarianceResult,
    ml: MLAnomalyResult,
):
    """Print full anomaly detection report to console."""
    print(f"\n{'='*60}")
    print(f"  ANOMALY DETECTION REPORT")
    print(f"{'='*60}")
    drift.summary()
    variance.summary()
    ml.summary()

    print(f"\n{'─'*60}")
    print(f"  OVERALL STATUS")
    print(f"{'─'*60}")

    issues = []
    if drift.has_changepoints:
        issues.append(f"  🔴 {len(drift.changepoints)} change-point(s) detected")
    if drift.has_drift:
        issues.append(f"  🟠 {len(drift.drift_flags)} rolling mean drift point(s)")
    if variance.has_variance_shift:
        issues.append(f"  🟠 Variance shift detected (Levene p={variance.levene_pvalue:.4f})")
    if variance.variance_flags:
        issues.append(f"  🟡 {len(variance.variance_flags)} high-variance window(s)")
    if ml.consensus_flags:
        issues.append(f"  🔴 {len(ml.consensus_flags)} consensus ML anomaly batch(es)")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ No anomalies detected across all methods.")

    print(f"{'='*60}")


def plot_anomaly_dashboard(
    values: np.ndarray,
    batch_ids: list[str],
    drift: DriftResult,
    variance: VarianceResult,
    ml: MLAnomalyResult,
    measurement_name: str = "Measurement",
    save_path: str = None,
):
    """
    Full anomaly detection dashboard with 4 panels:
    1. Drift detection (rolling mean + change-points)
    2. Variance monitoring (sliding window σ)
    3. ML anomaly scores (Isolation Forest)
    4. Consensus anomaly overlay on raw data
    """
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values))

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Anomaly Detection Dashboard — {measurement_name}",
        fontsize=14, fontweight="bold"
    )
    gs = GridSpec(4, 1, figure=fig, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # ── Panel 1: Drift Detection ───────────────────────────────────
    ax1.plot(x, values, "b-", linewidth=0.8, alpha=0.5, label="Raw")
    ax1.plot(x, drift.rolling_mean, "orange", linewidth=2.0, label="Rolling Mean")

    overall_mean = np.nanmean(values)
    ax1.axhline(overall_mean, color="green", linestyle="--",
                linewidth=1.2, label="Overall Mean")
    ax1.axhline(overall_mean + drift.drift_threshold,
                color="red", linestyle=":", linewidth=1.0, label="Drift Threshold")
    ax1.axhline(overall_mean - drift.drift_threshold,
                color="red", linestyle=":", linewidth=1.0)

    for cp in drift.changepoints:
        ax1.axvline(cp, color="purple", linestyle="--",
                    linewidth=1.5, alpha=0.7,
                    label="Change-point" if cp == drift.changepoints[0] else "")

    for idx in drift.drift_flags:
        ax1.plot(idx, drift.rolling_mean[idx], "r^",
                 markersize=6, alpha=0.7, zorder=5)

    ax1.set_title(f"Drift Detection  |  "
                  f"{len(drift.changepoints)} change-point(s)  |  "
                  f"{len(drift.drift_flags)} drift flag(s)")
    ax1.set_ylabel(measurement_name)
    ax1.legend(fontsize=7, loc="upper right", ncol=3)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Variance Monitoring ──────────────────────────────
    ax2.plot(variance.window_indices, variance.window_stds,
             "b-o", markersize=5, linewidth=1.5, label="Window σ")
    ax2.axhline(variance.baseline_std, color="green",
                linestyle="--", linewidth=1.2, label=f"Baseline σ={variance.baseline_std:.3f}")
    ax2.axhline(variance.baseline_std * 2, color="red",
                linestyle=":", linewidth=1.0, label="2× Threshold")

    for idx in variance.variance_flags:
        nearest = min(variance.window_indices,
                      key=lambda wi: abs(wi - idx))
        wi_pos  = variance.window_indices.index(nearest)
        ax2.plot(nearest, variance.window_stds[wi_pos],
                 "rv", markersize=10, zorder=5)

    lev_status = "⚠️ SIGNIFICANT" if variance.levene_significant else "✅ OK"
    ax2.set_title(f"Variance Monitor  |  "
                  f"Levene p={variance.levene_pvalue:.4f} {lev_status}  |  "
                  f"{len(variance.variance_flags)} spike(s)")
    ax2.set_ylabel("Window σ")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: ML Anomaly Scores ─────────────────────────────────
    ax3.plot(x, ml.anomaly_scores, "b-", linewidth=0.8,
             alpha=0.6, label="IF Score")
    threshold_score = np.percentile(ml.anomaly_scores, 5)
    ax3.axhline(threshold_score, color="red", linestyle="--",
                linewidth=1.2, label="Anomaly Threshold (5th pct)")

    for idx in ml.isolation_forest_flags:
        ax3.plot(idx, ml.anomaly_scores[idx], "r.",
                 markersize=8, alpha=0.5)
    for idx in ml.consensus_flags:
        ax3.plot(idx, ml.anomaly_scores[idx], "rv",
                 markersize=10, zorder=5, label="Consensus" if idx == ml.consensus_flags[0] else "")

    ax3.set_title(f"ML Anomaly Scores  |  "
                  f"IF={len(ml.isolation_forest_flags)}  |  "
                  f"SVM={len(ml.one_class_svm_flags)}  |  "
                  f"Consensus={len(ml.consensus_flags)}")
    ax3.set_ylabel("Anomaly Score")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Consensus Overlay ─────────────────────────────────
    ax4.plot(x, values, "b-o", markersize=3,
             linewidth=1.0, alpha=0.7, label="Values")

    if ml.consensus_flags:
        ax4.scatter(ml.consensus_flags,
                    values[ml.consensus_flags],
                    color="red", s=80, zorder=5,
                    label=f"Consensus Anomalies ({len(ml.consensus_flags)})")

    if drift.changepoints:
        for cp in drift.changepoints:
            ax4.axvline(cp, color="purple", linestyle="--",
                        linewidth=1.5, alpha=0.6,
                        label="Change-point" if cp == drift.changepoints[0] else "")

    ax4.set_title("Consensus Anomaly Overlay — All Methods Combined")
    ax4.set_ylabel(measurement_name)
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(True, alpha=0.3)

    # X-axis labels on bottom panel only
    step = max(1, len(batch_ids) // 20)
    ax4.set_xticks(x[::step])
    ax4.set_xticklabels(batch_ids[::step], rotation=45,
                        ha="right", fontsize=7)
    ax4.set_xlabel("Batch ID")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[anomaly_reporter] Dashboard saved to {save_path}")

    plt.show()