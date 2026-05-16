import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core.charts import ControlLimits


# Control chart constant for n=2 (moving range of 2)
D4 = 3.267
D3 = 0.0
d2 = 1.128


def compute_imr_limits(values: np.ndarray) -> tuple[ControlLimits, ControlLimits]:
    """
    Compute control limits for Individuals (I) and Moving Range (MR) charts.
    Returns (i_limits, mr_limits)
    """
    values = np.asarray(values, dtype=float)

    # Moving ranges
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr)

    # Individuals limits
    x_bar = np.mean(values)
    sigma_est = mr_bar / d2

    i_limits = ControlLimits(
        center_line=x_bar,
        ucl=x_bar + 3 * sigma_est,
        lcl=x_bar - 3 * sigma_est,
        sigma_1=x_bar + 1 * sigma_est,
        sigma_2=x_bar + 2 * sigma_est,
    )

    # Moving range limits
    mr_limits = ControlLimits(
        center_line=mr_bar,
        ucl=D4 * mr_bar,
        lcl=D3 * mr_bar,
        sigma_1=mr_bar + 1 * (mr_bar / d2),
        sigma_2=mr_bar + 2 * (mr_bar / d2),
    )

    return i_limits, mr_limits


def detect_violations(
    values: np.ndarray,
    limits: ControlLimits
) -> dict[str, list[int]]:
    """Flag points beyond UCL/LCL (Rule 1 violations only — full rules in Phase 4)."""
    violations = {"beyond_ucl": [], "beyond_lcl": []}
    for i, v in enumerate(values):
        if v > limits.ucl:
            violations["beyond_ucl"].append(i)
        if v < limits.lcl:
            violations["beyond_lcl"].append(i)
    return violations


def plot_imr_chart(
    values: np.ndarray,
    batch_ids: list[str],
    measurement_name: str = "Measurement",
    usl: float = None,
    lsl: float = None,
    save_path: str = None,
):
    """
    Plot a full I-MR control chart with:
    - Zone shading (1σ, 2σ, 3σ)
    - UCL/LCL lines
    - Spec limit overlays
    - Violation highlights
    """
    values = np.asarray(values, dtype=float)
    mr = np.abs(np.diff(values))
    mr_padded = np.concatenate([[np.nan], mr])

    i_limits, mr_limits = compute_imr_limits(values)
    i_violations = detect_violations(values, i_limits)
    mr_violations = detect_violations(mr, mr_limits)

    x = np.arange(len(values))
    x_mr = np.arange(1, len(values))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"I-MR Control Chart — {measurement_name}", fontsize=14, fontweight="bold")

    # ── Individuals Chart ──────────────────────────────────────────
    # Zone shading
    ax1.axhspan(i_limits.center_line, i_limits.sigma_1,  alpha=0.08, color="green")
    ax1.axhspan(i_limits.sigma_1,     i_limits.sigma_2,  alpha=0.08, color="yellow")
    ax1.axhspan(i_limits.sigma_2,     i_limits.ucl,      alpha=0.08, color="red")
    ax1.axhspan(i_limits.center_line, i_limits.center_line - (i_limits.sigma_1 - i_limits.center_line), alpha=0.08, color="green")
    ax1.axhspan(i_limits.center_line - (i_limits.sigma_1 - i_limits.center_line),
                i_limits.center_line - (i_limits.sigma_2 - i_limits.center_line), alpha=0.08, color="yellow")
    ax1.axhspan(i_limits.center_line - (i_limits.sigma_2 - i_limits.center_line),
                i_limits.lcl, alpha=0.08, color="red")

    # Control lines
    ax1.axhline(i_limits.ucl,         color="red",    linestyle="--", linewidth=1.2, label="UCL/LCL")
    ax1.axhline(i_limits.lcl,         color="red",    linestyle="--", linewidth=1.2)
    ax1.axhline(i_limits.center_line, color="green",  linestyle="-",  linewidth=1.5, label="CL")
    ax1.axhline(i_limits.sigma_2,     color="orange", linestyle=":",  linewidth=0.8, label="±2σ")
    ax1.axhline(i_limits.center_line - (i_limits.sigma_2 - i_limits.center_line),
                color="orange", linestyle=":", linewidth=0.8)
    ax1.axhline(i_limits.sigma_1,     color="blue",   linestyle=":",  linewidth=0.8, label="±1σ")
    ax1.axhline(i_limits.center_line - (i_limits.sigma_1 - i_limits.center_line),
                color="blue", linestyle=":", linewidth=0.8)

    # Spec limits
    if usl is not None:
        ax1.axhline(usl, color="purple", linestyle="-.", linewidth=1.2, label="USL/LSL")
    if lsl is not None:
        ax1.axhline(lsl, color="purple", linestyle="-.", linewidth=1.2)

    # Data line
    ax1.plot(x, values, "b-o", markersize=4, linewidth=1.2, label="Individual")

    # Violations
    for idx in i_violations["beyond_ucl"] + i_violations["beyond_lcl"]:
        ax1.plot(idx, values[idx], "rv", markersize=10, zorder=5)

    ax1.set_ylabel(measurement_name)
    ax1.set_title("Individuals (I) Chart")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Moving Range Chart ─────────────────────────────────────────
    ax2.axhline(mr_limits.ucl,         color="red",   linestyle="--", linewidth=1.2, label="UCL")
    ax2.axhline(mr_limits.center_line, color="green", linestyle="-",  linewidth=1.5, label="MR̄")
    ax2.axhline(mr_limits.lcl,         color="red",   linestyle="--", linewidth=1.2, label="LCL")

    ax2.plot(x_mr, mr, "b-o", markersize=4, linewidth=1.2, label="Moving Range")

    for idx in mr_violations["beyond_ucl"]:
        ax2.plot(idx, mr[idx - 1], "rv", markersize=10, zorder=5)

    ax2.set_ylabel("Moving Range")
    ax2.set_title("Moving Range (MR) Chart")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # X-axis labels
    step = max(1, len(batch_ids) // 20)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(batch_ids[::step], rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Batch ID")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[imr_chart] Chart saved to {save_path}")

    plt.show()
    return i_limits, mr_limits