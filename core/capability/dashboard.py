import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from core.capability.indices import CapabilityResult


def plot_capability_dashboard(
    result: CapabilityResult,
    values: np.ndarray,
    save_path: str = None,
):
    """
    Full capability dashboard with:
    - Process distribution vs spec limits
    - Cp/Cpk gauge
    - Capability indices table
    - Normal probability plot
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Process Capability Dashboard — {result.measurement_name}",
        fontsize=15, fontweight="bold"
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)
    ax_dist  = fig.add_subplot(gs[0, :2])   # Distribution plot (wide)
    ax_gauge = fig.add_subplot(gs[0, 2])    # Cpk gauge
    ax_table = fig.add_subplot(gs[1, :2])   # Indices table
    ax_prob  = fig.add_subplot(gs[1, 2])    # Normal probability plot

    # ── 1. Distribution Plot ───────────────────────────────────────
    x_min = min(result.lsl - 1, values.min() - 1)
    x_max = max(result.usl + 1, values.max() + 1)
    x_range = np.linspace(x_min, x_max, 500)

    # Fitted normal curve (within sigma)
    curve_within  = stats.norm.pdf(x_range, result.mean, result.std_within)
    curve_overall = stats.norm.pdf(x_range, result.mean, result.std_overall)

    ax_dist.hist(values, bins=30, density=True, alpha=0.4,
                 color="steelblue", edgecolor="white", label="Data")
    ax_dist.plot(x_range, curve_within,  "b-",  linewidth=2.0, label="Normal (within σ)")
    ax_dist.plot(x_range, curve_overall, "g--", linewidth=1.5, label="Normal (overall σ)")

    # Spec limit shading
    ax_dist.axvline(result.usl, color="red",    linestyle="--", linewidth=1.5, label="USL/LSL")
    ax_dist.axvline(result.lsl, color="red",    linestyle="--", linewidth=1.5)
    ax_dist.axvline(result.mean, color="green", linestyle="-",  linewidth=1.5, label="Mean")

    # Shade out-of-spec tails
    x_below = x_range[x_range < result.lsl]
    x_above = x_range[x_range > result.usl]
    ax_dist.fill_between(x_below, stats.norm.pdf(x_below, result.mean, result.std_overall),
                         alpha=0.3, color="red", label="OOS Region")
    ax_dist.fill_between(x_above, stats.norm.pdf(x_above, result.mean, result.std_overall),
                         alpha=0.3, color="red")

    ax_dist.set_title("Process Distribution vs Spec Limits")
    ax_dist.set_xlabel(result.measurement_name)
    ax_dist.set_ylabel("Density")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # ── 2. Cpk Gauge ───────────────────────────────────────────────
    gauge_val = min(result.cpk, 2.0)   # cap at 2.0 for display
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
    thresholds = [0, 1.0, 1.33, 2.0]
    labels_g = ["Not Capable\n(< 1.0)", "Marginal\n(1.0–1.33)", "Capable\n(> 1.33)"]

    theta_start = np.pi
    theta_end   = 0

    for i in range(3):
        t1 = theta_start - (thresholds[i]   / 2.0) * np.pi
        t2 = theta_start - (thresholds[i+1] / 2.0) * np.pi
        theta = np.linspace(t1, t2, 100)
        ax_gauge.fill_between(
            np.cos(theta), np.sin(theta),
            0.6 * np.cos(theta), 0.6 * np.sin(theta),
            color=colors[i], alpha=0.8
        )

    # Needle
    needle_angle = theta_start - (gauge_val / 2.0) * np.pi
    ax_gauge.annotate(
        "", xy=(0.75 * np.cos(needle_angle), 0.75 * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2.5)
    )

    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(-0.3, 1.2)
    ax_gauge.set_aspect("equal")
    ax_gauge.axis("off")
    ax_gauge.set_title(f"Cpk Gauge\n{result.cpk:.3f} — {result.status}", fontsize=10)
    ax_gauge.text(0, -0.2, f"Cpk = {result.cpk:.3f}",
                  ha="center", fontsize=12, fontweight="bold",
                  color=result.status_color)

    # ── 3. Indices Table ───────────────────────────────────────────
    ax_table.axis("off")

    table_data = [
        ["Index",     "Value",              "Benchmark",   "Status"],
        ["Cp",        f"{result.cp:.3f}",   "≥ 1.33",
         "✅" if result.cp >= 1.33 else "⚠️" if result.cp >= 1.0 else "❌"],
        ["Cpk",       f"{result.cpk:.3f}",  "≥ 1.33",
         "✅" if result.cpk >= 1.33 else "⚠️" if result.cpk >= 1.0 else "❌"],
        ["Cpu",       f"{result.cpu:.3f}",  "≥ 1.33", ""],
        ["Cpl",       f"{result.cpl:.3f}",  "≥ 1.33", ""],
        ["Pp",        f"{result.pp:.3f}",   "≥ 1.33",
         "✅" if result.pp >= 1.33 else "⚠️" if result.pp >= 1.0 else "❌"],
        ["Ppk",       f"{result.ppk:.3f}",  "≥ 1.33",
         "✅" if result.ppk >= 1.33 else "⚠️" if result.ppk >= 1.0 else "❌"],
        ["Sigma Level", f"{result.sigma_level:.2f}σ", "≥ 4σ", ""],
        ["PPM Expected", f"{result.ppm_expected:,.0f}", "< 6,210", ""],
        ["N",         f"{result.n}",         "—",           ""],
        ["Mean",      f"{result.mean:.4f}",  f"Target: {(result.usl+result.lsl)/2:.1f}", ""],
    ]

    col_widths = [0.2, 0.2, 0.25, 0.15]
    col_positions = [0.05, 0.25, 0.45, 0.70]
    row_height = 0.08

    for r_idx, row in enumerate(table_data):
        y = 1.0 - r_idx * row_height
        bg_color = "#f0f0f0" if r_idx == 0 else ("white" if r_idx % 2 == 0 else "#fafafa")
        ax_table.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.06), 0.96, row_height,
            boxstyle="round,pad=0.01", facecolor=bg_color, edgecolor="lightgray"
        ))
        for c_idx, cell in enumerate(row):
            weight = "bold" if r_idx == 0 else "normal"
            ax_table.text(col_positions[c_idx], y - 0.02, cell,
                          fontsize=9, fontweight=weight, va="center")

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.set_title("Capability Indices Summary", fontsize=10, fontweight="bold")

    # ── 4. Normal Probability Plot ─────────────────────────────────
    (osm, osr), (slope, intercept, r) = stats.probplot(values, dist="norm")
    ax_prob.plot(osm, osr,        "bo", markersize=3, alpha=0.6, label="Data")
    ax_prob.plot(osm, slope * np.array(osm) + intercept,
                 "r-", linewidth=1.5, label=f"Fit (R²={r**2:.3f})")
    ax_prob.set_title("Normal Probability Plot")
    ax_prob.set_xlabel("Theoretical Quantiles")
    ax_prob.set_ylabel("Sample Values")
    ax_prob.legend(fontsize=8)
    ax_prob.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[dashboard] Chart saved to {save_path}")

    plt.show()
    return fig