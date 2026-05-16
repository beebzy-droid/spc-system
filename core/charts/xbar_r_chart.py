import numpy as np
import matplotlib.pyplot as plt
from core.charts import ControlLimits


# Control chart constants by subgroup size
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
D3 = {2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


def compute_xbar_r_limits(
    subgroups: list[list[float]]
) -> tuple[ControlLimits, ControlLimits]:
    """
    Compute X̄-R control limits from subgroups.
    Each subgroup is a list of measurements (n = 2 to 10).
    """
    n = len(subgroups[0])
    assert 2 <= n <= 10, "Subgroup size must be between 2 and 10 for X̄-R chart."

    means = np.array([np.mean(sg) for sg in subgroups])
    ranges = np.array([np.max(sg) - np.min(sg) for sg in subgroups])

    x_bar_bar = np.mean(means)
    r_bar = np.mean(ranges)

    xbar_limits = ControlLimits(
        center_line=x_bar_bar,
        ucl=x_bar_bar + A2[n] * r_bar,
        lcl=x_bar_bar - A2[n] * r_bar,
        sigma_1=x_bar_bar + (A2[n] * r_bar) / 3,
        sigma_2=x_bar_bar + (A2[n] * r_bar) * 2 / 3,
    )

    r_limits = ControlLimits(
        center_line=r_bar,
        ucl=D4[n] * r_bar,
        lcl=D3[n] * r_bar,
        sigma_1=r_bar + (D4[n] * r_bar) / 3,
        sigma_2=r_bar + (D4[n] * r_bar) * 2 / 3,
    )

    return xbar_limits, r_limits


def plot_xbar_r_chart(
    subgroups: list[list[float]],
    subgroup_labels: list[str] = None,
    measurement_name: str = "Measurement",
    save_path: str = None,
):
    """Plot X̄-R control chart."""
    means  = np.array([np.mean(sg) for sg in subgroups])
    ranges = np.array([np.max(sg) - np.min(sg) for sg in subgroups])
    xbar_limits, r_limits = compute_xbar_r_limits(subgroups)

    x = np.arange(len(subgroups))
    labels = subgroup_labels or [str(i) for i in x]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"X̄-R Control Chart — {measurement_name}", fontsize=14, fontweight="bold")

    for ax, data, limits, title, ylabel in [
        (ax1, means,  xbar_limits, "X̄ (Mean) Chart",  "Subgroup Mean"),
        (ax2, ranges, r_limits,    "R (Range) Chart",  "Subgroup Range"),
    ]:
        ax.axhline(limits.ucl,         color="red",    linestyle="--", linewidth=1.2, label="UCL/LCL")
        ax.axhline(limits.lcl,         color="red",    linestyle="--", linewidth=1.2)
        ax.axhline(limits.center_line, color="green",  linestyle="-",  linewidth=1.5, label="CL")
        ax.axhline(limits.sigma_2,     color="orange", linestyle=":",  linewidth=0.8, label="±2σ")
        ax.axhline(limits.sigma_1,     color="blue",   linestyle=":",  linewidth=0.8, label="±1σ")
        ax.plot(x, data, "b-o", markersize=4, linewidth=1.2)

        for i, v in enumerate(data):
            if v > limits.ucl or v < limits.lcl:
                ax.plot(i, v, "rv", markersize=10, zorder=5)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    step = max(1, len(labels) // 20)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Subgroup")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[xbar_r_chart] Chart saved to {save_path}")
    plt.show()
    return xbar_limits, r_limits