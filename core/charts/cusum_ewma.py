import numpy as np
import matplotlib.pyplot as plt


def compute_cusum(
    values: np.ndarray,
    target: float = None,
    k: float = 0.5,
    h: float = 4.0,
) -> dict:
    """
    Compute two-sided CUSUM.
    k = allowance (slack), h = decision interval (threshold).
    """
    values = np.asarray(values, dtype=float)
    if target is None:
        target = np.mean(values)

    sigma = np.std(values, ddof=1)
    cusum_pos = np.zeros(len(values))
    cusum_neg = np.zeros(len(values))

    for i in range(1, len(values)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - target) / sigma - k)
        cusum_neg[i] = max(0, cusum_neg[i-1] - (values[i] - target) / sigma - k)

    threshold = h
    signals_pos = np.where(cusum_pos > threshold)[0].tolist()
    signals_neg = np.where(cusum_neg > threshold)[0].tolist()

    return {
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        "threshold": threshold,
        "signals_pos": signals_pos,
        "signals_neg": signals_neg,
    }


def compute_ewma(
    values: np.ndarray,
    lambda_: float = 0.2,
    L: float = 3.0,
) -> dict:
    """
    Compute EWMA statistic and control limits.
    lambda_ = smoothing factor, L = sigma multiplier.
    """
    values = np.asarray(values, dtype=float)
    mu = np.mean(values)
    sigma = np.std(values, ddof=1)

    ewma = np.zeros(len(values))
    ewma[0] = values[0]
    for i in range(1, len(values)):
        ewma[i] = lambda_ * values[i] + (1 - lambda_) * ewma[i-1]

    ucl = mu + L * sigma * np.sqrt(lambda_ / (2 - lambda_))
    lcl = mu - L * sigma * np.sqrt(lambda_ / (2 - lambda_))

    signals = np.where((ewma > ucl) | (ewma < lcl))[0].tolist()

    return {
        "ewma": ewma,
        "ucl": ucl,
        "lcl": lcl,
        "center_line": mu,
        "signals": signals,
    }


def plot_cusum_ewma(
    values: np.ndarray,
    batch_ids: list[str],
    measurement_name: str = "Measurement",
    target: float = None,
    save_path: str = None,
):
    """Plot CUSUM and EWMA charts side by side."""
    cusum = compute_cusum(values, target=target)
    ewma  = compute_ewma(values)

    x = np.arange(len(values))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"CUSUM & EWMA Charts — {measurement_name}", fontsize=14, fontweight="bold")

    # CUSUM
    ax1.plot(x, cusum["cusum_pos"], "b-o", markersize=3, linewidth=1.2, label="CUSUM+")
    ax1.plot(x, cusum["cusum_neg"], "g-o", markersize=3, linewidth=1.2, label="CUSUM−")
    ax1.axhline(cusum["threshold"], color="red", linestyle="--", linewidth=1.2, label=f"H={cusum['threshold']}")
    for idx in cusum["signals_pos"]:
        ax1.plot(idx, cusum["cusum_pos"][idx], "rv", markersize=10, zorder=5)
    for idx in cusum["signals_neg"]:
        ax1.plot(idx, cusum["cusum_neg"][idx], "r^", markersize=10, zorder=5)
    ax1.set_title("CUSUM Chart")
    ax1.set_ylabel("Cumulative Sum")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # EWMA
    ax2.plot(x, values, "b-o", markersize=3, linewidth=0.8, alpha=0.4, label="Raw")
    ax2.plot(x, ewma["ewma"], "b-", linewidth=1.8, label="EWMA")
    ax2.axhline(ewma["ucl"],         color="red",   linestyle="--", linewidth=1.2, label="UCL/LCL")
    ax2.axhline(ewma["lcl"],         color="red",   linestyle="--", linewidth=1.2)
    ax2.axhline(ewma["center_line"], color="green", linestyle="-",  linewidth=1.5, label="CL")
    for idx in ewma["signals"]:
        ax2.plot(idx, ewma["ewma"][idx], "rv", markersize=10, zorder=5)
    ax2.set_title("EWMA Chart")
    ax2.set_ylabel(measurement_name)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    step = max(1, len(batch_ids) // 20)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(batch_ids[::step], rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Batch ID")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[cusum_ewma] Chart saved to {save_path}")
    plt.show()
    return cusum, ewma