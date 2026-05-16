import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core.rules.western_electric import RuleViolation, run_all_rules


SEVERITY_COLORS = {
    "warning":  "#f0ad4e",   # orange
    "action":   "#d9534f",   # red
    "critical": "#8B0000",   # dark red
}

SEVERITY_ORDER = {"critical": 0, "action": 1, "warning": 2}


def print_rule_report(
    violations: list[RuleViolation],
    measurement_name: str = "Measurement",
):
    """Print a formatted rule violation report to console."""
    print(f"\n{'='*60}")
    print(f"  Western Electric Rules Report — {measurement_name}")
    print(f"{'='*60}")

    if not violations:
        print("  ✅ No violations detected. Process is in control.")
        print(f"{'='*60}")
        return

    # Sort by severity
    sorted_v = sorted(violations, key=lambda v: SEVERITY_ORDER[v.severity])

    total_violations = sum(v.count for v in sorted_v)
    print(f"  Total rules triggered : {len(sorted_v)}")
    print(f"  Total violation points: {total_violations}")
    print(f"{'─'*60}")

    for v in sorted_v:
        icon = "🔴" if v.severity == "critical" else "🟠" if v.severity == "action" else "🟡"
        print(f"\n  {icon} Rule {v.rule_number} — {v.rule_name} [{v.severity.upper()}]")
        print(f"     {v.description}")
        print(f"     Points    : {v.count} violation(s)")
        print(f"     Indices   : {v.violated_indices[:10]}")
        print(f"     Root Cause: {v.root_cause_hint}")

    print(f"\n{'='*60}")


def plot_rule_violations(
    values: np.ndarray,
    batch_ids: list[str],
    violations: list[RuleViolation],
    measurement_name: str = "Measurement",
    center: float = None,
    sigma: float = None,
    save_path: str = None,
):
    """
    Plot control chart with all Western Electric violations highlighted
    by rule number and severity color.
    """
    values = np.asarray(values, dtype=float)

    if center is None:
        center = np.mean(values)
    if sigma is None:
        mr = np.abs(np.diff(values))
        sigma = np.mean(mr) / 1.128

    ucl = center + 3 * sigma
    lcl = center - 3 * sigma
    s1  = center + 1 * sigma
    s2  = center + 2 * sigma
    s1n = center - 1 * sigma
    s2n = center - 2 * sigma

    x = np.arange(len(values))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        f"Western Electric Rules — {measurement_name}",
        fontsize=14, fontweight="bold"
    )

    # ── Zone shading ───────────────────────────────────────────────
    ax1.axhspan(s2,  ucl, alpha=0.08, color="red")
    ax1.axhspan(s1,  s2,  alpha=0.08, color="orange")
    ax1.axhspan(center, s1, alpha=0.08, color="green")
    ax1.axhspan(s1n, center, alpha=0.08, color="green")
    ax1.axhspan(s2n, s1n, alpha=0.08, color="orange")
    ax1.axhspan(lcl, s2n, alpha=0.08, color="red")

    # ── Control lines ──────────────────────────────────────────────
    for y, color, style, lw, label in [
        (ucl,    "red",    "--", 1.5, "UCL/LCL"),
        (lcl,    "red",    "--", 1.5, None),
        (center, "green",  "-",  1.8, "CL"),
        (s2,     "orange", ":",  1.0, "±2σ"),
        (s2n,    "orange", ":",  1.0, None),
        (s1,     "blue",   ":",  0.8, "±1σ"),
        (s1n,    "blue",   ":",  0.8, None),
    ]:
        ax1.axhline(y, color=color, linestyle=style, linewidth=lw,
                    label=label if label else "")

    # ── Data line ──────────────────────────────────────────────────
    ax1.plot(x, values, "b-o", markersize=3, linewidth=1.0,
             alpha=0.7, label="Values", zorder=3)

    # ── Violation markers ──────────────────────────────────────────
    plotted_rules = {}
    for v in violations:
        color = SEVERITY_COLORS[v.severity]
        for idx in v.violated_indices:
            if idx < len(values):
                ax1.plot(idx, values[idx], "v", color=color,
                         markersize=12, zorder=5, alpha=0.85)
                ax1.annotate(
                    f"R{v.rule_number}",
                    xy=(idx, values[idx]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=6.5, color=color, fontweight="bold"
                )
        if v.rule_number not in plotted_rules:
            plotted_rules[v.rule_number] = mpatches.Patch(
                color=color,
                label=f"Rule {v.rule_number}: {v.rule_name}"
            )

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    rule_patches = list(plotted_rules.values())
    ax1.legend(handles=handles + rule_patches,
               loc="upper right", fontsize=7, ncol=2)
    ax1.set_ylabel(measurement_name)
    ax1.set_title("Control Chart with Rule Violations")
    ax1.grid(True, alpha=0.3)

    # ── Rule summary bar chart ─────────────────────────────────────
    if violations:
        sorted_v = sorted(violations, key=lambda v: v.rule_number)
        rule_labels = [f"R{v.rule_number}" for v in sorted_v]
        rule_counts = [v.count for v in sorted_v]
        bar_colors  = [SEVERITY_COLORS[v.severity] for v in sorted_v]

        bars = ax2.bar(rule_labels, rule_counts, color=bar_colors,
                       edgecolor="white", linewidth=0.8)
        for bar, count in zip(bars, rule_counts):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1, str(count),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2.set_title("Violation Count by Rule")
        ax2.set_ylabel("# Violations")
        ax2.set_xlabel("Rule")
        ax2.grid(True, alpha=0.3, axis="y")

        legend_patches = [
            mpatches.Patch(color=SEVERITY_COLORS["critical"], label="Critical"),
            mpatches.Patch(color=SEVERITY_COLORS["action"],   label="Action"),
            mpatches.Patch(color=SEVERITY_COLORS["warning"],  label="Warning"),
        ]
        ax2.legend(handles=legend_patches, fontsize=8, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "✅ No violations detected",
                 ha="center", va="center", fontsize=12,
                 transform=ax2.transAxes)
        ax2.axis("off")

    # X-axis labels
    step = max(1, len(batch_ids) // 20)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(batch_ids[::step], rotation=45,
                        ha="right", fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[rule_reporter] Chart saved to {save_path}")

    plt.show()