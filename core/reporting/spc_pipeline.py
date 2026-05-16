import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from core.ingestion import load_csv
from core.preprocessor import preprocess
from core.schema import SpecLimits
from core.charts.imr_chart import compute_imr_limits, plot_imr_chart
from core.charts.xbar_r_chart import plot_xbar_r_chart
from core.charts.cusum_ewma import plot_cusum_ewma
from core.capability.indices import compute_all_capabilities
from core.capability.dashboard import plot_capability_dashboard
from core.rules.western_electric import run_all_rules
from core.rules.rule_reporter import print_rule_report, plot_rule_violations
from core.anomaly.drift_detector import run_drift_detection
from core.anomaly.variance_monitor import run_variance_monitoring
from core.anomaly.ml_detector import run_ml_detection
from core.anomaly.anomaly_reporter import print_anomaly_report, plot_anomaly_dashboard
from core.reporting.alert_engine import generate_alerts
from core.reporting.report_builder import build_html_report


def run_full_spc_pipeline(
    csv_path: str,
    measurement_cols: list[str],
    process_param_cols: list[str],
    spec_limits: dict[str, SpecLimits],
    chart_dir: str = "data/processed",
    report_path: str = "data/processed/spc_report.html",
    audit_path: str = "data/processed/audit_trail.json",
    show_charts: bool = True,
):
    """
    Run the complete AI-Assisted SPC pipeline:
    Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
    """
    print("\n" + "="*65)
    print("  AI-ASSISTED SPC SYSTEM — FULL PIPELINE")
    print("="*65)

    # ── Phase 1: Load & Preprocess ────────────────────────────────
    print("\n📦 Phase 1 — Data Layer")
    df = load_csv(csv_path)
    df = preprocess(df, measurement_cols, spec_limits)
    batch_ids = df["batch_id"].tolist()
    print(f"  ✅ {len(df)} clean records ready")

    # ── Phase 2: Control Charts ───────────────────────────────────
    print("\n📊 Phase 2 — Control Charts")
    for col in measurement_cols:
        values = df[col].values
        spec   = spec_limits[col]

        plot_imr_chart(
            values=values, batch_ids=batch_ids,
            measurement_name=col,
            usl=spec.usl, lsl=spec.lsl,
            save_path=f"{chart_dir}/imr_chart.png",
        ) if show_charts else None

        subgroup_size = 5
        n = len(values) - (len(values) % subgroup_size)
        subgroups = values[:n].reshape(-1, subgroup_size).tolist()
        plot_xbar_r_chart(
            subgroups=subgroups,
            subgroup_labels=batch_ids[:n:subgroup_size],
            measurement_name=col,
            save_path=f"{chart_dir}/xbar_r_chart.png",
        ) if show_charts else None

        plot_cusum_ewma(
            values=values, batch_ids=batch_ids,
            measurement_name=col, target=spec.target,
            save_path=f"{chart_dir}/cusum_ewma_chart.png",
        ) if show_charts else None

    print("  ✅ Control charts generated")

    # ── Phase 3: Capability ───────────────────────────────────────
    print("\n📈 Phase 3 — Process Capability")
    capability = compute_all_capabilities(df, spec_limits)
    for name, cap in capability.items():
        plot_capability_dashboard(
            result=cap,
            values=df[name].values,
            save_path=f"{chart_dir}/capability_{name}.png",
        ) if show_charts else None
        cap.summary()
    print("  ✅ Capability indices computed")

    # ── Phase 4: Western Electric Rules ──────────────────────────
    print("\n📋 Phase 4 — Western Electric Rules")
    violations = {}
    for col in measurement_cols:
        values = df[col].values
        i_limits, _ = compute_imr_limits(values)
        sigma = (i_limits.ucl - i_limits.center_line) / 3
        v = run_all_rules(values, i_limits.center_line, sigma)
        violations[col] = v
        print_rule_report(v, col)
        plot_rule_violations(
            values=values, batch_ids=batch_ids,
            violations=v, measurement_name=col,
            center=i_limits.center_line, sigma=sigma,
            save_path=f"{chart_dir}/we_rules_{col}.png",
        ) if show_charts else None
    print("  ✅ Western Electric rules evaluated")

    # ── Phase 5: Anomaly Detection ────────────────────────────────
    print("\n🤖 Phase 5 — Anomaly Detection")
    drift_results    = {}
    variance_results = {}
    for col in measurement_cols:
        values = df[col].values
        drift_results[col]    = run_drift_detection(values, col)
        variance_results[col] = run_variance_monitoring(values, col)

    ml_result = run_ml_detection(df, measurement_cols, process_param_cols)

    for col in measurement_cols:
        print_anomaly_report(drift_results[col], variance_results[col], ml_result)
        plot_anomaly_dashboard(
            values=df[col].values, batch_ids=batch_ids,
            drift=drift_results[col],
            variance=variance_results[col],
            ml=ml_result,
            measurement_name=col,
            save_path=f"{chart_dir}/anomaly_{col}.png",
        ) if show_charts else None
    print("  ✅ Anomaly detection complete")

    # ── Phase 6: Alerts & Reporting ───────────────────────────────
    print("\n🚨 Phase 6 — Alerts & Reporting")
    alert_summary = generate_alerts(
        capability=capability,
        violations=violations,
        drift=drift_results,
        variance=variance_results,
        ml=ml_result,
    )
    alert_summary.print_summary()
    alert_summary.save_audit_trail(audit_path)

    report_file = build_html_report(
        alert_summary=alert_summary,
        capability=capability,
        violations=violations,
        drift=drift_results,
        variance=variance_results,
        ml=ml_result,
        chart_dir=chart_dir,
        output_path=report_path,
    )

    # ── Final Summary ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  🎉 PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f"  📄 HTML Report  : {report_file}")
    print(f"  📋 Audit Trail  : {audit_path}")
    print(f"  🔴 Critical     : {alert_summary.critical_count}")
    print(f"  🟠 Action       : {alert_summary.action_count}")
    print(f"  🟡 Warning      : {alert_summary.warning_count}")
    print(f"  Overall Status  : {ALERT_TIERS[alert_summary.overall_status]['icon']} "
          f"{alert_summary.overall_status}")
    print(f"{'='*65}")

    return alert_summary, report_file


from core.reporting.alert_engine import ALERT_TIERS