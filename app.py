import sys
import os
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Assisted SPC System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .phase-header {
        background: #f0f4f8;
        padding: 10px 16px;
        border-left: 4px solid #1a1a2e;
        border-radius: 4px;
        margin: 16px 0 8px 0;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .status-critical { color: #8B0000; font-weight: bold; }
    .status-warning  { color: #f0ad4e; font-weight: bold; }
    .status-ok       { color: #5cb85c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏭 AI-Assisted SPC System</h1>
    <p>Statistical Process Control for Batch Manufacturing</p>
    <p style="opacity:0.7; font-size:0.85em;">
        Phases 1–6 — Control Charts · Cp/Cpk · Western Electric Rules ·
        ML Anomaly Detection · Automated Alerts
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=60)
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Use sample data", "Upload your own CSV"]
    )

    uploaded_file = None
    if data_source == "Upload your own CSV":
        uploaded_file = st.file_uploader(
            "Upload batch CSV",
            type=["csv"],
            help="Must contain: batch_id, timestamp, measurement columns"
        )

    st.markdown("---")
    st.subheader("📏 Spec Limits — Measurement 1")
    m1_usl = st.number_input("USL", value=53.0, step=0.5, key="m1_usl")
    m1_lsl = st.number_input("LSL", value=47.0, step=0.5, key="m1_lsl")
    m1_target = st.number_input("Target", value=50.0, step=0.5, key="m1_target")

    st.markdown("---")
    st.subheader("📏 Spec Limits — Measurement 2")
    m2_usl = st.number_input("USL", value=106.0, step=0.5, key="m2_usl")
    m2_lsl = st.number_input("LSL", value=94.0, step=0.5, key="m2_lsl")
    m2_target = st.number_input("Target", value=100.0, step=0.5, key="m2_target")

    st.markdown("---")
    st.subheader("🤖 ML Settings")
    contamination = st.slider(
        "Anomaly contamination rate",
        min_value=0.01, max_value=0.15,
        value=0.05, step=0.01,
        help="Expected % of anomalous batches"
    )

    st.markdown("---")
    run_button = st.button(
        "▶ Run Full SPC Pipeline",
        type="primary",
        use_container_width=True
    )


# ── Main Content ──────────────────────────────────────────────────
if not run_button:
    # Welcome screen
    st.markdown("### 👋 Welcome to the AI-Assisted SPC System")
    st.markdown("""
    This system runs a **full 6-phase SPC pipeline** on your batch data:

    | Phase | Module | What It Does |
    |---|---|---|
    | 1 | Data Layer | Ingest, validate, preprocess batch records |
    | 2 | Control Charts | I-MR, X̄-R, CUSUM, EWMA charts |
    | 3 | Capability | Cp, Cpk, Pp, Ppk with dashboard |
    | 4 | WE Rules | 8 Western Electric rules engine |
    | 5 | ML Anomaly | Isolation Forest + One-Class SVM |
    | 6 | Alerts | Tiered alerts + audit trail |

    **👈 Configure settings in the sidebar and click Run to start.**
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **4 Chart Types**\nI-MR · X̄-R · CUSUM · EWMA")
    with col2:
        st.info("🤖 **2 ML Models**\nIsolation Forest · One-Class SVM")
    with col3:
        st.info("🚨 **3 Alert Tiers**\nWarning · Action · Critical")

else:
    # ── Load Data ─────────────────────────────────────────────────
    try:
        from core.ingestion import load_csv, dataframe_to_dataset
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
        from core.anomaly.anomaly_reporter import plot_anomaly_dashboard
        from core.reporting.alert_engine import generate_alerts, ALERT_TIERS

        MEASUREMENT_COLS   = ["measurement_1", "measurement_2"]
        PROCESS_PARAM_COLS = ["temperature", "pressure"]
        SPEC_LIMITS = {
            "measurement_1": SpecLimits(
                "measurement_1",
                usl=m1_usl, lsl=m1_lsl, target=m1_target
            ),
            "measurement_2": SpecLimits(
                "measurement_2",
                usl=m2_usl, lsl=m2_lsl, target=m2_target
            ),
        }

        # Load data
        with st.spinner("📦 Phase 1 — Loading and preprocessing data..."):
            if data_source == "Upload your own CSV" and uploaded_file:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                df = load_csv(tmp_path)
            else:
                df = load_csv("sample_data/batches.csv")

            df = preprocess(df, MEASUREMENT_COLS, SPEC_LIMITS)
            batch_ids = df["batch_id"].tolist()

        # ── Phase 1 Summary ───────────────────────────────────────
        st.markdown(
            '<div class="phase-header">📦 Phase 1 — Data Layer</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Batches", len(df))
        col2.metric("Measurements", len(MEASUREMENT_COLS))
        col3.metric(
            "OOS Batches (M1)",
            int(df["measurement_1_oos"].sum())
        )
        col4.metric(
            "OOS Batches (M2)",
            int(df["measurement_2_oos"].sum())
        )
        st.success(f"✅ {len(df)} clean records ready for analysis")

        # ── Phase 2 — Control Charts ──────────────────────────────
        st.markdown(
            '<div class="phase-header">📊 Phase 2 — Control Charts</div>',
            unsafe_allow_html=True
        )

        for col in MEASUREMENT_COLS:
            with st.spinner(f"Generating charts for {col}..."):
                values = df[col].values
                spec   = SPEC_LIMITS[col]

                tab1, tab2, tab3 = st.tabs([
                    f"📈 I-MR — {col}",
                    f"📊 X̄-R — {col}",
                    f"📉 CUSUM/EWMA — {col}"
                ])

                with tab1:
                    fig, _ = plt.subplots()
                    plt.close(fig)
                    i_lim, mr_lim = plot_imr_chart(
                        values=values,
                        batch_ids=batch_ids,
                        measurement_name=col,
                        usl=spec.usl, lsl=spec.lsl,
                    )
                    st.pyplot(plt.gcf())
                    plt.close("all")

                with tab2:
                    subgroup_size = 5
                    n = len(values) - (len(values) % subgroup_size)
                    subgroups = values[:n].reshape(
                        -1, subgroup_size
                    ).tolist()
                    plot_xbar_r_chart(
                        subgroups=subgroups,
                        subgroup_labels=batch_ids[:n:subgroup_size],
                        measurement_name=col,
                    )
                    st.pyplot(plt.gcf())
                    plt.close("all")

                with tab3:
                    plot_cusum_ewma(
                        values=values,
                        batch_ids=batch_ids,
                        measurement_name=col,
                        target=spec.target,
                    )
                    st.pyplot(plt.gcf())
                    plt.close("all")

        # ── Phase 3 — Capability ──────────────────────────────────
        st.markdown(
            '<div class="phase-header">📈 Phase 3 — Process Capability</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Computing capability indices..."):
            capability = compute_all_capabilities(df, SPEC_LIMITS)

        cap_col1, cap_col2 = st.columns(2)
        for idx, (name, cap) in enumerate(capability.items()):
            col_ref = cap_col1 if idx == 0 else cap_col2
            with col_ref:
                st.markdown(f"**{name}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cp",  f"{cap.cp:.3f}")
                c2.metric("Cpk", f"{cap.cpk:.3f}")
                c3.metric("Sigma", f"{cap.sigma_level:.2f}σ")
                c4.metric("PPM", f"{cap.ppm_expected:,.0f}")

                if cap.cpk < 1.0:
                    st.error(f"❌ NOT CAPABLE — Cpk={cap.cpk:.3f}")
                elif cap.cpk < 1.33:
                    st.warning(f"⚠️ MARGINAL — Cpk={cap.cpk:.3f}")
                else:
                    st.success(f"✅ CAPABLE — Cpk={cap.cpk:.3f}")

                plot_capability_dashboard(
                    result=cap, values=df[name].values
                )
                st.pyplot(plt.gcf())
                plt.close("all")

        # ── Phase 4 — Western Electric Rules ─────────────────────
        st.markdown(
            '<div class="phase-header">📋 Phase 4 — Western Electric Rules</div>',
            unsafe_allow_html=True
        )

        violations = {}
        for col in MEASUREMENT_COLS:
            values   = df[col].values
            i_limits, _ = compute_imr_limits(values)
            sigma    = (i_limits.ucl - i_limits.center_line) / 3
            v        = run_all_rules(values, i_limits.center_line, sigma)
            violations[col] = v

            with st.expander(
                f"📋 {col} — {len(v)} rule(s) triggered", expanded=True
            ):
                if not v:
                    st.success("✅ No violations detected")
                else:
                    for rule in v:
                        if rule.severity == "critical":
                            st.error(
                                f"🔴 Rule {rule.rule_number} — "
                                f"{rule.rule_name} | "
                                f"{rule.count} violation(s) | "
                                f"{rule.root_cause_hint}"
                            )
                        elif rule.severity == "action":
                            st.warning(
                                f"🟠 Rule {rule.rule_number} — "
                                f"{rule.rule_name} | "
                                f"{rule.count} violation(s) | "
                                f"{rule.root_cause_hint}"
                            )
                        else:
                            st.info(
                                f"🟡 Rule {rule.rule_number} — "
                                f"{rule.rule_name} | "
                                f"{rule.count} violation(s) | "
                                f"{rule.root_cause_hint}"
                            )

                plot_rule_violations(
                    values=values,
                    batch_ids=batch_ids,
                    violations=v,
                    measurement_name=col,
                    center=i_limits.center_line,
                    sigma=sigma,
                )
                st.pyplot(plt.gcf())
                plt.close("all")

        # ── Phase 5 — Anomaly Detection ───────────────────────────
        st.markdown(
            '<div class="phase-header">🤖 Phase 5 — ML Anomaly Detection</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Running ML anomaly detection..."):
            drift_results    = {}
            variance_results = {}
            for col in MEASUREMENT_COLS:
                values = df[col].values
                drift_results[col] = run_drift_detection(
                    values, col
                )
                variance_results[col] = run_variance_monitoring(
                    values, col
                )
            ml_result = run_ml_detection(
                df, MEASUREMENT_COLS,
                PROCESS_PARAM_COLS,
                contamination=contamination
            )

        for col in MEASUREMENT_COLS:
            with st.expander(
                f"🤖 {col} — Anomaly Results", expanded=True
            ):
                dr = drift_results[col]
                vr = variance_results[col]

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Change-points", len(dr.changepoints))
                a2.metric("Drift Flags",   len(dr.drift_flags))
                a3.metric(
                    "Levene p-value",
                    f"{vr.levene_pvalue:.4f}"
                )
                a4.metric(
                    "ML Consensus",
                    len(ml_result.consensus_flags)
                )

                plot_anomaly_dashboard(
                    values=df[col].values,
                    batch_ids=batch_ids,
                    drift=dr,
                    variance=vr,
                    ml=ml_result,
                    measurement_name=col,
                )
                st.pyplot(plt.gcf())
                plt.close("all")

        # ── Phase 6 — Alerts ──────────────────────────────────────
        st.markdown(
            '<div class="phase-header">🚨 Phase 6 — Alert Summary</div>',
            unsafe_allow_html=True
        )

        alert_summary = generate_alerts(
            capability=capability,
            violations=violations,
            drift=drift_results,
            variance=variance_results,
            ml=ml_result,
        )

        status = alert_summary.overall_status
        if status == "CRITICAL":
            st.error(
                f"🔴 OVERALL STATUS: CRITICAL — "
                f"Immediate investigation required"
            )
        elif status == "ACTION":
            st.warning(
                f"🟠 OVERALL STATUS: ACTION — "
                f"Engineer notification required"
            )
        elif status == "WARNING":
            st.warning(
                f"🟡 OVERALL STATUS: WARNING — "
                f"Increased sampling recommended"
            )
        else:
            st.success("✅ OVERALL STATUS: OK — Process in control")

        # Alert counts
        a1, a2, a3 = st.columns(3)
        a1.metric("🔴 Critical", alert_summary.critical_count)
        a2.metric("🟠 Action",   alert_summary.action_count)
        a3.metric("🟡 Warning",  alert_summary.warning_count)

        # Full alert table
        st.markdown("#### 📋 Full Alert Log")
        alert_data = []
        for alert in sorted(
            alert_summary.alerts,
            key=lambda a: ALERT_TIERS[a.tier]["level"],
            reverse=True
        ):
            alert_data.append({
                "Tier":        alert.tier,
                "Source":      alert.source,
                "Measurement": alert.measurement,
                "Message":     alert.message,
                "Root Cause":  alert.root_cause_hint,
                "Action":      alert.action,
            })

        if alert_data:
            st.dataframe(
                pd.DataFrame(alert_data),
                use_container_width=True,
                hide_index=True,
            )

        st.success(
            "🎉 Full SPC Pipeline Complete! "
            "All 6 phases executed successfully."
        )

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        st.exception(e)