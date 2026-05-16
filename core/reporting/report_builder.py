import os
import base64
from datetime import datetime
from core.reporting.alert_engine import AlertSummary, ALERT_TIERS
from core.capability.indices import CapabilityResult
from core.rules.western_electric import RuleViolation
from core.anomaly.drift_detector import DriftResult
from core.anomaly.variance_monitor import VarianceResult
from core.anomaly.ml_detector import MLAnomalyResult


def _img_to_base64(filepath: str) -> str:
    """Convert image file to base64 string for embedding in HTML."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _img_tag(filepath: str, width: str = "100%") -> str:
    """Return an HTML img tag with embedded base64 image."""
    b64 = _img_to_base64(filepath)
    if not b64:
        return f'<p style="color:gray;">[ Chart not found: {filepath} ]</p>'
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{width}; border-radius:8px; '
            f'box-shadow:0 2px 8px rgba(0,0,0,0.15);" />')


def build_html_report(
    alert_summary: AlertSummary,
    capability: dict[str, CapabilityResult],
    violations: dict[str, list[RuleViolation]],
    drift: dict[str, DriftResult],
    variance: dict[str, VarianceResult],
    ml: MLAnomalyResult,
    chart_dir: str = "data/processed",
    output_path: str = "data/processed/spc_report.html",
):
    """Generate a full HTML SPC report with embedded charts."""

    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = alert_summary.overall_status
    color  = ALERT_TIERS[status]["color"]
    icon   = ALERT_TIERS[status]["icon"]

    # ── Capability rows ────────────────────────────────────────────
    cap_rows = ""
    for name, cap in capability.items():
        row_color = (
            "#ffe0e0" if cap.cpk < 1.0 else
            "#fff3cd" if cap.cpk < 1.33 else
            "#e6f4ea"
        )
        cap_rows += f"""
        <tr style="background:{row_color}">
            <td><b>{name}</b></td>
            <td>{cap.cp:.3f}</td>
            <td><b>{cap.cpk:.3f}</b></td>
            <td>{cap.pp:.3f}</td>
            <td>{cap.ppk:.3f}</td>
            <td>{cap.sigma_level:.2f}σ</td>
            <td>{cap.ppm_expected:,.0f}</td>
            <td>{cap.status}</td>
        </tr>"""

    # ── WE Rules rows ──────────────────────────────────────────────
    rule_rows = ""
    severity_colors = {
        "critical": "#ffe0e0",
        "action":   "#fff3cd",
        "warning":  "#fffbe6",
    }
    for name, rule_list in violations.items():
        if not rule_list:
            rule_rows += f"""
            <tr style="background:#e6f4ea">
                <td>{name}</td>
                <td colspan="4">✅ No violations detected</td>
            </tr>"""
        for v in rule_list:
            rule_rows += f"""
            <tr style="background:{severity_colors.get(v.severity, 'white')}">
                <td>{name}</td>
                <td>Rule {v.rule_number} — {v.rule_name}</td>
                <td>{v.severity.upper()}</td>
                <td>{v.count} point(s)</td>
                <td style="font-size:0.85em">{v.root_cause_hint}</td>
            </tr>"""

    # ── Alert rows ─────────────────────────────────────────────────
    alert_rows = ""
    tier_bg = {
        "CRITICAL": "#ffe0e0",
        "ACTION":   "#fff3cd",
        "WARNING":  "#fffbe6",
        "OK":       "#e6f4ea",
    }
    for alert in sorted(
        alert_summary.alerts,
        key=lambda a: ALERT_TIERS[a.tier]["level"],
        reverse=True
    ):
        alert_rows += f"""
        <tr style="background:{tier_bg.get(alert.tier, 'white')}">
            <td>{alert.icon} <b>{alert.tier}</b></td>
            <td>{alert.source}</td>
            <td>{alert.measurement}</td>
            <td>{alert.message}</td>
            <td style="font-size:0.85em">{alert.root_cause_hint}</td>
            <td style="font-size:0.85em">{alert.action}</td>
        </tr>"""

    # ── Chart sections ─────────────────────────────────────────────
    chart_sections = ""
    measurements = list(capability.keys())

    for col in measurements:
        imr_img     = _img_tag(f"{chart_dir}/imr_chart.png")
        xbar_img    = _img_tag(f"{chart_dir}/xbar_r_chart.png")
        cusum_img   = _img_tag(f"{chart_dir}/cusum_ewma_chart.png")
        cap_img     = _img_tag(f"{chart_dir}/capability_{col}.png")
        rules_img   = _img_tag(f"{chart_dir}/we_rules_{col}.png")
        anomaly_img = _img_tag(f"{chart_dir}/anomaly_{col}.png")

        chart_sections += f"""
        <div class="section">
            <h2>📊 {col} — Charts</h2>
            <div class="chart-grid">
                <div class="chart-card">
                    <h3>I-MR Control Chart</h3>
                    {imr_img}
                </div>
                <div class="chart-card">
                    <h3>X̄-R Control Chart</h3>
                    {xbar_img}
                </div>
                <div class="chart-card">
                    <h3>CUSUM & EWMA</h3>
                    {cusum_img}
                </div>
                <div class="chart-card">
                    <h3>Process Capability Dashboard</h3>
                    {cap_img}
                </div>
                <div class="chart-card">
                    <h3>Western Electric Rules</h3>
                    {rules_img}
                </div>
                <div class="chart-card">
                    <h3>Anomaly Detection</h3>
                    {anomaly_img}
                </div>
            </div>
        </div>"""

    # ── Full HTML ──────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPC Report — {ts}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f4f6f9;
            color: #333;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            padding: 30px 40px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 8px; }}
        .header p  {{ opacity: 0.8; font-size: 0.95em; }}
        .status-badge {{
            display: inline-block;
            background: {color};
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
            margin-top: 12px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-top: 4px solid #ddd;
        }}
        .kpi-card.critical {{ border-top-color: #8B0000; }}
        .kpi-card.action   {{ border-top-color: #d9534f; }}
        .kpi-card.warning  {{ border-top-color: #f0ad4e; }}
        .kpi-card.ok       {{ border-top-color: #5cb85c; }}
        .kpi-card .number  {{
            font-size: 2.5em;
            font-weight: bold;
            color: #1a1a2e;
        }}
        .kpi-card .label   {{
            font-size: 0.85em;
            color: #666;
            margin-top: 4px;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            font-size: 1.3em;
            color: #1a1a2e;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f0f0f0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        th {{
            background: #1a1a2e;
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 9px 12px;
            border-bottom: 1px solid #f0f0f0;
        }}
        tr:hover {{ filter: brightness(0.97); }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .chart-card {{
            border: 1px solid #e8e8e8;
            border-radius: 8px;
            padding: 16px;
            background: #fafafa;
        }}
        .chart-card h3 {{
            font-size: 0.95em;
            color: #555;
            margin-bottom: 10px;
        }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 0.85em;
            margin-top: 30px;
            padding: 20px;
        }}
        @media (max-width: 900px) {{
            .kpi-grid    {{ grid-template-columns: repeat(2, 1fr); }}
            .chart-grid  {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>

<!-- Header -->
<div class="header">
    <h1>🏭 AI-Assisted SPC Report</h1>
    <p>Batch Manufacturing — Statistical Process Control</p>
    <p>Generated: {ts}</p>
    <div class="status-badge">{icon} OVERALL STATUS: {status}</div>
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
    <div class="kpi-card critical">
        <div class="number">{alert_summary.critical_count}</div>
        <div class="label">🔴 Critical Alerts</div>
    </div>
    <div class="kpi-card action">
        <div class="number">{alert_summary.action_count}</div>
        <div class="label">🟠 Action Alerts</div>
    </div>
    <div class="kpi-card warning">
        <div class="number">{alert_summary.warning_count}</div>
        <div class="label">🟡 Warning Alerts</div>
    </div>
    <div class="kpi-card ok">
        <div class="number">{len(capability)}</div>
        <div class="label">📊 Measurements Analyzed</div>
    </div>
</div>

<!-- Capability Summary -->
<div class="section">
    <h2>📈 Process Capability Summary</h2>
    <table>
        <tr>
            <th>Measurement</th>
            <th>Cp</th>
            <th>Cpk</th>
            <th>Pp</th>
            <th>Ppk</th>
            <th>Sigma Level</th>
            <th>PPM Expected</th>
            <th>Status</th>
        </tr>
        {cap_rows}
    </table>
</div>

<!-- Western Electric Rules -->
<div class="section">
    <h2>📋 Western Electric Rules Summary</h2>
    <table>
        <tr>
            <th>Measurement</th>
            <th>Rule</th>
            <th>Severity</th>
            <th>Violations</th>
            <th>Root Cause Hint</th>
        </tr>
        {rule_rows}
    </table>
</div>

<!-- Alert Log -->
<div class="section">
    <h2>🚨 Full Alert Log</h2>
    <table>
        <tr>
            <th>Tier</th>
            <th>Source</th>
            <th>Measurement</th>
            <th>Message</th>
            <th>Root Cause Hint</th>
            <th>Required Action</th>
        </tr>
        {alert_rows}
    </table>
</div>

<!-- Charts -->
{chart_sections}

<!-- Footer -->
<div class="footer">
    <p>AI-Assisted SPC System — Auto-generated report</p>
    <p>Generated: {ts} | Phases 1–6 Complete</p>
</div>

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[report_builder] HTML report saved to {output_path}")
    return output_path