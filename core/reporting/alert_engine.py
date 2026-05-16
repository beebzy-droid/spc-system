import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from core.rules.western_electric import RuleViolation
from core.capability.indices import CapabilityResult
from core.anomaly.drift_detector import DriftResult
from core.anomaly.variance_monitor import VarianceResult
from core.anomaly.ml_detector import MLAnomalyResult


# Alert tier definitions
ALERT_TIERS = {
    "CRITICAL": {
        "level": 3,
        "color": "#8B0000",
        "icon": "🔴",
        "action": "STOP LINE — Immediate investigation required",
    },
    "ACTION": {
        "level": 2,
        "color": "#d9534f",
        "icon": "🟠",
        "action": "INVESTIGATE — Engineer notification required",
    },
    "WARNING": {
        "level": 1,
        "color": "#f0ad4e",
        "icon": "🟡",
        "action": "MONITOR — Increased sampling recommended",
    },
    "OK": {
        "level": 0,
        "color": "#5cb85c",
        "icon": "✅",
        "action": "Process in control",
    },
}


@dataclass
class Alert:
    """Single alert record."""
    timestamp: str
    tier: str                   # CRITICAL, ACTION, WARNING, OK
    source: str                 # Which module triggered this
    measurement: str
    message: str
    batch_indices: list[int]
    root_cause_hint: str = ""

    @property
    def icon(self) -> str:
        return ALERT_TIERS[self.tier]["icon"]

    @property
    def color(self) -> str:
        return ALERT_TIERS[self.tier]["color"]

    @property
    def action(self) -> str:
        return ALERT_TIERS[self.tier]["action"]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "tier": self.tier,
            "source": self.source,
            "measurement": self.measurement,
            "message": self.message,
            "batch_indices": self.batch_indices,
            "root_cause_hint": self.root_cause_hint,
            "action": self.action,
        }


@dataclass
class AlertSummary:
    """Collection of all alerts for a full batch run."""
    alerts: list[Alert] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.alerts if a.tier == "CRITICAL")

    @property
    def action_count(self) -> int:
        return sum(1 for a in self.alerts if a.tier == "ACTION")

    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.alerts if a.tier == "WARNING")

    @property
    def overall_status(self) -> str:
        if self.critical_count > 0:
            return "CRITICAL"
        elif self.action_count > 0:
            return "ACTION"
        elif self.warning_count > 0:
            return "WARNING"
        return "OK"

    def add(self, alert: Alert):
        self.alerts.append(alert)

    def print_summary(self):
        tier = self.overall_status
        info = ALERT_TIERS[tier]
        print(f"\n{'='*65}")
        print(f"  ALERT SUMMARY  —  Generated: {self.generated_at}")
        print(f"{'='*65}")
        print(f"  Overall Status : {info['icon']} {tier}")
        print(f"  Required Action: {info['action']}")
        print(f"{'─'*65}")
        print(f"  🔴 Critical : {self.critical_count}")
        print(f"  🟠 Action   : {self.action_count}")
        print(f"  🟡 Warning  : {self.warning_count}")
        print(f"{'─'*65}")
        for alert in sorted(self.alerts,
                            key=lambda a: ALERT_TIERS[a.tier]["level"],
                            reverse=True):
            print(f"\n  {alert.icon} [{alert.tier}] {alert.source} — {alert.measurement}")
            print(f"     {alert.message}")
            if alert.root_cause_hint:
                print(f"     Hint   : {alert.root_cause_hint}")
            print(f"     Action : {alert.action}")
        print(f"{'='*65}")

    def save_audit_trail(self, filepath: str):
        """Save full audit trail as JSON for compliance."""
        audit = {
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "summary": {
                "critical": self.critical_count,
                "action": self.action_count,
                "warning": self.warning_count,
            },
            "alerts": [a.to_dict() for a in self.alerts],
        }
        with open(filepath, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"[alert_engine] Audit trail saved to {filepath}")


def generate_alerts(
    capability: dict[str, CapabilityResult],
    violations: dict[str, list[RuleViolation]],
    drift: dict[str, DriftResult],
    variance: dict[str, VarianceResult],
    ml: MLAnomalyResult,
) -> AlertSummary:
    """
    Generate tiered alerts from all Phase outputs.
    Combines capability, WE rules, drift, variance, and ML results.
    """
    summary = AlertSummary()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Capability alerts ──────────────────────────────────────────
    for name, cap in capability.items():
        if cap.cpk < 1.0:
            summary.add(Alert(
                timestamp=ts, tier="CRITICAL",
                source="Capability (Cpk)",
                measurement=name,
                message=f"Cpk={cap.cpk:.3f} below 1.0 — "
                        f"process producing defects. "
                        f"PPM={cap.ppm_expected:,.0f}",
                batch_indices=[],
                root_cause_hint="Reduce process spread or re-center mean toward target",
            ))
        elif cap.cpk < 1.33:
            summary.add(Alert(
                timestamp=ts, tier="WARNING",
                source="Capability (Cpk)",
                measurement=name,
                message=f"Cpk={cap.cpk:.3f} marginal (1.0–1.33) — "
                        f"process at risk",
                batch_indices=[],
                root_cause_hint="Monitor closely — improve centering to reach Cpk ≥ 1.33",
            ))

    # ── Western Electric rule alerts ───────────────────────────────
    severity_map = {
        "critical": "CRITICAL",
        "action":   "ACTION",
        "warning":  "WARNING",
    }
    for name, rule_list in violations.items():
        for v in rule_list:
            summary.add(Alert(
                timestamp=ts,
                tier=severity_map[v.severity],
                source=f"WE Rule {v.rule_number} ({v.rule_name})",
                measurement=name,
                message=f"{v.description} — "
                        f"{v.count} violation point(s)",
                batch_indices=v.violated_indices,
                root_cause_hint=v.root_cause_hint,
            ))

    # ── Drift alerts ───────────────────────────────────────────────
    for name, dr in drift.items():
        if dr.has_changepoints:
            summary.add(Alert(
                timestamp=ts, tier="ACTION",
                source="Drift Detection (PELT)",
                measurement=name,
                message=f"{len(dr.changepoints)} change-point(s) detected "
                        f"at indices {dr.changepoints}",
                batch_indices=dr.changepoints,
                root_cause_hint="Process level shift — check material lot, "
                                "equipment settings, or operator change",
            ))
        if dr.has_drift:
            summary.add(Alert(
                timestamp=ts, tier="WARNING",
                source="Drift Detection (Rolling Mean)",
                measurement=name,
                message=f"{len(dr.drift_flags)} rolling mean drift flag(s)",
                batch_indices=dr.drift_flags,
                root_cause_hint="Gradual mean drift — monitor trend and "
                                "check for tool wear or temperature creep",
            ))

    # ── Variance alerts ────────────────────────────────────────────
    for name, vr in variance.items():
        if vr.levene_significant:
            summary.add(Alert(
                timestamp=ts, tier="ACTION",
                source="Variance Monitor (Levene)",
                measurement=name,
                message=f"Significant variance shift detected "
                        f"(p={vr.levene_pvalue:.4f})",
                batch_indices=[],
                root_cause_hint="Process variance changed — check for "
                                "new material lot, equipment wear, or "
                                "environmental changes",
            ))
        if vr.variance_flags:
            summary.add(Alert(
                timestamp=ts, tier="WARNING",
                source="Variance Monitor (Sliding Window)",
                measurement=name,
                message=f"{len(vr.variance_flags)} high-variance "
                        f"window(s) detected",
                batch_indices=vr.variance_flags,
                root_cause_hint="Localized variance spike — check for "
                                "intermittent equipment issues",
            ))

    # ── ML consensus alerts ────────────────────────────────────────
    if ml.consensus_flags:
        summary.add(Alert(
            timestamp=ts, tier="CRITICAL",
            source="ML Consensus (IF + OC-SVM)",
            measurement=ml.measurement_name,
            message=f"{len(ml.consensus_flags)} batch(es) flagged as "
                    f"anomalous by both ML models: "
                    f"{ml.consensus_flags}",
            batch_indices=ml.consensus_flags,
            root_cause_hint="Multivariate anomaly — batch profile is "
                            "statistically abnormal across multiple "
                            "measurements and process parameters",
        ))

    return summary