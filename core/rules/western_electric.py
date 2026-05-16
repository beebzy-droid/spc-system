import numpy as np
from dataclasses import dataclass, field


@dataclass
class RuleViolation:
    """Represents a single Western Electric rule violation."""
    rule_number: int
    rule_name: str
    description: str
    violated_indices: list[int]
    severity: str           # "warning", "action", "critical"
    root_cause_hint: str

    @property
    def count(self) -> int:
        return len(self.violated_indices)

    def summary(self):
        print(f"  Rule {self.rule_number} — {self.rule_name}")
        print(f"    Description : {self.description}")
        print(f"    Violations  : {self.count} point(s) at indices {self.violated_indices[:10]}")
        print(f"    Severity    : {self.severity.upper()}")
        print(f"    Likely Cause: {self.root_cause_hint}")


def _compute_zones(
    values: np.ndarray,
    center: float,
    sigma: float,
) -> np.ndarray:
    """
    Assign zone labels to each point:
    Zone A = beyond ±2σ (up to ±3σ)
    Zone B = between ±1σ and ±2σ
    Zone C = within ±1σ
    Returns array of zone distances in sigma units.
    """
    return (values - center) / sigma


def check_rule_1(z: np.ndarray) -> list[int]:
    """Rule 1: 1 point beyond ±3σ (out of control)."""
    return [i for i, v in enumerate(z) if abs(v) > 3.0]


def check_rule_2(z: np.ndarray) -> list[int]:
    """Rule 2: 9 consecutive points on same side of centerline."""
    violations = []
    n = len(z)
    for i in range(8, n):
        window = z[i-8:i+1]
        if all(v > 0 for v in window) or all(v < 0 for v in window):
            violations.append(i)
    return violations


def check_rule_3(z: np.ndarray) -> list[int]:
    """Rule 3: 6 points in a row steadily increasing or decreasing."""
    violations = []
    n = len(z)
    for i in range(5, n):
        window = z[i-5:i+1]
        if all(window[j] < window[j+1] for j in range(5)):
            violations.append(i)
        elif all(window[j] > window[j+1] for j in range(5)):
            violations.append(i)
    return violations


def check_rule_4(z: np.ndarray) -> list[int]:
    """Rule 4: 14 points in a row alternating up and down."""
    violations = []
    n = len(z)
    for i in range(13, n):
        window = z[i-13:i+1]
        alternating = all(
            (window[j] > window[j-1]) != (window[j+1] > window[j])
            for j in range(1, 13)
        )
        if alternating:
            violations.append(i)
    return violations


def check_rule_5(z: np.ndarray) -> list[int]:
    """Rule 5: 2 out of 3 consecutive points beyond ±2σ (same side)."""
    violations = []
    n = len(z)
    for i in range(2, n):
        window = z[i-2:i+1]
        beyond_pos = sum(1 for v in window if v > 2.0)
        beyond_neg = sum(1 for v in window if v < -2.0)
        if beyond_pos >= 2 or beyond_neg >= 2:
            violations.append(i)
    return violations


def check_rule_6(z: np.ndarray) -> list[int]:
    """Rule 6: 4 out of 5 consecutive points beyond ±1σ (same side)."""
    violations = []
    n = len(z)
    for i in range(4, n):
        window = z[i-4:i+1]
        beyond_pos = sum(1 for v in window if v > 1.0)
        beyond_neg = sum(1 for v in window if v < -1.0)
        if beyond_pos >= 4 or beyond_neg >= 4:
            violations.append(i)
    return violations


def check_rule_7(z: np.ndarray) -> list[int]:
    """Rule 7: 15 points in a row within ±1σ (stratification)."""
    violations = []
    n = len(z)
    for i in range(14, n):
        window = z[i-14:i+1]
        if all(abs(v) < 1.0 for v in window):
            violations.append(i)
    return violations


def check_rule_8(z: np.ndarray) -> list[int]:
    """Rule 8: 8 points beyond ±1σ on both sides (mixture)."""
    violations = []
    n = len(z)
    for i in range(7, n):
        window = z[i-7:i+1]
        if all(abs(v) > 1.0 for v in window):
            violations.append(i)
    return violations


# Rule metadata — descriptions, severity, root cause hints
RULE_METADATA = {
    1: {
        "name": "Beyond 3σ",
        "description": "1 point beyond ±3σ control limits",
        "severity": "critical",
        "root_cause_hint": "Special cause event — equipment failure, wrong material, measurement error, operator mistake",
    },
    2: {
        "name": "9-Point Run",
        "description": "9 consecutive points on same side of centerline",
        "severity": "action",
        "root_cause_hint": "Process mean shift — tool wear, raw material change, gradual temperature drift",
    },
    3: {
        "name": "6-Point Trend",
        "description": "6 points in a row steadily increasing or decreasing",
        "severity": "action",
        "root_cause_hint": "Gradual drift — tool wear, operator fatigue, reagent degradation, temperature creep",
    },
    4: {
        "name": "14-Point Alternating",
        "description": "14 points alternating up and down",
        "severity": "warning",
        "root_cause_hint": "Two alternating process streams — over-adjustment, two machines/operators alternating",
    },
    5: {
        "name": "2-of-3 Beyond 2σ",
        "description": "2 of 3 consecutive points beyond ±2σ on same side",
        "severity": "action",
        "root_cause_hint": "Process shift beginning — early warning of mean drift or sudden input change",
    },
    6: {
        "name": "4-of-5 Beyond 1σ",
        "description": "4 of 5 consecutive points beyond ±1σ on same side",
        "severity": "warning",
        "root_cause_hint": "Sustained process shift — systematic bias from calibration drift or material lot change",
    },
    7: {
        "name": "15-Point Stratification",
        "description": "15 consecutive points within ±1σ (too good)",
        "severity": "warning",
        "root_cause_hint": "Stratification — data mixed from multiple streams, measurement rounding, or incorrect sigma",
    },
    8: {
        "name": "8-Point Mixture",
        "description": "8 consecutive points beyond ±1σ on both sides",
        "severity": "warning",
        "root_cause_hint": "Mixture of two processes — two machines, two shifts, or bimodal input distribution",
    },
}


def run_all_rules(
    values: np.ndarray,
    center: float = None,
    sigma: float = None,
) -> list[RuleViolation]:
    """
    Run all 8 Western Electric rules on a data series.
    Returns list of RuleViolation objects for each triggered rule.
    """
    values = np.asarray(values, dtype=float)

    if center is None:
        center = np.mean(values)
    if sigma is None:
        mr = np.abs(np.diff(values))
        sigma = np.mean(mr) / 1.128      # sigma from moving range

    z = _compute_zones(values, center, sigma)

    rule_checks = {
        1: check_rule_1,
        2: check_rule_2,
        3: check_rule_3,
        4: check_rule_4,
        5: check_rule_5,
        6: check_rule_6,
        7: check_rule_7,
        8: check_rule_8,
    }

    violations = []
    for rule_num, check_fn in rule_checks.items():
        indices = check_fn(z)
        if indices:
            meta = RULE_METADATA[rule_num]
            violations.append(RuleViolation(
                rule_number=rule_num,
                rule_name=meta["name"],
                description=meta["description"],
                violated_indices=indices,
                severity=meta["severity"],
                root_cause_hint=meta["root_cause_hint"],
            ))

    return violations