import numpy as np
from dataclasses import dataclass
from core.schema import SpecLimits


@dataclass
class CapabilityResult:
    """Stores all capability indices for one measurement."""
    measurement_name: str
    usl: float
    lsl: float
    mean: float
    std_within: float       # Short-term sigma (from moving range)
    std_overall: float      # Long-term sigma (from all data)
    n: int                  # Sample size

    # Short-term indices
    cp: float = 0.0
    cpl: float = 0.0
    cpu: float = 0.0
    cpk: float = 0.0

    # Long-term indices
    pp: float = 0.0
    ppl: float = 0.0
    ppu: float = 0.0
    ppk: float = 0.0

    # Yield estimates
    sigma_level: float = 0.0
    ppm_expected: float = 0.0     # Defects per million

    @property
    def status(self) -> str:
        """Color-coded status based on Cpk."""
        if self.cpk >= 1.33:
            return "✅ CAPABLE"
        elif self.cpk >= 1.0:
            return "⚠️  MARGINAL"
        else:
            return "❌ NOT CAPABLE"

    @property
    def status_color(self) -> str:
        if self.cpk >= 1.33:
            return "green"
        elif self.cpk >= 1.0:
            return "orange"
        else:
            return "red"

    def summary(self):
        print(f"\n{'='*50}")
        print(f"  Capability Report — {self.measurement_name}")
        print(f"{'='*50}")
        print(f"  N               : {self.n}")
        print(f"  Mean            : {self.mean:.4f}")
        print(f"  USL / LSL       : {self.usl} / {self.lsl}")
        print(f"  Spec Range      : {self.usl - self.lsl:.4f}")
        print(f"  σ (within)      : {self.std_within:.4f}")
        print(f"  σ (overall)     : {self.std_overall:.4f}")
        print(f"{'─'*50}")
        print(f"  Cp              : {self.cp:.3f}")
        print(f"  Cpl             : {self.cpl:.3f}")
        print(f"  Cpu             : {self.cpu:.3f}")
        print(f"  Cpk             : {self.cpk:.3f}  ← {self.status}")
        print(f"{'─'*50}")
        print(f"  Pp              : {self.pp:.3f}")
        print(f"  Ppl             : {self.ppl:.3f}")
        print(f"  Ppu             : {self.ppu:.3f}")
        print(f"  Ppk             : {self.ppk:.3f}")
        print(f"{'─'*50}")
        print(f"  Sigma Level     : {self.sigma_level:.2f}σ")
        print(f"  Expected PPM    : {self.ppm_expected:,.0f}")
        print(f"{'='*50}")


def compute_capability(
    values: np.ndarray,
    spec: SpecLimits,
) -> CapabilityResult:
    """
    Compute full capability indices for one measurement column.

    Short-term sigma (within) — estimated from moving range (like control charts)
    Long-term sigma (overall) — computed from all data points
    """
    from scipy import stats

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    mean = np.mean(values)

    # Short-term sigma — from moving range (consistent with control charts)
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr)
    d2 = 1.128
    std_within = mr_bar / d2

    # Long-term sigma — overall standard deviation
    std_overall = np.std(values, ddof=1)

    # ── Short-term indices (Cp, Cpk) ──────────────────────────────
    cp  = (spec.usl - spec.lsl) / (6 * std_within)
    cpu = (spec.usl - mean)     / (3 * std_within)
    cpl = (mean - spec.lsl)     / (3 * std_within)
    cpk = min(cpu, cpl)

    # ── Long-term indices (Pp, Ppk) ───────────────────────────────
    pp  = (spec.usl - spec.lsl) / (6 * std_overall)
    ppu = (spec.usl - mean)     / (3 * std_overall)
    ppl = (mean - spec.lsl)     / (3 * std_overall)
    ppk = min(ppu, ppl)

    # ── Sigma level & PPM ─────────────────────────────────────────
    sigma_level = cpk * 3
    z_usl = (spec.usl - mean) / std_overall
    z_lsl = (mean - spec.lsl) / std_overall
    ppm_expected = (stats.norm.sf(z_usl) + stats.norm.cdf(-z_lsl)) * 1_000_000

    return CapabilityResult(
        measurement_name=spec.measurement_name,
        usl=spec.usl,
        lsl=spec.lsl,
        mean=mean,
        std_within=std_within,
        std_overall=std_overall,
        n=n,
        cp=cp, cpl=cpl, cpu=cpu, cpk=cpk,
        pp=pp, ppl=ppl, ppu=ppu, ppk=ppk,
        sigma_level=sigma_level,
        ppm_expected=ppm_expected,
    )


def compute_all_capabilities(
    df,
    spec_limits: dict,
) -> dict[str, CapabilityResult]:
    """Run capability analysis on all measurement columns."""
    results = {}
    for name, spec in spec_limits.items():
        if name in df.columns:
            results[name] = compute_capability(df[name].values, spec)
    return results