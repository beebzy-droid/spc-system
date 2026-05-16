import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ControlLimits:
    """Stores control limits for any chart type."""
    center_line: float
    ucl: float          # Upper Control Limit
    lcl: float          # Lower Control Limit
    sigma_1: float      # ±1σ zone boundary
    sigma_2: float      # ±2σ zone boundary

    def summary(self):
        print(f"  Center Line : {self.center_line:.4f}")
        print(f"  UCL (3σ)    : {self.ucl:.4f}")
        print(f"  LCL (3σ)    : {self.lcl:.4f}")
        print(f"  ±2σ boundary: {self.sigma_2:.4f}")
        print(f"  ±1σ boundary: {self.sigma_1:.4f}")