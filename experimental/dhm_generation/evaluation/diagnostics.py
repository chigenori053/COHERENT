"""
Diagnostics Module

Automatic diagnostics for experiment runs.
Classifies outcomes as SUCCESS, WARN, or INVALID.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class DiagnosticResult:
    status: str  # SUCCESS, WARN, INVALID
    flags: List[str]
    details: Dict[str, Any]

class ExperimentDiagnostics:
    def __init__(self, config):
        self.config = config

    def check_result(self, 
                     target_symbol: str, 
                     top_result: str, 
                     top_score: float, 
                     margin: float,
                     final_norm: float,
                     is_ddim_deterministic: bool = True) -> DiagnosticResult:
        
        flags = []
        status = "SUCCESS"
        
        # --- INVALID Handling (Spec Violations) ---
        
        # L1: Normalization Failure
        if abs(final_norm - 1.0) > self.config.norm_tolerance:
            status = "INVALID"
            flags.append("L1_NORM_VIOLATION")

        # L2: Determinism check (passed via argument if known)
        if not is_ddim_deterministic and self.config.mode == "ddim":
             status = "INVALID"
             flags.append("L2_DDIM_NON_DETERMINISTIC")

        if status == "INVALID":
            return DiagnosticResult(status, flags, {"norm": final_norm})

        # --- WARN Handling (Quality Issues) ---
        
        # P1: Incorrect Prediction
        if target_symbol != top_result:
            status = "WARN"
            flags.append("P1_INCORRECT_SYMBOL")
            
        # P2: Low Margin
        if margin < self.config.resonance_margin_threshold:
            if status == "SUCCESS": status = "WARN" # Don't downgrade INVALID
            flags.append("P2_LOW_MARGIN")
            
        return DiagnosticResult(status, flags, {
            "target": target_symbol,
            "predicted": top_result,
            "score": top_score,
            "margin": margin
        })
