"""
COHERENT Evaluation System

Defines immutable evaluation profiles for Static and Causal memory.
Implements Spec v1.0 Section 4.2 and 5.2.
"""

from dataclasses import dataclass
from typing import FrozenSet

@dataclass(frozen=True)
class StaticEvaluationProfile:
    """
    Evaluation Profile for Static Memory (E_S).
    """
    resonance_strength: float    # R
    validation_confidence: float # V
    generalization_score: float  # G
    redundancy_score: float      # D
    utilization_score: float     # U

@dataclass(frozen=True)
class CausalEvaluationProfile:
    """
    Evaluation Profile for Causal Memory (E_C).
    """
    causal_stability: float        # C_s
    decision_consistency: float    # C_d
    entropy_reduction: float       # C_e
    counterfactual_robustness: float # C_r
