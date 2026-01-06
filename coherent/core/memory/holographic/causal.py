"""
Causal Holographic Memory (Layer-C)

Decision-Theoretic Arbiter for Memory Transitions.
Implements Spec v1.0.
"""

import numpy as np
import math
import time
from typing import List, Tuple, Any, Dict, Optional
from enum import Enum, auto
from dataclasses import dataclass, field
from .base import HolographicMemoryBase
from .evaluation import CausalEvaluationProfile

class Action(Enum):
    PROMOTE = auto()
    RETAIN = auto()
    SUPPRESS = auto()
    DEFER_REVIEW = auto()

@dataclass
class DecisionState:
    resonance_score: float
    margin: float
    repetition_count: int
    entropy_estimate: float
    memory_origin: str # 'Dynamic' or 'Static'
    historical_conflict_rate: float = 0.0

@dataclass
class CausalTrace:
    state_snapshot: DecisionState
    probability: float
    eu_scores: Dict[str, float]
    final_decision: str
    timestamp: float = field(default_factory=time.time)

class CausalHolographicMemory(HolographicMemoryBase):
    def __init__(self):
        self._transitions: List[Dict[str, Any]] = []
        self._decision_logs: List[CausalTrace] = []
        
        # 5.2 Utility Table (State x Action)
        # States: VALID, INVALID
        self.UTILITY_MATRIX = {
            Action.PROMOTE:      (1.0, -10.0),
            Action.RETAIN:       (0.5, -0.5),
            Action.SUPPRESS:     (-1.0, 0.5),
            Action.DEFER_REVIEW: (0.0,  0.0)
        }
        
        # 6. Weights for P(VALID | S)
        self.W = {
            'resonance': 10.0,
            'margin': 5.0,
            'entropy': -0.5, # Negative impact
            'conflict': -2.0, # Negative impact
            'bias': -4.0 # Base bias to be conservative
        }

    def evaluate_decision(self, state: DecisionState) -> Action:
        """
        Evaluate a memory state and return the optimal action.
        Follows Spec v1.0 Section 7.
        """
        # 1. Estimate Probability
        p_valid = self._estimate_probability(state)
        
        # 2. Compute Expected Utility
        eu_scores = {}
        for action in Action:
            u_valid, u_invalid = self.UTILITY_MATRIX[action]
            eu = p_valid * u_valid + (1.0 - p_valid) * u_invalid
            eu_scores[action.name] = eu
            
        # 3. Select Action (ArgMax)
        # Sort by utility desc
        sorted_actions = sorted(eu_scores.items(), key=lambda x: x[1], reverse=True)
        best_action_name = sorted_actions[0][0]
        final_action = Action[best_action_name]
        
        # 4. Log Trace
        trace = CausalTrace(
            state_snapshot=state,
            probability=p_valid,
            eu_scores=eu_scores,
            final_decision=best_action_name
        )
        self._log_trace(trace)
        
        return final_action

    def _estimate_probability(self, s: DecisionState) -> float:
        """
        P = sigmoid(w1*res + w2*margin - w3*entropy - w4*conflict + bias)
        """
        logit = (
            self.W['resonance'] * s.resonance_score +
            self.W['margin'] * s.margin +
            self.W['entropy'] * s.entropy_estimate + # W['entropy'] is negative
            self.W['conflict'] * s.historical_conflict_rate + # W['conflict'] is negative
            self.W['bias']
        )
        return 1.0 / (1.0 + math.exp(-logit))

    def _log_trace(self, trace: CausalTrace):
        """Append trace to internal log. In production, flush to disk."""
        self._decision_logs.append(trace)

    # --- Legacy / Base Methods ---

    def add(self, state: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        raise NotImplementedError("Use evaluate_decision() or add_transition()")

    def add_transition(self, source: np.ndarray, target: np.ndarray, context: np.ndarray = None, profile: CausalEvaluationProfile = None) -> None:
        """
        Register a transition.
        Stored Entry: { 'source': vec, 'target': vec, 'context': vec, 'profile': profile }
        """
        if profile is None:
            profile = CausalEvaluationProfile(1.0, 1.0, 0.1, 1.0) # Default robust

        entry = {
            'source': self.normalize(source),
            'target': self.normalize(target),
            'context': self.normalize(context) if context is not None else None,
            'profile': profile
        }
        self._transitions.append(entry)

    def evaluate_relation(self,
                        failure_rate: float,
                        decision_entropy: float,
                        entropy_delta: float,
                        counterfactual_failure: float) -> CausalEvaluationProfile:
        """
        Evaluate a causal relation.
        Calculates Cs, Cd, Ce, Cr.
        """
        # C_s: Causal Stability = 1 - failure_rate
        c_s = 1.0 - failure_rate
        
        # C_d: Decision Consistency = 1 - H(A). 
        # Provided decision_entropy is H(A)
        c_d = 1.0 - decision_entropy
        
        # C_e: Entropy Reduction = delta
        c_e = entropy_delta
        
        # C_r: Counterfactual Robustness = 1 - failure_rate(counterfactual)
        c_r = 1.0 - counterfactual_failure
        
        return CausalEvaluationProfile(c_s, c_d, c_e, c_r)

    def check_update_rules(self, profile: CausalEvaluationProfile) -> str:
        """
        Returns 'STORE', 'DEACTIVATE', 'REMOVE', 'IGNORE'.
        Spec 5.4.
        """
        # Thresholds
        TH_Cs = 0.8
        TH_Cd = 0.7
        
        # Store if: C_s > θ_Cs ∧ C_d > θ_Cd ∧ C_e > 0
        if (profile.causal_stability > TH_Cs and 
            profile.decision_consistency > TH_Cd and 
            profile.entropy_reduction > 0):
            return 'STORE'
            
        # Deactivate if: C_r decreases persistently (Simplification: if C_r is low)
        if profile.counterfactual_robustness < 0.5:
            return 'DEACTIVATE'
            
        # Remove if: C_e ≈ 0
        if abs(profile.entropy_reduction) < 0.01:
            return 'REMOVE'
            
        return 'IGNORE'

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[Any, float]]:
        """
        Query Pattern: Given Source (query_vector), predict Target.
        Returns: List[(TargetVector, ResonanceWithSource)]
        """
        query_vector = self.normalize(query_vector)
        results = []
        
        for entry in self._transitions:
            # Measure resonance with Source (+ Context if applicable)
            # For v1, simple Source matching
            score = self.compute_resonance(query_vector, entry['source'])
            
            # Use profile to filter or weigh? Spec doesn't strictly say, but implies trusted relations guide decisions.
            # We return raw resonance for now.
            
            results.append((entry['target'], score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
