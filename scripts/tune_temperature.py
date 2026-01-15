
import sys
import os
import numpy as np
from typing import List, Tuple

# Add project root to path
sys.path.append(os.getcwd())

from coherent.core.cognitive_core import CognitiveCore, CognitiveStateVector, DecisionType
from unittest.mock import MagicMock

def simulate_metrics(tau: float, scores: List[float], core: CognitiveCore) -> CognitiveStateVector:
    """
    Simulate metrics calculation for a given tau and raw resonance scores.
    """
    core.tau = tau
    # Mock recall results: list of (id, score)
    results = [(f"id_{i}", s) for i, s in enumerate(scores)]
    # Query vector doesn't matter for the metric calc itself in current impl
    query_vec = np.zeros(64) 
    
    return core._calculate_metrics(results, query_vec)

def run_tuning():
    # Setup Core
    core = CognitiveCore(experience_manager=MagicMock())
    
    # Define Scenarios (Raw Resonance Scores)
    scenarios = {
        "Clear Winner": [1.0, 0.2, 0.1, 0.1, 0.05],
        "Ambiguous (High Conflict)": [0.9, 0.85, 0.2, 0.1, 0.1],
        "Weak Match (Low Confidence)": [0.4, 0.3, 0.3, 0.2, 0.2],
        "Flat (High Entropy)": [0.5, 0.5, 0.5, 0.5, 0.5]
    }
    
    tau_values = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    print(f"{'Scenario':<25} | {'Tau':<5} | {'H':<5} | {'C':<5} | {'R':<5} | {'Decision':<10}")
    print("-" * 75)
    
    for name, scores in scenarios.items():
        print(f"--- {name} ---")
        for tau in tau_values:
            state = simulate_metrics(tau, scores, core)
            decision = core._make_decision(state)
            
            h_str = f"{state.entropy:.2f}"
            c_str = f"{state.confidence:.2f}"
            r_str = f"{state.recall_reliability:.2f}"
            d_str = decision.decision_type.value
            
            print(f"{'':<25} | {tau:<5.1f} | {h_str:<5} | {c_str:<5} | {r_str:<5} | {d_str:<10}")

if __name__ == "__main__":
    run_tuning()
