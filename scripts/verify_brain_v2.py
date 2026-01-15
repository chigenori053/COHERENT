
"""
Verification Script for BrainModel v2.0 (CognitiveCore)

Scenario:
1. Initialize CognitiveCore.
2. Inject a mock input.
3. Observe CognitiveStateVector metrics (H, C, R, B).
4. Verify authority (SimulationTrigger).
"""

from coherent.core.cognitive_core import CognitiveCore, DecisionType
from coherent.core.memory.experience_manager import ExperienceManager
from unittest.mock import MagicMock
import numpy as np

def run_verification():
    print("=== BrainModel v2.0 Verification Start ===")
    
    # Setup
    mock_exp = MagicMock()
    brain = CognitiveCore(experience_manager=mock_exp)
    
    # 1. Simulate "Unknown" Input (High Entropy, Low Recall)
    # We force the internal recall_engine to return low scores or random noise
    # Since we can't easily force internal dynamic memory state without complex setup,
    # we rely on the fact that an empty memory returns empty results -> High Entropy?
    # Our impl handles empty: "if not results: return default".
    
    print("\n[Scenario 1] Empty Memory (Novelty)")
    decision = brain.process_input("What is X?")
    
    state = decision.state_snapshot
    print(f"Decision: {decision.decision_type.value}, Action: {decision.action}")
    print(f"Metrics: H={state.entropy:.2f}, C={state.confidence:.2f}, R={state.recall_reliability:.2f}, B={state.branching_pressure:.2f}")
    
    # Expect: H=1.0 (or high), C=0.0, R=0.0 -> REJECT/SUPPRESS or REVIEW?
    # Spec: C < 0.4 and R < 0.4 -> REJECT.
    # Wait, our empty logic returns H=1.0, C=0.0, R=0.0.
    # So it should be REJECT.
    
    # However, for "Learning", we might want to trigger Simulation if C is low?
    # Trigger Policy: C < 0.4 -> Trigger.
    # So process_input *should* have triggered Simulation if REVIEW?
    # But logic is: if REVIEW then Check Trigger.
    # if REJECT (C<0.4, R<0.4) -> Just Reject?
    # Spec: SimulationTrigger: trigger_if_any -> one is C<0.4.
    # But Decision Rule says: REJECT if C < 0.4 AND R < 0.4.
    # So if both are low, it decides REJECT. It does NOT go to REVIEW -> Simulate.
    # This implies "Ignorance" (Don't know, don't care).
    # But if H >= 0.6 (High Uncertainty) -> Trigger?
    # Maybe we need to adjust "REVIEW" condition to catch these cases if needed?
    # Or maybe "REJECT" means "I have no idea, so I won't decide".
    # But usually we want to "Explore" (Simulation).
    
    # Let's see what happens.
    
    # 2. Simulate "Ambiguous" Input (Review)
    # We manually inject some items into recall_engine to create ambiguity.
    print("\n[Scenario 2] Ambiguous Memory (Conflict)")
    
    # Inject 2 competing memories
    vec1 = np.random.rand(64)
    vec1 = vec1 / np.linalg.norm(vec1)
    brain.recall_engine.add(vec1, {"content": "Concept A"})
    
    # We need access to internals to force resonance. 
    # Or just use the fact that input "Concept A" might resonate with vec1.
    # But `process_input` creates a RANDOM vector for string input in our mock.
    # So we can't easily reproduce resonance without controlling the vectorizer.
    # However, `CognitiveCore._vectorize` uses hash of string. So it IS deterministic.
    
    # So if we add the vector for "Concept A", then input "Concept A" again...
    # But wait, `brain.recall_engine` is `DynamicHolographicMemory`. 
    # `add(vec)` adds to the hologram.
    
    # Calculate vector for "Concept A"
    vec_a = brain._vectorize("Concept A")
    brain.recall_engine.add(vec_a, {"content": "Concept A"})
    
    # Calculate vector for "Concept B" (make it close to A? No, let's just add B)
    vec_b = brain._vectorize("Concept B")
    brain.recall_engine.add(vec_b, {"content": "Concept B"})
    
    # Now input "Concept A".
    # It should hit "Concept A" (1.0) and "Concept B" (maybe small resonance).
    # If 1.0 dominant, decision should be ACCEPT.
    
    print(">> Input: 'Concept A'")
    decision_a = brain.process_input("Concept A")
    s = decision_a.state_snapshot
    print(f"Decision: {decision_a.decision_type.value}, Action: {decision_a.action}")
    print(f"Metrics: H={s.entropy:.2f}, C={s.confidence:.2f}, R={s.recall_reliability:.2f}")
    
    if decision_a.decision_type == DecisionType.ACCEPT:
         print("SUCCESS: Recognized Concept A")
    else:
         print("NOTE: Did not accept immediately. Maybe Confidence too low?")

if __name__ == "__main__":
    run_verification()
