
import numpy as np
import unittest
from coherent.core.memory.holographic.static import StaticHolographicMemory, StaticEvaluationProfile
from coherent.core.memory.holographic.causal import CausalHolographicMemory, CausalEvaluationProfile

class TestEvaluationSystem(unittest.TestCase):
    def test_static_evaluation(self):
        mem = StaticHolographicMemory()
        
        # Test High Quality Profile
        profile_good = StaticEvaluationProfile(
            resonance_strength=1.0,
            validation_confidence=0.95,
            generalization_score=0.8,
            redundancy_score=0.1,
            utilization_score=0.0
        )
        action = mem.check_update_rules(profile_good)
        self.assertEqual(action, 'STORE')
        
        # Test Redundant Profile
        profile_redundant = StaticEvaluationProfile(
            resonance_strength=1.0,
            validation_confidence=0.95,
            generalization_score=0.8,
            redundancy_score=0.9, # High Redundancy
            utilization_score=0.0
        )
        action = mem.check_update_rules(profile_redundant)
        self.assertEqual(action, 'MERGE')

    def test_causal_evaluation(self):
        mem = CausalHolographicMemory()
        
        # Test Robust Causal Relation
        # Cs > 0.8, Cd > 0.7, Ce > 0
        profile_robust = mem.evaluate_relation(
            failure_rate=0.1,    # Cs = 0.9
            decision_entropy=0.2,# Cd = 0.8
            entropy_delta=0.5,   # Ce = 0.5
            counterfactual_failure=0.1 # Cr = 0.9
        )
        action = mem.check_update_rules(profile_robust)
        self.assertEqual(action, 'STORE')
        
        # Test Invalid (Zero Entropy Reduction)
        profile_useless = mem.evaluate_relation(
            failure_rate=0.1,
            decision_entropy=0.2,
            entropy_delta=0.0,   # Ce = 0
            counterfactual_failure=0.1
        )
        action = mem.check_update_rules(profile_useless)
        self.assertEqual(action, 'REMOVE')

if __name__ == '__main__':
    unittest.main()
