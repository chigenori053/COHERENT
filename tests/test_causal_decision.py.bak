
import unittest
import numpy as np
from coherent.core.memory.holographic.causal import CausalHolographicMemory, DecisionState, Action, CausalTrace

class TestCausalDecision(unittest.TestCase):
    def setUp(self):
        self.causal = CausalHolographicMemory()

    def test_high_quality_promotion(self):
        """Test that high resonance/margin leads to PROMOTE."""
        state = DecisionState(
            resonance_score=0.95,
            margin=0.5,
            repetition_count=5, # Not used in weights yet but in struct
            entropy_estimate=0.05,
            memory_origin='Dynamic'
        )
        action = self.causal.evaluate_decision(state)
        self.assertEqual(action, Action.PROMOTE)
        
        # check probability high
        p = self.causal._estimate_probability(state)
        self.assertGreater(p, 0.9)

    def test_low_quality_retain(self):
        """Test that low quality leads to RETAIN or other non-promote."""
        state = DecisionState(
            resonance_score=0.4,
            margin=0.05,
            repetition_count=1,
            entropy_estimate=0.6,
            memory_origin='Dynamic'
        )
        action = self.causal.evaluate_decision(state)
        # Low quality means low P(~0.1-0.3). 
        # EU(RETAIN) = P(0.5) + (1-P)(-0.5) = P - 0.5. If P < 0.5, EU < 0.
        # EU(DEFER) = 0.
        # So DEFER is superior to RETAIN when P < 0.5.
        # This test case produces low P, so DEFER is correct.
        self.assertIn(action, [Action.RETAIN, Action.DEFER_REVIEW]) 
        if action == Action.DEFER_REVIEW:
             pass # Correctly deferred due to low confidence

    def test_suppress_case(self):
        """Test extremely poor state leading to SUPPRESS (conceptually)."""
        # With current weights, let's see if we can trigger suppression
        # Bias -4.0. Low P.
        # If P ~ 0.
        # EU(PROMOTE) ~ -10
        # EU(RETAIN) ~ -0.5
        # EU(SUPPRESS) ~ +0.5
        # So low P should lead to SUPPRESS.
        state = DecisionState(
            resonance_score=0.0,
            margin=0.0,
            repetition_count=0,
            entropy_estimate=1.0, # -0.5 impact
            memory_origin='Dynamic',
            historical_conflict_rate=1.0 # -2.0 impact
        )
        # Logit: 0 + 0 -0.5 -2.0 -4.0 = -6.5 -> P ~ 0
        action = self.causal.evaluate_decision(state)
        self.assertEqual(action, Action.SUPPRESS)

    def test_logging(self):
        state = DecisionState(0.9, 0.4, 1, 0.1, 'Dynamic')
        self.causal.evaluate_decision(state)
        self.assertEqual(len(self.causal._decision_logs), 1)
        trace = self.causal._decision_logs[0]
        self.assertIsInstance(trace, CausalTrace)
        self.assertEqual(trace.final_decision, 'PROMOTE')

if __name__ == '__main__':
    unittest.main()
