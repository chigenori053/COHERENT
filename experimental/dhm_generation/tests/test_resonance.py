"""
Tests for Resonance and EvaluationLogic

Verifies:
- Resonance calculation basics
- Symbol matching correctness
"""

import numpy as np
import unittest
from experimental.dhm_generation.evaluation.resonance import calculate_resonance, evaluate_symbol_match

class TestResonance(unittest.TestCase):
    def test_resonance_basic(self):
        # Create random normalized vectors
        v1 = np.array([1, 0, 0], dtype=np.complex128)
        v2 = np.array([0, 1, 0], dtype=np.complex128)
        
        # Orthogonal -> 0 resonance
        self.assertAlmostEqual(calculate_resonance(v1, v2), 0.0)
        
        # Self -> 1.0 resonance
        self.assertAlmostEqual(calculate_resonance(v1, v1), 1.0)
        
        # Scaled (though should be normalized in practice)
        v3 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        # Norm = sqrt(0.25*4) = 1.0
        self.assertAlmostEqual(calculate_resonance(v3, v3), 1.0)

    def test_evaluate_symbol_match(self):
        # Setup: Dictionary of symbol vectors
        # A: [1,0], B: [0,1]
        candidates = {
            "A": np.array([1, 0], dtype=np.complex128),
            "B": np.array([0, 1], dtype=np.complex128)
        }
        
        # Query closer to A
        query = np.array([0.9, 0.1], dtype=np.complex128)
        query /= np.linalg.norm(query)
        
        results = evaluate_symbol_match(query, candidates)
        
        # Should be sorted
        self.assertEqual(results[0][0], "A")
        self.assertEqual(results[1][0], "B")
        self.assertGreater(results[0][1], results[1][1])
