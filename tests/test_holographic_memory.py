"""
Tests for Holographic Memory Architecture (3-Layer)
"""

import unittest
import numpy as np
from coherent.core.memory.holographic.encoder import HolographicEncoder
from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.static import StaticHolographicMemory
from coherent.core.memory.holographic.causal import CausalHolographicMemory
from coherent.core.memory.holographic.orchestrator import MemoryOrchestrator

class TestHolographicMemory(unittest.TestCase):
    def setUp(self):
        self.encoder = HolographicEncoder(dimension=512)
        self.dynamic = DynamicHolographicMemory(capacity=10)
        self.static = StaticHolographicMemory()
        self.causal = CausalHolographicMemory()
        self.orchestrator = MemoryOrchestrator(
            self.dynamic, self.static, self.causal, promotion_threshold=0.8
        )

    def test_encoder_deterministic(self):
        """Test that encoder produces same vector for same input."""
        v1 = self.encoder.encode_attribute("test")
        v2 = self.encoder.encode_attribute("test")
        self.assertTrue(np.allclose(v1, v2))
        
        v3 = self.encoder.encode_attribute("other")
        # Should be nearly orthogonal (low resonance)
        resonance = abs(np.vdot(v1, v3)) / (np.linalg.norm(v1) * np.linalg.norm(v3))
        self.assertLess(resonance, 0.3)

    def test_dynamic_memory_add_query(self):
        """Test adding and retrieving from Dynamic Memory."""
        vec = self.encoder.encode_attribute("apple")
        self.dynamic.add(vec, metadata={"content": "apple"})
        
        # Exact query
        results = self.dynamic.query(vec)
        self.assertTrue(results)
        content, score = results[0]
        self.assertEqual(content, "apple")
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_static_memory_recognition(self):
        """Test that Static memory stores and recognizes items."""
        vec = self.encoder.encode_attribute("definition:A")
        self.static.add(vec, metadata={"id": "A"})
        
        results = self.static.query(vec)
        self.assertTrue(results)
        id_key, score = results[0]
        self.assertEqual(id_key, "A")
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_causal_memory_prediction(self):
        """Test Sequence Prediction (A -> B)."""
        vec_a = self.encoder.encode_attribute("State_A")
        vec_b = self.encoder.encode_attribute("State_B")
        
        self.causal.add_transition(source=vec_a, target=vec_b)
        
        # Query with A, expect B
        results = self.causal.query(vec_a)
        self.assertTrue(results)
        
        predicted_vec, score = results[0]
        # Resonance between predicted and actual B should be high (it IS actual B)
        # But we need to compare vectors
        sim = self.orchestrator.static.compute_resonance(predicted_vec, vec_b)
        self.assertAlmostEqual(sim, 1.0, places=5)
        # Resonance score returned by query is "similarity between Query(A) and Source(A)"
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_orchestrator_flow(self):
        """Test basic flow in Orchestrator."""
        vec = self.encoder.encode_attribute("input_X")
        
        # 1. Process Input -> Should go to Dynamic
        self.orchestrator.process_input(vec, metadata={"content": "input_X"})
        res_dynamic = self.dynamic.query(vec)
        self.assertEqual(res_dynamic[0][0], "input_X")
        
        # 2. Promote to Static
        self.orchestrator.promote_to_static(vec, metadata={"id": "X_Item"})
        
        # 3. Recall
        recalled = self.orchestrator.recall(vec)
        self.assertIsNotNone(recalled["static"])
        self.assertEqual(recalled["static"][0], "X_Item")

if __name__ == '__main__':
    unittest.main()
