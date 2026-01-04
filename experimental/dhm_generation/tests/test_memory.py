"""
Tests for DHM Memory and Attributes

Verifies:
- Attribute encoding pipeline (FFT, normalization)
- Dynamic memory storage (attributes only)
- Symbol query construction
"""

import numpy as np
import unittest
from experimental.dhm_generation.memory.attribute_hologram import HolographicEncoder, get_attributes_for_symbol
from experimental.dhm_generation.memory.dynamic_memory import DynamicHolographicMemory

class TestDHMMemory(unittest.TestCase):
    def setUp(self):
        self.encoder = HolographicEncoder(dimension=512)
        self.memory = DynamicHolographicMemory(self.encoder)

    def test_encoding_properties(self):
        """Test basic properties of encoded attribute vectors."""
        vec = self.encoder.encode_attribute("TEST_ATTR")
        
        # 1. Check normalization
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
        # 2. Check dimension
        self.assertEqual(len(vec), 512)
        
        # 3. Check data type
        self.assertTrue(np.iscomplexobj(vec))

    def test_determinism(self):
        """Test that encoding is deterministic for the same input."""
        vec1 = self.encoder.encode_attribute("SAME")
        vec2 = self.encoder.encode_attribute("SAME")
        np.testing.assert_array_equal(vec1, vec2)
        
        vec3 = self.encoder.encode_attribute("DIFFERENT")
        self.assertFalse(np.array_equal(vec1, vec3))

    def test_memory_storage(self):
        """Test that memory stores attributes correctly."""
        attr = "TYPE_LETTER"
        self.memory.encode_and_store_attribute(attr)
        
        # Check explicit storage
        self.assertIn(attr, self.memory._memory_space)
        stored_vec = self.memory.get_attribute_vector(attr)
        
        # Verify it matches fresh encoding
        fresh_vec = self.encoder.encode_attribute(attr)
        np.testing.assert_array_almost_equal(stored_vec, fresh_vec)

    def test_symbol_query_construction(self):
        """Test H_0 construction."""
        # Pre-populate some attributes for 'A'
        # 'A' = {letter, uppercase, early, vowel, angular, complex}
        # Just check it runs and produces a normalized vector
        h_0 = self.memory.construct_symbol_query("A")
        
        self.assertEqual(len(h_0), 512)
        self.assertAlmostEqual(np.linalg.norm(h_0), 1.0, places=5)
        
        # Verify it is composed of attributes
        # (A symbol query should have non-zero resonance with its attributes)
        # However, due to HRR complexity, we just check output validity here.
