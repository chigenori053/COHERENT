
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from coherent.engine.knowledge_registry import KnowledgeRegistry, KnowledgeNode
from coherent.memory.optical_store import OpticalFrequencyStore

class TestOpticalKnowledge:
    
    @pytest.fixture
    def mock_store(self):
        # We want to test real integration with OpticalFrequencyStore but mock embedding to avoid heavy models?
        # Actually OpticalFrequencyStore uses real torch.
        # But Embedder in factory uses SentenceTransformer.
        # We should mock Embedder only.
        pass

    @patch("coherent.memory.factory.get_embedder")
    def test_indexing_and_retrieval(self, mock_get_embedder, tmp_path):
        # Setup Mock Embedder
        mock_embedder = MagicMock()
        # Mock embedding: return distinct vectors
        # "Algebra" -> [1.0, 0.0, ...]
        # "Calculus" -> [0.0, 1.0, ...]
        def side_effect(text):
            vec = [0.0] * 384
            if "algebra" in text.lower():
                vec[0] = 1.0
            elif "calculus" in text.lower():
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            return vec
        
        mock_embedder.embed_text.side_effect = side_effect
        mock_get_embedder.return_value = mock_embedder
        
        # Create dummy rules
        root = tmp_path / "rules"
        root.mkdir()
        
        rule_file = root / "algebra.yaml"
        rule_file.write_text("""
- id: alg_01
  description: "Basic algebra addition"
  pattern_before: "x + x"
  pattern_after: "2*x"
  domain: algebra
""")
        
        # Initialize Registry (this triggers indexing)
        # Mock engine as well
        mock_engine = MagicMock()
        
        # Manually reset the factory singleton before and after
        import coherent.memory.factory as factory
        original_store = factory._STORE_INSTANCE
        factory._STORE_INSTANCE = OpticalFrequencyStore(vector_dim=384, capacity=1000)
        
        try:
            registry = KnowledgeRegistry(root, mock_engine)
            
            store = factory.get_vector_store()

            assert isinstance(store, OpticalFrequencyStore)
            assert store.current_count == 1
            assert store.index_to_id[0] == "alg_01"
            
            # Test Search through Registry
            # Query "search algebra" -> [1, 0, ...]
            results = registry.semantic_search("algebra rule")
            
            assert len(results) > 0
            assert results[0]["id"] == "alg_01"
            assert results[0]["score"] > 0.0
            
            # Test Ambiguity
            assert "ambiguity" in results[0]
        
        finally:
             factory._STORE_INSTANCE = original_store
        
    def test_empty_registry(self, tmp_path):
         mock_engine = MagicMock()
         registry = KnowledgeRegistry(tmp_path, mock_engine)
         # Should not crash
         assert len(registry.rules_by_id) == 0

