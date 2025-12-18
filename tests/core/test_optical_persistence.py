
import pytest
import torch
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from coherent.memory.optical_store import OpticalFrequencyStore
from coherent.tools.ingest_knowledge import main as ingest_main

class TestOpticalPersistence:
    
    def test_save_load_cycle(self, tmp_path):
        # 1. Setup Store and Add Data
        store = OpticalFrequencyStore(vector_dim=4, capacity=10)
        
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        metadatas = [{"name": "A"}, {"name": "B"}]
        ids = ["id_a", "id_b"]
        
        store.add("test", vectors, metadatas, ids)
        
        # Verify added
        assert store.current_count == 2
        
        # 2. Save
        save_path = tmp_path / "memory.pt"
        store.save(str(save_path))
        
        assert save_path.exists()
        
        # 3. Load into New Store
        new_store = OpticalFrequencyStore(vector_dim=4, capacity=10)
        new_store.load(str(save_path))
        
        # 4. Verify Content
        assert new_store.current_count == 2
        assert new_store.index_to_id[0] == "id_a"
        assert new_store.index_to_metadata[1]["name"] == "B"
        
        # Verify Optical Memory Tensor equality
        # Note: We can't compare directly if 'new_store' initialized with zeros then loaded.
        # But we can check if data matches.
        assert torch.allclose(store.optical_layer.optical_memory, new_store.optical_layer.optical_memory)

    @patch("coherent.tools.ingest_knowledge.get_vector_store")
    @patch("coherent.tools.ingest_knowledge.KnowledgeRegistry")
    def test_ingest_tool(self, mock_registry, mock_get_store, tmp_path):
        # Setup Mocks
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        
        input_dir = tmp_path / "rules"
        input_dir.mkdir()
        output_file = tmp_path / "out.pt"
        
        # Mock sys.argv
        with patch("sys.argv", ["ingest_knowledge.py", "--input", str(input_dir), "--output", str(output_file)]):
            ingest_main()
            
        # Verify Interactions
        mock_registry.assert_called_once()
        mock_store.save.assert_called_with(str(output_file))

