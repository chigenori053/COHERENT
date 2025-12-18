
import pytest
import torch
from coherent.engine.tensor.engine import TensorLogicEngine

class TestOpticalMigration:
    def test_optical_initialization(self):
        # 1. Initialize
        vocab_size = 10
        engine = TensorLogicEngine(vocab_size=vocab_size, embedding_dim=16)
        
        # Check if optical core is present
        assert hasattr(engine, 'optical')
        assert engine.optical.memory_capacity == 100
        assert engine.optical.input_dim == 16
        
        # Check basic forward pass doesn't crash
        # Batch size 1, Sequence length 5
        dummy_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        resonance = engine(dummy_input)
        
        assert resonance.shape == (1, 100) # [Batch, Capacity]
        assert resonance.is_complex() is False # Resonance energy should be real
        assert (resonance >= 0).all() # Energy must be non-negative

    def test_rule_registration_and_expansion(self):
        engine = TensorLogicEngine(vocab_size=10, embedding_dim=16)
        initial_capacity = engine.optical.memory_capacity
        
        # Register rules up to capacity
        # We manually advance next_rule_idx to simulate full memory
        engine.next_rule_idx = initial_capacity
        
        # Register one more
        engine.register_rule("OverflowRule")
        
        # Check expansion
        assert engine.optical.memory_capacity > initial_capacity
        assert "OverflowRule" in engine.rule_index_map
        assert engine.rule_index_map["OverflowRule"] == initial_capacity

    def test_prediction_logic(self):
        engine = TensorLogicEngine(vocab_size=10, embedding_dim=16)
        
        # Register a rule
        engine.register_rule("RuleA")
        idx_a = engine.rule_index_map["RuleA"]
        
        # Hack: Manually set the memory vector for RuleA to be identical to embedding of token 1
        # This ensures maximum resonance when input is token 1
        with torch.no_grad():
            # Get embedding for token 1
            # Note: Engine pools embeddings. For single token, it's just the embedding.
            token_1_emb = engine.embeddings(torch.tensor([1])) # [1, Emb]
            # Convert to complex as logic engine does (Real -> Complex)
            token_1_complex = token_1_emb.type(torch.cfloat)
            
            # Set memory. Need to transpose/conjugate? 
            # forward: input * memory.conj().t()
            # If input = M, then M * M.conj().t() = |M|^2
            # So memory row should be token_1_complex.
            # engine.optical.optical_memory is [Capacity, Dim]
            engine.optical.optical_memory.data[idx_a] = token_1_complex.squeeze(0)
            
        # Predict with input token 1
        input_tensor = torch.tensor([[1]])
        predicted = engine.predict_rules(input_tensor, top_k=1)
        
        assert predicted == ["RuleA"]
        
        # Evaluate state goodness
        score = engine.evaluate_state(input_tensor)
        assert score.item() > 0.0 # Should be high resonance

    def test_similarity_complex(self):
        engine = TensorLogicEngine(vocab_size=10, embedding_dim=16)
        
        # Sim(1, 1) should be approx 1.0
        sim = engine.get_similarity(1, 1)
        assert 0.99 < sim <= 1.0001
