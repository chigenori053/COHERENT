import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from coherent.optical.layer import OpticalInterferenceEngine

class TensorLogicEngine(nn.Module):
    """
    Optical Logic Engine (formerly TensorLogicEngine).
    Uses Optical Interference to compute resonance between state vectors and rules.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Complex Embedding: We need amplitude and phase
        # Approach: Standard embedding -> complex conversion
        # padding_idx=0
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Optical Core
        # Initial capacity 100, grows as rules are registered
        self.optical = OpticalInterferenceEngine(memory_capacity=100, input_dim=embedding_dim)
        
        # Map rule IDs to memory indices
        self.rule_index_map: Dict[str, int] = {}
        self.next_rule_idx = 0
        
        self.to(device)
        self.optical.to(device)
        
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Compute resonance energy for each memory slot.
        Args:
            state_tensor: [Batch, Seq] or [Seq] token IDs
        Returns:
            Resonance Energy: [Batch, MemoryCapacity]
        """
        # 1. State Encoding -> Complex Probe Wave
        # [Batch, Seq] -> [Batch, Seq, Emb]
        embedded = self.embeddings(state_tensor)
        
        # Pooling to get a single vector per sequence
        if embedded.dim() == 3:
            # [Batch, Seq, Emb] -> [Batch, Emb]
            state_vector = embedded.mean(dim=1)
        else:
            # [Seq, Emb] -> [Emb] -> [1, Emb] (add batch dim for consistency logic)
            state_vector = embedded.mean(dim=0, keepdim=True)

        # Convert to Complex (Holographic)
        # Simple method: Real -> Amplitude, Phase 0
        # Better: Learn phase? For now, keep it simple.
        # Ensure it is complex float
        probe_wave = state_vector.type(torch.cfloat)
        
        # 2. Optical Interference
        resonance = self.optical(probe_wave)
        
        return resonance

    def register_rule(self, rule_id: str):
        """
        Registers a rule by assigning it a slot in the optical memory.
        """
        if rule_id in self.rule_index_map:
            return

        if self.next_rule_idx >= self.optical.memory_capacity:
            # Expand memory
            expand_by = max(10, self.optical.memory_capacity // 2)
            self.optical.expand_memory(expand_by)
            
        idx = self.next_rule_idx
        self.rule_index_map[rule_id] = idx
        self.next_rule_idx += 1
        
        # In a real system, we might initialize this slot with a specific vector
        # representing the rule's premise. Here, it stays random (initialized)
        # until trained, or we could set it if we had an embedding for the rule.

    def predict_rules(self, state_tensor: torch.Tensor, top_k: int = 5) -> List[str]:
        """
        Predicts rules based on resonance intensity.
        """
        if not self.rule_index_map:
            return []
            
        with torch.no_grad():
            # [Batch, Capacity] - assume single batch input for prediction usually
            resonance = self.forward(state_tensor)
            
        # Get score for current item (assume batch index 0)
        scores = resonance[0] 
        
        # Get active indices
        active_indices = []
        active_ids = []
        for rid, idx in self.rule_index_map.items():
            active_indices.append(idx)
            active_ids.append(rid)
            
        if not active_indices:
            return []
            
        # Extract scores for registered rules only
        rule_scores = scores[active_indices]
        
        # Sort
        k = min(top_k, len(rule_scores))
        top_indices = torch.argsort(rule_scores, descending=True)[:k]
        
        return [active_ids[i] for i in top_indices]

    def evaluate_state(self, state_tensors: torch.Tensor) -> torch.Tensor:
        """
        Evaluates state 'goodness' via total resonance/energy or ambiguity.
        """
        resonance = self.forward(state_tensors)
        # Use total energy as a metric of 'sense' (how much it resonates with knowns)
        return resonance.sum(dim=1)

    def get_similarity(self, term_a_idx: int, term_b_idx: int) -> float:
        """
        Computes cosine similarity (real part of interference) between term embeddings.
        """
        idx_a = torch.tensor([term_a_idx], device=self.device)
        idx_b = torch.tensor([term_b_idx], device=self.device)
        
        with torch.no_grad():
            vec_a = self.embeddings(idx_a).type(torch.cfloat)
            vec_b = self.embeddings(idx_b).type(torch.cfloat)
            
            # Complex dot product similarity
            # <u, v> / (|u||v|)
            dot = torch.sum(vec_a * vec_b.conj(), dim=-1)
            norm_a = torch.norm(vec_a, dim=-1)
            norm_b = torch.norm(vec_b, dim=-1)
            
            sim = dot / (norm_a * norm_b + 1e-8)
            
        return float(sim.abs().item()) # Return magnitude of similarity

    def update_embeddings(self, vocab_size: int):
        current_vocab, dim = self.embeddings.weight.shape
        if vocab_size > current_vocab:
            new_embeddings = nn.Embedding(vocab_size, dim, padding_idx=0).to(self.device)
            with torch.no_grad():
                new_embeddings.weight[:current_vocab] = self.embeddings.weight
            self.embeddings = new_embeddings
