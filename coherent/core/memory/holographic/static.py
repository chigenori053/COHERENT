"""
Static Holographic Memory (Layer-S)

Long-term / Structural Memory.
Features:
- Stores stable, validated representations (e.g. definitions)
- Immutable (mostly)
- High resonance threshold for retrieval
"""

import numpy as np
from typing import List, Tuple, Any, Dict
from .base import HolographicMemoryBase
from .evaluation import StaticEvaluationProfile

class StaticHolographicMemory(HolographicMemoryBase):
    def __init__(self):
        # Key-Value storage: ID -> (Vector, Metadata, EvaluationProfile)
        self._storage: Dict[str, Tuple[np.ndarray, Dict[str, Any], StaticEvaluationProfile]] = {}
        # Usage stats: ID -> successful_recalls
        self._recall_stats: Dict[str, int] = {}
        self._total_recalls = 0

    def add(self, state: np.ndarray, metadata: Dict[str, Any] = None, profile: StaticEvaluationProfile = None) -> None:
        """
        Register a stable state to Static Memory.
        Requires a unique ID in metadata.
        If profile is provided, it is stored. Otherwise a default one is created.
        """
        if not metadata or 'id' not in metadata:
            raise ValueError("Static Memory requires 'id' in metadata.")
        
        id_key = metadata['id']
        normalized_state = self.normalize(state)
        
        if profile is None:
            # Default/Placeholder profile for legacy adds
            profile = StaticEvaluationProfile(1.0, 1.0, 1.0, 0.0, 0.0)

        self._storage[id_key] = (normalized_state, metadata, profile)
        if id_key not in self._recall_stats:
            self._recall_stats[id_key] = 0

    def evaluate_experience(self, 
                          candidate_vector: np.ndarray, 
                          validation_signal: float, 
                          generalization_score: float) -> StaticEvaluationProfile:
        """
        Evaluate a candidate experience against current static memory.
        Calculates R, V, G, D, U.
        Note: R is resonance with itself (input checking) or query? 
        Spec: R(q, m) = |q . m*|. If we are evaluating a candidate to STORE, q=candidate.
        But 'm' is the stored memory. 
        Actually, for a new candidate, 'Resonance Strength' usually means how well it resonates with the *Recall Query* that triggered it, 
        OR if it's self-consistent.
        Spec 4.3: R(q, m) = | q . m* |. 
        If this is an evaluation for *storage*, R might be effectively 1.0 (self-resonance) or relevant to the context.
        Let's assume R is passed or computed against the context. 
        However, the interface asks to evaluate the *candidate*.
        Let's calc D (Redundancy) here accurately.
        """
        candidate_vector = self.normalize(candidate_vector)
        
        # D: Redundancy = max resonance with existing memories
        max_redundancy = 0.0
        for vec, _, _ in self._storage.values():
            score = self.compute_resonance(candidate_vector, vec)
            if score > max_redundancy:
                max_redundancy = score
                
        # U: Utilization. For a NEW candidate, this is 0.
        utilization = 0.0
        
        # R: Resonance. If we treat the candidate as the 'memory' m, and we don't have a 'q'.
        # Assuming R is high (1.0) because we are considering it for promotion (it was retrieved/created).
        # OR, we take R as input if we had a query context.
        # For this function signature, let's assume R=1.0 for self-consistency as we promote it.
        resonance = 1.0 # Placeholder
        
        return StaticEvaluationProfile(
            resonance_strength=resonance,
            validation_confidence=validation_signal,
            generalization_score=generalization_score,
            redundancy_score=max_redundancy,
            utilization_score=utilization
        )

    def check_update_rules(self, profile: StaticEvaluationProfile) -> str:
        """
        Returns 'STORE', 'MERGE', 'IGNORE' based on Spec 4.4.
        """
        # Thresholds (Configurable)
        TH_R = 0.8
        TH_V = 0.9
        TH_G = 0.5
        TH_D = 0.85 

        # Store if: R > θ_R ∧ V > θ_V ∧ G > θ_G
        is_high_quality = (profile.resonance_strength > TH_R and 
                           profile.validation_confidence > TH_V and 
                           profile.generalization_score > TH_G)

        # Merge if: D > θ_D (and G_new > G_existing - handled by caller or assumed check)
        is_redundant = profile.redundancy_score > TH_D
        
        if is_redundant:
            return 'MERGE'
        elif is_high_quality:
            return 'STORE'
        else:
            return 'IGNORE'

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[Any, float]]:
        """
        Query static memory.
        """
        query_vector = self.normalize(query_vector)
        results = []
        
        self._total_recalls += 1
        
        for id_key, (vec, meta, profile) in self._storage.items():
            score = self.compute_resonance(query_vector, vec)
            results.append((id_key, score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Simplified Utilization tracking: Top-1 is "recalled"
        if results:
            top_id = results[0][0]
            self._recall_stats[top_id] += 1
            
        return results[:top_k]
    
    def get_vector(self, id_key: str) -> np.ndarray:
        if id_key not in self._storage:
            raise KeyError(f"{id_key} not found in Static Memory.")
        return self._storage[id_key][0]
