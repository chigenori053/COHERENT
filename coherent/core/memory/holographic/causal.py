"""
Causal Holographic Memory (Layer-C)

Transition / Sequence Memory.
Features:
- Stores transitions: Source -> Target (optionally with Context)
- Enables prediction (Query Source -> Retrieve Target)
"""

import numpy as np
from typing import List, Tuple, Any, Dict
from .base import HolographicMemoryBase

class CausalHolographicMemory(HolographicMemoryBase):
    def __init__(self):
        # Storing transitions. 
        # Simplest form: Source Vector binds to Target Vector?
        # Or explicit storage of pairs?
        # Specification says "CausalEntry: source, target, context".
        # Let's verify via resonance against Source.
        self._transitions: List[Dict[str, Any]] = []

    def add(self, state: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """
        Generic add (interface compliance).
        But preferred usage is add_transition.
        If used, assumes state is Source, and Target is in metadata?
        Let's implement explicit add_transition instead.
        """
        raise NotImplementedError("Use add_transition() for Causal Memory.")

    def add_transition(self, source: np.ndarray, target: np.ndarray, context: np.ndarray = None) -> None:
        """
        Register a transition.
        Stored Entry: { 'source': vec, 'target': vec, 'context': vec }
        """
        entry = {
            'source': self.normalize(source),
            'target': self.normalize(target),
            'context': self.normalize(context) if context is not None else None
        }
        self._transitions.append(entry)

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[Any, float]]:
        """
        Query Pattern: Given Source (query_vector), predict Target.
        Returns: List[(TargetVector, ResonanceWithSource)]
        
        The 'content' returned here is the TARGET state (vector or ID?), associated with the score.
        Since we return (Any, float), we return the (TargetVector, score).
        """
        query_vector = self.normalize(query_vector)
        results = []
        
        for entry in self._transitions:
            # Measure resonance with Source (+ Context if applicable)
            # For v1, simple Source matching
            score = self.compute_resonance(query_vector, entry['source'])
            
            # TODO: Context modulation
            
            # Return Target Vector as the "Content"
            results.append((entry['target'], score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
