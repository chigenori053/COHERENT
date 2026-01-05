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

class StaticHolographicMemory(HolographicMemoryBase):
    def __init__(self):
        # Key-Value storage: ID -> Vector
        # This layer acts like a Dictionary/Encyclopedia
        self._storage: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    def add(self, state: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """
        Register a stable state to Static Memory.
        Requires a unique ID in metadata (e.g. 'id', 'char', 'symbol').
        """
        if not metadata or 'id' not in metadata:
            raise ValueError("Static Memory requires 'id' in metadata.")
        
        id_key = metadata['id']
        normalized_state = self.normalize(state)
        
        # Overwrite or check conflict? 
        # For now, allow overwrite (updates definition).
        self._storage[id_key] = (normalized_state, metadata)

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[Any, float]]:
        """
        Query static memory.
        """
        query_vector = self.normalize(query_vector)
        results = []
        
        for id_key, (vec, meta) in self._storage.items():
            score = self.compute_resonance(query_vector, vec)
            results.append((id_key, score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_vector(self, id_key: str) -> np.ndarray:
        if id_key not in self._storage:
            raise KeyError(f"{id_key} not found in Static Memory.")
        return self._storage[id_key][0]
