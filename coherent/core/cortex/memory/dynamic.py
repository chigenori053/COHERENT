"""
Dynamic Holographic Memory (Layer-D)

Short-term / Working Memory.
Features:
- High plasticity (instant write)
- Capacity limited (FIFO or Decay)
- Stores noisy/intermediate states
"""

import numpy as np
from typing import List, Tuple, Any, Dict, Deque
from collections import deque
from .base import HolographicMemoryBase

class DynamicHolographicMemory(HolographicMemoryBase):
    def __init__(self, capacity: int = 100, decay_rate: float = 0.0):
        """
        Args:
            capacity: Maximum number of items to hold active.
            decay_rate: Not fully implemented in v1, placeholder for forgetting curve.
        """
        self.capacity = capacity
        # Storing tuples of (vector, metadata)
        # Using deque for efficient FIFO auto-discard if needed, 
        # though we might want random access for resonance.
        self._storage: Deque[Tuple[np.ndarray, Dict[str, Any]]] = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """
        Add state to dynamic memory. 
        Oldest entries are automatically removed if capacity is exceeded (via deque maxlen).
        """
        meta = metadata or {}
        # Normalize before storage to ensure consistent resonance
        normalized_state = self.normalize(state)
        self._storage.append((normalized_state, meta))

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[Any, float]]:
        """
        Query dynamic memory.
        Returns entries sorted by resonance.
        """
        query_vector = self.normalize(query_vector)
        results = []
        
        # Linear scan (acceptable for small STM capacity)
        for vec, meta in self._storage:
            score = self.compute_resonance(query_vector, vec)
            # Content identifier is usually in metadata, e.g., 'symbol' or 'content'
            content = meta.get('content', meta) 
            results.append((content, score))
            
        # Sort desc by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_recent_items(self, n: int = 5) -> List[Tuple[np.ndarray, Dict]]:
        """Retrieve n most recent items (LIFO)."""
        # deque is right-ended (newest at end)
        return list(reversed(self._storage))[:n]

    def clear(self):
        """Clear all items from dynamic memory."""
        self._storage.clear()
