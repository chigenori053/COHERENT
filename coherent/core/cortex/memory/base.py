"""
Holographic Memory Base

Defines the common interface for all memory layers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
import numpy as np

class HolographicMemoryBase(ABC):
    """
    Abstract Base Class for Holographic Memory Layers.
    Enforces a common contract for adding, querying, and resonance calculation.
    """

    @abstractmethod
    def add(self, state: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """
        Add a holographic state vector to the memory.
        
        Args:
            state: Complex-valued holographic vector (1D array).
            metadata: Optional dictionary for tracking (e.g., origin, timestamp).
        """
        pass

    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[Any, float]]:
        """
        Query the memory with a probe vector.
        
        Args:
            query_vector: Probe vector.
            top_k: Number of top results to return.
            
        Returns:
            List of (content_id/content, resonance_score).
        """
        pass

    def compute_resonance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Computes the resonance (similarity) between two holographic vectors.
        Standard metric: Cosine Similarity of real parts (or Hermitian dot product magnitude).
        
        In DHM context, we typically use the real part of the dot product (if normalized) 
        or simply modulus of dot product. 
        
        Using: ABS(Dot(A, B)) / (Norm(A)*Norm(B))
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
            
        # Hermitian dot product: sum(a_i * conj(b_i))
        dot_product = np.vdot(vec_a, vec_b) 
        
        # Resonance is the magnitude of the projection
        resonance = np.abs(dot_product) / (norm_a * norm_b)
        return float(resonance)

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        """Helper to L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return vec
        return vec / norm
