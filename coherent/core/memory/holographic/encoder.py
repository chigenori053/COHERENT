"""
Holographic Encoder

Standard encoder for converting attributes/symbols into holographic vectors.
Migrated from experimental.dhm_generation.memory.attribute_hologram.
"""

import numpy as np
import hashlib
from typing import List

class HolographicEncoder:
    """
    Handles the encoding of string attributes into holographic vectors (complex128).
    Pipeline: Seed(Name) -> Real Vector -> FFT -> Normalize
    """
    def __init__(self, dimension: int = 2048):
        self.dimension = dimension

    def encode_attribute(self, attribute_name: str) -> np.ndarray:
        """
        Encode an attribute string into a holographic vector.
        """
        # 1. Deterministic seed from attribute name
        seed = int(hashlib.sha256(attribute_name.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        # 2. Generate real vector (Gaussian)
        real_vec = rng.standard_normal(self.dimension)
        
        # 3. FFT -> complex vector
        complex_vec = np.fft.fft(real_vec)
        
        # 4. L2 normalization
        return self.normalize(complex_vec)

    def create_superposition(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Create a query/state by BINDING (element-wise product) vectors.
        Note: terminology 'superposition' can be ambiguous. 
        In this DHM context (as established in implementations), 
        combining attributes to form a symbol uses Hadamard Product (Binding).
        Summation (Superposition) is used for storing multiple items in memory.
        
        This method corresponds to `construct_symbol_query` logic:
        Symbol = Normalize( Attr1 * Attr2 * ... )
        """
        if not vectors:
            raise ValueError("No vectors provided for binding.")
            
        result = vectors[0]
        for vec in vectors[1:]:
            result = result * vec  # Hadamard product (Binding)
            # Numerical stability normalization step
            result = self.normalize(result)
            
        return result

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            # Avoid division by zero, though in high-dim Gaussian this is rare
            return vec 
        return vec / norm
