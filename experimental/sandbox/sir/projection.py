"""
SIR Vector Projection
Projects SIR structures into fixed-dimensional vectors for DHM input.
Uses Holographic Reduced Representation (HRR) principles.
"""

import numpy as np
import hashlib
from .models import SIR

class SIRProjector:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def _hash_to_seed(self, hash_str: str) -> int:
        """Converts hex hash to integer seed."""
        return int(hash_str[:8], 16)

    def _generate_vector(self, hash_str: str) -> np.ndarray:
        """Generates a random vector based on the hash (HRR encoding)."""
        seed = self._hash_to_seed(hash_str)
        rng = np.random.RandomState(seed)
        # Generate random formulation (e.g., Gaussian)
        # For simplicity in Sandbox v1.0, we use real-valued Gaussian.
        # DHM typically uses Complex, but we start with structural projection.
        return rng.randn(self.dimension)

    def project(self, sir: SIR) -> np.ndarray:
        """
        Projects SIR into a single vector s.
        s = sum(f(E)) + sum(f(R)) + sum(f(O)) + sum(f(C))
        """
        # Ensure signatures are computed
        if not sir.structure_signature.graph_hash:
            sir.recompute_signature()

        # Initialize superposition vector
        s_vector = np.zeros(self.dimension)

        # 1. Project Entities
        # We need the entity_map for structural features again. 
        # Ideally SIR models should cache this, but recomputing is cheap for sandbox.
        entity_map = {e.id: e.get_structure_feature() for e in sir.semantic_core.entities}
        
        for entity_hash in entity_map.values():
            s_vector += self._generate_vector(entity_hash)

        # 2. Project Relations
        for rel in sir.semantic_core.relations:
            h = rel.get_structure_feature(entity_map)
            s_vector += self._generate_vector(h)

        # 3. Project Operations
        for op in sir.semantic_core.operations:
            h = op.get_structure_feature(entity_map)
            s_vector += self._generate_vector(h)

        # 4. Project Constraints
        for const in sir.semantic_core.constraints:
            h = const.get_structure_feature(entity_map)
            s_vector += self._generate_vector(h)

        return s_vector
