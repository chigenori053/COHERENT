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
        
        Note: Modified for C2-Final to be Content-Aware (Specific).
        Includes label in hashing to distinguish 'A=B' from 'B=C'.
        """
        # Ensure signatures are computed
        if not sir.structure_signature.graph_hash:
            sir.recompute_signature()

        # Initialize superposition vector
        s_vector = np.zeros(self.dimension)
        
        # Helper for specific hash (Structure + Content)
        def _get_specific_entity_hash(entity) -> str:
            # We mix label into the hash to ensure 'A' != 'B'
            data = {
                "struct": entity.get_structure_feature(),
                "label": entity.label
            }
            # Use stable_hash from models if available, or just hashlib here
            from .models import stable_hash
            return stable_hash(data)

        # 1. Project Entities
        # Build map: id -> specific_hash
        entity_map = {e.id: _get_specific_entity_hash(e) for e in sir.semantic_core.entities}
        
        for entity_hash in entity_map.values():
            s_vector += self._generate_vector(entity_hash)

        # 2. Project Relations
        for rel in sir.semantic_core.relations:
            # Relation.get_structure_feature uses the map we passed.
            # Since entity_map now has specific hashes, the relation hash becomes specific too.
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
