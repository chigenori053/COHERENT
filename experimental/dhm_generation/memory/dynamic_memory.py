"""
Dynamic Holographic Memory (DHM) Module

Implements the memory storage that STRICTLY only stores attributes, not symbols.
"""

import numpy as np
from typing import Dict, List, Optional
from .attribute_hologram import HolographicEncoder, get_attributes_for_symbol

class DynamicHolographicMemory:
    def __init__(self, encoder: HolographicEncoder, attribute_lookup_fn=None):
        self.encoder = encoder
        # The core memory storage: Maps attribute_name -> Holographic Vector
        self._memory_space: Dict[str, np.ndarray] = {}
        self._is_sealed = False
        
        # Default to alphabet mapping if not provided
        if attribute_lookup_fn is None:
            self.attribute_lookup_fn = get_attributes_for_symbol
        else:
            self.attribute_lookup_fn = attribute_lookup_fn

    def encode_and_store_attribute(self, attribute_name: str) -> None:
        """Generates and stores the hologram for a single attribute."""
        if attribute_name in self._memory_space:
            return # Already stored
        
        h_attr = self.encoder.encode_attribute(attribute_name)
        self._memory_space[attribute_name] = h_attr

    def populate_known_attributes(self, all_attributes: List[str]) -> None:
        """Populate memory with a list of known attributes."""
        for attr in all_attributes:
            self.encode_and_store_attribute(attr)

    def get_attribute_vector(self, attribute_name: str) -> np.ndarray:
        """Retrieve vector for a specific attribute."""
        if attribute_name not in self._memory_space:
            # Lazy loading is permitted by spec (generating from seed),
            # but let's enforce storage check to be explicit about memory contents.
            self.encode_and_store_attribute(attribute_name)
            
        return self._memory_space[attribute_name]

    def construct_symbol_query(self, symbol: str) -> np.ndarray:
        """
        Constructs the seed state H_0 for a symbol by combining its attributes.
        This is a GENERATIVE step, not a retrieval of a stored symbol.
        
        H_0 = Normalize( ⊙ H_attr )
        """
        attributes = self.attribute_lookup_fn(symbol)
        vectors = [self.get_attribute_vector(attr) for attr in attributes]
        
        # Use the encoder's binding logic (Hadamard product + Normalize)
        h_0 = self.encoder.create_superposition(vectors)
        return h_0

    def inspect_memory(self) -> List[str]:
        """Returns a list of keys currently in memory. For verification."""
        return list(self._memory_space.keys())
