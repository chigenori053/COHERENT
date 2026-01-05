"""
Stage 2 Encoder
Implements Spec 4 Binding Procedure.
"""

import numpy as np
from experimental.dhm_generation.memory.attribute_hologram import HolographicEncoder as AtomicEncoder

class Stage2Encoder:
    def __init__(self, dimension=2048):
        self.atomic = AtomicEncoder(dimension)
    
    def encode_sequence(self, sequence_list):
        """
        Spec 4:
        H_temp = h(a1)
        for k in 2..L:
             H_temp = Normalize(H_temp * h(ak))
        """
        if not sequence_list:
            return np.zeros(self.atomic.dimension, dtype=np.complex128)
            
        # 1. First item
        h_curr = self.atomic.encode_attribute(sequence_list[0])
        
        # 2. Sequential Binding
        for token in sequence_list[1:]:
            h_next = self.atomic.encode_attribute(token)
            
            # Binding: Hadamard Product
            bound = h_curr * h_next
            
            # Normalize
            norm = np.linalg.norm(bound)
            if norm > 1e-9:
                h_curr = bound / norm
            else:
                h_curr = bound
                
        return h_curr
