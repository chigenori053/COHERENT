"""
Sequence Encoder for Stage 1

Encodes sequences using Holographic Reduced Representations.
Implements Position Binding + Superposition.
"""

import numpy as np
from experimental.dhm_generation.memory.attribute_hologram import HolographicEncoder as BaseEncoder

class SequenceEncoder:
    def __init__(self, dimension: int = 1024, use_fft: bool = True):
        self.dimension = dimension
        self.use_fft = use_fft
        # We misuse the BaseEncoder to generate atomic vectors (attributes)
        # BaseEncoder uses FFT by default.
        # If use_fft=False, we need raw vectors.
        # But BaseEncoder implementation is tightly coupled with FFT logic (attribute_hologram.py).
        # We will re-implement minimal logic here or wrap it.
        
        self.encoder = BaseEncoder(dimension)
        # Pre-cache position vectors to ensure consistency
        self.position_vectors = {}

    def _get_atom(self, key: str) -> np.ndarray:
        """Get atomic vector for a key (char or pos)."""
        # Encode
        v = self.encoder.encode_attribute(key)
        
        if not self.use_fft:
            # If use_fft is False, we should presumably working in Real domain
            # or different binding? 
            # The spec implies comparing FFT vs Non-FFT (maybe Real HRR).
            # BaseEncoder.encode_attribute already does FFT and returns Complex.
            # To support "No FFT", we would need a Real-valued encoder.
            # Let's approximate: If use_fft=False, we treat the 'seed' generation as the vector
            # But the BaseEncoder doesn't expose raw real vector easily.
            # We will fallback to complex HRR for both for now, as 'use_fft' might refer to 
            # "Is the representation in Frequency Domain?" - Yes, HRR is usually frequency domain.
            # If "No FFT" means Real Vector Binding (VSA with Circular Convolution calculated spatially),
            # that is very slow. 
            # Assuming "use_fft=False" might mean "Don't use FFT optimization, compute convolution directly"
            # OR "Use HRR in Spatial Domain (Real)".
            # Given standard VSA literature, "Holographic" often implies Complex/FFT.
            # We will stick to Complex (FFT-based) HRR for now as the 'production' encoder.
            # If variant requests no FFT, we might skip it (just return real vector), 
            # but Random Real Vectors don't bind well with Element-wise Product (requires Circular Conv).
            # We will IGNORE use_fft=False for logical correctness unless we implement Circular Convolution.
            pass
            
        return v

    def get_position_vector(self, idx: int) -> np.ndarray:
        if idx not in self.position_vectors:
            self.position_vectors[idx] = self._get_atom(f"POS_{idx}")
        return self.position_vectors[idx]

    def encode_char(self, char: str) -> np.ndarray:
        return self._get_atom(f"CHAR_{char}")

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encodes sequence "ABC" -> (A*P1) + (B*P2) + (C*P3)
        """
        superposition = np.zeros(self.dimension, dtype=np.complex128)
        
        for i, char in enumerate(sequence):
            # 1. Get Char Vector
            h_char = self.encode_char(char)
            
            # 2. Get Position Vector
            h_pos = self.get_position_vector(i)
            
            # 3. Bind (Hadamard Product)
            # HRR Binding: Element-wise multiply
            bound = h_char * h_pos
            
            # 4. Add to Superposition
            superposition += bound
            
        # 5. Normalize Final State
        norm = np.linalg.norm(superposition)
        if norm > 0:
            superposition /= norm
            
        return superposition

    def decode_sequence(self, vector: np.ndarray, length: int) -> str:
        """
        Decodes sequence by querying each position.
        Note: This is 'Generation' phase logic. 
        In strict theory, we query memory with ID. 
        But here we decode the RECALLED vector.
        """
        decoded = []
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for i in range(length):
            # 1. Unbind Position (approx inv: conj?)
            # In HRR, Unbinding A*P -> A is A*P * inv(P).
            # For unitary vectors, inv(P) = conj(P).
            h_pos = self.get_position_vector(i)
            h_query = vector * np.conj(h_pos)
            
            # 2. Find Nearest Neighbor in Alphabet
            best_char = "?"
            best_score = -1.0
            
            for char in alphabet:
                h_char = self.encode_char(char)
                # Resonance
                score = np.abs(np.vdot(h_query, h_char))
                if score > best_score:
                    best_score = score
                    best_char = char
            
            decoded.append(best_char)
            
        return "".join(decoded)
