"""
Attribute Hologram Module

Defines attribute categories, alphabet mapping, and holographic encoding logic.
Strictly separates definition from storage.
"""

import numpy as np
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

from ..config import DEFAULT_CONFIG

# --- 3. Attribute System ---

ATTRIBUTE_CATEGORIES = {
    "TYPE": ["letter"],
    "CASE": ["uppercase"],
    "POSITION": ["early", "middle", "late"],
    "ROLE": ["vowel", "consonant"],
    "SHAPE": ["angular", "round", "mixed"],
    "COMPLEXITY": ["simple", "intermediate", "complex"],
    "PHONETIC_CLASS": ["vowel", "plosive", "fricative", "nasal", "approximant", "affricate"],
}

# --- 4. Alphabet Attribute Mapping ---
# Static lookup table, used ONLY to build queries.

ALPHABET_MAPPING: Dict[str, Dict[str, str]] = {
    "A": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "vowel", "SHAPE": "angular", "COMPLEXITY": "complex", "PHONETIC_CLASS": "vowel"},
    "B": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "consonant", "SHAPE": "round", "COMPLEXITY": "complex", "PHONETIC_CLASS": "plosive"},
    "C": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "consonant", "SHAPE": "round", "COMPLEXITY": "simple", "PHONETIC_CLASS": "plosive"},
    "D": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "consonant", "SHAPE": "mixed", "COMPLEXITY": "simple", "PHONETIC_CLASS": "plosive"},
    "E": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "vowel", "SHAPE": "angular", "COMPLEXITY": "intermediate", "PHONETIC_CLASS": "vowel"},
    "F": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "intermediate", "PHONETIC_CLASS": "fricative"},
    "G": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "consonant", "SHAPE": "mixed", "COMPLEXITY": "complex", "PHONETIC_CLASS": "plosive"},
    "H": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "simple", "PHONETIC_CLASS": "fricative"},
    "I": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "early", "ROLE": "vowel", "SHAPE": "angular", "COMPLEXITY": "simple", "PHONETIC_CLASS": "vowel"},
    "J": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "consonant", "SHAPE": "round", "COMPLEXITY": "simple", "PHONETIC_CLASS": "affricate"},
    "K": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "intermediate", "PHONETIC_CLASS": "plosive"},
    "L": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "simple", "PHONETIC_CLASS": "approximant"},
    "M": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "complex", "PHONETIC_CLASS": "nasal"},
    "N": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "intermediate", "PHONETIC_CLASS": "nasal"},
    "O": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "vowel", "SHAPE": "round", "COMPLEXITY": "simple", "PHONETIC_CLASS": "vowel"},
    "P": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "consonant", "SHAPE": "round", "COMPLEXITY": "intermediate", "PHONETIC_CLASS": "plosive"},
    "Q": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "middle", "ROLE": "consonant", "SHAPE": "round", "COMPLEXITY": "complex", "PHONETIC_CLASS": "plosive"}, # Q is roughly O + tail
    "R": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "mixed", "COMPLEXITY": "complex", "PHONETIC_CLASS": "approximant"},
    "S": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "round", "COMPLEXITY": "simple", "PHONETIC_CLASS": "fricative"},
    "T": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "simple", "PHONETIC_CLASS": "plosive"},
    "U": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "vowel", "SHAPE": "round", "COMPLEXITY": "simple", "PHONETIC_CLASS": "vowel"},
    "V": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "simple", "PHONETIC_CLASS": "fricative"},
    "W": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "complex", "PHONETIC_CLASS": "approximant"},
    "X": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "simple", "PHONETIC_CLASS": "affricate"},
    "Y": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "mixed", "COMPLEXITY": "intermediate", "PHONETIC_CLASS": "approximant"},
    "Z": {"TYPE": "letter", "CASE": "uppercase", "POSITION": "late", "ROLE": "consonant", "SHAPE": "angular", "COMPLEXITY": "complex", "PHONETIC_CLASS": "fricative"},
}

def get_attributes_for_symbol(symbol: str) -> List[str]:
    """Retrieve attribute list for a given symbol."""
    if symbol not in ALPHABET_MAPPING:
        raise ValueError(f"Symbol {symbol} not found in mapping.")
    
    attr_dict = ALPHABET_MAPPING[symbol]
    # Return list of attribute values
    return list(attr_dict.values())


class HolographicEncoder:
    """
    Handles the encoding of string attributes into holographic vectors (complex128).
    """
    def __init__(self, dimension: int = DEFAULT_CONFIG.dimension):
        self.dimension = dimension

    def encode_attribute(self, attribute_name: str) -> np.ndarray:
        """
        Encode an attribute string into a holographic vector.
        Pipeline: Seed(Name) -> Real Vector -> Zero Pad -> FFT -> Normalize
        """
        # 1. Deterministic seed from attribute name
        seed = int(hashlib.sha256(attribute_name.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        # 2. Generate real vector (using a normal distribution for spectral diversity)
        # Using dimension // 2 points initially for FFT symmetry properties if real input
        # But spec says: "Generate real vector -> Zero-pad to D -> FFT"
        
        # Let's interpret "Zero-pad to D" as sparse generation or simply filling a buffer.
        # A common valid HRR approach: Random phases or Real Gaussian -> FFT.
        
        # Implementation per spec:
        # Generate real vector. Size isn't strictly specified, let's assume D.
        real_vec = rng.standard_normal(self.dimension)
        
        # 3. Zero-pad to D (Already D, assuming full density. If "Zero-pad" meant sparse, we'd adjust)
        # If the intent is to have low frequency content, we might generate fewer points then pad.
        # However, "Attribute Encoding Pipeline" usually implies full rank for orthogonality.
        # Let's stick to standard HRR: Real Gaussian -> FFT -> Normalize
        
        # 4. FFT -> complex vector
        complex_vec = np.fft.fft(real_vec)
        
        # 5. L2 normalization
        norm = np.linalg.norm(complex_vec)
        if norm < 1e-9:
            raise ValueError(f"Vector norm too small for {attribute_name}")
            
        return complex_vec / norm

    def create_superposition(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Create a superposition of vectors.
        Note: The spec mentions "H_0 = Normalize( ⊙_{attr} H_attr )" for query construction.
        The symbol ⊙ usually means Hadamard (element-wise) product which is binding (like XOR),
        while + is superposition (like OR).
        
        For HRR, "Attributes of a symbol" forming the symbol typically uses BINDING (element-wise mul)
        if we consider the symbol to be the conjunction of attributes.
        
        Spec Section 7: "H_0 = Normalize( ⊙_{attr ∈ Attributes(c)} H_attr )"
        This confirms ELEMENT-WISE PRODUCT (Binding).
        """
        if not vectors:
            raise ValueError("No vectors provided for superposition.")
            
        result = vectors[0]
        for vec in vectors[1:]:
            result = result * vec  # Hadamard product
            # Numerical stability: normalize at each step to prevent underflow
            # because L2-normalized vectors have small component magnitudes (~1/sqrt(D)),
            # and repeated product decays magnitude exponentially.
            norm_step = np.linalg.norm(result)
            if norm_step > 1e-15:
                result = result / norm_step
            
        # Normalize
        norm = np.linalg.norm(result)
        if norm < 1e-9:
             # Fallback or error? A zero vector implies orthogonality failure or destructive interference.
             raise ValueError("Resulting vector has near-zero norm.")
             
        return result / norm

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return vec # Or error
        return vec / norm
