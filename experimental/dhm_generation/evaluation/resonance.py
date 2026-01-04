"""
Resonance and Evaluation Module

Implements the resonance function and symbol extraction logic.
"""

import numpy as np
from typing import Dict, List, Tuple

def calculate_resonance(h_query: np.ndarray, h_memory: np.ndarray) -> float:
    """
    R(H_q, H_m) = | H_q . conj(H_m) |
    Calculates the magnitude of the projection of query onto memory.
    Assumes vectors are L2 normalized, so this is equivalent to cosine similarity magnitude.
    """
    # Dot product: sum(q_i * conj(m_i))
    # Note: numpy.vdot(a, b) does conj(a) * b.
    # We want q . conj(m).
    # np.dot(q, np.conj(m)) is simplest.
    
    dot_prod = np.dot(h_query, np.conj(h_memory))
    return float(np.abs(dot_prod))

def evaluate_symbol_match(
    h_final: np.ndarray, 
    all_symbol_holograms: Dict[str, np.ndarray]
) -> List[Tuple[str, float]]:
    """
    Computes resonance against all candidate symbols and returns sorted results.
    """
    results = []
    for symbol, h_cand in all_symbol_holograms.items():
        score = calculate_resonance(h_final, h_cand)
        results.append((symbol, score))
        
    # Sort descending by score
    results.sort(key=lambda x: x[1], reverse=True)
    return results
