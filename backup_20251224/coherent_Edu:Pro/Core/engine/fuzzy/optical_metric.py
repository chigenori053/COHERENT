"""Optical similarity metric using complex interference."""

from __future__ import annotations

import torch
from .encoder import MLVector

class OpticalSimilarityMetric:
    """
    Computes similarity using Optical Interference principles.
    treats vectors as complex wave amplitudes.
    Similarity = Resonance Energy / (Energy_1 * Energy_2)
    """

    def similarity(self, v1: MLVector, v2: MLVector) -> float:
        # Convert to tensors
        # v1.data is tuple[float]
        t1 = torch.tensor(v1.data, dtype=torch.float32)
        t2 = torch.tensor(v2.data, dtype=torch.float32)
        
        # Convert to Complex (Holographic Projection)
        # Real -> Amplitude, Phase -> 0 (for now)
        # In future, phase could encode sequencing or structural depth
        c1 = t1.type(torch.cfloat)
        c2 = t2.type(torch.cfloat)
        
        # Interference / Dot Product
        # <u, v> = u * v.conj()
        interference = torch.sum(c1 * c2.conj())
        
        # Resonance = Intensity (Magnitude Squared)
        resonance = interference.abs().pow(2)
        
        # Normalization (Energy)
        # E = |u|^2
        e1 = torch.sum(c1.abs().pow(2))
        e2 = torch.sum(c2.abs().pow(2))
        
        if e1 == 0 or e2 == 0:
            return 0.0
            
        # Normalized Resonance
        # For unit vectors, this is cos^2(theta)
        # Standard cosine sim is cos(theta).
        # We return sqrt(Resonance) to align with linear expectation [0, 1]
        # or keep it squared to punish weak matches?
        # Let's return the linear magnitude for compatibility with existing thresholds.
        # i.e. |<u,v>| / (|u||v|)
        
        metric = interference.abs() / (torch.sqrt(e1) * torch.sqrt(e2))
        
        return float(metric.item())
