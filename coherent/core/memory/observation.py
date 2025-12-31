from dataclasses import dataclass

@dataclass
class Observation:
    """
    Basic observation metrics derived from the hologram.
    All values should be normalized (continuously) where appropriate.
    """
    max_resonance: float
    resonance_mean: float
    resonance_variance: float
    phase_distance: float
    interference_score: float
    snr: float
    novelty_score: float
    memory_density: float

@dataclass
class MicroVariationObservation:
    """
    Detailed observation metrics for comparing with a specific candidate for micro-variation.
    Used when a potential duplicate or variant is detected.
    """
    semantic_overlap: float
    phase_offset: float
    contextual_divergence: float
    applicability_delta: float
    recall_conflict_rate: float
