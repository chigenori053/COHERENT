from typing import Dict
from .observation import Observation, MicroVariationObservation
from .types import StateDistribution

class UniquenessInference:
    def infer(self, obs: Observation) -> StateDistribution:
        """
        Returns P(TrulyUnique, Variant, Redundant, Noisy) based on observation.
        Initial implementation uses a heuristic mapping to probabilities.
        """
        # Placeholder logic as per spec (Step 4 allows "initial implementation Softmax / Naive Bayes compatible")
        
        # Heuristic scoring
        score_unique = obs.novelty_score * (1.0 - obs.max_resonance)
        score_variant = obs.max_resonance * (1.0 - obs.novelty_score)
        score_redundant = obs.max_resonance * obs.memory_density
        score_noisy = obs.interference_score
        
        # Normalize to probability
        total = score_unique + score_variant + score_redundant + score_noisy + 1e-9
        
        probs = {
            "TrulyUnique": score_unique / total,
            "Variant": score_variant / total,
            "Redundant": score_redundant / total,
            "Noisy": score_noisy / total
        }
        
        return StateDistribution(probs=probs)

class MicroVariationInference:
    def infer(self, obs: Observation, micro: MicroVariationObservation) -> StateDistribution:
        """
        Returns P(Meaningful, Benign, Redundant, Harmful) based on micro-observation.
        """
        # Heuristic scoring for micro-variation
        score_meaningful = micro.semantic_overlap * micro.applicability_delta
        score_benign = micro.semantic_overlap * (1.0 - micro.contextual_divergence)
        score_redundant = 1.0 - micro.phase_offset
        score_harmful = micro.recall_conflict_rate
        
        total = score_meaningful + score_benign + score_redundant + score_harmful + 1e-9
        
        probs = {
            "Meaningful": score_meaningful / total,
            "Benign": score_benign / total,
            "Redundant": score_redundant / total,
            "Harmful": score_harmful / total
        }
        
        return StateDistribution(probs=probs)
