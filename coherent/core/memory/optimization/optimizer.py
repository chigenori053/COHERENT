from typing import Any, Tuple

from ..types import Action
from ..observation import Observation, MicroVariationObservation
from ..inference import UniquenessInference, MicroVariationInference
from ..utility import UtilityTable
from ..human.utility_shaping import HumanUtilityShaper
from ..decision import DecisionEngine
from ..logging.decision_log import DecisionLog
from .micro_variation import MicroVariationOptimizer
from ..hologram.encoder import HolographicEncoder

class MemorySpaceOptimizer:
    def __init__(self, 
                 uniqueness_inference: UniquenessInference,
                 micro_inference: MicroVariationInference,
                 decision_engine: DecisionEngine,
                 micro_optimizer: MicroVariationOptimizer,
                 encoder: HolographicEncoder):
        self.uniqueness_inference = uniqueness_inference
        self.micro_inference = micro_inference
        self.decision_engine = decision_engine
        self.micro_optimizer = micro_optimizer
        self.encoder = encoder

    def process(self, hologram: Any, context_obs: Observation) -> Tuple[Action, DecisionLog]:
        """
        Orchestrates the optimization process for a new hologram.
        
        1. Observe (Already passed as context_obs or derived here? 
           Spec says `obs = self.observe(hologram_new)`. 
           Ideally we need an 'Observer' component. 
           I'll assume context_obs is passed or we have a method to derive it.)
        2. Infer State
        3. Micro-Observation (if applicable)
        4. Evaluate EU
        5. Decide
        6. Return Action & Log
        """
        
        # 1. Observation (Passed in or derived)
        # For this implementation, we assume 'context_obs' is the primary observation
        obs = context_obs
        
        # 2. Inference
        state_dist = self.uniqueness_inference.infer(obs)
        
        # 3. Micro-Observation Check
        # If the state suggests potential redundancy/variance, we calculate micro-obs.
        # This logic flow depends on how expensive micro-obs is.
        # For simplicity, if P(Variant) + P(Redundant) > threshold (soft check via Utility?)
        # Strictly "All EU -> argmax".
        # So we should probably always compute Micro-Obs if we have a candidate target?
        # Or maybe the 'Observation' already includes data about "nearest neighbor".
        
        # Let's assume we proceed to decision with the broad state first.
        # If the decision is 'MERGE_SOFT' or 'VARIANT_LINK' or 'ABSORB',
        # we might need the specific target which implies micro-obs was done.
        
        # To strictly follow "Probability x Utility -> argmax", we compute EU using the state_dist.
        eu = self.decision_engine.compute_expected_utility(state_dist)
        
        # 4. Decide Base Action
        action = self.decision_engine.decide(eu)
        
        # 5. Logging
        log = DecisionLog(
            observation=obs,
            state_distribution=state_dist,
            expected_utility=eu,
            action=action
        )
        
        return action, log

    def optimize_micro(self, 
                       base_obs: Observation, 
                       micro_obs: MicroVariationObservation, 
                       existing_hologram: Any, 
                       new_hologram: Any) -> Tuple[Action, DecisionLog]:
        """
        Specific flow for when a collision/relation is detected and we need micro-optimization.
        """
        action = self.micro_optimizer.optimize(base_obs, micro_obs, existing_hologram, new_hologram)
        
        # Log this micro-decision?
        # Ideally yes.
        # Re-construct Log.
        # (This is simplified, real log would need EU from micro-optimizer)
        
        return action, DecisionLog(
            observation=base_obs,
            micro_observation=micro_obs,
            action=action,
            execution_context={"type": "micro_optimization"}
        )
