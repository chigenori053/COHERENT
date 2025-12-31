from ..types import Action, StateDistribution
from ..observation import Observation, MicroVariationObservation
from ..inference import MicroVariationInference
from ..decision import DecisionEngine
from ..utility import UtilityTable

class MicroVariationOptimizer:
    def __init__(self, inference: MicroVariationInference, decision_engine: DecisionEngine):
        self.inference = inference
        self.decision_engine = decision_engine

    def optimize(self, 
                 base_obs: Observation, 
                 micro_obs: MicroVariationObservation, 
                 existing_hologram: object, 
                 new_hologram: object) -> Action:
        """
        Determines the optimal action for a detected micro-variation.
        Does NOT return 'Accept' or 'Review' typically, but rather 'Merge', 'Variant', 'Absorb', 'Reject'.
        However, it uses the standard DecisionEngine, so it follows the utility landscape.
        """
        
        # 1. Infer specific micro-state
        state_dist = self.inference.infer(base_obs, micro_obs)
        
        # 2. Compute Utility for this micro-context
        eu = self.decision_engine.compute_expected_utility(state_dist)
        
        # 3. Decide
        action = self.decision_engine.decide(eu)
        
        return action
