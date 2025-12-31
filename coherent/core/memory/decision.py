from .types import Action, StateDistribution, ExpectedUtility
from .utility import UtilityTable
from .human.utility_shaping import HumanUtilityShaper

class DecisionEngine:
    def __init__(self, utility_table: UtilityTable, human_shaper: HumanUtilityShaper = None):
        self.utility_table = utility_table
        self.human_shaper = human_shaper

    def compute_expected_utility(self, state_dist: StateDistribution) -> ExpectedUtility:
        """
        Computes E[U(a)] = Sum_{s} P(s) * U(a, s)
        """
        eu_values = {}
        
        # Iterate over all possible actions defined in Enum
        for action in Action:
            expected_val = 0.0
            for state, prob in state_dist.probs.items():
                # Get base utility
                u = self.utility_table.base_utility(action, state)
                
                # Apply human shaping if applicable (per state-action pair ideally, 
                # but simplistic application here would be:
                # effective_u = u + shaping_bias
                if self.human_shaper:
                    # Retrieve modifier for this specific state-action
                    # (Implementation detail: HumanUtilityShaper might need a direct accessor)
                    modifiers = self.human_shaper.modifiers.get(state, {})
                    u += modifiers.get(action, 0.0)
                
                expected_val += prob * u
            
            eu_values[action] = expected_val
            
        return ExpectedUtility(values=eu_values)

    def decide(self, eu: ExpectedUtility) -> Action:
        """
        Returns argmax Action.
        """
        if not eu.values:
            return Action.REVIEW # Fallback safety
            
        return max(eu.values, key=eu.values.get)
