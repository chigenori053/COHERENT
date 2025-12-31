from typing import Dict, Any
from .types import Action

class UtilityTable:
    def __init__(self):
        # Default utility matrix: Action x State
        # This could be loaded from YAML/JSON
        self.matrix: Dict[str, Dict[Action, float]] = {
            "TrulyUnique": {
                Action.STORE_NEW: 1.0,
                Action.MERGE_SOFT: 0.2,
                Action.VARIANT_LINK: 0.5,
                Action.REVIEW: 0.1,
                Action.ABSORB: -0.5,
                Action.REJECT: -1.0
            },
            "Variant": {
                Action.STORE_NEW: -0.5,
                Action.MERGE_SOFT: 0.6,
                Action.VARIANT_LINK: 1.0, # High utility for linking variants
                Action.REVIEW: 0.3,
                Action.ABSORB: 0.1,
                Action.REJECT: -0.5
            },
            "Redundant": {
                Action.STORE_NEW: -1.0,
                Action.MERGE_SOFT: 0.8,
                Action.VARIANT_LINK: 0.2,
                Action.REVIEW: 0.0,
                Action.ABSORB: 0.9, # High utility for absorbing redundancy
                Action.REJECT: 0.0
            },
            "Noisy": {
                Action.STORE_NEW: -1.0,
                Action.MERGE_SOFT: -0.5,
                Action.VARIANT_LINK: -0.5,
                Action.REVIEW: 0.5,
                Action.ABSORB: -0.5,
                Action.REJECT: 1.0 # High utility for rejecting noise
            },
            # Micro-Variation States
            "Meaningful": {
                Action.VARIANT_LINK: 1.0,
                Action.MERGE_SOFT: 0.5,
                Action.ABSORB: -0.2
            },
            "Benign": {
                Action.VARIANT_LINK: 0.5,
                Action.MERGE_SOFT: 0.8,
                Action.ABSORB: 0.2
            },
            # Redundant state is shared or similar
            "Harmful": {
                Action.REJECT: 1.0,
                Action.REVIEW: 0.8,
                Action.STORE_NEW: -1.0
            }
        }

    def base_utility(self, action: Action, state: str) -> float:
        """
        Returns the base utility of taking `action` given `state`.
        Returns 0.0 if not defined (neutral).
        """
        state_utils = self.matrix.get(state, {})
        return state_utils.get(action, 0.0)


