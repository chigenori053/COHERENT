from typing import Dict, Any, Optional
from ..types import Action

class HumanUtilityShaper:
    def __init__(self):
        # Stores delta modifiers: { StateName: { Action: DeltaFloat } }
        self.modifiers: Dict[str, Dict[Action, float]] = {}

    def set_modifier(self, state: str, action: Action, delta: float):
        """
        Sets a utility modifier. 
        e.g. set_modifier("TrulyUnique", Action.STORE_NEW, 0.5)
        This increases the utility of storing unique items by 0.5.
        """
        if state not in self.modifiers:
            self.modifiers[state] = {}
        self.modifiers[state][action] = delta

    def get_modifier(self, state: str, action: Action) -> float:
        return self.modifiers.get(state, {}).get(action, 0.0)
        
    def apply(self, base_utils: Dict[Action, float], state: str) -> Dict[Action, float]:
        """
        Applies modifiers to a dictionary of {Action: Utility}.
        Used if we want to get a shaped vector for a specific state.
        """
        new_utils = base_utils.copy()
        if state in self.modifiers:
            for act, delta in self.modifiers[state].items():
                if act in new_utils:
                    new_utils[act] += delta
        return new_utils
