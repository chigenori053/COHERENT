from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any

class Action(Enum):
    STORE_NEW = "store_new"
    MERGE_SOFT = "merge_soft"
    VARIANT_LINK = "variant_link"
    REVIEW = "review"
    ABSORB = "absorb"
    REJECT = "reject"
    # Additional actions for maintenance or future extensibility could be added here
    # but strictly following the spec for now.

@dataclass
class StateDistribution:
    """
    Represents the probability distribution over possible states inferred from observation.
    e.g., {"TrulyUnique": 0.8, "Variant": 0.1, ...}
    """
    probs: Dict[str, float] = field(default_factory=dict)

    def get(self, state: str) -> float:
        return self.probs.get(state, 0.0)

@dataclass
class ExpectedUtility:
    """
    Represents the calculated expected utility for each possible Action.
    """
    values: Dict[Action, float] = field(default_factory=dict)

    def get(self, action: Action) -> float:
        return self.values.get(action, -float('inf'))
