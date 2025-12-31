from dataclasses import dataclass, field
from typing import Dict, Any
import time

from ..types import Action, StateDistribution, ExpectedUtility
from ..observation import Observation, MicroVariationObservation

@dataclass
class DecisionLog:
    """
    Log record for a single optimization decision.
    Contains everything needed to re-calculate (replay) the decision logic.
    """
    timestamp: float = field(default_factory=time.time)
    observation: Observation = None
    micro_observation: MicroVariationObservation = None
    state_distribution: StateDistribution = None
    expected_utility: ExpectedUtility = None
    action: Action = None
    execution_context: Dict[str, Any] = field(default_factory=dict) # e.g. input ID, user metadata

    def to_dict(self) -> Dict[str, Any]:
        # Helper for serialization
        return {
            "timestamp": self.timestamp,
            "observation": vars(self.observation) if self.observation else None,
            "micro_observation": vars(self.micro_observation) if self.micro_observation else None,
            "state_distribution": vars(self.state_distribution) if self.state_distribution else None,
            "expected_utility": vars(self.expected_utility) if self.expected_utility else None,
            "action": self.action.value if self.action else None,
            "execution_context": self.execution_context
        }
