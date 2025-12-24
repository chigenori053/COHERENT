
from dataclasses import dataclass, field
from typing import Any, Dict
from .action_types import ActionType

@dataclass
class Action:
    type: ActionType
    name: str = "unknown"
    inputs: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
