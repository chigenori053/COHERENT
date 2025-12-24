
from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class State:
    task_goal: Any = None
    initial_inputs: List[Any] = field(default_factory=list)
    current_expression: str = ""
    
    @classmethod
    def from_string(cls, expr: str):
        return cls(current_expression=expr)
