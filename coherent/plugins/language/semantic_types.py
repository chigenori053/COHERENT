
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional

class TaskType(Enum):
    SOLVE = auto()
    VERIFY = auto()
    EXPLAIN = auto()
    PLOT = auto()
    UNKNOWN = auto()

@dataclass
class InputItem:
    value: str
    type: str = "math"

@dataclass
class SemanticIR:
    task: TaskType
    inputs: List[InputItem] = field(default_factory=list)
