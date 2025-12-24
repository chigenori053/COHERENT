
from enum import Enum, auto

class ActionType(Enum):
    APPLY_RULE = auto()
    CALL_TOOL = auto()
    RECALL = auto()
    ASK = auto()
    FINAL = auto()
    REJECT = auto()
