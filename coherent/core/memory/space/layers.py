from dataclasses import dataclass, field
from typing import Any, Optional
from ..types import Action
from ..logging.decision_log import DecisionLog

@dataclass
class ProcessingResult:
    """
    Encapsulates the result of a MemorySpace operation.
    Corresponds to the "Accept / Review / Reject" layer concept - 
    this object carries the final status (Action) and the proof (Log).
    """
    action: Action
    log: DecisionLog
    hologram_ref: Optional[Any] = None # Reference to the stored/merged hologram
    message: str = ""
    is_success: bool = True

    @property
    def status(self) -> str:
        return self.action.value
