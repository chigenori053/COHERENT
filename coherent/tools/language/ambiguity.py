from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ClarificationRequest(BaseModel):
    """
    Represents a request for clarification when an input is ambiguous.
    """
    original_text: str
    ambiguity_score: float
    possible_intents: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = "The request is ambiguous. Please clarify."
