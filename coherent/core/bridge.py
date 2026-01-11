"""
Core Bridge: Interface between Cortex (Core A) and Logic (Core B).
Defines shared data structures for handoff.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass
class InterchangeData:
    """
    Standard packet for Cortex -> Logic handoff.
    """
    hypothesis_id: str
    abstract_pattern: Any # HolographicTensor or abstract symbol
    resonance_score: float
    entropy: float
    rationale: str # Metadata or explanation
    
    # Optional feedback channel from Logic
    execution_result: Optional[bool] = None
    error_metric: Optional[float] = None
