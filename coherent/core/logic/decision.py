
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

class DecisionState(Enum):
    """
    Trinary Logic Decision States for COHERENT Core.
    """
    ACCEPT = "ACCEPT"   # 確定: Constraints satisfied, Single Candidate.
    REVIEW = "REVIEW"   # 保留: Constraints satisfied, Multiple Candidates (Ambiguity).
    REJECT = "REJECT"   # 拒絶: Constraint violation or Contradiction.

@dataclass
class ReviewState:
    """
    Detailed state information for REVIEW decisions.
    """
    candidate_set: List[int]
    unresolved_constraints: List[str] = field(default_factory=list)
    wait_condition: str = "More Information Required"

@dataclass
class DecisionRecord:
    """
    Standardized record for any decision event in the reasoning chain.
    """
    step_id: int
    target_cell: Tuple[int, int]
    candidate_set: List[int] # List of integers (typically) for Sudoku
    decision_state: DecisionState
    decision_reason: str
    supporting_constraints: List[str] = field(default_factory=list)
    
    # State Transition Tracking
    decision_state_before: Optional[DecisionState] = None
    decision_state_after: Optional[DecisionState] = None
    
    # Detailed Review Info (Only populated if state is REVIEW)
    review_details: Optional[ReviewState] = None

    # Recall / Compute visualization
    recall_attempted: bool = False
    recall_confidence: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Helper for JSON serialization."""
        return {
            "step_id": self.step_id,
            "target_cell": self.target_cell,
            "candidate_set": self.candidate_set,
            "decision_state": self.decision_state.value,
            "decision_reason": self.decision_reason,
            "supporting_constraints": self.supporting_constraints,
            "decision_state_before": self.decision_state_before.value if self.decision_state_before else None,
            "decision_state_after": self.decision_state_after.value if self.decision_state_after else None,
            "recall_attempted": self.recall_attempted,
            "recall_confidence": self.recall_confidence,
            "review_details": {
                "candidate_set": self.review_details.candidate_set,
                "unresolved_constraints": self.review_details.unresolved_constraints,
                "wait_condition": self.review_details.wait_condition
            } if self.review_details else None
        }
