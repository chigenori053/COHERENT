from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class BranchStatus(Enum):
    ACTIVE = "active"
    PRUNED = "pruned"
    CONVERGED = "converged"
    TERMINATED = "terminated"

@dataclass
class UtilityState:
    validity: float = 0.0
    progress: float = 0.0
    novelty: float = 0.0
    confidence: float = 0.0
    risk: float = 0.0
    expected_utility: float = 0.0

@dataclass
class Hypothesis:
    """
    Represents a proposed reasoning step (candidate).
    """
    id: str
    rule_id: str
    current_expr: str
    next_expr: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    explanation: Optional[str] = None

@dataclass
class ThoughtBranch:
    # Identity
    branch_id: int
    parent_branch_id: Optional[int] = None

    # Lifecycle
    status: BranchStatus = BranchStatus.ACTIVE
    step_count: int = 0

    # Core State
    current_ast: Any = None
    goal_ast: Any = None
    last_action: Optional[str] = None

    # Branch-local working memory
    working_memory: Dict[str, Any] = field(default_factory=dict)
    hypothesis_queue: List[Any] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    local_constraints: Dict[str, Any] = field(default_factory=dict)
    trace_log: List[str] = field(default_factory=list)

    # Recall context
    recall_threshold: float = 0.7
    resonance_history: List[float] = field(default_factory=list)
    recalled_memory_ids: List[str] = field(default_factory=list)

    # Policy knobs
    exploration_bias: float = 0.5  # 0.0=conservative, 1.0=exploratory
    lookahead_depth: int = 2
    risk_preference: float = 0.5   # 0.0=risk-averse, 1.0=risk-seeking

    # Evaluation
    utility: UtilityState = field(default_factory=UtilityState)
