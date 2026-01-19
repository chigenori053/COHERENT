"""
ObservationCore (BrainModel v2.0)

Purpose:
    Observes the behavior of the BrainModel (Inference, Decision, Expression, Context)
    and externalizes its state as "description" rather than "evaluation".
    It does NOT improve performance, make judgments, or alter inference.
    Its sole purpose is to verify whether the BrainModel is in a state where it can grow soundly.
    
    See: docs/specs/observation_core_spec_v1.0.md
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import datetime
import logging
import json

# --- 1. Data Structures ---

class ObservationState(Enum):
    """
    State Labels representing the "Health" or "Status" of the inference process.
    Not judgments (Good/Bad), but descriptive states.
    """
    STRUCTURALLY_STABLE = "STRUCTURALLY_STABLE"           # Normal operation
    STRUCTURAL_DRIFT_DETECTED = "STRUCTURAL_DRIFT_DETECTED" # Logic might be shifting
    
    RECALL_UNDERUTILIZED = "RECALL_UNDERUTILIZED"         # Not using memory enough
    RECALL_EMPTY = "RECALL_EMPTY"                         # No memory found
    
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"                 # Entropy is high
    DOGMATIC_CERTAINTY = "DOGMATIC_CERTAINTY"             # Too high confidence with low evidence
    
    ETHICALLY_SENSITIVE = "ETHICALLY_SENSITIVE"           # Potential ethical risk
    CONTEXT_INSUFFICIENT = "CONTEXT_INSUFFICIENT"         # Missing necessary context
    
    SIMULATION_TRIGGERED = "SIMULATION_TRIGGERED"         # Simulation was used
    SIMULATION_FAILED = "SIMULATION_FAILED"               # Simulation failed

@dataclass
class InferenceEvent:
    """
    Represents a single "Inference" unit to be observed.
    """
    event_id: str
    session_id: str
    timestamp: float
    
    # 1. Inputs & Context
    input_content: str  # Serialized input
    context_tags: List[str]
    
    # 2. Process Info
    input_modality: str # "text", "image", etc.
    recall_source: str  # "Holographic", "None"
    recall_score: float # Max resonance/similarity
    
    # 3. Decision & Metrics
    decision_type: str  # "ACCEPT", "REVIEW", "REJECT"
    confidence_score: float
    entropy_score: float
    
    # 4. Outcome
    final_action: str
    
    # Optional extensions
    details: Dict[str, Any] = field(default_factory=dict)


class ObservationResult:
    """
    The output of the observation process.
    """
    def __init__(self, event_id: str, states: List[ObservationState]):
        self.event_id = event_id
        self.states = states
        self.timestamp = datetime.datetime.now().timestamp()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "states": [s.value for s in self.states]
        }
    
    def __str__(self):
        return f"[Observation] Event {self.event_id}: {', '.join([s.value for s in self.states])}"


# --- 2. Observation Core ---

class ObservationCore:
    def __init__(self):
        self.logger = logging.getLogger("ObservationCore")
        # In MVP, we might simple log to a file or stdout
        # Ideally, this should hook into a separate monitor
        
    def observe(self, event: InferenceEvent) -> ObservationResult:
        """
        Main entry point. Receives an event, evaluates it, and emits a state.
        Does NOT alter the event or the control flow.
        """
        try:
            # 1. Evaluate State
            states = self._evaluate_state(event)
            
            # 2. Construct Result
            result = ObservationResult(event.event_id, states)
            
            # 3. Emit / Persist
            self._emit(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Observation failed: {e}")
            # Fail safe: return empty result, do not crash application
            return ObservationResult(event.event_id, [])

    def _evaluate_state(self, event: InferenceEvent) -> List[ObservationState]:
        """
        Pure logic to map Event -> States.
        """
        states = []
        
        # --- A. Structural Health ---
        
        # Uncertainty
        if event.entropy_score > 0.6:
            states.append(ObservationState.HIGH_UNCERTAINTY)
        
        # Dogmatism (High Confidence but Low Recall)
        if event.confidence_score > 0.9 and event.recall_score < 0.2:
            states.append(ObservationState.DOGMATIC_CERTAINTY)
            
        # Recall Health
        if event.recall_score < 0.1:
            states.append(ObservationState.RECALL_EMPTY)
        elif event.recall_score < 0.3:
            states.append(ObservationState.RECALL_UNDERUTILIZED)
            
        # --- B. Simulation Status (if present in details) ---
        if event.details.get("simulation_active", False):
            states.append(ObservationState.SIMULATION_TRIGGERED)
            if event.details.get("simulation_status") == "FAILURE":
                states.append(ObservationState.SIMULATION_FAILED)
        
        # Default State if nothing special
        if not states:
            states.append(ObservationState.STRUCTURALLY_STABLE)
            
        return states

    def _emit(self, result: ObservationResult):
        """
        Output the observation.
        """
        # MVP: Log to logger
        self.logger.info(str(result))
        
        # Future: Write to dedicated 'brain_health.log' or stream to dashboard
