"""
CognitiveCore (BrainModel v2.0)

Role:
    Decision Authority & Cognitive Subject.
    Manages Recall, Reasoning, Decision, and Memory Persistence.
    
Internal State:
    Governed by CognitiveStateVector (H, C, R, B).
"""

import numpy as np
import logging
import uuid
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from coherent.core.simulation_core import SimulationCore
from coherent.core.memory.experience_manager import ExperienceManager
from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory

# --- 1. Internal State Metrics Schema ---

@dataclass
class CognitiveStateVector:
    """
    Formal mapping of CognitiveCore Internal State.
    See: cognitive_state.yaml / .schema.json
    """
    # 1. Entropy (Uncertainty) H(t) range [0, 1]
    entropy: float
    
    # 2. Confidence C(t) range [0, 1]
    confidence: float
    margin_confidence: float # C_delta
    concentration_confidence: float # C_H
    
    # 3. Recall Reliability R(t) range [0, 1]
    recall_reliability: float
    
    # 4. Branching Pressure B_tilde(t) range [0, 1]
    branching_pressure: float
    
    # Metadata
    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())

class DecisionType(Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"

@dataclass
class CognitiveDecision:
    decision_type: DecisionType
    action: str # "PROMOTE", "SUPPRESS", "ACTIVATE_SIMULATION"
    reason: str
    state_snapshot: CognitiveStateVector

# --- 2. Cognitive Core Class ---

class CognitiveCore:
    def __init__(self, experience_manager: ExperienceManager):
        self.logger = logging.getLogger("CognitiveCore")
        
        # Submodules
        self.experience_memory = experience_manager
        self.recall_engine = DynamicHolographicMemory(capacity=100) # Resonance Field
        self.simulation_core = SimulationCore() # Execution Unit
        
        # Hyperparameters (from Spec)
        self.tau = 1.0       # Softmax temperature
        self.alpha = 0.6     # Confidence composition weight
        self.theta_recall = 0.65 # Recall sigmoid center
        self.kappa = 10.0    # Recall sigmoid slope
        self.epsilon = 0.05  # Branching threshold
        
        # Internal State
        self.current_state: Optional[CognitiveStateVector] = None

    def process_input(self, input_signal: Any) -> CognitiveDecision:
        """
        Main Cognitive Loop:
        1. Recall (Resonance)
        2. Reasoning (Candidate Generation) -> Not fully impl in MVP, assumed from Recall
        3. State Metrics Calculation
        4. Decision Making
        5. (Optional) Simulation Activation
        6. Memory Update
        """
        session_id = str(uuid.uuid4())
        
        # 1. Recall & Resonance
        # Assume input_signal is converted to vector. 
        # For MVP, we mock vector generation or use a simple hash/random if no parser.
        # In real system, this comes from Parser/Encoder.
        query_vec = self._vectorize(input_signal)
        
        # Query Holographic Memory
        recall_results = self.recall_engine.query(query_vec, top_k=10)
        
        # 2. Compute State Metrics
        state_vector = self._calculate_metrics(recall_results, query_vec)
        self.current_state = state_vector
        self.logger.info(f"CognitiveState: H={state_vector.entropy:.2f}, C={state_vector.confidence:.2f}, R={state_vector.recall_reliability:.2f}")

        # 3. Decision Logic
        decision = self._make_decision(state_vector)
        
        # 4. Simulation Trigger (if needed)
        if decision.decision_type == DecisionType.REVIEW:
             # Check Simulation Trigger Policy
             if self._should_activate_simulation(state_vector):
                 self.logger.info("Activating SimulationCore...")
                 sim_result = self.simulation_core.execute_request({
                     "domain": "numeric", # Detect from input in real logic
                     "input_context": input_signal
                 })
                 
                 # Re-evaluate logic based on Sim result? 
                 # For MVP, we just accept Sim result if success.
                 if sim_result.get("status") == "SUCCESS":
                     decision = CognitiveDecision(
                         decision_type=DecisionType.ACCEPT,
                         action="PROMOTE_SIMULATED",
                         reason=f"Simulation verified: {sim_result.get('result')}",
                         state_snapshot=state_vector
                     )
        
        # 5. Experience Update (Learning)
        if decision.decision_type == DecisionType.ACCEPT:
            self._learn_experience(input_signal, decision)
            
        return decision

    # --- Metrics Calculation Implementation ---

    def _calculate_metrics(self, recall_results: List[Tuple[Any, float]], query_vec: np.ndarray) -> CognitiveStateVector:
        """
        Implements equations from Spec 3.0 - 6.0
        """
        # Extract Scores s_i(t)
        # If no results (empty memory), we simulate high entropy/low confidence
        if not recall_results:
            return CognitiveStateVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        scores = np.array([score for _, score in recall_results])
        n = len(scores)
        
        # 1. Normalize Distribution p_i(t) (Softmax)
        # p_i = exp(s_i/tau) / sum(...)
        exps = np.exp(scores / self.tau)
        p = exps / np.sum(exps)
        
        # 2. Entropy H(t)
        # H = - (1/log n) * sum(p * log p)
        # Avoid log(0)
        p_safe = np.clip(p, 1e-10, 1.0)
        if n > 1:
            raw_entropy = -np.sum(p * np.log(p_safe))
            max_entropy = np.log(n)
            H = raw_entropy / max_entropy
        else:
            H = 0.0 # Single item = 0 entropy
            
        # 3. Confidence C(t)
        # Sort p descending
        p_sorted = np.sort(p)[::-1]
        
        # Margin C_delta = p[0] - p[1]
        p_1 = p_sorted[0]
        p_2 = p_sorted[1] if n > 1 else 0.0
        C_delta = p_1 - p_2
        
        # Concentration C_H = 1 - H
        C_H = 1.0 - H
        
        # Composite C = alpha * C_delta + (1-alpha) * C_H
        C = self.alpha * C_delta + (1.0 - self.alpha) * C_H
        
        # 4. Recall Reliability R(t)
        # R = sigmoid(kappa * (I_max - theta))
        # I_max is the top raw resonance score (assuming normalized 0-1 from Engine)
        I_max = scores[0] if n > 0 else 0.0
        # Ensure I_max is 0-1. In Cosine sim it is -1 to 1. Assume we use 0-1 for reliability.
        # If I_max < 0, implies no match.
        I_max = max(0.0, I_max)
        
        R = 1.0 / (1.0 + np.exp(-self.kappa * (I_max - self.theta_recall)))
        
        # 5. Branching Pressure B_tilde(t)
        # B_epsilon = count(p_i >= epsilon)
        B_epsilon = np.sum(p >= self.epsilon)
        
        # Norm B = log(1 + B_eps) / log(1 + B_max) where B_max = n
        # Actually n is top_k (10).
        B_tilde = np.log(1.0 + B_epsilon) / np.log(1.0 + 10.0) # Assuming max branch is k=10
        
        return CognitiveStateVector(
            entropy=float(H),
            confidence=float(C),
            margin_confidence=float(C_delta),
            concentration_confidence=float(C_H),
            recall_reliability=float(R),
            branching_pressure=float(B_tilde)
        )

    def _make_decision(self, state: CognitiveStateVector) -> CognitiveDecision:
        """
        Implements Spec 8.1 Decision Rules
        """
        # Accept: C >= 0.75 AND R >= 0.6
        if state.confidence >= 0.75 and state.recall_reliability >= 0.6:
            return CognitiveDecision(DecisionType.ACCEPT, "PROMOTE", "High Confidence & Reliable Recall", state)
            
        # Reject: C < 0.4 AND R < 0.4
        elif state.confidence < 0.4 and state.recall_reliability < 0.4:
            return CognitiveDecision(DecisionType.REJECT, "SUPPRESS", "Low Confidence & Unreliable Recall", state)
            
        # Review (everything else)
        else:
            return CognitiveDecision(DecisionType.REVIEW, "DEFER_REVIEW", "Ambiguous State", state)

    def _should_activate_simulation(self, state: CognitiveStateVector) -> bool:
        """
        Spec 8.2 Simulation Trigger Policy
        Trigger if ANY:
         - R < 0.4 (Recall Failure)
         - H >= 0.6 (High Uncertainty)
         - B >= 0.7 (Branching Explosion)
         - C < 0.4 (Low Confidence)
        """
        if state.recall_reliability < 0.4: return True
        if state.entropy >= 0.6: return True
        if state.branching_pressure >= 0.7: return True
        if state.confidence < 0.4: return True
        return False
        
    def _learn_experience(self, input_signal: Any, decision: CognitiveDecision):
        # Store result in ExperienceMemory (Layer 3)
        # Spec 9: Log H, C, R, B
        meta = {
            "metrics": {
                "H": decision.state_snapshot.entropy,
                "C": decision.state_snapshot.confidence,
                "R": decision.state_snapshot.recall_reliability,
                "B": decision.state_snapshot.branching_pressure
            },
            "timestamp": decision.state_snapshot.timestamp,
            "decision": decision.decision_type.value
        }
        # Assuming ExperienceManager has a log or save method
        # self.experience_memory.log_experience(input_signal, meta)
        pass

    def _vectorize(self, input_signal: Any) -> np.ndarray:
        # Mock vectorizer
        # In real impl, use Parser/Embedding
        # For repeatable tests, we might want a determinstic hash if input is str
        if isinstance(input_signal, str):
             import hashlib
             seed = int(hashlib.md5(input_signal.encode()).hexdigest(), 16) % (2**32)
             rng = np.random.default_rng(seed)
             vec = rng.random(64)
             return vec / np.linalg.norm(vec)
        return np.random.rand(64)
