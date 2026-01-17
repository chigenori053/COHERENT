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
from dataclasses import asdict

from coherent.core.simulation_core import SimulationCore
from coherent.core.memory.experience_manager import ExperienceManager
from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.simulation_core import SimulationCore
from coherent.core.memory.experience_manager import ExperienceManager
from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.reasoning_engine import ReasoningEngine, Hypothesis

# --- 0. Tracing / Visualization Schema ---

@dataclass
class CognitiveEvent:
    step: str # "Recall", "Reasoning", "Decision", "Simulation", "Experience"
    description: str
    metrics: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CognitiveTrace:
    session_id: str
    input_content: Any
    events: List[CognitiveEvent] = field(default_factory=list)
    final_decision: Optional['CognitiveDecision'] = None


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
        self.reasoning_engine = ReasoningEngine() # Reasoning Layer
        self.simulation_core = SimulationCore() # Execution Unit
        
        # Hyperparameters (from Spec)
        self.tau = 0.25      # Softmax temperature (Tuned 2026-01-15)
        self.alpha = 0.6     # Confidence composition weight
        self.theta_recall = 0.65 # Recall sigmoid center
        self.kappa = 10.0    # Recall sigmoid slope
        self.epsilon = 0.05  # Branching threshold
        
        # Internal State
        self.current_state: Optional[CognitiveStateVector] = None
        self.current_trace: Optional[CognitiveTrace] = None # Visualization Trace

    def process_input(self, input_signal: Any, context_config: Optional[Dict[str, Any]] = None) -> 'CognitiveDecision':
        """
        Main Cognitive Loop:
        1. Recall (Resonance)
        2. Reasoning (Candidate Generation)
        3. State Metrics Calculation
        4. Decision Making
        5. (Optional) Simulation Activation
        6. Memory Update
        """
        if context_config is None:
            context_config = {}
            
        session_id = str(uuid.uuid4())
        # Flatten input for display if it's a dict (multimodal)
        display_content = str(input_signal)[:100] + "..." if len(str(input_signal)) > 100 else str(input_signal)
        self.current_trace = CognitiveTrace(session_id=session_id, input_content=display_content)
        
        # 1. Recall & Resonance
        # Assume input_signal is converted to vector. 
        query_vec = self._vectorize(input_signal)
        
        # Query Holographic Memory
        recall_results = self.recall_engine.query(query_vec, top_k=10)
        
        self._trace_event("Recall", "Query Holographic Memory", {
            "top_score": recall_results[0][1] if recall_results else 0.0,
            "count": len(recall_results)
        }, {"top_k": [{"content": str(c), "score": float(s)} for c, s in recall_results]})
        
        # 2. Reasoning (Candidate Generation)
        hypotheses = self.reasoning_engine.generate_hypotheses(recall_results, query_vec)
        
        self._trace_event("Reasoning", "Generate Hypotheses", {
            "count": len(hypotheses),
            "abductive": any(h.source == "Abduction" for h in hypotheses)
        }, {"hypotheses": [asdict(h) for h in hypotheses]}) # Need dataclasses.asdict import if strictly typed, but let's assume dict conversion or simple list
        
        # 3. Compute State Metrics
        state_vector = self._calculate_metrics(hypotheses)
        self.current_state = state_vector
        self.logger.info(f"CognitiveState: H={state_vector.entropy:.2f}, C={state_vector.confidence:.2f}, R={state_vector.recall_reliability:.2f}")

        # 4. Decision Logic
        decision = self._make_decision(state_vector)
        
        self._trace_event("Decision", f"Made Decision: {decision.decision_type.name}", {
            "entropy": state_vector.entropy,
            "confidence": state_vector.confidence,
            "recall_reliability": state_vector.recall_reliability,
            "branching_pressure": state_vector.branching_pressure,
            "decision": decision.decision_type.name 
        }, {"reason": decision.reason, "action": decision.action})

        
        # 5. Simulation Trigger (if needed)
        if decision.decision_type == DecisionType.REVIEW:
             # Check Simulation Trigger Policy
             # Check explicit enable/disable flag (Default: True)
             simulation_enabled = context_config.get("enable_simulation", True)
             
             if simulation_enabled and self._should_activate_simulation(state_vector):
                 self.logger.info("Activating SimulationCore...")
                 
                 self._trace_event("Simulation", "Triggered SimulationCore", {"trigger": True}, {})
                 
                 # Detect Domain
                 detected_domain = self._detect_domain(input_signal)
                 
                 sim_result = self.simulation_core.execute_request({
                     "domain": detected_domain,
                     "input_context": input_signal
                 })
                 
                 self._trace_event("Simulation", "Execution Complete", {"status": sim_result.get("status")}, sim_result)
                 
                 # Re-evaluate logic based on Sim result? 
                 # For MVP, we just accept Sim result if success.
                 decision = self._update_decision_after_simulation(decision, sim_result)
                 
                 self._trace_event("Decision", "Updated Decision after Simulation", {
                    "decision": decision.decision_type.name
                 }, {"reason": decision.reason})
        
        # 6. Experience Update (Learning)
        if decision.decision_type == DecisionType.ACCEPT:
            self._learn_experience(input_signal, decision)
             # Trace persistence status based on whether it was skipped or not?
             # _learn_experience traces if skipped. If not skipped, we trace here?
             # Let's move persistence tracing inside _learn_experience completely?
             # For now, trace persistence intention.
            self._trace_event("Experience", "Learned Experience", {"persisted": decision.state_snapshot.confidence >= 0.6}, {})
            
        self.current_trace.final_decision = decision
        return decision

    def _trace_event(self, step: str, desc: str, metrics: Dict, details: Dict):
        if self.current_trace:
            evt = CognitiveEvent(step, desc, metrics, details=details)
            self.current_trace.events.append(evt)

    # --- Metrics Calculation Implementation ---

    def _calculate_metrics(self, hypotheses: List[Hypothesis]) -> CognitiveStateVector:
        """
        Implements equations from Spec 3.0 - 6.0
        Now uses Hypotheses instead of raw scores.
        """
        # Extract Scores s_i(t)
        if not hypotheses:
            return CognitiveStateVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        scores = np.array([h.score for h in hypotheses])
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
            
        # Reject: C < 0.2 AND R < 0.2 (Relaxed from 0.4 to allow Novelty/Simulation)
        elif state.confidence < 0.2 and state.recall_reliability < 0.2:
            return CognitiveDecision(DecisionType.REJECT, "SUPPRESS", "Extremely Low Confidence & Reliability", state)
            
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
        seed_source = ""
        
        if isinstance(input_signal, str):
             seed_source = input_signal
        elif isinstance(input_signal, dict):
             # Handle Multimodal Dict {"text": "...", "image": ...}
             # Use text part for seed
             seed_source = input_signal.get("text", "") + str(input_signal.get("image_name", ""))
        else:
             seed_source = str(input_signal)

        if seed_source:
             import hashlib
             seed = int(hashlib.md5(seed_source.encode('utf-8', errors='ignore')).hexdigest(), 16) % (2**32)
             rng = np.random.default_rng(seed)
             vec = rng.random(64)
             return vec / np.linalg.norm(vec)
             
    def _detect_domain(self, input_signal: Any) -> str:
        """
        Simple heuristic to detect simulation domain.
        """
        # Multimodal / Vision detection
        if isinstance(input_signal, dict):
            if "image_name" in input_signal or "image" in input_signal:
                return "vision"
            # Fallback for dict: extract 'text' content for text logic
            text = str(input_signal.get("text", input_signal)).lower()
        else:
            text = str(input_signal).lower()
        
        # Coding / Text Generation Keywords
        coding_keywords = [
            "code", "function", "generate", "list", "translate", "言葉", "語", 
            "program", "script", "display", "print", "show", "hello"
        ]
        
        # Numeric / Math Keywords
        numeric_keywords = [
            "calculate", "solve", "math", "factor", "prime", "comput", "equation", "素因数", "計算"
        ]
        # Common Math Operators
        if any(op in text for op in ["+", "-", "*", "/", "="]) and any(char.isdigit() for char in text):
             # Weak signal, check if it looks like a sentence
             pass
             
        for kw in numeric_keywords:
            if kw in text:
                return "numeric"
                
        for kw in coding_keywords:
            if kw in text:
                return "coding"
                
        # Fallback based on content
        # If it has digits and operators, maybe numeric?
        import re
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', text):
            return "numeric"
            
        return "numeric" # Default to numeric for now? Or "general"?


    def _update_decision_after_simulation(self, decision: CognitiveDecision, simulation_result: Dict) -> CognitiveDecision:
        """
        Updates the decision based on simulation outcome.
        OPT-1: Guards against ACCEPTing 'Unknown' results.
        """
        status = simulation_result.get("status", "FAILURE")
        result_content = simulation_result.get("result", "")
        detected_class = simulation_result.get("detected_class", "")
        
        if status == "SUCCESS":
            # OPT-1: Vision Safety Guard
            if detected_class == "Unknown" or "Unknown Object" in str(result_content):
                 return CognitiveDecision(
                     decision_type=DecisionType.REVIEW, 
                     action="DEFER_REVIEW", 
                     reason=f"Vision result unresolved: {detected_class}", 
                     state_snapshot=decision.state_snapshot
                 )

            return CognitiveDecision(
                decision_type=DecisionType.ACCEPT, 
                action="PROMOTE_SIMULATED", 
                reason=f"Simulation verified: {result_content}", 
                state_snapshot=decision.state_snapshot
            )
        else:
            return CognitiveDecision(
                decision_type=DecisionType.REJECT, 
                action="SUPPRESS_FAILED_SIM", 
                reason=f"Simulation failed: {result_content}", 
                state_snapshot=decision.state_snapshot
            )

    def _learn_experience(self, input_signal: Any, decision: CognitiveDecision):
        """
        Store result in ExperienceMemory.
        OPT-3: Enforce Persistence Threshold (Confidence >= 0.6)
        """
        # OPT-3 Check Confidence Logic
        confidence = decision.state_snapshot.confidence
        # Threshold 0.6 as per Spec (Phase 1)
        if confidence < 0.6:
             # Skip persistence
             self._trace_event("Experience", "Learning Skipped", {"reason": "Low Confidence", "confidence": confidence}, {})
             return
        
        # Original Logic
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
        self.experience_memory.log_experience(input_signal, meta)
        pass
