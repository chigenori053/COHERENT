"""
Task Gate Config & Logic (MVP)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
from typing import Optional, List, Dict, Any

class TaskType(Enum):
    TRANSFORM = "TRANSFORM"   # Translation, Paraphrasing, Format Conversion
    RETRIEVAL = "RETRIEVAL"   # Definition, Fact fetching
    REASONING = "REASONING"   # Comparison, Judgment, Explanation
    SIMULATION = "SIMULATION" # Hypothesis testing, Future prediction

class RouteType(Enum):
    FAST_PATH = "FAST_PATH"
    FULL_PATH = "FULL_PATH"

class EscalationReason(Enum):
    NONE = "NONE"
    MIXED_TASK = "MIXED_TASK"            # Transform + Reasoning
    SUBJECTIVE_OUTPUT = "SUBJECTIVE_OUTPUT" # 'Nicely', 'Appropriately'
    DECISION_REQUIRED = "DECISION_REQUIRED" # Phase 1 (Stub)
    INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT" # Phase 1 (Stub)

@dataclass
class TaskGateDecision:
    task_type: TaskType
    route: RouteType
    complexity_score: float
    simulation_allowed: bool
    confidence_required: bool
    reason: str
    escalation_reason: EscalationReason = EscalationReason.NONE

class TaskGate:
    """
    Task Gate Mechanism (MVP)
    
    Purpose:
        Determines if a task should be processed via Fast Path (lightweight)
        or Full Path (cognitive heavy lifting).
    """

    def __init__(self):
        # Configuration Thresholds
        self.complexity_threshold = 50.0  # Threshold for routing to FULL_PATH if complexity is high
        
        # Rule Definitions
        self.transform_patterns = [
            r"translate", r"paraphrase", r"convert", r"format", r"summarize",
            r"翻訳", r"言い換え", r"変換", r"フォーマット", r"語で", r"要約", r"in (english|japanese|json)"
        ]
        self.retrieval_patterns = [
            r"define", r"what is", r"meaning of", r"explain the term",
            r"とは", r"意味", r"定義", r"教えて"
        ]
        
    def assess_task(self, input_signal: Any, metadata: Optional[Dict] = None) -> TaskGateDecision:
        """
        Main entry point for task assessment.
        """
        raw_text = self._extract_text(input_signal)
        
        # 1. Classification
        task_type = self._classify_task_type(raw_text)
        
        # 2. Complexity Evaluation
        complexity = self._evaluate_complexity(raw_text)
        
        # 3. Route Decision
        route, esc_reason = self._decide_route(task_type, complexity, raw_text)
        
        # 4. Permissions
        sim_allowed = (task_type == TaskType.SIMULATION) or (task_type == TaskType.REASONING)
        conf_required = (route == RouteType.FULL_PATH)
        
        return TaskGateDecision(
            task_type=task_type,
            route=route,
            complexity_score=complexity,
            simulation_allowed=sim_allowed,
            confidence_required=conf_required,
            reason=f"Type={task_type.name}, Score={complexity:.1f}, Esc={esc_reason.name}",
            escalation_reason=esc_reason
        )

    def _extract_text(self, input_signal: Any) -> str:
        if isinstance(input_signal, str):
            return input_signal
        elif isinstance(input_signal, dict):
            return str(input_signal.get("text", input_signal))
        return str(input_signal)

    def _classify_task_type(self, text: str) -> TaskType:
        """
        Rule-based classification.
        """
        text_lower = text.lower()
        
        # TRANSFORM check
        for pattern in self.transform_patterns:
            if re.search(pattern, text_lower):
                return TaskType.TRANSFORM
                
        # RETRIEVAL check
        for pattern in self.retrieval_patterns:
            if re.search(pattern, text_lower):
                return TaskType.RETRIEVAL
                
        # Default fallback to REASONING (Safe default)
        return TaskType.REASONING

    def _evaluate_complexity(self, text: str) -> float:
        """
        Heuristic complexity scoring.
        Formula: w1*len + w2*depth + ...
        """
        # 1. Token Count (approx)
        token_count = len(text.split())
        
        # 2. Syntax Depth (Mock: purely based on commas/clauses)
        # Count punctuation that indicates clauses
        syntax_depth = text.count(',') + text.count(';') + text.count('However') + text.count('but')
        
        # 3. Abstract/Condition terms
        condition_terms = [
            "if", "when", "unless", "suppose", "assume",
            "もし", "場合", "仮定", "条件"
        ]
        cond_count = sum(1 for t in condition_terms if t in text)
        
        # Weights (Arbitrary for MVP)
        w1 = 1.5  # Length weight (Increased)
        w2 = 5.0  # Clause weight
        w3 = 10.0 # Condition weight
        
        score = (w1 * token_count) + (w2 * syntax_depth) + (w3 * cond_count)
        return score

    def _decide_route(self, task_type: TaskType, complexity: float, text: str) -> tuple[RouteType, EscalationReason]:
        """
        Route logic with Escalation Check (Phase 0).
        """
        # Default Decision
        route = RouteType.FULL_PATH
        if task_type in [TaskType.TRANSFORM, TaskType.RETRIEVAL]:
            if complexity < self.complexity_threshold:
                route = RouteType.FAST_PATH
        
        # Phase 0 Escalation Check
        if route == RouteType.FAST_PATH:
            esc_reason = self._check_escalation_phase0(text, task_type)
            if esc_reason != EscalationReason.NONE:
                return RouteType.FULL_PATH, esc_reason
            return RouteType.FAST_PATH, EscalationReason.NONE
            
        return RouteType.FULL_PATH, EscalationReason.NONE

    def _check_escalation_phase0(self, text: str, initial_type: TaskType) -> EscalationReason:
        """
        Phase 0 Escalation Logic:
        1. Mixed Tasks (Transform + Reasoning Trigger)
        2. Subjective Output ("nicely", "appropriately", etc.)
        """
        text_lower = text.lower()
        
        # 1. Mixed Task Check
        # If classified as TRANSFORM/RETRIEVAL, but contains REASONING terms like "why", "explain" (deeper explanation), "compare"
        reasoning_triggers = ["why", "because", "compare", "evaluate", "judgement", "なぜ", "理由", "比較", "評価"]
        
        has_reasoning_trigger = any(t in text_lower for t in reasoning_triggers)
        
        if initial_type in [TaskType.TRANSFORM, TaskType.RETRIEVAL] and has_reasoning_trigger:
             # Exception: "Explain" might be in RETRIEVAL patterns ("explain the term").
             # We need to distinguish "explain definition" vs "explain why".
             # For MVP, let's say "why" or "compare" is the strong signal.
             return EscalationReason.MIXED_TASK
             
        # 2. Subjective Output Check
        subjective_terms = ["nicely", "appropriately", "well", "good", "bad", "beautifully", "適切に", "上手く", "きれいに"]
        if any(t in text_lower for t in subjective_terms):
            return EscalationReason.SUBJECTIVE_OUTPUT
            
        return EscalationReason.NONE
