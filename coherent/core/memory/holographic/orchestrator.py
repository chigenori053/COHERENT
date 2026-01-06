"""
Memory Orchestrator

Manages the interaction between Dynamic, Static, and Causal memory layers.
Implements the Promotion Rule and Orchestration Logic.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from .dynamic import DynamicHolographicMemory
from .static import StaticHolographicMemory
from .dynamic import DynamicHolographicMemory
from .static import StaticHolographicMemory
from .causal import CausalHolographicMemory, DecisionState, Action

class MemoryOrchestrator:
    def __init__(self, 
                 dynamic: DynamicHolographicMemory,
                 static: StaticHolographicMemory,
                 causal: CausalHolographicMemory,
                 promotion_threshold: float = 0.85):
        self.dynamic = dynamic
        self.static = static
        self.causal = causal
        self.promotion_threshold = promotion_threshold

    def process_input(self, state: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """
        Main entry point for new information.
        1. Store in Dynamic
        2. Check for Static Promotion
        3. Induce Causal Link (if applicable - simplified for v1)
        """
        # 1. Dynamic Storage
        self.dynamic.add(state, metadata)
        
        # 2. Causal Decision (Replaces Simplified Promotion)
        # Gather metrics for Decision State
        
        # Check recognition in Static
        static_res = self.static.query(state, top_k=2)
        top_score = static_res[0][1] if static_res else 0.0
        second_score = static_res[1][1] if len(static_res) > 1 else 0.0
        margin = top_score - second_score
        
        # Entropy Estimate: For now, we can use 1 - max_resonance as a proxy for "uncertainty/entropy"
        # High resonance -> Low entropy/uncertainty.
        entropy_est = 1.0 - top_score
        
        # Repetition: In a real system, Dynamic tracks this. Here we default to 1 for single-shot.
        repetition = 1 
        
        decision_state = DecisionState(
            resonance_score=top_score,
            margin=margin,
            repetition_count=repetition,
            entropy_estimate=entropy_est,
            memory_origin="Dynamic",
            historical_conflict_rate=0.0 # Placeholder
        )
        
        action = self.causal.evaluate_decision(decision_state)
        
        if action == Action.PROMOTE:
            # Execute Promotion
            self.promote_to_static(state, metadata)
            # print(f">> DECISION: PROMOTE (Score={top_score:.2f})")
            
        elif action == Action.RETAIN:
            # Stay in Dynamic (Already done in step 1)
            pass
            
        elif action == Action.SUPPRESS:
            # Remove from Dynamic? Or just tag as suppressed?
            # For now, do nothing (Dynamic is FIFO)
            pass
            
        elif action == Action.DEFER_REVIEW:
            # Log for review
            pass

    def promote_to_static(self, state: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Explicitly promote a state to Static Memory.
        Integrates with Evaluation System (Spec v1.0).
        """
        # Extract signals (if available, else default to trusted)
        val_signal = metadata.get('validation_confidence', 1.0)
        gen_score = metadata.get('generalization_score', 1.0)
        
        # Evaluate Experience
        profile = self.static.evaluate_experience(state, val_signal, gen_score)
        
        # Check Update Rules
        decision = self.static.check_update_rules(profile)
        
        if decision == 'STORE':
            self.static.add(state, metadata, profile)
            return True
        elif decision == 'MERGE':
            # For Key-Value static memory, MERGE implies update/overwrite or fusion.
            # v1: Overwrite with new profile
            self.static.add(state, metadata, profile)
            return True
        else:
            # IGNORE / REJECT
            return False

    def register_transition(self, source_state: np.ndarray, target_state: np.ndarray) -> None:
        """Register a transition in Causal Memory."""
        self.causal.add_transition(source_state, target_state)

    def recall(self, query_state: np.ndarray) -> Dict[str, Any]:
        """
        Unified Recall.
        1. Query Static (Definition/Identity)
        2. Query Causal (Prediction)
        3. Query Dynamic (Context/Recent)
        """
        static_res = self.static.query(query_state, top_k=1)
        causal_res = self.causal.query(query_state, top_k=1)
        dynamic_res = self.dynamic.query(query_state, top_k=1)
        
        return {
            "dynamic": dynamic_res[0] if dynamic_res else None,
            "static": static_res[0] if static_res else None,
            "causal": causal_res[0] if causal_res else None
        }

    def reset_dynamic(self):
        """Reset the dynamic memory layer."""
        self.dynamic.clear()
