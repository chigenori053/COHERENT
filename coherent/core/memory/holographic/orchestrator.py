"""
Memory Orchestrator

Manages the interaction between Dynamic, Static, and Causal memory layers.
Implements the Promotion Rule and Orchestration Logic.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from .dynamic import DynamicHolographicMemory
from .static import StaticHolographicMemory
from .causal import CausalHolographicMemory

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
        
        # 2. Promotion Logic (Simplified)
        # In a real loop, this checks stability over time.
        # Here, we check if the input ALREADY resonates with Static (Recognition)
        # or if it's a candidate for new Static entry (Learning).
        # For this refactor, we focus on the structure existence.
        
        # Check if already known in Static
        results = self.static.query(state, top_k=1)
        if results and results[0][1] >= self.promotion_threshold:
            # Recognized!
            pass 
        else:
            # Not recognized. If this were a learning agent, we'd check stability.
            # For now, explicit promotion is assumed to be external (or via specific method).
            pass

    def promote_to_static(self, state: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Explicitly promote a state to Static Memory."""
        # Sanity check: Does it conflict?
        # TODO: checking conflict
        self.static.add(state, metadata)
        return True

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
            "static": static_res[0] if static_res else None,
            "causal": causal_res[0] if causal_res else None,
            "dynamic": dynamic_res[0] if dynamic_res else None
        }
