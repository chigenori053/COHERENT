from typing import Any, Dict, List, Optional
import logging

from ..types import Action
from ..observation import Observation
from ..optimization.optimizer import MemorySpaceOptimizer
from ..hologram.encoder import HolographicEncoder
from ..hologram.merge import SoftMerger
from ..hologram.variant_link import VariantLink
from .layers import ProcessingResult
from .store import AcceptStore, ReviewStore, RejectStore
from .router import MemoryRouter
from ..logging.decision_log import DecisionLog

class MemorySpace:
    def __init__(self, 
                 optimizer: MemorySpaceOptimizer, 
                 merger: SoftMerger,
                 encoder: HolographicEncoder):
        self.optimizer = optimizer
        self.merger = merger
        self.encoder = encoder
        
        # Sub-Area Architecture
        self.accept_store = AcceptStore()
        self.review_store = ReviewStore()
        self.reject_store = RejectStore()
        
        self.router = MemoryRouter(
            self.accept_store, 
            self.review_store, 
            self.reject_store
        )
        
        self.logger = logging.getLogger(__name__)

    def store(self, raw_data: Any, context_obs: Observation, forced_action: Optional[Action] = None) -> ProcessingResult:
        """
        Main entry point to store new information.
        If forced_action is provided (e.g. from CommitController), it overrides internal optimization.
        """
        # 0. Encode
        hologram = self.encoder.encode(raw_data)
        
        # 1. Optimize / Decide
        if forced_action:
            action = forced_action
            # Create a dummy or partial log for the forced action
            # Ideally we still run process() to get the log but ignore the action?
            # For efficiency and authority, we just log the override.
            decision_log = DecisionLog(
                observation=context_obs,
                state_distribution=None, # Not computed
                expected_utility=None,   # Not computed
                action=action
            )
        else:
            action, decision_log = self.optimizer.process(hologram, context_obs)
        
        # 2. Prepare Result
        # Note: In a real system, we'd handle ID generation and linking details here or in the Router.
        # For now, we generate a ref ID to track it.
        hologram_ref = str(id(hologram))
        if action == Action.REVIEW:
            hologram_ref = "REVIEW_" + hologram_ref
            
        success = action != Action.REJECT
        
        result = ProcessingResult(
            action=action,
            log=decision_log,
            hologram_ref=hologram_ref,
            message=f"Action {action.value} executed via Router.",
            is_success=success
        )
        
        # 3. Route (Execute Action)
        self.router.route(result, hologram)
        
        return result

    def retrieve(self, query_obs: Observation, context: Optional[str] = None) -> Any:
        """
        Retrieves memory.
        Guard-1: Resonance Isolation - By default only searches AcceptStore.
        Guard-4: Review Recall Restriction - Handled by ReviewStore.recall() check.
        """
        # Primary source: AcceptStore
        results = {}
        accept_res = self.accept_store.recall(query_obs, context)
        if isinstance(accept_res, dict):
            results.update(accept_res)
            
        # Optional: Check ReviewStore if context allows
        if context in ReviewStore.ALLOWED_CONTEXTS:
            review_res = self.review_store.recall(query_obs, context)
            if isinstance(review_res, dict):
                results.update(review_res) # Or keep separate depending on API needs
                
        # RejectStore is implicitly ignored / Guard-5
        
        return results
