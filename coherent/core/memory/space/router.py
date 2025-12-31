from typing import Any, Dict
from ..types import Action
from .layers import ProcessingResult
from .store import AcceptStore, ReviewStore, RejectStore

class MemoryRouter:
    """
    Routes processing results to the appropriate Memory Store.
    Enforces 'DecisionEngine One-Way Control' (Guard-3).
    """
    def __init__(self, 
                 accept_store: AcceptStore, 
                 review_store: ReviewStore, 
                 reject_store: RejectStore):
        self.accept_store = accept_store
        self.review_store = review_store
        self.reject_store = reject_store

    def route(self, result: ProcessingResult, hologram: Any) -> None:
        """
        Routes the hologram based on the decided Action in result.
        """
        action = result.action
        
        # Determine Key (using ID generation logic typically found in Space, 
        # but Router needs to know where to put it. 
        # Ideally Space handles ID gen, but simpler here for now.)
        # Using hologram_ref if available, else generating one.
        key = result.hologram_ref 
        if not key:
            key = str(id(hologram))

        # Mapping Logic
        if action in [Action.STORE_NEW, Action.MERGE_SOFT, Action.VARIANT_LINK]:
            self.accept_store.write(key, hologram)
            
        elif action == Action.REVIEW:
            self.review_store.write(key, hologram)
            
        elif action in [Action.REJECT, Action.ABSORB]:
            # Absorb might mean 'do nothing' or 'store as redundant in reject for stats'
            # Spec says "U(Reject, Match) << ...", Absorb is semantically close to "Don't add to Accept".
            # If ABSORB, we might strictly technically NOT write execution, 
            # but for safety let's put it in Reject/Trash or just Ignore.
            # Plan said: REJECT, ABSORB -> RejectStore.
            self.reject_store.write(key, hologram)
            
        else:
            # Fallback
            self.review_store.write(f"UNMAPPED_{key}", hologram)
