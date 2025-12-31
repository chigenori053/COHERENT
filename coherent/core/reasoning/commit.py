from typing import Any, Optional
import logging
from .types import ThoughtBranch, BranchStatus
from ..memory.types import Action
from ..memory.space.memory_space import MemorySpace
from ..memory.space.layers import ProcessingResult
from ..memory.observation import Observation

# Spec 7.5
V_COMMIT = 0.85

class CommitController:
    """
    Handles final routing to MemorySpace.
    Enforces 'Converged Only' and 'Authority Separation'.
    """
    def __init__(self, memory_space: MemorySpace):
        self.memory_space = memory_space
        self.logger = logging.getLogger(__name__)

    def commit(self, branch: ThoughtBranch) -> Optional[ProcessingResult]:
        """
        Commits the branch result to MemorySpace.
        Applies final validation gate.
        """
        if branch.status != BranchStatus.CONVERGED:
            self.logger.warning(f"Attempted to commit non-converged branch {branch.branch_id}")
            return None

        # Final Validation Gate
        validity = branch.utility.validity
        target_store = Action.STORE_NEW # Default to Accept path (Abstracted Action)
        
        # Determine Routing
        if validity >= V_COMMIT:
            # Accepted
            target_store = Action.STORE_NEW # Maps to AcceptStore in MemorySpace
        else:
            # Downgrade to Review
            target_store = Action.REVIEW # Maps to ReviewStore
            self.logger.info(f"Branch {branch.branch_id} downgraded to REVIEW (Validity {validity} < {V_COMMIT})")

        # Create a synthetic observation for storage context
        # In a full system, this would come from the branch's last state validation
        obs = Observation(
            max_resonance=branch.utility.confidence, # using confidence as proxy
            resonance_mean=0.5,
            resonance_variance=0.1,
            phase_distance=0.0,
            interference_score=branch.utility.risk,
            snr=1.0,
            novelty_score=branch.utility.novelty,
            memory_density=0.5
        )
        
        # We need to construct the data to store.
        # Assuming branch.current_ast is the payload.
        # But we also need to tell MemorySpace *which* Action to use? 
        # MemorySpace uses optimizer to decide action. 
        # BUT Spec says: "Orchestrator invokes ... DecisionEngine" and "CommitController -> MemorySpace Layer".
        # If MemorySpace.store() runs its own optimizer, it might disagree.
        # However, the user request says "referencing current architecture".
        # MemorySpace.store() takes raw_data and uses its internal optimizer/router.
        
        # TO COMPLY WITH "CommitController routs to layer" AND existing MemorySpace:
        # We need a way to force the Action/Store in MemorySpace, OR rely on MemorySpace to agree.
        # 
        # Given the "Sub-Area Design" we just built:
        # MemorySpace.store calls Optimizer -> Decision -> Router.
        # The Optimizer should ideally respect the high-level intent.
        # But `store()` as implemented doesn't take an override Action.
        
        # FOR NOW: We rely on MemorySpace's standard flow. 
        # Ideally, we would extend MemorySpace to accept a `direct_commit(data, action)` method.
        # The spec says "MemoryRouter.route(decision, ...)" is internal.
        
        # Workaround reusing existing public API:
        # We store. MemorySpace logs it. 
        # The internal logic of MemorySpace (MicroVariation, etc.) runs inside.
        
        return self.memory_space.store(branch.current_ast, obs, forced_action=target_store)
