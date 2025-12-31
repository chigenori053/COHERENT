from typing import List, Any, Optional
import logging
from .types import ThoughtBranch, BranchStatus
from .scheduler import BranchScheduler
from .controllers import BranchEvaluator, Pruner, ConvergenceController, ForkController
from .commit import CommitController
from ..memory.space.memory_space import MemorySpace

class ParallelThoughtOrchestrator:
    """
    Cognitive OS Kernel.
    Manages the lifecycle of ThoughtBranches deterministically.
    """
    def __init__(self, memory_space: MemorySpace):
        self.memory_space = memory_space
        self.branches: List[ThoughtBranch] = []
        self.next_branch_id = 0
        
        # Components
        self.scheduler = BranchScheduler()
        self.evaluator = BranchEvaluator()
        self.pruner = Pruner()
        self.convergence_ctrl = ConvergenceController()
        self.fork_ctrl = ForkController()
        self.commit_ctrl = CommitController(memory_space)
        
        self.logger = logging.getLogger(__name__)

    def start(self, initial_ast: Any, max_ticks: int = 100) -> Optional[Any]:
        """
        Main Loop (Spec Section 11).
        """
        # Initialize Root Branch
        root = self._create_branch(initial_ast)
        self.branches.append(root)
        
        for tick in range(max_ticks):
            active_count = sum(1 for b in self.branches if b.status == BranchStatus.ACTIVE)
            if active_count == 0:
                self.logger.info("All branches terminated or pruned.")
                break

            # 1) Scheduler Tick
            self.scheduler.tick(self.branches)
            
            # 2) Evaluator (Update Utility)
            for b in self.branches:
                if b.status == BranchStatus.ACTIVE:
                    self.evaluator.update_utility(b)
            
            # 3) Pruner
            self.pruner.prune(self.branches)
            
            # 4) Fork Controller (Optional diversification)
            # Need to iterate over a copy or use index to avoid modification issues
            current_active = [b for b in self.branches if b.status == BranchStatus.ACTIVE]
            for b in current_active:
                if self.fork_ctrl.should_fork(b, len(self.branches)):
                    self._fork(b)
            
            # 5) Convergence Check
            converged_branch = self.convergence_ctrl.check_convergence(self.branches)
            if converged_branch:
                converged_branch.status = BranchStatus.CONVERGED
                self.logger.info(f"System converged on branch {converged_branch.branch_id}")
                
                # Commit
                result = self.commit_ctrl.commit(converged_branch)
                return result

        self.logger.info("Orchestrator finished without convergence.")
        return None

    def _create_branch(self, ast: Any, parent_id: Optional[int] = None) -> ThoughtBranch:
        """Helper to create a new branch."""
        b = ThoughtBranch(branch_id=self.next_branch_id, parent_branch_id=parent_id)
        b.current_ast = ast
        self.next_branch_id += 1
        return b

    def _fork(self, parent: ThoughtBranch) -> None:
        """Fork constraint logic and cloning."""
        if len(self.branches) >= ForkController.MAX_BRANCH:
            return

        child = self._create_branch(parent.current_ast, parent_id=parent.branch_id)
        # Inherit state (deep copy needed in real app?)
        # For now, shallow reference inheritance for MVP structure
        child.working_memory = parent.working_memory.copy()
        child.utility = parent.utility # Copy utility snapshot? 
        # Usually reset some utility fields or inherited? 
        # Spec says "inherits parent state". 
        
        # Mutate policy for exploration (simple example)
        child.exploration_bias = parent.exploration_bias + 0.1
        
        self.branches.append(child)
        self.logger.info(f"Forked branch {child.branch_id} from {parent.branch_id}")
