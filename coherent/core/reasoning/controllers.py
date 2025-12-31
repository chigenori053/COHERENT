import math
from typing import List, Optional, Any
from .types import ThoughtBranch, BranchStatus, UtilityState

# -------------------------------------------------------------------------
# Utility Model (Fixed Defaults)
# -------------------------------------------------------------------------
W_VALIDITY   = 0.35
W_PROGRESS   = 0.25
W_CONFIDENCE = 0.20
W_NOVELTY    = 0.15
W_RISK       = 0.30

# Thresholds
U_MIN       = -0.10   # prune immediately
U_SURVIVE   =  0.05   # minimum survival utility
U_CONVERGE  =  0.65   # single-branch converge threshold
R_MAX       =  0.70   # max tolerated risk
STEP_LIMIT  =  64     # default per-branch step cap
H_MIN       =  0.25   # entropy convergence

class UtilityAggregator:
    @staticmethod
    def compute_expected_utility(u: UtilityState) -> float:
        return (
            W_VALIDITY * u.validity +
            W_PROGRESS * u.progress +
            W_CONFIDENCE * u.confidence +
            W_NOVELTY * u.novelty -
            W_RISK * u.risk
        )

# -------------------------------------------------------------------------
# Controllers
# -------------------------------------------------------------------------

class BranchEvaluator:
    """
    Observer that updates branch utility based on external signals (e.g. Validation).
    """
    def update_utility(self, branch: ThoughtBranch, validation_result: Any = None) -> None:
        # In a real run, this takes a validation_result object. 
        # For structure, we ensure EU is up to date with whatever is in UtilityState.
        branch.utility.expected_utility = UtilityAggregator.compute_expected_utility(branch.utility)


class Pruner:
    """
    Eliminates branches based on thresholds and dominance.
    """
    def prune(self, branches: List[ThoughtBranch]) -> None:
        for b in branches:
            if b.status != BranchStatus.ACTIVE:
                continue

            # 1. Threshold Pruning
            if b.utility.expected_utility < U_MIN:
                b.status = BranchStatus.PRUNED
                b.trace_log.append(f"Pruned: EU {b.utility.expected_utility:.2f} < {U_MIN}")
                continue

            if b.utility.risk > R_MAX:
                b.status = BranchStatus.PRUNED
                b.trace_log.append(f"Pruned: Risk {b.utility.risk:.2f} > {R_MAX}")
                continue

            if b.step_count > STEP_LIMIT:
                b.status = BranchStatus.TERMINATED
                b.trace_log.append(f"Terminated: Step Limit {STEP_LIMIT}")
                continue

            # 2. Dominance Pruning (Pareto)
            if self._is_dominated(b, branches):
                b.status = BranchStatus.PRUNED
                b.trace_log.append("Pruned: Dominated by peer branch")

    def _is_dominated(self, target: ThoughtBranch, pool: List[ThoughtBranch]) -> bool:
        """
        Returns True if target is strictly dominated by any other branch in pool.
        """
        for other in pool:
            if other is target or other.status != BranchStatus.ACTIVE:
                continue
            
            u_t = target.utility
            u_o = other.utility

            # Check weak dominance on all factors
            if (u_o.validity >= u_t.validity and
                u_o.progress >= u_t.progress and
                u_o.confidence >= u_t.confidence and
                u_o.novelty >= u_t.novelty):
                
                # Check for strict dominance in at least one
                if (u_o.validity > u_t.validity or
                    u_o.progress > u_t.progress or
                    u_o.confidence > u_t.confidence or
                    u_o.novelty > u_t.novelty):
                    return True
        return False


class ConvergenceController:
    """
    Checks if the system has converged.
    """
    def check_convergence(self, branches: List[ThoughtBranch]) -> Optional[ThoughtBranch]:
        active_branches = [b for b in branches if b.status == BranchStatus.ACTIVE]
        if not active_branches:
            return None

        # 1. Dominant Utility Convergence
        best_branch = max(active_branches, key=lambda b: b.utility.expected_utility)
        if best_branch.utility.expected_utility > U_CONVERGE:
            return best_branch

        # 2. Entropy Convergence
        entropy = self._compute_entropy(active_branches)
        if entropy < H_MIN and best_branch.utility.expected_utility > U_SURVIVE:
            # Low entropy means one branch is vastly superior (or all are)
            # If the best one is decent, converge.
            return best_branch

        return None

    def _compute_entropy(self, branches: List[ThoughtBranch]) -> float:
        # Exclude non-positive EU branches to avoid log domain error and noise
        eus = [b.utility.expected_utility for b in branches if b.utility.expected_utility > 0]
        total = sum(eus)
        if total == 0 or len(eus) <= 1:
            return 0.0

        entropy = 0.0
        for eu in eus:
            p = eu / total
            entropy -= p * math.log(p)
        return entropy

class ForkController:
    """
    Triggers branching logic.
    For MVP/Skeleton, implements basic max-branch check.
    """
    MAX_BRANCH = 5
    A_FORK = 0.40

    def should_fork(self, branch: ThoughtBranch, branch_count: int) -> bool:
        if branch_count >= self.MAX_BRANCH:
            return False
            
        # Example condition: Ambiguity high but Validity exists
        # NOTE: Using 'risk' as proxy for ambiguity as per Spec 7.3
        if branch.utility.risk > self.A_FORK and branch.utility.validity > 0:
            return True
            
        return False
