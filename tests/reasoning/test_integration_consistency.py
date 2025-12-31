import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coherent.core.reasoning.commit import CommitController, V_COMMIT
from coherent.core.reasoning.types import ThoughtBranch, UtilityState, BranchStatus
from coherent.core.memory.space.memory_space import MemorySpace
from coherent.core.memory.optimization.optimizer import MemorySpaceOptimizer
from coherent.core.memory.hologram.encoder import HolographicEncoder
from coherent.core.memory.hologram.merge import SoftMerger
from coherent.core.memory.types import Action

class TestIntegrationConsistency(unittest.TestCase):

    def setUp(self):
        # Real MemorySpace with Real Components (no mocks for internals)
        # We need to trust the stack.
        
        # Mocks for dependencies of MemorySpace
        # Optimizer signature: (decision_engine, micro_optimizer, encoder, inference, unique_inference)
        # Based on error: TypeError: MemorySpaceOptimizer.__init__() missing 3 required positional arguments: 'decision_engine', 'micro_optimizer', and 'encoder'
        # Waiting for view_file to confirm exact signature, but providing mocks as requested by error.
        self.optimizer = MemorySpaceOptimizer(
            uniqueness_inference=MagicMock(),
            micro_inference=MagicMock(),
            decision_engine=MagicMock(),
            micro_optimizer=MagicMock(),
            encoder=MagicMock()
        )
        # We need to control the decision engine inside optimizer for this test 
        # OR we modify MemorySpace to allow overriding.
        # But first, let's see what happens with default behavior if we can mock the decision engine response.
        
        self.optimizer.decision_engine = MagicMock()
        
        merger = SoftMerger()
        encoder = HolographicEncoder()
        
        self.memory_space = MemorySpace(self.optimizer, merger, encoder)
        self.commit_controller = CommitController(self.memory_space)

    def test_force_review_downgrade(self):
        """
        Verify that if CommitController decides REVIEW (validity < V_COMMIT),
        MemorySpace actually stores it in ReviewStore.
        
        Current Issue Hypothesis: MemorySpace.store() ignores CommitController's intent 
        and re-runs optimizer. If optimizer says STORE_NEW, it goes to AcceptStore.
        """
        # Branch that should be downgraded
        branch = ThoughtBranch(branch_id=1, status=BranchStatus.CONVERGED)
        branch.utility.validity = V_COMMIT - 0.1 # 0.75, so should Review
        branch.utility.confidence = 0.9
        branch.utility.risk = 0.0
        branch.current_ast = "downgrade_payload"
        
        # Mock Optimizer to return STORE_NEW (simulating disagreement)
        # If MemorySpace respects its own optimizer over CommitController, this will go to AcceptStore.
        self.optimizer.process = MagicMock(return_value=(Action.STORE_NEW, MagicMock()))
        
        self.commit_controller.commit(branch)
        
        # Check where it went
        in_review = self.memory_space.review_store.recall("stub_id", context="human_override_review")
        in_accept = self.memory_space.accept_store.recall("stub_id")
        
        # If Integrity is maintained, it should be in ReviewStore.
        # But based on current code, MemorySpace follows Optimizer -> AcceptStore.
        # So we expect this test to FAIL if my hypothesis is correct.
        
        # We search by values since ID is generated inside
        found_review = any("downgrade_payload" == v for v in self.memory_space.review_store.data.values())
        found_accept = any("downgrade_payload" == v for v in self.memory_space.accept_store.data.values())
        
        if found_accept and not found_review:
             self.fail("Integrity Failure: CommitController wanted Review, but MemorySpace put in Accept!")

if __name__ == '__main__':
    unittest.main()
