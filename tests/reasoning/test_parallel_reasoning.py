import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coherent.core.reasoning.types import ThoughtBranch, BranchStatus
from coherent.core.reasoning.orchestrator import ParallelThoughtOrchestrator
from coherent.core.reasoning.controllers import Pruner, U_MIN, R_MAX
from coherent.core.memory.space.memory_space import MemorySpace

class TestParallelReasoning(unittest.TestCase):

    def setUp(self):
        # Mock MemorySpace
        self.mock_memory = MagicMock(spec=MemorySpace)
        self.orchestrator = ParallelThoughtOrchestrator(self.mock_memory)

    def test_branch_creation_and_step(self):
        """Verify scheduler ticks increment step count."""
        self.orchestrator.start("init_ast", max_ticks=1)
        # Should have 1 branch
        self.assertEqual(len(self.orchestrator.branches), 1)
        # Should have stepped once
        self.assertEqual(self.orchestrator.branches[0].step_count, 1)

    def test_pruning_thresholds(self):
        """Verify pruning by utility thresholds."""
        # Create a branch manually
        b1 = self.orchestrator._create_branch("ast")
        b1.utility.expected_utility = U_MIN - 0.1 # Below min
        self.orchestrator.branches.append(b1)
        
        b2 = self.orchestrator._create_branch("ast2")
        b2.utility.risk = R_MAX + 0.1 # Above max risk
        self.orchestrator.branches.append(b2)
        
        self.orchestrator.pruner.prune(self.orchestrator.branches)
        
        self.assertEqual(b1.status, BranchStatus.PRUNED)
        self.assertEqual(b2.status, BranchStatus.PRUNED)

    def test_convergence_commit(self):
        """Verify successful convergence triggers commit."""
        # Setup a scenario where branch is awesome
        b1 = self.orchestrator._create_branch("winning_ast")
        b1.utility.expected_utility = 0.9 # High utility
        b1.utility.validity = 0.9
        
        self.orchestrator.branches.append(b1)
        
        # Run orchestrator
        # Step 1: tick -> eval -> prune -> convergence check
        # Since EU > 0.65 (default), it should converge immediately
        result = self.orchestrator.start("dummy", max_ticks=2)
        
        # Check commit called
        self.mock_memory.store.assert_called_once()
        self.assertEqual(b1.status, BranchStatus.CONVERGED)

    def test_dominance_pruning(self):
        """Verify Pareto dominance logic."""
        # Winner
        winner = self.orchestrator._create_branch("winner")
        winner.utility.validity = 0.9
        winner.utility.progress = 0.9
        winner.utility.confidence = 0.9
        winner.utility.novelty = 0.9
        
        # Loser (strictly dominated)
        loser = self.orchestrator._create_branch("loser")
        loser.utility.validity = 0.1
        loser.utility.progress = 0.1
        loser.utility.confidence = 0.1
        loser.utility.novelty = 0.1
        
        self.orchestrator.branches = [winner, loser]
        self.orchestrator.pruner.prune(self.orchestrator.branches)
        
        self.assertEqual(winner.status, BranchStatus.ACTIVE)
        self.assertEqual(loser.status, BranchStatus.PRUNED)

if __name__ == '__main__':
    unittest.main()
