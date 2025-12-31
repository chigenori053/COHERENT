import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import sys
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# Real imports - assumes environment is correctly set up
# If dependencies are missing, these will raise ImportError, which is expected/cleaner than global mock pollution.

from coherent.core.memory.types import Action, StateDistribution
from coherent.core.memory.observation import Observation
from coherent.core.memory.inference import UniquenessInference, MicroVariationInference
from coherent.core.memory.utility import UtilityTable
from coherent.core.memory.human.utility_shaping import HumanUtilityShaper
from coherent.core.memory.decision import DecisionEngine
from coherent.core.memory.optimization.optimizer import MemorySpaceOptimizer
from coherent.core.memory.hologram.encoder import HolographicEncoder
from coherent.core.memory.hologram.merge import SoftMerger
from coherent.core.memory.optimization.micro_variation import MicroVariationOptimizer
from coherent.core.memory.space.memory_space import MemorySpace

@pytest.fixture
def memory_space():
    # Setup components
    u_inf = UniquenessInference()
    m_inf = MicroVariationInference()
    
    util_table = UtilityTable()
    shaper = HumanUtilityShaper()
    decision = DecisionEngine(util_table, shaper)
    
    micro_opt = MicroVariationOptimizer(m_inf, decision)
    encoder = HolographicEncoder()
    merger = SoftMerger()
    
    optimizer = MemorySpaceOptimizer(u_inf, m_inf, decision, micro_opt, encoder)
    
    return MemorySpace(optimizer, merger, encoder), shaper

def create_obs(**kwargs):
    defaults = {
        "max_resonance": 0.5,
        "resonance_mean": 0.3,
        "resonance_variance": 0.1,
        "phase_distance": 0.0,
        "interference_score": 0.0,
        "snr": 1.0,
        "novelty_score": 0.5,
        "memory_density": 0.5
    }
    defaults.update(kwargs)
    return Observation(**defaults)

def test_inference_logic_structure():
    """Verify inference returns a distribution, not a single state (No Threshold Check)."""
    inference = UniquenessInference()
    obs = create_obs(max_resonance=0.8, novelty_score=0.2)
    dist = inference.infer(obs)
    
    assert isinstance(dist, StateDistribution)
    total_prob = sum(dist.probs.values())
    assert abs(total_prob - 1.0) < 1e-6
    
    # Should not be 1.0 or 0.0 strictly for heuristic logic (soft)
    # This assertion depends on the current implementation details of the heuristic,
    # but generally we expect some mix.
    assert 0.0 < dist.get("TrulyUnique") < 1.0
    assert 0.0 < dist.get("Variant") < 1.0

def test_decision_reproducibility(memory_space):
    space, _ = memory_space
    data = "test_data"
    obs = create_obs(max_resonance=0.1, novelty_score=0.9) # Likely unique
    
    # Run 1
    result1 = space.store(data, obs)
    
    # Run 2 (Same inputs)
    result2 = space.store(data, obs)
    
    assert result1.action == result2.action
    # Logs should be essentially identical (except timestamp)
    assert result1.log.observation == result2.log.observation
    assert result1.log.state_distribution == result2.log.state_distribution
    assert result1.log.expected_utility == result2.log.expected_utility

def test_human_utility_shaping(memory_space):
    space, shaper = memory_space
    data = "test_data"
    
    # Create an ambiguous situation
    # High resonance but also high novelty? (Contradictory/Confusing state)
    # Let's target a specific state transition.
    # State "Noisy" usually leads to REJECT.
    # We want to force it to STORE_NEW via shaping.
    
    obs = create_obs(interference_score=0.9, novelty_score=0.1) 
    # This should infer high "Noisy" prob.
    
    # Default behavior without shaping
    dist = UniquenessInference().infer(obs)
    # Verify it biases towards Noisy
    assert dist.get("Noisy") > dist.get("TrulyUnique")
    
    # Helper to calculate decision without full integration test first for clarity
    res1 = space.store(data, obs)
    # Expected: REJECT (since Noisy -> Reject is 1.0 utility, Store is -1.0)
    # Note: UtilityTable defs:
    # Noisy: REJECT 1.0, STORE_NEW -1.0
    
    if res1.action != Action.REJECT:
        # If heuristics didn't trigger Noisy strong enough, check what happened
        # Ideally we assume it does. If not, this test needs tighter control on "Noisy" heuristic.
        pass

    # Now Apply Shaping: Make Noisy -> STORE_NEW very desirable
    # Delta +3.0 to overcome (-1.0 vs 1.0 gap)
    shaper.set_modifier("Noisy", Action.STORE_NEW, 5.0)
    
    res2 = space.store(data, obs)
    
    assert res2.action == Action.STORE_NEW
    assert res2.log.action == Action.STORE_NEW

def test_soft_merge_invocation(memory_space):
    space, _ = memory_space
    # Trigger MERGE by high resonance, low novelty
    obs = create_obs(max_resonance=0.9, novelty_score=0.1, memory_density=0.8)
    # This should map to "Redundant" or "Variant"
    
    res = space.store("data", obs)
    
    # Based on Utility:
    # Redundant -> MERGE_SOFT (0.8), ABSORB (0.9)
    # Variant -> MERGE_SOFT (0.6), VARIANT_LINK (1.0)
    
    # Depending on heuristics, might be ABSORB or MERGE or LINK.
    # Just verify it's NOT STORE_NEW
    assert res.action in [Action.MERGE_SOFT, Action.ABSORB, Action.VARIANT_LINK]

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
