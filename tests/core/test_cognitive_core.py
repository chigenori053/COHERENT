
import pytest
import numpy as np
from unittest.mock import MagicMock
from coherent.core.cognitive_core import CognitiveCore, CognitiveStateVector, DecisionType

@pytest.fixture
def cognitive_core():
    # Mock ExperienceManager
    mock_exp = MagicMock()
    core = CognitiveCore(experience_manager=mock_exp)
    # Mock SimulationCore to avoid external dependencies
    core.simulation_core = MagicMock()
    return core

def test_metrics_calculation_single_dominant(cognitive_core):
    """
    Test case: Single dominant memory (Entropy should be low, Confidence high)
    """
    results = [("id1", 10.0), ("id2", 1.0), ("id3", 1.0)]
    query = np.zeros(64)
    
    state = cognitive_core._calculate_metrics(results, query)
    
    assert state.entropy < 0.25
    assert state.confidence > 0.8
    assert state.margin_confidence > 0.8
    assert state.concentration_confidence > 0.8

def test_metrics_calculation_high_entropy(cognitive_core):
    """
    Test case: Uniform distribution (Entropy high, Confidence low)
    """
    results = [("id1", 5.0), ("id2", 5.0), ("id3", 5.0), ("id4", 5.0)]
    query = np.zeros(64)
    
    state = cognitive_core._calculate_metrics(results, query)
    
    assert abs(state.entropy - 1.0) < 0.01
    assert state.confidence < 0.1

def test_recall_reliability_logic(cognitive_core):
    # High score
    results_high = [("id1", 1.0)] 
    state_high = cognitive_core._calculate_metrics(results_high, np.zeros(64))
    assert state_high.recall_reliability > 0.9
    
    # Low score
    results_low = [("id1", 0.4)]
    state_low = cognitive_core._calculate_metrics(results_low, np.zeros(64))
    assert state_low.recall_reliability < 0.2

def test_decision_logic_accept(cognitive_core):
    # Case: High Confidence, High R -> ACCEPT
    state = CognitiveStateVector(
        entropy=0.1, confidence=0.8, margin_confidence=0.8, 
        concentration_confidence=0.9, recall_reliability=0.7, branching_pressure=0.1
    )
    decision = cognitive_core._make_decision(state)
    assert decision.decision_type == DecisionType.ACCEPT
    assert decision.action == "PROMOTE"

def test_decision_logic_reject(cognitive_core):
    # Case: Low Confidence -> REJECT
    state = CognitiveStateVector(
        entropy=0.9, confidence=0.3, margin_confidence=0.1, 
        concentration_confidence=0.1, recall_reliability=0.3, branching_pressure=0.9
    )
    decision = cognitive_core._make_decision(state)
    assert decision.decision_type == DecisionType.REJECT
    assert decision.action == "SUPPRESS"

def test_decision_logic_review(cognitive_core):
    # Case: Middle Ground -> REVIEW
    state = CognitiveStateVector(
        entropy=0.5, confidence=0.5, margin_confidence=0.4, 
        concentration_confidence=0.5, recall_reliability=0.8, branching_pressure=0.5
    )
    decision = cognitive_core._make_decision(state)
    assert decision.decision_type == DecisionType.REVIEW

def test_simulation_trigger_logic(cognitive_core):
    # Trigger if R < 0.4 or H >= 0.6 or B >= 0.7 or C < 0.4
    
    # Safe state - No trigger
    state_safe = CognitiveStateVector(
        entropy=0.2, confidence=0.8, margin_confidence=0.8, 
        concentration_confidence=0.8, recall_reliability=0.8, branching_pressure=0.2
    )
    assert cognitive_core._should_activate_simulation(state_safe) == False
    
    # Trigger cases
    # 1. Low Recall
    state_low_r = CognitiveStateVector(0.2, 0.8, 0.8, 0.8, 0.3, 0.2)
    assert cognitive_core._should_activate_simulation(state_low_r) == True
    
    # 2. High Uncertainty (Entropy)
    state_high_h = CognitiveStateVector(0.7, 0.5, 0.5, 0.5, 0.8, 0.2)
    assert cognitive_core._should_activate_simulation(state_high_h) == True
    
    # 3. Branching Pressure
    state_high_b = CognitiveStateVector(0.2, 0.8, 0.8, 0.8, 0.8, 0.8)
    assert cognitive_core._should_activate_simulation(state_high_b) == True

