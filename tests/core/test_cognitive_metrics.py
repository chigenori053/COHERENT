
import pytest
import numpy as np
from unittest.mock import MagicMock
from coherent.core.cognitive_core import CognitiveCore, CognitiveStateVector, DecisionType

@pytest.fixture
def cognitive_core():
    # Mock ExperienceManager
    mock_exp = MagicMock()
    core = CognitiveCore(experience_manager=mock_exp)
    return core

def test_metrics_calculation_single_dominant(cognitive_core):
    """
    Test case: Single dominant memory (Entropy should be low, Confidence high)
    """
    # Simulate scores: one very high, others low
    # e.g. [10.0, 1.0, 1.0 ...]
    results = [("id1", 10.0), ("id2", 1.0), ("id3", 1.0)]
    query = np.zeros(64)
    
    state = cognitive_core._calculate_metrics(results, query)
    
    # Entropy should be low
    assert state.entropy < 0.25
    
    # Confidence should be high
    # p1 will be close to 1.0, p2 close to 0.0
    # C_delta approx 1.0
    # C_H approx 1.0
    assert state.confidence > 0.8
    assert state.margin_confidence > 0.8
    assert state.concentration_confidence > 0.8

def test_metrics_calculation_high_entropy(cognitive_core):
    """
    Test case: Uniform distribution (Entropy high, Confidence low)
    """
    # Equal scores
    results = [("id1", 5.0), ("id2", 5.0), ("id3", 5.0), ("id4", 5.0)]
    query = np.zeros(64)
    
    state = cognitive_core._calculate_metrics(results, query)
    
    # Entropy should be 1.0 (normalized normalized entropy calculation check needed)
    # H = -sum(p log p) / log(n)
    # If p are equal, sum(p log p) = log(1/n) = -log n
    # H = -(-log n) / log n = 1.0
    assert abs(state.entropy - 1.0) < 0.01
    
    # Confidence should be low
    # C_delta = 0 (top two are equal)
    # C_H = 0 (1 - 1.0)
    assert state.confidence < 0.1

def test_recall_reliability_logic(cognitive_core):
    # R depends on I_max (top score)
    # theta=0.65, kappa=10.0
    
    # High score
    results_high = [("id1", 1.0)] # Score 1.0
    state_high = cognitive_core._calculate_metrics(results_high, np.zeros(64))
    # sigmoid(10 * (1.0 - 0.65)) = sigmoid(3.5) ~ 0.97
    assert state_high.recall_reliability > 0.9
    
    # Low score
    results_low = [("id1", 0.4)] # Score 0.4
    state_low = cognitive_core._calculate_metrics(results_low, np.zeros(64))
    # sigmoid(10 * (0.4 - 0.65)) = sigmoid(-2.5) ~ 0.07
    assert state_low.recall_reliability < 0.2

def test_decision_logic(cognitive_core):
    # Case 1: High Confidence, High R -> ACCEPT
    state_accept = CognitiveStateVector(
        entropy=0.1, confidence=0.8, margin_confidence=0.8, 
        concentration_confidence=0.9, recall_reliability=0.7, branching_pressure=0.1
    )
    decision = cognitive_core._make_decision(state_accept)
    assert decision.decision_type == DecisionType.ACCEPT
    
    # Case 2: Low Confidence -> REJECT
    state_reject = CognitiveStateVector(
        entropy=0.9, confidence=0.3, margin_confidence=0.1, 
        concentration_confidence=0.1, recall_reliability=0.3, branching_pressure=0.9
    )
    decision = cognitive_core._make_decision(state_reject)
    assert decision.decision_type == DecisionType.REJECT
    
    # Case 3: Middle Ground -> REVIEW
    state_review = CognitiveStateVector(
        entropy=0.5, confidence=0.5, margin_confidence=0.4, 
        concentration_confidence=0.5, recall_reliability=0.8, branching_pressure=0.5
    )
    decision = cognitive_core._make_decision(state_review)
    assert decision.decision_type == DecisionType.REVIEW

def test_simulation_trigger(cognitive_core):
    # Trigger if R < 0.4 or H >= 0.6 or B >= 0.7 or C < 0.4
    
    # Safe state - No trigger
    state_safe = CognitiveStateVector(
        entropy=0.2, confidence=0.8, margin_confidence=0.8,
        concentration_confidence=0.8, recall_reliability=0.8, branching_pressure=0.2
    )
    assert cognitive_core._should_activate_simulation(state_safe) == False
    
    # Low Recall -> Trigger
    state_low_r = CognitiveStateVector(state_safe.entropy, 0.8, 0.8, 0.8, 0.3, 0.2)
    assert cognitive_core._should_activate_simulation(state_low_r) == True
    
    # High Entropy -> Trigger
    state_high_h = CognitiveStateVector(0.7, 0.5, 0.5, 0.3, 0.8, 0.2)
    assert cognitive_core._should_activate_simulation(state_high_h) == True
