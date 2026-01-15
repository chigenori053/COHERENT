
import pytest
import numpy as np
from unittest.mock import MagicMock, ANY
from coherent.core.cognitive_core import CognitiveCore, DecisionType

@pytest.fixture
def brain_system():
    # Real CognitiveCore & SimulationCore (via default init)
    # But mock ExperienceManager to avoid file I/O
    mock_exp = MagicMock()
    core = CognitiveCore(experience_manager=mock_exp)
    return core, mock_exp

def test_end_to_end_flow_with_simulation(brain_system):
    core, mock_exp = brain_system
    
    # 1. Setup ambiguous state to trigger Review -> Simulation
    # We need to mock the metrics calculation or memory state to force this.
    # Since _calculate_metrics is internal, we can check if we can mock the recall_engine result.
    
    # But recall_engine is real DynamicHolographicMemory.
    # Let's inject conflicting items to creating high entropy.
    # This was difficult in verification script, but we can try injecting many random vectors.
    
    # Alternative: Mock _calculate_metrics on the instance to return a specific state.
    # This is a cleaner way to test the "Flow" rather than the "Math" (which is tested in unit tests).
    
    # Create a Mock State causing REVIEW and TRIGGER
    # High Entropy (0.8), Low Confidence (0.2), High Branching
    mock_state = MagicMock()
    mock_state.entropy = 0.8
    mock_state.confidence = 0.2
    # To pass "Should Activate"?
    # trigger_if_any: H>=0.6 -> True.
    mock_state.recall_reliability = 0.8
    mock_state.branching_pressure = 0.8
    mock_state.timestamp = 123456789.0
    
    # Patch _calculate_metrics
    core._calculate_metrics = MagicMock(return_value=mock_state)
    
    # Also patch SimulationCore to return success immediately
    core.simulation_core.execute_request = MagicMock(return_value={
        "status": "SUCCESS",
        "result": "Simulated 42"
    })
    
    # Act
    decision = core.process_input("Complex Question")
    
    # Assert
    # 1. Should have called SimulationCore
    core.simulation_core.execute_request.assert_called_once()
    
    # 2. Decision should be updated to ACCEPT (if logic sets it to promote simulated)
    # Implementation of process_input:
    # if REVIEW:
    #    if trigger:
    #       sim_result = exec()
    #       if success: decision = ACCEPT
    
    assert decision.decision_type == DecisionType.ACCEPT
    assert decision.action == "PROMOTE_SIMULATED"
    assert "Simulated 42" in decision.reason

def test_simulation_authority_check(brain_system):
    """
    Verify that SimulationCore is not called if Decision is already ACCEPT or REJECT.
    """
    core, mock_exp = brain_system
    
    # Mock efficient state (High Confidence) -> ACCEPT
    mock_state = MagicMock()
    mock_state.confidence = 0.9
    mock_state.recall_reliability = 0.9
    mock_state.entropy = 0.1
    
    core._calculate_metrics = MagicMock(return_value=mock_state)
    core.simulation_core.execute_request = MagicMock()
    
    decision = core.process_input("Simple Question")
    
    assert decision.decision_type == DecisionType.ACCEPT
    # Should NOT trigger simulation
    core.simulation_core.execute_request.assert_not_called()
