import pytest
import datetime
from coherent.core.observation_core import ObservationCore, InferenceEvent, ObservationState, ObservationResult

@pytest.fixture
def observation_core():
    return ObservationCore()

def test_observation_stable(observation_core):
    """Test that valid/normal metrics result in STRUCTURALLY_STABLE."""
    event = InferenceEvent(
        event_id="test_stable",
        session_id="sess_1",
        timestamp=datetime.datetime.now().timestamp(),
        input_content="Hello",
        context_tags=[],
        input_modality="text",
        recall_source="Holographic",
        recall_score=0.8,
        decision_type="ACCEPT",
        confidence_score=0.9,
        entropy_score=0.2,
        final_action="PROMOTE",
        details={}
    )
    
    result = observation_core.observe(event)
    assert isinstance(result, ObservationResult)
    assert ObservationState.STRUCTURALLY_STABLE in result.states
    assert ObservationState.HIGH_UNCERTAINTY not in result.states

def test_observation_high_uncertainty(observation_core):
    """Test high entropy triggers HIGH_UNCERTAINTY."""
    event = InferenceEvent(
        event_id="test_uncertain",
        session_id="sess_2",
        timestamp=datetime.datetime.now().timestamp(),
        input_content="???",
        context_tags=[],
        input_modality="text",
        recall_source="Holographic",
        recall_score=0.5,
        decision_type="REVIEW",
        confidence_score=0.4,
        entropy_score=0.8, # High Entropy
        final_action="DEFER_REVIEW",
        details={}
    )
    
    result = observation_core.observe(event)
    assert ObservationState.HIGH_UNCERTAINTY in result.states

def test_observation_recall_empty(observation_core):
    """Test low recall score triggers RECALL_EMPTY."""
    event = InferenceEvent(
        event_id="test_empty",
        session_id="sess_3",
        timestamp=datetime.datetime.now().timestamp(),
        input_content="Unknown",
        context_tags=[],
        input_modality="text",
        recall_source="None",
        recall_score=0.05, # Extremely low recall
        decision_type="REJECT",
        confidence_score=0.1,
        entropy_score=0.5,
        final_action="SUPPRESS",
        details={}
    )
    
    result = observation_core.observe(event)
    assert ObservationState.RECALL_EMPTY in result.states

def test_observation_dogmatic(observation_core):
    """Test high confidence but low recall triggers DOGMATIC_CERTAINTY."""
    event = InferenceEvent(
        event_id="test_dogmatic",
        session_id="sess_4",
        timestamp=datetime.datetime.now().timestamp(),
        input_content="Dogma",
        context_tags=[],
        input_modality="text",
        recall_source="Holographic",
        recall_score=0.15, # Low recall
        decision_type="ACCEPT",
        confidence_score=0.95, # High confidence
        entropy_score=0.1,
        final_action="PROMOTE",
        details={}
    )
    
    result = observation_core.observe(event)
    assert ObservationState.DOGMATIC_CERTAINTY in result.states

def test_observation_simulation(observation_core):
    """Test simulation flags."""
    event = InferenceEvent(
        event_id="test_sim",
        session_id="sess_5",
        timestamp=datetime.datetime.now().timestamp(),
        input_content="Sim",
        context_tags=[],
        input_modality="text",
        recall_source="Holographic",
        recall_score=0.4,
        decision_type="ACCEPT",
        confidence_score=0.7,
        entropy_score=0.3,
        final_action="PROMOTE_SIMULATED",
        details={
            "simulation_active": True,
            "simulation_status": "SUCCESS"
        }
    )
    
    result = observation_core.observe(event)
    assert ObservationState.SIMULATION_TRIGGERED in result.states
    assert ObservationState.SIMULATION_FAILED not in result.states
    
    # Test Failed Simulation
    event_fail = InferenceEvent(
        event_id="test_sim_fail",
        session_id="sess_6",
        timestamp=datetime.datetime.now().timestamp(),
        input_content="SimFail",
        context_tags=[],
        input_modality="text",
        recall_source="Holographic",
        recall_score=0.4,
        decision_type="REJECT",
        confidence_score=0.7,
        entropy_score=0.3,
        final_action="SUPPRESS_FAILED_SIM",
        details={
            "simulation_active": True,
            "simulation_status": "FAILURE"
        }
    )
    result_fail = observation_core.observe(event_fail)
    assert ObservationState.SIMULATION_TRIGGERED in result_fail.states
    assert ObservationState.SIMULATION_FAILED in result_fail.states
