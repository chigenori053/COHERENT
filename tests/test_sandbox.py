import pytest
import shutil
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock

from coherent.experimental.sandbox import Sandbox
from coherent.core.memory.holographic.orchestrator import MemoryOrchestrator
# Attempt to import Action/DecisionState, mocking if needed or strictly relying on what Orchestrator uses
# Orchestrator uses: from .causal import CausalHolographicMemory, DecisionState, Action
from coherent.core.memory.holographic.causal import Action, DecisionState

# Helper to clean up
@pytest.fixture
def clean_sandbox():
    path = Path("tests_sandbox_output")
    if path.exists():
        shutil.rmtree(path)
    yield path
    if path.exists():
        shutil.rmtree(path)

def test_sandbox_directory_creation(clean_sandbox):
    sandbox = Sandbox(root_dir=str(clean_sandbox))
    assert clean_sandbox.exists()
    assert (clean_sandbox / "outputs").exists()
    assert (clean_sandbox / "logs").exists()
    assert (clean_sandbox / "traces").exists()
    assert (clean_sandbox / "metadata").exists()

def test_sandbox_logging(clean_sandbox):
    sandbox = Sandbox(root_dir=str(clean_sandbox))
    
    # Simulate decision
    # DecisionState is a dataclass, but we can verify generic object logging too if not importable
    # But we imported it.
    ds = DecisionState(
        resonance_score=0.9,
        margin=0.1,
        repetition_count=1,
        entropy_estimate=0.1,
        memory_origin="Dynamic",
        historical_conflict_rate=0.0
    )
    
    action = Action.PROMOTE
    metadata = {"test": "data"}
    
    sandbox.capture_decision(ds, action, metadata)
    
    log_file = clean_sandbox / "logs" / "decisions.jsonl"
    assert log_file.exists()
    
    with open(log_file, "r") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["type"] == "decision"
        assert str(action) in data["action"] # Action might serialize as "Action.PROMOTE"
        assert data["metadata"]["test"] == "data"
        assert "decision_state" in data
        assert data["decision_state"]["resonance_score"] == 0.9

def test_orchestrator_integration(clean_sandbox):
    # Mock dependencies for Orchestrator
    dynamic = MagicMock()
    static = MagicMock()
    causal = MagicMock()
    
    # Mock return values
    static.query.return_value = []
    causal.evaluate_decision.return_value = Action.RETAIN
    
    sandbox = Sandbox(root_dir=str(clean_sandbox))
    
    orch = MemoryOrchestrator(dynamic, static, causal, sandbox=sandbox)
    
    state = np.array([1, 2, 3])
    orch.process_input(state, {"source": "test"})
    
    # Check if files created
    # Inputs
    input_log = clean_sandbox / "logs" / "inputs.jsonl"
    assert input_log.exists()
    
    # Decisions
    decision_log = clean_sandbox / "logs" / "decisions.jsonl"
    assert decision_log.exists()
    
    # Verify content of input log
    with open(input_log, "r") as f:
        data = json.loads(f.readline())
        assert data["type"] == "input"
        assert data["metadata"]["source"] == "test"

def test_non_invasive(clean_sandbox):
    """Verify that Orchestrator behavior remains same mainly (no crashes)."""
    dynamic = MagicMock()
    static = MagicMock()
    causal = MagicMock()
    static.query.return_value = []
    causal.evaluate_decision.return_value = Action.RETAIN
    
    # With Sandbox
    sandbox = Sandbox(root_dir=str(clean_sandbox))
    orch = MemoryOrchestrator(dynamic, static, causal, sandbox=sandbox)
    state = np.array([1, 2, 3])
    orch.process_input(state)
    
    # Without Sandbox
    orch_no_sb = MemoryOrchestrator(dynamic, static, causal)
    orch_no_sb.process_input(state)
    
    # Just asserting no exceptions and calls made
    assert causal.evaluate_decision.call_count == 2
