import pytest
from coherent.core.logic.controller import LogicController, LogicTask

def test_registry_initialization():
    """Test that default engines are registered."""
    controller = LogicController()
    assert "arithmetic" in controller._engines
    assert "symbolic" in controller._engines
    assert "validation" in controller._engines

def test_orchestration_compute():
    """Test dynamic orchestration for computation."""
    controller = LogicController()
    # Task: Compute 1 + 1 via Symbolic Engine (default for compute)
    task = LogicTask(task_type="compute", content="1 + 1")
    result = controller.orchestrate(task)
    # Symbolic engine usually returns a string or sympy object, let's just check it runs
    # Assuming SymbolicEngine.evaluate returns "2" or similar
    assert result == "2" or str(result) == "2"

def test_orchestration_routing_hint():
    """Test routing with explicit engine hint."""
    controller = LogicController()
    # Task: Compute via arithmetic engine explicitly
    # ArithmeticEngine might use numeric_eval or similar.
    # Note: ArithmeticEngine in this codebase might not have 'evaluate' matching Symbolic.
    # We strictly test routing logic here.
    
    # Mocking for isolation if needed, but let's try integration first.
    # If ArithmeticEngine doesn't support 'evaluate', we expect it might fail or return None/Error
    # based on the current implementation of orchestrate.
    
    # Let's inspect what orchestrate does for 'arithmetic'.
    # It currently passes 'arithmetic' type but falls back to "Executed..." string if no specific handler.
    # Let's verify the fallback or specific handler.
    
    task = LogicTask(task_type="arithmetic", content="2+2", engine_hint="arithmetic")
    result = controller.orchestrate(task)
    # The current implementation for 'arithmetic' task type is 'pass', so it returns "Executed..."
    assert "Executed arithmetic on arithmetic" in str(result)

def test_legacy_wrappers():
    """Test backward compatibility wrappers."""
    controller = LogicController()
    # execute_compute wrapper
    result = controller.execute_compute("3 * 3")
    assert result == "9" or str(result) == "9"
