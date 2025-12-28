import pytest
from coherent.core.core_runtime import CoreRuntime
from coherent.core.computation_engine import ComputationEngine
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.validation_engine import ValidationEngine
from coherent.core.hint_engine import HintEngine
from coherent.tools.language.experience import ExperienceManager

# Mock dependencies
class MockValidationEngine(ValidationEngine):
    def __init__(self):
        pass
    def validate_step(self, before, after, context=None):
        return {"valid": False, "status": "unknown"}

class MockHintEngine(HintEngine):
    def __init__(self):
        pass

def create_runtime_with_memory():
    # We rely on CoreRuntime lazy initialization of OpticalStore in Phase 2
    # But for tests we might want to ensure it uses a fresh in-memory store if possible.
    # The default CoreRuntime implementation creates OpticalFrequencyStore(capacity=1000).
    # We should ensure torch/sentence-transformers are available or mocked.
    # If they are not available, ExperienceManager disables itself.
    
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = MockValidationEngine()
    hint_engine = MockHintEngine()
    
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine)
    return runtime

def test_experience_storage_and_recall():
    runtime = create_runtime_with_memory()
    
    # Check if enabled
    if not runtime.experience_manager.is_enabled():
        pytest.skip("Test skipped: Optical/Semantic dependencies not available.")
    
    query = "Solve 2 + 3"
    
    # 1. First Execution (Cold)
    # This should compute and then store.
    result_cold = runtime.process_natural_language(query)
    
    assert "result" in result_cold
    assert result_cold["result"] == 5
    assert not result_cold.get("recalled", False)
    
    # Verify it was stored
    assert len(runtime.optical_store.index_to_id) > 0, "Optical store should not be empty after execution"
    
    # 2. Second Execution (Warm/Recall)
    # This should find the stored experience and return it.
    print("Testing Recall...")
    result_warm = runtime.process_natural_language(query)
    print(f"Warm Result: {result_warm}")
    
    assert "result" in result_warm
    assert result_warm["result"] == 5
    assert result_warm.get("recalled") is True, f"Should be recalled. Got: {result_warm}"
    
    # 3. Similar Query (Generalization)
    query_var = "solve 2 + 3" # Case change
    result_var = runtime.process_natural_language(query_var)
    assert result_var.get("recalled") is True, f"Variance should be recalled. Got: {result_var}"


def test_experience_different_query():
    runtime = create_runtime_with_memory()
    if not runtime.experience_manager.is_enabled():
        pytest.skip("Test skipped: Optical/Semantic dependencies not available.")
        
    # Store one experience
    runtime.process_natural_language("Solve 10 + 10")
    
    # Query something completely different
    result = runtime.process_natural_language("Solve 5 + 5")
    
    assert "result" in result
    assert result["result"] == 10
    assert not result.get("recalled", False) # Should NOT be recall
