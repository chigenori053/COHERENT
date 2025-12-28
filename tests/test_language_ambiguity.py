import pytest
from unittest.mock import MagicMock
from coherent.core.core_runtime import CoreRuntime
from coherent.tools.language.experience import ExperienceUnit
from coherent.tools.language.models import SemanticIR, IntentType
from coherent.tools.language.ambiguity import ClarificationRequest

class MockExperienceManager:
    def __init__(self, high_ambiguity=False):
        self.high_ambiguity = high_ambiguity

    def recall_experience(self, query: str, threshold: float = 0.95):
        # Return a mock experience unit with specific ambiguity score
        score = 0.8 if self.high_ambiguity else 0.0
        
        # Mock Semantic IR
        sir = SemanticIR(task=IntentType.SOLVE)
        result_data = {"result": 42, "ambiguity_score": score}
        
        unit = ExperienceUnit(
             query_text="mock query",
             sir=sir,
             result=result_data
        )
        return unit

    def store_experience(self, text, sir, result):
        pass

def create_mock_runtime():
    comp_engine = MagicMock()
    comp_engine.symbolic_engine = MagicMock()
    
    # CoreRuntime init expects these
    val_engine = MagicMock()
    hint_engine = MagicMock()
    
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine)
    return runtime

def test_ambiguity_clarification_flow():
    # 1. Setup Runtime with mocked manager that returns High Ambiguity
    runtime = create_mock_runtime()
    
    # Inject mock manager
    runtime.experience_manager = MockExperienceManager(high_ambiguity=True)
    
    # 2. Execute Query
    # The actual query "ambiguous query" will likely result in an "Unknown action" error 
    # from the SemanticParser/Router since the parser is real and checks for keywords.
    result = runtime.process_natural_language("ambiguous query")
    
    # 3. Verify Fallback Behavior (Phase 4 Change)
    # The system should NOT return a ClarificationRequest just because Recall was ambiguous.
    # It should fall back to Parsing.
    print(f"Result: {result}")
    
    # Check that we did NOT recall (fallback used)
    assert not result.get("recalled"), "Should have fallen back to parsing due to high ambiguity"
    
    # Since "ambiguous query" is not valid math syntax, we expect an error or unknown action
    # But crucially, NOT a ClarificationRequest from the recall phase.
    assert "ambiguity_score" not in result or result.get("recalled") is False
    # If it falls through to 'error', that's success for this test (checking flow bypass)
    assert "error" in result or "result" in result or "explanation_needed" in result
    
def test_normal_flow_low_ambiguity():
    # 1. Setup Runtime with Low Ambiguity
    runtime = create_mock_runtime()
    runtime.experience_manager = MockExperienceManager(high_ambiguity=False)
    
    # 2. Execute Query
    result = runtime.process_natural_language("clear query")
    
    # 3. Verify Normal Result (Recalled)
    assert result.get("recalled") is True
    assert result["result"] == 42
    # Ambiguity score might be present but low, or absent if we didn't inject it deeply enough in non-ambiguous paths?
    # In my implementation, I just return {**recalled.result} so it should be there.
    assert result.get("ambiguity_score", 0.0) <= 0.3
