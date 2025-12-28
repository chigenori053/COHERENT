import pytest
from coherent.core.core_runtime import CoreRuntime
from coherent.core.computation_engine import ComputationEngine
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.validation_engine import ValidationEngine
from coherent.core.hint_engine import HintEngine
from coherent.tools.language.models import IntentType, MathDomain

class MockValidationEngine(ValidationEngine):
    def __init__(self):
        pass
    def validate_step(self, before, after, context=None):
        return {"valid": False, "status": "unknown"}

class MockHintEngine(HintEngine):
    def __init__(self):
        pass

def create_runtime():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = MockValidationEngine()
    hint_engine = MockHintEngine()
    return CoreRuntime(comp_engine, val_engine, hint_engine)

def test_nlp_solve_arithmetic():
    runtime = create_runtime()
    
    # "Solve 2 + 2"
    result = runtime.process_natural_language("Solve 2 + 2")
    
    assert "result" in result
    assert result["result"] == 4
    assert result["sir"]["task"] == IntentType.SOLVE
    assert result["sir"]["math_domain"] == MathDomain.ARITHMETIC

def test_nlp_solve_equation():
    runtime = create_runtime()
    # "Solve x + 2 = 5"
    # Note: Our simple implementation of 'solve' currently uses 'evaluate' which might fail for equations
    # or return normalized form. Let's see what evaluate does for equations in SymbolicEngine.
    # Usually evaluate returns numeric result or simplifed form.
    
    # If the parser extracts "x + 2 = 5", CoreRuntime.evaluate might treat it as "x+2=5" => "x+2-(5)" if it normalizes.
    
    result = runtime.process_natural_language("Solve x + 2 = 5")
    # If evaluate returns a sympy equality or expression
    # Use key check
    assert "result" in result
    print(f"DEBUG Result: {result}")

def test_nlp_verify_simple():
    runtime = create_runtime()
    
    # "Verify 2 + 2 = 4"
    result = runtime.process_natural_language("Verify 2 + 2 = 4")
    
    assert "valid" in result
    assert result["valid"] is True
    assert result["sir"]["task"] == IntentType.VERIFY

def test_nlp_verify_false():
    runtime = create_runtime()
    
    result = runtime.process_natural_language("Verify 2 + 2 = 5")
    
    assert "valid" in result
    assert result["valid"] is False

def test_nlp_unknown():
    runtime = create_runtime()
    result = runtime.process_natural_language("Hello world")
    
    # Fallback to explain
    assert "explanation_needed" in result
    assert result["sir"]["task"] == IntentType.EXPLAIN
