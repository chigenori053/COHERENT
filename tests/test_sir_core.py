
import pytest
import numpy as np
from coherent.core.cortex.controller import CortexController
from coherent.core.sir import SIR

def test_cortex_observe_via_sir():
    """Verify CortexController can process input via SIR."""
    cortex = CortexController()
    
    # Test 1: Math Input
    input_str = "x + y"
    result = cortex.observe_via_sir(input_str, modality="math")
    
    # Check Structure
    assert "sir" in result
    assert "vector" in result
    assert "hash" in result
    
    sir = result["sir"]
    assert isinstance(sir, SIR)
    assert sir.modality == "math"
    assert len(sir.semantic_core.entities) > 0
    # "x", "y" should be entities
    labels = [e.label for e in sir.semantic_core.entities]
    assert "x" in labels
    assert "y" in labels
    
    # Check Vector
    vector = result["vector"]
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (1024,)
    
    # Check Loop/Commutativity via Cortex
    input_rev = "y + x"
    result_rev = cortex.observe_via_sir(input_rev, modality="math")
    
    # Hash should match for x+y vs y+x (commutative add)
    assert result["hash"] == result_rev["hash"]
    
    # Vector should match
    dist = np.linalg.norm(vector - result_rev["vector"])
    assert dist < 1e-9

def test_sir_package_exports():
    """Verify coherent.core.sir exposes necessary classes."""
    import coherent.core.sir as sir_pkg
    assert hasattr(sir_pkg, "SIR")
    assert hasattr(sir_pkg, "SIRFactory")
    assert hasattr(sir_pkg, "SIRProjector")
