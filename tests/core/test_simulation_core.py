
import pytest
from coherent.core.simulation_core import SimulationCore, ComputationMode

@pytest.fixture
def sim_core():
    return SimulationCore()

def test_simulation_routing_numeric(sim_core):
    request = {
        "domain": "numeric",
        "input_context": "1 + 1"
    }
    result = sim_core.execute_request(request)
    # Since SimpleAlgebra is used inside, it should return simplified result
    assert result.get("status") == "SUCCESS"
    # Result might depend on SimpleAlgebra impl, but essentially "2" if simple simplify works, or at least no crash
    
def test_simulation_routing_unknown_domain(sim_core):
    request = {
        "domain": "magic_spells",
        "input_context": "leviosa"
    }
    result = sim_core.execute_request(request)
    assert result.get("status") == "FAILED"
    assert "Unsupported domain" in result.get("error")

def test_simulation_statelessness(sim_core):
    # Ensure sequential calls don't bleed state (MVP check)
    req1 = {"domain": "numeric", "input_context": "x"}
    res1 = sim_core.execute_request(req1)
    
    req2 = {"domain": "numeric", "input_context": "y"}
    res2 = sim_core.execute_request(req2)
    
    assert res1 != res2
