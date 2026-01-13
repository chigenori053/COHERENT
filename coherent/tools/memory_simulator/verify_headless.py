from model import MemorySpaceSystem
from scenarios import setup_scenario_a, setup_scenario_b, setup_scenario_c

def verify():
    print("Starting Headless Verification...")
    sys = MemorySpaceSystem(size=64)
    
    # --- Scenario A ---
    print("\n--- Verifying Scenario A (Resonance) ---")
    setup_scenario_a(sys)
    sys.step()
    res_a = sys.resonance_history[-1]
    print(f"Resonance A: {res_a:.4f}")
    assert res_a > 10.0, "Scenario A should have significant resonance"
    
    # --- Scenario B ---
    print("\n--- Verifying Scenario B (SHM Block) ---")
    setup_scenario_b(sys)
    sys.step()
    res_b = sys.resonance_history[-1]
    print(f"Resonance B: {res_b:.4f}")
    
    # B should be less than A significantly because half the field (including target) is blocked
    # Actually setup_a puts target at 0.3, 0.3. Block mask from setup_b blocks x < 0.5.
    # So Targe is BLOCKED. Distractor A (0.7) is visible.
    # Resonance should be lower than A (roughly 2/3? or less if Target was strong?)
    # All DHMs have similar amplitude in setup.
    assert res_b < res_a, "Scenario B resonance should be lower than A due to masking"
    
    # --- Scenario C ---
    print("\n--- Verifying Scenario C (CHM Violation) ---")
    setup_scenario_c(sys)
    sys.chm.gate_state = 0.0 # Closed
    sys.step()
    res_c = sys.resonance_history[-1]
    print(f"Resonance C: {res_c:.4f}")
    
    assert res_c < 0.001, "Scenario C resonance should be ~0 (Gate Closed)"
    
    print("\nSUCCESS: All Scenarios Verified.")

if __name__ == "__main__":
    verify()
