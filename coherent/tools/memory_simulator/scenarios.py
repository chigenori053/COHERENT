import numpy as np
from model import MemorySpaceSystem, DHM, SHM, CHM

def setup_scenario_a(sys: MemorySpaceSystem):
    """
    Scenario A: Single Resonance
    1 Query
    3 DHM bands active (Horizontal).
    Spaced vertically to avoid full overlap, but allowing interference if widths are large.
    """
    sys.reset()
    
    # DHM 1: The Target (Freq 1.0, Y=0.5)
    # Horizontal bands -> X doesn't define position, only phase start.
    dhm1 = DHM(id="Target", center_y=0.5, frequency=2.0, width=0.1)
    
    # DHM 2: Distractor A (Freq 5.0, Y=0.2)
    dhm2 = DHM(id="DistractorA", center_y=0.2, frequency=5.0, width=0.1)
    
    # DHM 3: Distractor B (Freq 1.5, Y=0.8)
    dhm3 = DHM(id="DistractorB", center_y=0.8, frequency=1.5, width=0.1)
    
    sys.add_dhm(dhm1)
    sys.add_dhm(dhm2)
    sys.add_dhm(dhm3)
    
    # Disable SHM/CHM for pure resonance test
    sys.shm.active = False
    sys.chm.active = False
    
    return "Scenario A: Single Resonance Setup. 3 Horizontal DHM bands active."

def setup_scenario_b(sys: MemorySpaceSystem):
    """
    Scenario B: Static Filter Block
    Same as A, but SHM is active.
    Since we have horizontal bands, we can block a vertical region to show interruption.
    """
    setup_scenario_a(sys)
    
    sys.shm.active = True
    
    # Define mask: Vertical Bar Block in the middle? 
    # Or block the right side?
    # Let's block the center vertical strip [0.4, 0.6] to show interruption.
    def vertical_bar_mask(X, Y):
        # 0 if 0.4 < x < 0.6 else 1
        condition = (X > 0.4) & (X < 0.6)
        return np.where(condition, 0.0, 1.0)
    
    sys.shm.mask_func = vertical_bar_mask
    
    return "Scenario B: SHM Filter Active. Blocking Center Vertical Strip."

def setup_scenario_c(sys: MemorySpaceSystem):
    """
    Scenario C: Causal Violation
    Same as A, but CHM Gate is closed (or oscillating).
    """
    setup_scenario_a(sys)
    
    sys.chm.active = True
    # Gate is closed initially
    sys.chm.gate_state = 0.0
    
    # Can define update logic in simulation loop or here?
    # For this simple sim, we can just toggle it manually or set it fixed.
    # "Expect: band interruption"
    
    return "Scenario C: Causal Gate Closed (0.0). No signal should pass."
