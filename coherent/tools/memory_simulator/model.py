import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Callable, Optional

# Constants
DEFAULT_SIZE = 64
DEFAULT_DECAY = 0.95
THRESHOLD_RESONANCE = 0.7

@dataclass
class DHM:
    """Dynamic Holographic Memory: A wave source."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    center_x: float = 0.5
    center_y: float = 0.5
    frequency: float = 1.0  # omega
    phase_offset: float = 0.0 # phi_0
    amplitude: float = 1.0 # A_i
    width: float = 0.1 # Spatial spread (standard deviation of Gaussian envelope)
    
    # User Feedback: "Horizontal extending light band"
    # A(x,y) = A0 * exp(-(y - center_y)^2 / (2 * width^2))
    # This creates a horizontal strip.
    
    def get_field(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        # Distance from center vertical (y-axis)
        dist_y = Y - self.center_y
        
        # Gaussian Amplitude (Horizontal Band)
        # A(x,y) = A0 * exp(-dy^2 / (2 * width^2))
        amp_spatial = self.amplitude * np.exp(-(dist_y**2) / (2 * self.width**2))
        
        # Phase evolution
        # Add a spatial frequency component 'k' along X to make it look like a wave
        # Ψ = A(y) * exp(i * (k*x + omega*t + phi))
        # k can be derived from frequency or fixed. Let's make it proportional to frequency.
        k = self.frequency * 5.0 # Arbitrary wave number scaling
        
        phase_val = k * X + self.frequency * t + self.phase_offset
        
        return amp_spatial * np.exp(1j * phase_val)

@dataclass
class SHM:
    """Static Holographic Memory: A spatial filter/mask."""
    id: str = "SHM_Filter"
    active: bool = True
    # Mask function: takes X, Y returns [0, 1]
    mask_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    def apply(self, field_data: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if not self.active or self.mask_func is None:
            return field_data
        
        # Calculate mask
        F = self.mask_func(X, Y)
        return field_data * F

@dataclass
class CHM:
    """Causal Holographic Memory: A temporal gate based on conditions."""
    id: str = "CHM_Gate"
    active: bool = True
    
    # Condition: function of t (or external state) -> bool/float [0,1]
    gate_state: float = 1.0
    
    def update(self, t: float, external_context: dict):
        """Update logic for the gate based on time or causal rules."""
        # Simple implementation: logic defined externally or hardcoded for specific scenarios
        pass

    def apply(self, field_data: np.ndarray) -> np.ndarray:
        return field_data * self.gate_state

class MemorySpaceSystem:
    def __init__(self, size=DEFAULT_SIZE):
        self.size = size
        # Coordinate grid [0, 1]
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.time: float = 0.0
        self.field: np.ndarray = np.zeros((size, size), dtype=np.complex128)
        
        self.dhms: List[DHM] = []
        self.shm: SHM = SHM()
        self.chm: CHM = CHM()
        
        # Stability / Layer 3 tracking
        # We track "resonance energy" over time?
        self.resonance_history: List[float] = []

    def reset(self):
        self.time = 0.0
        self.field[:] = 0
        self.dhms.clear()
        self.shm = SHM()
        self.chm = CHM()
        self.resonance_history.clear()

    def add_dhm(self, dhm: DHM):
        self.dhms.append(dhm)

    def step(self, dt: float = 0.1):
        self.time += dt
        
        # --- Layer 1: Resonance Field (Superposition) ---
        # Sum of all DHMs
        # Reset field for fresh calculation at t (memoryless propagation + persistence?)
        # Spec says "Time-evolving", usually implies integration or fresh sum.
        # "Ψ = Σ Ψ_i" implies fresh sum of current states of generators.
        
        new_field = np.zeros((self.size, self.size), dtype=np.complex128)
        
        for dhm in self.dhms:
            wave = dhm.get_field(self.X, self.Y, self.time)
            new_field += wave
            
        # --- Layer 2: Modulation ---
        # 1. Static Filter (SHM)
        new_field = self.shm.apply(new_field, self.X, self.Y)
        
        # 2. Causal Gate (CHM)
        # Assuming CHM state is updated externally or we enforce rules here
        new_field = self.chm.apply(new_field)
        
        self.field = new_field
        
        # --- Layer 3: Stabilization ---
        # "If resonance > θ: stabilize()..."
        # We compute global resonance as total energy or max amplitude?
        # "R = |⟨Ψ_query , Ψ⟩|" implies a dot product with a query.
        # Use total energy as a proxy for "something is happening" for visualization
        total_energy = np.sum(np.abs(self.field))
        self.resonance_history.append(total_energy)

