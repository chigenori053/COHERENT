import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from .models import (
    TraceEvent, EncodeEndEvent, RecallTopKEvent, InterferenceTopKEvent
)

@dataclass
class VisualizerState:
    # 2D Holographic Signal
    amplitude_map: np.ndarray = field(default_factory=lambda: np.zeros((32, 32)))
    phase_map: np.ndarray = field(default_factory=lambda: np.zeros((32, 32)))
    
    # 3D Memory Space Topology
    # Fixed X, Y for Memory Slots (random or fixed grid for visualization)
    memory_coords: Dict[str, tuple] = field(default_factory=dict)
    
    # 3D Activation (Z-axis)
    memory_activation: Dict[str, float] = field(default_factory=dict)
    
    # Phase Tracking
    memory_phase: Dict[str, str] = field(default_factory=dict)
    
    last_event_seq: int = -1

class StateManager:
    def __init__(self):
        self.state = VisualizerState()
        # Pre-generate coordinates for potential memory slots (mock topology)
        self._init_topology()

    def _init_topology(self, count=100):
        # Generate stable random coords for visualization consistency
        rng = np.random.default_rng(42)
        # Mock memory IDs
        # In real scenario, we'd receive all known IDs at start or discover them
        # For now, we allow dynamic addition
        pass
    
    def _assign_coords_if_needed(self, mem_id: str):
        if mem_id not in self.state.memory_coords:
            rng = np.random.default_rng(hash(mem_id) & 0xFFFFFF)
            self.state.memory_coords[mem_id] = (rng.uniform(-10, 10), rng.uniform(-10, 10))
            self.state.memory_activation[mem_id] = 0.0

    def process_event(self, event: TraceEvent) -> VisualizerState:
        if event.event_seq <= self.state.last_event_seq:
            # Ignore out of order or duplicate for now
            pass
        
        self.state.last_event_seq = event.event_seq

        if isinstance(event, EncodeEndEvent):
            self._update_2d_signal(event)
        elif isinstance(event, RecallTopKEvent):
            self._update_recall_activation(event)
        elif isinstance(event, InterferenceTopKEvent):
            self._update_interference_signal(event)
        
        return self.state

    def _update_2d_signal(self, event: EncodeEndEvent):
        # Reshape 1024 -> 32x32
        real = np.array(event.H_complex.real)
        imag = np.array(event.H_complex.imag)
        
        # Ensure correct size (pad or truncate)
        target_size = 32 * 32
        current_size = len(real)
        
        if current_size != target_size:
            # Simple handling: pad with zeros or truncate
            if current_size < target_size:
                real = np.pad(real, (0, target_size - current_size))
                imag = np.pad(imag, (0, target_size - current_size))
            else:
                real = real[:target_size]
                imag = imag[:target_size]
        
        c_sig = real + 1j * imag
        grid = c_sig.reshape((32, 32))
        
        self.state.amplitude_map = np.abs(grid)
        self.state.phase_map = np.angle(grid)

    def _update_recall_activation(self, event: RecallTopKEvent):
        # Decay previous activations
        for k in self.state.memory_activation:
            self.state.memory_activation[k] *= 0.9 # Decay factor
            
        # Update TopK
        for item in event.topK:
            self._assign_coords_if_needed(item.mem_id)
            # Z = resonance
            self.state.memory_activation[item.mem_id] = item.resonance
            if item.phase:
                self.state.memory_phase[item.mem_id] = item.phase

    def _update_interference_signal(self, event: InterferenceTopKEvent):
        # Spec 3.2.5: Reshape interference tensor to 32x32
        # Assuming event contains the combined interference vector or list of components?
        # Model has 'interference': List[InterferenceItem]
        # We should sum them or visualize the first one?
        # "Reshape interference tensor to 32x32" implies a single composite view or switching to interference view.
        # Let's sum complex components for the visual.
        
        total_real = np.zeros(1024)
        total_imag = np.zeros(1024)
        
        valid_items = False
        
        for item in event.interference:
            self._assign_coords_if_needed(item.mem_id)
            
            r = np.array(item.complex.real)
            i = np.array(item.complex.imag)
            
            # Simple validation on length
            if len(r) == 1024:
                total_real += r
                total_imag += i
                valid_items = True

        if valid_items:
             c_sig = total_real + 1j * total_imag
             grid = c_sig.reshape((32, 32))
             self.state.amplitude_map = np.abs(grid) # Update main view to show interference pattern
             self.state.phase_map = np.angle(grid)
