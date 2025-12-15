import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import os

class OpticalScoringLayer(nn.Module):
    """
    Simulates optical interference to score potential rules using PyTorch.
    Learns the relationship between AST structure (Phase/Amplitude) and Rule Applicability.
    """

    def __init__(self, weights_path: Optional[str] = None, input_dim: int = 64, output_dim: int = 100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize complex weights for wave simulation
        # Real part: Amplitude, Imaginary part: Phase
        # We explicitly manage weights as a Parameter for autograd
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim, dtype=torch.cfloat))
        
        if weights_path:
            self._load_weights(weights_path)

    def _load_weights(self, path: str):
        if os.path.exists(path):
            try:
                # Load state dict or raw tensor
                # For compatibility with potential old numpy files, we check extension
                if path.endswith('.npy'):
                    np_weights = np.load(path)
                    with torch.no_grad():
                        self.weights.copy_(torch.from_numpy(np_weights))
                else:
                    state_dict = torch.load(path)
                    self.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load weights from {path}. Using random initialization. Error: {e}")

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Performs the optical transformation.
        
        Args:
            input_tensor: The feature tensor of the AST. Shape: [Batch, InputDim] or [InputDim]
            
        Returns:
            intensity: Tensor of intensity scores.
            ambiguity: Float representing signal ambiguity.
        """
        # Ensure input is at least 1D, potentially add batch dim if missing
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0) # [1, Dim]
            
        # Ensure complex type for interference (Phase + Amplitude)
        if not input_tensor.is_complex():
            input_tensor = input_tensor.type(torch.cfloat)

        # Linear transformation (Optical Propagation)
        # Signal S = x * W^T
        # input: [B, In], weights: [Out, In] -> [B, Out]
        signal = torch.matmul(input_tensor, self.weights.t())
        
        # Intensity I = |S|^2
        intensity = signal.abs() ** 2
        
        # Calculate Ambiguity (entropy) on the first item in batch (usually batch=1 for inference)
        # For batch training, we might average it, but for compatibility with predict signature:
        ambiguity = self._calculate_ambiguity(intensity[0])
        
        return intensity, ambiguity

    def _calculate_ambiguity(self, intensity: torch.Tensor) -> float:
        """
        Calculates ambiguity based on the entropy of the normalized intensity distribution.
        """
        with torch.no_grad():
            total_energy = intensity.sum()
            if total_energy == 0:
                return 1.0 
                
            probs = intensity / total_energy
            epsilon = 1e-10
            # Shannon Entropy
            entropy = -torch.sum(probs * torch.log(probs + epsilon))
            
            # Normalize
            max_entropy = np.log(len(intensity))
            if max_entropy == 0:
                return 0.0
                
            normalized_ambiguity = entropy.item() / max_entropy
            return float(normalized_ambiguity)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
