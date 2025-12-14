import numpy as np
from typing import List, Tuple, Optional
import os

class OpticalScoringLayer:
    """
    Simulates optical interference to score potential rules.
    """

    def __init__(self, weights_path: Optional[str] = None, input_dim: int = 64, output_dim: int = 100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = self._load_weights(weights_path)

    def _load_weights(self, path: Optional[str]) -> np.ndarray:
        """
        Loads weights from file or initializes random weights for simulation.
        The weights represent the 'optical medium' transmission matrix.
        We use Complex weights to simulate phase and amplitude (optical interference).
        """
        if path and os.path.exists(path):
            try:
                # Assuming .npy format for now
                return np.load(path)
            except Exception:
                print(f"Warning: Could not load weights from {path}. Using random initialization.")
        
        # Initialize complex weights for wave simulation
        # Real part: Amplitude, Imaginary part: Phase
        real = np.random.randn(self.output_dim, self.input_dim)
        imag = np.random.randn(self.output_dim, self.input_dim)
        return real + 1j * imag

    def predict(self, input_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Performs the optical transformation.
        
        Args:
            input_vector: The feature vector of the AST.
            
        Returns:
            scores: A numpy array of intensity scores for each rule index.
            ambiguity: A float representing the signal ambiguity (entropy/variance).
        """
        # Ensure input dimensions match
        if input_vector.shape[0] != self.input_dim:
            # Pad or trim if mismatch (simple handling for MVP)
            if input_vector.shape[0] < self.input_dim:
                input_vector = np.pad(input_vector, (0, self.input_dim - input_vector.shape[0]))
            else:
                input_vector = input_vector[:self.input_dim]

        # Linear transformation (Optical Propagation)
        # Signal S = W * v
        output_signal = np.dot(self.weights, input_vector)
        
        # Intensity I = |S|^2
        intensity = np.abs(output_signal) ** 2
        
        # Calculate Ambiguity
        # High entropy = high ambiguity (energy spread across many rules)
        # Low entropy = low ambiguity (energy concentrated on few rules)
        ambiguity = self._calculate_ambiguity(intensity)
        
        return intensity, ambiguity

    def _calculate_ambiguity(self, intensity: np.ndarray) -> float:
        """
        Calculates ambiguity based on the entropy of the normalized intensity distribution.
        """
        total_energy = np.sum(intensity)
        if total_energy == 0:
            return 1.0 # Max ambiguity if no signal
            
        # Normalize to get probability distribution
        probs = intensity / total_energy
        
        # Shannon Entropy
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probs * np.log(probs + epsilon))
        
        # Normalize entropy by max possible entropy (log(N)) to get 0-1 range
        max_entropy = np.log(len(intensity))
        if max_entropy == 0:
            return 0.0
            
        normalized_ambiguity = entropy / max_entropy
        return float(normalized_ambiguity)
