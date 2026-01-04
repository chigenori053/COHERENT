"""
Core Plotting Logic

Implements the projection logic (PCA) strictly according to Spec 3.
"""

import numpy as np
import io

# Optional import: sklearn is required for visualization
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

class DHMStateProjector:
    def __init__(self, output_dim=2):
        self.output_dim = output_dim
        self.pca = None
        
    def _complex_to_real_concat(self, complex_vectors: np.ndarray) -> np.ndarray:
        """
        Converts complex vectors (N, D) into real vectors (N, 2D)
        by concatenating real and imaginary parts.
        Spec 3.1
        """
        # vectors shape: (N, D)
        real_part = np.real(complex_vectors)
        imag_part = np.imag(complex_vectors)
        
        # Concatenate along the feature dimension
        return np.concatenate([real_part, imag_part], axis=1)

    def fit(self, all_states: np.ndarray):
        """
        Fit PCA on a representative set of states.
        Spec 3.2: Same PCA basis must be used for all H_t and H_c.
        """
        if PCA is None:
            raise ImportError("scikit-learn is required for visualization.")
            
        real_data = self._complex_to_real_concat(all_states)
        self.pca = PCA(n_components=self.output_dim)
        self.pca.fit(real_data)

    def transform(self, states: np.ndarray) -> np.ndarray:
        """
        Project states into 2D space using fitted PCA.
        """
        if self.pca is None:
            raise ValueError("Projector must be fitted first.")
            
        real_data = self._complex_to_real_concat(states)
        return self.pca.transform(real_data)

    def get_explained_variance(self):
        return self.pca.explained_variance_ratio_
