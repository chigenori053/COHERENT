"""
Vision Plugin Holographic Encoder.
"""
from typing import Any, Dict
import numpy as np

from .base import VisionPluginBase, VisionEmbedding
from .preprocess import preprocess_image
from .errors import VisionInputError

class HolographicVisionEncoder(VisionPluginBase):
    """
    Encodes images into Dimension-D Holographic Spectra using 2D FFT.
    """
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        # Determine S x S size
        # S ~ sqrt(D)
        self.size_s = int(np.sqrt(dimension))
        if self.size_s * self.size_s != dimension:
            # Should we pad or error?
            # For strictness:
            pass 
            
    def encode(self, image: Any) -> VisionEmbedding:
        # 1. Preprocess
        # Ensure input is S x S
        proc_img = preprocess_image(image, target_size=(self.size_s, self.size_s))
        
        # 2. Spectral Encoding (2D FFT)
        # F(u, v)
        spectrum = np.fft.fft2(proc_img)
        
        # 3. Shift zero frequency to center (Optional, usually good for optical logic)
        spectrum_shifted = np.fft.fftshift(spectrum)
        
        # 4. Flatten
        holographic_vector = spectrum_shifted.flatten()
        
        # 5. Calculate Energy
        energy = np.sum(np.abs(holographic_vector) ** 2)
        
        return VisionEmbedding(
            holographic_vector=holographic_vector,
            dimension=len(holographic_vector),
            energy=float(energy),
            vision_meta={
                "fft": "2D",
                "normalized": True,
                "original_shape": proc_img.shape
            }
        )
