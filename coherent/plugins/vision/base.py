"""
Vision Plugin Abstract Base Class.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple
import numpy as np

class VisionEmbedding(NamedTuple):
    """
    Structured output of the Vision Plugin.
    """
    holographic_vector: np.ndarray # Complex vector
    dimension: int
    energy: float
    vision_meta: Dict[str, Any]

class VisionPluginBase(ABC):
    """
    Abstract contract for Vision Plugins.
    """
    
    @abstractmethod
    def encode(self, image: Any) -> VisionEmbedding:
        """
        Encode an image into a holographic vector.
        
        Args:
            image: Input image (PIL.Image or np.ndarray)
            
        Returns:
            VisionEmbedding containing the complex feature vector.
        """
        raise NotImplementedError
