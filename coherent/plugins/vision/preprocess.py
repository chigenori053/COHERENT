"""
Vision Plugin Preprocessing Logic.
"""
from typing import Union, Tuple
import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    
from .errors import VisionInputError

def preprocess_image(
    image: Union["Image.Image", np.ndarray],
    target_size: Tuple[int, int] = (32, 32)
) -> np.ndarray:
    """
    Normalize image to grayscale, specific size, and [0,1] range.
    
    Args:
        image: PIL Image or Numpy Array (H, W, C)
        target_size: (Height, Width)
        
    Returns:
        np.ndarray: Normalized image (H, W) values in [0, 1]
    
    Raises:
        VisionInputError: If format is invalid or PIL missing.
    """
    # 1. Handle PIL Image
    if _PIL_AVAILABLE and isinstance(image, Image.Image):
        # Convert to Grayscale ('L')
        img_gray = image.convert('L')
        # Resize
        img_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)
        # Convert to numpy
        img_arr = np.array(img_resized, dtype=np.float32)
        # Normalize
        return img_arr / 255.0
        
    # 2. Handle Numpy Array
    elif isinstance(image, np.ndarray):
        # Allow basic checks
        img_arr = image.astype(np.float32)
        
        # If float and max > 1, assume 0-255
        if img_arr.max() > 1.0:
            img_arr /= 255.0
            
        # If 3 channels, simplistic RGB -> Grayscale
        # 0.299 R + 0.587 G + 0.114 B
        if img_arr.ndim == 3 and img_arr.shape[2] == 3:
            img_arr = np.dot(img_arr[...,:3], [0.299, 0.587, 0.114])
        elif img_arr.ndim == 3 and img_arr.shape[0] == 3:
            # Channel first?
             img_arr = np.dot(img_arr.transpose(1, 2, 0)[...,:3], [0.299, 0.587, 0.114])
        
        # Resize manually if needed? 
        # For MVP/Plugin, we assume user passes PIL or correct size ndarray if no PIL
        # If PIL not available, we can't easily resize without scipy/cv2
        if image.shape[:2] != target_size:
             if _PIL_AVAILABLE:
                  # Roundtrip via PIL to resize
                  im_pil = Image.fromarray((img_arr * 255).astype(np.uint8))
                  im_res = im_pil.resize(target_size, Image.Resampling.LANCZOS)
                  return np.array(im_res, dtype=np.float32) / 255.0
             else:
                  # Raise error or just warn?
                  # For now, strict.
                  pass 
        
        return img_arr
        
    else:
        raise VisionInputError(f"Unsupported image type: {type(image)}")
