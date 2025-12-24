
import pytest
import numpy as np
from coherent.plugins.vision import HolographicVisionEncoder, VisionInputError

def test_encoder_determinism():
    """Ensure same input yields same output."""
    encoder = HolographicVisionEncoder(dimension=1024)
    
    # Create random image 32x32
    # Ensure range [0, 255]
    img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    
    emb1 = encoder.encode(img)
    emb2 = encoder.encode(img)
    
    np.testing.assert_array_equal(emb1.holographic_vector, emb2.holographic_vector)
    assert emb1.energy == emb2.energy

def test_dimension():
    """Ensure output dimension is correct."""
    dim = 1024
    encoder = HolographicVisionEncoder(dimension=dim)
    img = np.zeros((32, 32), dtype=np.uint8)
    
    emb = encoder.encode(img)
    assert len(emb.holographic_vector) == dim
    assert emb.dimension == dim

def test_energy_consistency():
    """Parseval's theorem check (roughly)."""
    encoder = HolographicVisionEncoder(dimension=1024) # 32x32
    
    # Impulse image (center pixel 1, others 0) -> Flat spectrum
    img = np.zeros((32, 32), dtype=np.uint8)
    img[16, 16] = 255 # Only one pixel
    
    emb = encoder.encode(img)
    
    # Energy in spatial domain approx Energy in frequency domain
    # Note: numpy.fft.fft2 is unscaled, so Parseval is sum(|f|^2) = N * sum(|F|^2) or similar depending on norm
    # Standard DFT: sum(|x|^2) = (1/N) sum(|X|^2)
    # np.fft.fft2 returns unnormalized sum. 
    # With norm=None (default):
    # E_freq = sum(|X|^2) = N_pixels * sum(|x|^2)
    
    # Spatial Energy:
    # Preprocessing normalizes 255 -> 1.0
    # So one pixel is 1.0. sum(|x|^2) = 1.0
    
    # Frequency energy should be 32*32 * 1.0 = 1024
    
    # Let's check roughly
    assert np.isclose(emb.energy, 1024.0)

def test_preprocess_resize():
    """Test auto-resizing of numpy array."""
    encoder = HolographicVisionEncoder(dimension=1024) # expects 32x32
    
    # Input 64x64
    img_large = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    
    # Without PIL this might act differently or error if logic is specific
    # We implemented manual resize check that passes if matched, else tries PIL
    # If PIL is present it works. If not, it might fail or we assume environment has it.
    
    try:
        from PIL import Image
        emb = encoder.encode(img_large)
        assert emb.vision_meta["original_shape"] == (32, 32)
    except ImportError:
        pytest.skip("PIL not available")

def test_invalid_input():
    encoder = HolographicVisionEncoder()
    with pytest.raises(VisionInputError):
        encoder.encode("not an image")
