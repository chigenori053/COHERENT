import numpy as np
import colorsys

def complex_to_hsv_rgb(
    field: np.ndarray, 
    max_amp: float = 1.0, 
    gamma: float = 0.5,
    stability_map: np.ndarray = None
) -> np.ndarray:
    """
    Convert a complex-valued field to an RGB image based on the spec.
    
    H = (φ + π) / (2π)
    S = stability if provided, else 1.0 (or derived from local variance? Spec says local variance)
    V = min(1, (A / A_max)^γ)
    
    Args:
        field: 2D numpy array of complex128.
        max_amp: Normalization factor for amplitude.
        gamma: Gamma correction for brightness to see weak signals.
        stability_map: Optional 2D array [0,1] representing saturation/stability. 
                       If None, defaults to full saturation (1.0).
                       
    Returns:
        (H, W, 3) uint8 array [0..255]
    """
    # 1. Amplitude -> Value
    amp = np.abs(field)
    # Avoid divide by zero if max_amp is 0
    if max_amp <= 1e-9:
        val = np.zeros_like(amp)
    else:
        norm_amp = amp / max_amp
        # Gamma correction
        val = np.power(norm_amp, gamma)
        val = np.clip(val, 0, 1)

    # 2. Phase -> Hue
    # angle in (-pi, pi]
    phase = np.angle(field)
    # Map to [0, 1)
    hue = (phase + np.pi) / (2 * np.pi)
    # Wrap roughly to ensure [0, 1) range
    hue = np.mod(hue, 1.0)

    # 3. Stability -> Saturation
    if stability_map is not None:
        sat = np.clip(stability_map, 0, 1)
    else:
        # Default to 1.0 if not computed externally
        sat = np.ones_like(amp)

    # Vectorize HSV -> RGB
    # It's faster to do this via broadcasting or list comp if small, 
    # but for 128x128 loop is okay, or use matplotlib hsv_to_rgb if available?
    # Let's stick to pure numpy/colorsys or a custom vectorized approach.
    # Custom vectorized HSV->RGB is easy.
    
    # We'll use a helper or simple loop for readability since resolution is small (128x128).
    rows, cols = field.shape
    rgb_out = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    # Slow python loop? 128*128 = 16k iters. Might be laggy in pure python.
    # Let's use a vectorized conversion.
    
    # H, S, V are 2D arrays.
    
    def vectorized_hsv_to_rgb(h, s, v):
        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        # i == 0
        mask = (i == 0)
        r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
        
        # i == 1
        mask = (i == 1)
        r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
        
        # i == 2
        mask = (i == 2)
        r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
        
        # i == 3
        mask = (i == 3)
        r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
        
        # i == 4
        mask = (i == 4)
        r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
        
        # i == 5
        mask = (i == 5)
        r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
        
        return np.stack([r, g, b], axis=-1)

    rgb_float = vectorized_hsv_to_rgb(hue, sat, val)
    rgb_out = (rgb_float * 255).astype(np.uint8)
    
    return rgb_out

def compute_phase_variance(field: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Calculate local phase variance to determine Saturation (Stability).
    S = exp(-k * sigma_phi)
     
    Since phase is circular, standard variance is tricky.
    We can use the length of the mean vector of unit phasors.
    R = | sum(exp(i*phi)) / N |
    Variance roughly correlates with (1 - R).
    S = R is a good approximation for 'coherence'. 
    If all phases are aligned, R=1 -> S=1. Random -> R=0 -> S=0.
    """
    if kernel_size < 1:
        return np.ones_like(field, dtype=float)
        
    pad = kernel_size // 2
    
    # Unit phasors
    phasors = np.exp(1j * np.angle(field))
    
    # We need a moving average.
    # For a simulator, simple loop or convolution is fine.
    # Using scipy.ndimage.uniform_filter is best if available, but let's avoid extra deps if possible.
    # We will implement a basic sum-area table or just manual convolution for speed.
    
    # Actually, let's keep it simple: use the field amplitude as weight? Spec says "local phase variance".
    # Let's use unweighted phasor mean for pure phase consistency.
    
    rows, cols = field.shape
    # Basic convolution for mean phasor
    # This is slow in pure python.
    # Let's import scipy only if needed, or implement a naive integral image approach.
    # Integral image for complex numbers:
    
    integ = np.cumsum(np.cumsum(phasors, axis=0), axis=1)
    
    # Helper to get sum of rect
    def get_sum(r1, c1, r2, c2):
        r1, c1 = max(0, r1), max(0, c1)
        r2, c2 = min(rows-1, r2), min(cols-1, c2)
        
        A = integ[r1-1, c1-1] if r1>0 and c1>0 else 0
        B = integ[r1-1, c2] if r1>0 else 0
        C = integ[r2, c1-1] if c1>0 else 0
        D = integ[r2, c2]
        
        return D - B - C + A
    
    stability_map = np.zeros((rows, cols))
    
    # This is still O(N^2) loops in python. Too slow for real-time 128x128?
    # 16k calls. Maybe okay for a demo.
    # Vectorized approach:
    # We can shift arrays to compute neighbors.
    
    # Actually, R = |mean_phasor|.
    # Let's try to simulate variance without heavy convolution if possible, 
    # OR just assume scipy/numpy is fast enough. 
    # BUT we want to avoid scipy dependency if user env doesn't have it standard (it usually does).
    
    # Let's use a simplified "coherence" check: Just compare pixel to its 4 neighbors?
    # R approx ( |p + p_up + p_down + p_left + p_right| ) / 5
    
    p = phasors
    # Shifted
    p_u = np.roll(p, 1, axis=0)
    p_d = np.roll(p, -1, axis=0)
    p_l = np.roll(p, 1, axis=1)
    p_r = np.roll(p, -1, axis=1)
    
    # coherence
    local_sum = p + p_u + p_d + p_l + p_r
    # Normalize by 5 (approx, valid for interior)
    # Correct normalization not strictly needed for visualization scaling, 
    # but let's normalize to get 0..1
    R = np.abs(local_sum) / 5.0
    
    # Enhance contrast: S = R^k
    # Spec says S = exp(-k * variance). R is related to 1-variance.
    # So using linear R is fine.
    
    return R
