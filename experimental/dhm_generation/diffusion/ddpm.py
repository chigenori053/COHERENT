"""
DDPM (Denoising Diffusion Probabilistic Models) Implementation

Implements stochastic reverse diffusion.
Extends the DDIM logic by adding a noise term in the reverse step.
"""

import numpy as np
from .ddim import RefinementDiffusionDDIM

class RefinementDiffusionDDPM(RefinementDiffusionDDIM):
    def step(self, h_t: np.ndarray, t: int, h_0_clean: np.ndarray) -> np.ndarray:
        """
        DDPM Reverse Step: H_{t-1} = DDIM_step + sigma_t * noise
        
        Note: The strict DDPM formulation usually defines the mean differently than DDIM,
        but the Spec 8.4 says: "H_{t-1} = DDIM_step + sigma_t eta'_t".
        This implies a generalized diffusion where we take the deterministic trajectory
        and inject noise.
        """
        # 1. Calculate deterministic component (DDIM)
        # We assume the parent class step returns the normalized deterministic step.
        h_prev_det = super().step(h_t, t, h_0_clean)
        
        # 2. Add noise
        # sigma_t depends on the schedule. 
        # Standard DDPM posterier variance beta_tilde_t = (1-ab_{t-1})/(1-ab_t) * beta_t
        # Simplified: sigma_t = sqrt(1 - alpha_t) ? 
        # For this experiment, we might use a fixed parameter logic or derive from schedule.
        # Let's derive sigma_t based on standard eta parameter in generalized DDIM/DDPM.
        # If eta=1 (DDPM), sigma_t = sqrt( (1-ab_{t-1})/(1-ab_t) * (1 - ab_t/ab_{t-1}) )
        
        alpha_bar_t = self._get_alpha_bar(t)
        alpha_bar_prev = self._get_alpha_bar(t - 1)
        
        # Beta_t = 1 - alpha_t_step = 1 - (ab_t / ab_{t-1})
        # Let's approximate sigma roughly or use Spec 8.2 definition if present.
        # Spec 8.2 doesn't define sigma size.
        # Let's use eta=1.0 standard derivation.
        
        if t == 0:
            return h_prev_det
            
        # Variance
        # sigma = eta * sqrt( (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev) )
        # This is standard.
        sigma = 1.0 * np.sqrt( (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev) )
        
        # Generate Complex Noise
        real = np.random.standard_normal(h_t.shape)
        imag = np.random.standard_normal(h_t.shape)
        noise = (real + 1j * imag) / np.sqrt(2)
        
        h_prev = h_prev_det + sigma * noise
        
        # Enforce Normalization (Spec 9)
        norm = np.linalg.norm(h_prev)
        return h_prev / norm
