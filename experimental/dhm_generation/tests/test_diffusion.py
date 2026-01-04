"""
Tests for Diffusion Processes

Verifies:
- DDIM deterministic steps
- DDPM stochastic behavior
- Normalization preservation
"""

import numpy as np
import unittest
from experimental.dhm_generation.config import ExperimentConfig
from experimental.dhm_generation.diffusion.ddim import RefinementDiffusionDDIM
from experimental.dhm_generation.diffusion.ddpm import RefinementDiffusionDDPM

class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(
            dimension=512, # Smallest allowed dimension
            num_timesteps=10,
            alpha_start=0.9,
            alpha_end=0.1
        )
        self.ddim = RefinementDiffusionDDIM(self.config)
        self.ddpm = RefinementDiffusionDDPM(self.config)
        
        # Create a dummy clean state
        self.h_0 = np.random.random(512) + 1j * np.random.random(512)
        self.h_0 /= np.linalg.norm(self.h_0)

    def test_ddim_step_normalization(self):
        """Test that DDIM step preserves normalization."""
        # Create a noisy state h_t
        h_t = self.ddim.add_noise(self.h_0, t=5)
        self.assertAlmostEqual(np.linalg.norm(h_t), 1.0, places=5)
        
        # Reverse step
        h_prev = self.ddim.step(h_t, t=5, h_0_clean=self.h_0)
        self.assertAlmostEqual(np.linalg.norm(h_prev), 1.0, places=5)

    def test_ddim_reversibility_limit(self):
        """
        With pure oracle (knowing h_0), one step from small noise should move towards h_0.
        """
        # Small noise step (t=1, close to data)
        h_t = self.ddim.add_noise(self.h_0, t=1)
        h_prev = self.ddim.step(h_t, t=1, h_0_clean=self.h_0)
        
        # Check resonance improvement or similarity
        sim_t = np.abs(np.vdot(h_t, self.h_0))
        sim_prev = np.abs(np.vdot(h_prev, self.h_0))
        
        # Refinement should ideally increase or maintain high similarity to H_0
        # (Since we use H_0 as guide, we are effectively pulling it back)
        self.assertGreaterEqual(sim_prev, sim_t - 1e-4)

    def test_ddpm_stochasticity(self):
        """Test that DDPM adds noise in reverse step (unlike DDIM)."""
        h_t = self.ddpm.add_noise(self.h_0, t=5)
        
        # Run twice with different global seeds implicitly (numpy global)
        # Note: Spec says 'Seed controlled', so we rely on numpy state
        
        # 1. First run
        np.random.seed(111)
        res1 = self.ddpm.step(h_t, t=5, h_0_clean=self.h_0)
        
        # 2. Second run
        np.random.seed(222)
        res2 = self.ddpm.step(h_t, t=5, h_0_clean=self.h_0)
        
        # Should differ due to noise injection
        diffs = np.linalg.norm(res1 - res2)
        self.assertGreater(diffs, 1e-6, "DDPM should be stochastic")

    def test_ddim_determinism(self):
        """Test that DDIM is deterministic given same inputs."""
        h_t = self.ddim.add_noise(self.h_0, t=5)
        
        np.random.seed(111)
        res1 = self.ddim.step(h_t, t=5, h_0_clean=self.h_0)
        
        np.random.seed(222) # Seed shouldn't matter for DDIM step
        res2 = self.ddim.step(h_t, t=5, h_0_clean=self.h_0)
        
        np.testing.assert_array_almost_equal(res1, res2)
