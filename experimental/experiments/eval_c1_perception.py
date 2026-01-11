"""
C1: Perception / Representation Capability Evaluation
Objective: Verify that COHERENT transforms visual inputs into consistent, distinct, and robust scalar representations.
"""

import os
import sys
import numpy as np
import torch
import csv
import json
import time
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Adjust path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from coherent.core.cortex.representation.vision_encoder import HolographicVisionEncoder
    from coherent.core.cortex.memory.dynamic import DynamicHolographicMemory
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Configuration
REPORT_DIR = os.path.join(os.path.dirname(__file__), '../reports')
CSV_RESULTS_PATH = os.path.join(REPORT_DIR, 'c1_perception_results.csv')
REPORT_MD_PATH = os.path.join(REPORT_DIR, 'C1_PERCEPTION_REPORT.md')

@dataclass
class VisualSample:
    char: str
    label_class: str
    noise_level: float
    rotation: float
    image: Image.Image

class SyntheticVisualData:
    """Generates synthetic character images with controlled noise and rotation."""
    
    def __init__(self, size=(64, 64)):
        self.size = size
        self.font = self._load_font()

    def _load_font(self):
        # Try finding a standard font, fallback to default
        try:
            # Mac path
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except:
            return ImageFont.load_default()

    def generate(self, char: str, noise_level: float = 0.0, rotation: float = 0.0) -> VisualSample:
        # 1. Draw Char
        img = Image.new('L', self.size, color=0) # Black bg
        draw = ImageDraw.Draw(img)
        
        # Center text (approx)
        w, h = self.size
        # Get bounding box 
        bbox = draw.textbbox((0, 0), char, font=self.font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw.text(((w - text_w)/2, (h - text_h)/2), char, fill=255, font=self.font)
        
        # 2. Rotation
        if rotation != 0:
            img = img.rotate(rotation, resample=Image.BICUBIC)
            
        # 3. Noise (Salt and Pepper)
        if noise_level > 0:
            arr = np.array(img)
            mask = np.random.rand(*arr.shape) < noise_level
            # Random noise: sometimes 0, sometimes 255
            noise = np.random.randint(0, 256, arr.shape)
            # Apply only where mask is true
            arr = np.where(mask, noise, arr)
            img = Image.fromarray(arr.astype(np.uint8))

        return VisualSample(
            char=char,
            label_class=char,
            noise_level=noise_level,
            rotation=rotation,
            image=img
        )

class PerceptionExperiment:
    def __init__(self):
        self.encoder = HolographicVisionEncoder()
        # Storage for Canonical Representations: { 'A': HolographicTensor, ... }
        self.canonical_memory = {}
        os.makedirs(REPORT_DIR, exist_ok=True)

    def register_canonical(self, chars: List[str]):
        """Phase 1: Register clean, canonical representations."""
        print(print(f"--- Phase 1: Registration ---"))
        gen = SyntheticVisualData()
        for char in chars:
            sample = gen.generate(char, noise_level=0, rotation=0)
            tensor = self.encoder.encode(sample.image)
            self.canonical_memory[char] = tensor # Store the tensor directly
            print(f"Registered Canonical: {char}")

    def _cosine_sim_complex(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        # Sim(u, v) = |u . v*| / (|u||v|)
        dot = torch.vdot(v1, v2)
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return (torch.abs(dot) / (norm1 * norm2)).item()

    def run_consistency_test(self):
        """C1-1: Consistency (Small variations should have high resonance)"""
        print(f"\n--- Phase 2: C1-1 Consistency Test ---")
        gen = SyntheticVisualData()
        targets = ['A', 'B', '5']
        results = []
        
        for char in targets:
            # Low noise, small rotation variants
            variants = [
                {'noise': 0.05, 'rot': 0},
                {'noise': 0.0, 'rot': 10},
                {'noise': 0.1, 'rot': -5}
            ]
            
            for v in variants:
                sample = gen.generate(char, noise_level=v['noise'], rotation=v['rot'])
                vec = self.encoder.encode(sample.image)
                
                # Check resonance against self-canonical
                score = self._cosine_sim_complex(vec, self.canonical_memory[char])
                
                results.append({
                    'test': 'C1-1:Consistency',
                    'input': char,
                    'variant': f"Noise:{v['noise']:.2f},Rot:{v['rot']}",
                    'target': char,
                    'resonance': score,
                    'judgment': 'PASS' if score > 0.7 else 'FAIL' # Heuristic
                })
        return results

    def run_discrimination_test(self):
        """C1-2: Discrimination (Similar looking chars should be distinct)"""
        print(f"\n--- Phase 3: C1-2 Discrimination Test ---")
        gen = SyntheticVisualData()
        # Pairs: (Input, CanonicalTarget)
        pairs = [('0', 'O'), ('l', '1'), ('Q', 'O')]
        results = []
        
        for c1, c2 in pairs:
            # Ensure both registered
            if c1 not in self.canonical_memory: self.register_canonical([c1])
            if c2 not in self.canonical_memory: self.register_canonical([c2])
            
            # Generate input c1
            sample = gen.generate(c1)
            vec_c1 = self.encoder.encode(sample.image)
            
            # Compare with Canonical c2
            score = self._cosine_sim_complex(vec_c1, self.canonical_memory[c2])
            
            results.append({
                'test': 'C1-2:Discrimination',
                'input': c1,
                'variant': 'Clean',
                'target': c2,
                'resonance': score,
                'judgment': 'PASS' if score < 0.6 else 'FAIL' # Should distinguish
            })
        return results

    def run_robustness_test(self):
        """C1-3: Probabilistic Robustness (Resonance decay with noise)"""
        print(f"\n--- Phase 4: C1-3 Robustness Test ---")
        gen = SyntheticVisualData()
        char = 'X'
        if char not in self.canonical_memory: self.register_canonical([char])
        
        results = []
        noises = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # 0% to 100% noise
        
        for n in noises:
            sample = gen.generate(char, noise_level=n)
            vec = self.encoder.encode(sample.image)
            score = self._cosine_sim_complex(vec, self.canonical_memory[char])
            
            results.append({
                'test': 'C1-3:Robustness',
                'input': char,
                'variant': f"Noise:{n:.2f}",
                'target': char,
                'resonance': score,
                'judgment': 'INFO'
            })
        return results

    def run(self):
        # Setup
        chars = ['A', 'B', '5', '0', 'O', '1', 'l', 'Q', 'X']
        self.register_canonical(chars)
        
        all_results = []
        all_results.extend(self.run_consistency_test())
        all_results.extend(self.run_discrimination_test())
        all_results.extend(self.run_robustness_test())
        
        # Save CSV
        fields = ['test', 'input', 'variant', 'target', 'resonance', 'judgment']
        with open(CSV_RESULTS_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
            
        print(f"Results saved to {CSV_RESULTS_PATH}")
        self.generate_report(all_results)

    def generate_report(self, results):
        lines = ["# C1 Perception Verification Report", ""]
        lines.append(f"Date: {time.strftime('%Y-%m-%d')}")
        lines.append("## Summary")
        lines.append("Evaluated Logic: Visual input -> Holographic Vector -> Resonance Check.")
        lines.append("")
        
        lines.append("## Detailed Results")
        lines.append("| Test | Input | Variant | Target | Resonance | Judgment |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            lines.append(f"| {r['test']} | {r['input']} | {r['variant']} | {r['target']} | {r['resonance']:.4f} | {r['judgment']} |")
            
        with open(REPORT_MD_PATH, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Report saved to {REPORT_MD_PATH}")

if __name__ == "__main__":
    exp = PerceptionExperiment()
    exp.run()
