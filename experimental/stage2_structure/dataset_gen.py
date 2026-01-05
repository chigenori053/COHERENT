"""
Stage 2 Dataset & Candidate Generator
"""

import random
import string
from typing import List, Tuple, Dict

LETTERS = list(string.ascii_uppercase)
DIGITS = list(string.digits)
SYMBOLS = ['+', '-', '*', '/', '=', '#']

ALL_SYMBOLS = LETTERS + DIGITS + SYMBOLS
SYMBOL_TYPE_MAP = {}
for c in LETTERS: SYMBOL_TYPE_MAP[c] = 'LETTER'
for c in DIGITS: SYMBOL_TYPE_MAP[c] = 'DIGIT'
for c in SYMBOLS: SYMBOL_TYPE_MAP[c] = 'SYMBOL'

class Stage2DatasetGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _get_random(self, category):
        if category == 'LETTER': return self.rng.choice(LETTERS)
        if category == 'DIGIT': return self.rng.choice(DIGITS)
        if category == 'SYMBOL': return self.rng.choice(SYMBOLS)
        return self.rng.choice(ALL_SYMBOLS)

    def generate_target(self, length: int) -> List[str]:
        # Spec 3.2 patterns implies mixed types
        # We'll just generate random mixed sequences based on length
        # to ensure good coverage of heterogeneity.
        # Length 2: [L, D], [D, L], [L, S], [S, L]
        # Length 3: [L, D, L], [L, S, D], [D, S, L]
        
        # Simplified: Randomly pick type for each position, ensuring distinct types if L=2
        # Actually Spec 3.2 lists ALLOWED patterns.
        # We should stick to those or generalize.
        # Let's generalize: Any combination of heterogeneous types.
        
        seq = []
        types = ['LETTER', 'DIGIT', 'SYMBOL']
        last_type = None
        
        for _ in range(length):
            # Try to rotate types to be heterogeneous
            available = [t for t in types if t != last_type]
            if not available: available = types
            
            t = self.rng.choice(available)
            seq.append(self._get_random(t))
            last_type = t
            
        return seq

    def generate_candidate_set(self, target: List[str], size: int = 50) -> List[List[str]]:
        """
        Generates candidate pool C:
          - exact_target
          - structural_variants (order changed)
          - category_variants (symbol replaced same category)
          - random_distractors
        """
        candidates = []
        candidates.append(target) # Correct
        
        # Structural Variants (Permutations)
        # Swap 2 elements
        if len(target) >= 2:
            # Generate all swaps? Or random?
            # For small L (2-4), all permutations is small (2, 6, 24).
            # Let's add all unique permutations if possible.
            import itertools
            perms = list(itertools.permutations(target))
            # Limit to a few if too many (L=4 -> 24 is fine)
            for p in perms:
                p_list = list(p)
                if p_list != target and p_list not in candidates:
                    candidates.append(p_list)
        
        # Category Variants (Symbol replacement)
        # Same structure (e.g. Letter-Digit) but different values
        for _ in range(10):
            variant = []
            for char in target:
                ctype = SYMBOL_TYPE_MAP[char]
                # Pick different char of same type
                others = [c for c in (LETTERS if ctype=='LETTER' else (DIGITS if ctype=='DIGIT' else SYMBOLS)) if c != char]
                if others:
                    variant.append(self.rng.choice(others))
                else:
                    variant.append(char)
            if variant not in candidates:
                candidates.append(variant)

        # Random Distractors
        while len(candidates) < size:
            # Random sequence of same length
            dist = self.generate_target(len(target))
            if dist not in candidates:
                candidates.append(dist)
                
        return candidates
