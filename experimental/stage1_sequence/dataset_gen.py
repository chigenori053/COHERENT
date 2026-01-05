"""
Stage 1 Dataset Generator

Generates sequence datasets for Stage 1 Experiment.
"""

import random
import string
from typing import List, Tuple, Dict

CHAR_SET = string.ascii_uppercase

class DatasetGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_random_sequence(self, length: int) -> str:
        return "".join(self.rng.choice(CHAR_SET) for _ in range(length))

    def generate_dataset(self, length: int, count: int) -> List[Tuple[str, str]]:
        """
        Generates a dataset of (id, sequence) pairs.
        Distribution:
         - 50% Random
         - 30% Hamming Dist = 1 (derived from random set)
         - 10% Swap (derived)
         - 10% Edit Dist = 1 (Insert/Delete) -> Note: Changes length.
           For fixed length experiment, we will substitute this with 
           another Hamming=1 or Double Swap to maintain length, 
           OR we allow length mismatch IF the spec allows.
           Spec says "Output: string, length L". 
           If target is length L-1, output L will be wrong.
           WE WILL ASSUME FIXED LENGTH L for all items in "Per L" dataset.
           So 'Edit=1' types will be approximated by specific Substitutions or Swaps 
           that mimic close similarity, or we ignore Insert/Delete to strictly satisfy L.
           Let's distribute the last 10% to Hamming/Swap.
        
        Adjusted Distribution for Fixed L:
         - 50% Random
         - 40% Hamming = 1
         - 10% Swap
        """
        data = []
        
        # 1. Base Random Set (50%)
        base_count = int(count * 0.5)
        base_sequences = []
        for _ in range(base_count):
            seq = self.generate_random_sequence(length)
            base_sequences.append(seq)
            data.append(seq)
            
        # 2. Variants (50%)
        # We derive variants from base sequences to ensure interference
        
        # Hamming = 1 (40%)
        hamming_count = int(count * 0.4)
        for _ in range(hamming_count):
            origin = self.rng.choice(base_sequences)
            # Flip one char
            pos = self.rng.randint(0, length - 1)
            char = origin[pos]
            new_char = self.rng.choice([c for c in CHAR_SET if c != char])
            variant = origin[:pos] + new_char + origin[pos+1:]
            data.append(variant)
            
        # Swap (10%)
        swap_count = count - len(data) # Fill remainder
        for _ in range(swap_count):
            origin = self.rng.choice(base_sequences)
            if length < 2:
                # Should not happen for L>=2
                variant = origin
            else:
                pos = self.rng.randint(0, length - 2)
                # Swap pos and pos+1
                variant = list(origin)
                variant[pos], variant[pos+1] = variant[pos+1], variant[pos]
                variant = "".join(variant)
            data.append(variant)
            
        # Assign IDs
        # ID format: SEQ_{L}_{index}
        labeled_data = []
        for i, seq in enumerate(data):
            seq_id = f"SEQ_{length}_{i:04d}"
            labeled_data.append((seq_id, seq))
            
        return labeled_data

if __name__ == "__main__":
    # Test
    gen = DatasetGenerator()
    data = gen.generate_dataset(3, 10)
    for d in data:
        print(d)
