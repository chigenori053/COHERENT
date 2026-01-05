"""
Data Generation Utilities

Generates synthetic data for Stages 1 and 2.
"""

import random
import string
from typing import List

class DataGenerator:
    @staticmethod
    def generate_alphabet_sequence(length: int) -> str:
        """Stage 1: A-Z sequence."""
        return "".join(random.choices(string.ascii_uppercase, k=length))

    @staticmethod
    def generate_heterogeneous_sequence(length: int) -> str:
        """Stage 2: Mixed Alphanumeric + Symbols."""
        # A subset of printable characters (Letters, Digits, Punctuation)
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(random.choices(chars, k=length))

    @staticmethod
    def generate_batch_stage1(num_samples: int, min_len: int = 2, max_len: int = 5) -> List[str]:
        data = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            data.append(DataGenerator.generate_alphabet_sequence(length))
        return data

    @staticmethod
    def generate_batch_stage2(num_samples: int, min_len: int = 2, max_len: int = 4) -> List[str]:
        data = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            data.append(DataGenerator.generate_heterogeneous_sequence(length))
        return data
