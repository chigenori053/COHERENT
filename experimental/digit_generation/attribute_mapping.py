"""
Digit Attribute Mapping

Defines the attribute system and mapping for the Digit Generation Experiment.
Spec Sections 4 & 5.
"""

from typing import Dict, List

# --- 4. Attribute System ---

ATTRIBUTE_CATEGORIES = {
    "TYPE": ["digit"],
    "PARITY": ["even", "odd"],
    "MAGNITUDE": ["low", "mid", "high"],
    "PRIME_STATUS": ["prime", "composite", "neither"],
    "LOOP_COUNT": ["0loop", "1loop", "2loop"],
    "STROKE_CLASS": ["straight", "curved", "mixed"]
}

# --- 5. Digit -> Attribute Mapping ---

DIGIT_MAPPING: Dict[str, Dict[str, str]] = {
    "0": {"TYPE": "digit", "PARITY": "even", "MAGNITUDE": "low",  "PRIME_STATUS": "neither",   "LOOP_COUNT": "1loop", "STROKE_CLASS": "curved"},
    "1": {"TYPE": "digit", "PARITY": "odd",  "MAGNITUDE": "low",  "PRIME_STATUS": "neither",   "LOOP_COUNT": "0loop", "STROKE_CLASS": "straight"},
    "2": {"TYPE": "digit", "PARITY": "even", "MAGNITUDE": "low",  "PRIME_STATUS": "prime",     "LOOP_COUNT": "0loop", "STROKE_CLASS": "mixed"},
    "3": {"TYPE": "digit", "PARITY": "odd",  "MAGNITUDE": "low",  "PRIME_STATUS": "prime",     "LOOP_COUNT": "0loop", "STROKE_CLASS": "mixed"},
    "4": {"TYPE": "digit", "PARITY": "even", "MAGNITUDE": "mid",  "PRIME_STATUS": "composite", "LOOP_COUNT": "0loop", "STROKE_CLASS": "straight"},
    "5": {"TYPE": "digit", "PARITY": "odd",  "MAGNITUDE": "mid",  "PRIME_STATUS": "prime",     "LOOP_COUNT": "0loop", "STROKE_CLASS": "mixed"},
    "6": {"TYPE": "digit", "PARITY": "even", "MAGNITUDE": "mid",  "PRIME_STATUS": "composite", "LOOP_COUNT": "1loop", "STROKE_CLASS": "mixed"},
    "7": {"TYPE": "digit", "PARITY": "odd",  "MAGNITUDE": "high", "PRIME_STATUS": "prime",     "LOOP_COUNT": "0loop", "STROKE_CLASS": "straight"},
    "8": {"TYPE": "digit", "PARITY": "even", "MAGNITUDE": "high", "PRIME_STATUS": "composite", "LOOP_COUNT": "2loop", "STROKE_CLASS": "mixed"},
    "9": {"TYPE": "digit", "PARITY": "odd",  "MAGNITUDE": "high", "PRIME_STATUS": "composite", "LOOP_COUNT": "1loop", "STROKE_CLASS": "mixed"},
}

def get_attributes_for_digit(digit: str) -> List[str]:
    """Retrieve attribute list for a given digit."""
    if digit not in DIGIT_MAPPING:
        raise ValueError(f"Digit {digit} not found in mapping.")
    
    attr_dict = DIGIT_MAPPING[digit]
    # To ensure uniqueness across experiments (optional but good practice),
    # we could prefix values, but the Spec implies direct use of labels.
    # We will use the raw values as per Spec.
    return list(attr_dict.values())
