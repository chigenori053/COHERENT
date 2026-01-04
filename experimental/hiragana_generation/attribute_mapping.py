"""
Hiragana Attribute Mapping

Defines phonological attributes for Japanese Hiragana (Seion).
Spec Section 4 & 5.
"""

from typing import Dict, List, Tuple

# --- 4. Attribute System ---

ATTRIBUTE_CATEGORIES = {
    "SCRIPT": ["hiragana"],
    "VOWEL": ["a", "i", "u", "e", "o"],
    "CONSONANT": ["none", "k", "s", "t", "n", "h", "m", "y", "r", "w"],
    "VOICE": ["voiceless"],
    "SPECIAL_MARK": ["none", "nasal"]
}

# --- 5. Hiragana Mapping (Procedural Generation) ---

# Base rows
VOWELS = ["a", "i", "u", "e", "o"]
CONSONANTS = ["none", "k", "s", "t", "n", "h", "m", "y", "r", "w"]
# Note: 'n' here is for Na-row (na, ni, nu, ne, no)

HIRAGANA_TABLE = [
    # Vowels (none)
    ("none", "a", "あ"), ("none", "i", "い"), ("none", "u", "う"), ("none", "e", "え"), ("none", "o", "お"),
    # K
    ("k", "a", "か"), ("k", "i", "き"), ("k", "u", "く"), ("k", "e", "け"), ("k", "o", "こ"),
    # S
    ("s", "a", "さ"), ("s", "i", "し"), ("s", "u", "す"), ("s", "e", "せ"), ("s", "o", "そ"),
    # T
    ("t", "a", "た"), ("t", "i", "ち"), ("t", "u", "つ"), ("t", "e", "て"), ("t", "o", "と"),
    # N (Na-row)
    ("n", "a", "な"), ("n", "i", "に"), ("n", "u", "ぬ"), ("n", "e", "ね"), ("n", "o", "の"),
    # H
    ("h", "a", "は"), ("h", "i", "ひ"), ("h", "u", "ふ"), ("h", "e", "へ"), ("h", "o", "ほ"),
    # M
    ("m", "a", "ま"), ("m", "i", "み"), ("m", "u", "む"), ("m", "e", "め"), ("m", "o", "も"),
    # Y (ya, yu, yo) - yi, ye skip
    ("y", "a", "や"), ("y", "u", "ゆ"), ("y", "o", "よ"),
    # R
    ("r", "a", "ら"), ("r", "i", "り"), ("r", "u", "る"), ("r", "e", "れ"), ("r", "o", "ろ"),
    # W (wa, wo) - wi, wu, we skip
    ("w", "a", "わ"), ("w", "o", "を"),
    # N (Syllabic Nasal)
    ("special", "n", "ん") 
]

HIRAGANA_MAPPING: Dict[str, Dict[str, str]] = {}

def _build_mapping():
    for cons, vowel, char in HIRAGANA_TABLE:
        attrs = {
            "SCRIPT": "hiragana",
            "VOICE": "voiceless", # Default for Seion
            "SPECIAL_MARK": "none"
        }
        
        if char == "ん":
            # Special handling for 'n'
            attrs["VOWEL"] = "u" # Placeholder to satisfy category existense? Or create 'none'?
            # Spec says "Exactly one attribute from each category"
            # If we reuse 'u', we must distinguish by SPECIAL_MARK
            attrs["CONSONANT"] = "none"
            attrs["SPECIAL_MARK"] = "nasal"
        else:
            attrs["CONSONANT"] = cons
            attrs["VOWEL"] = vowel
            
        HIRAGANA_MAPPING[char] = attrs

_build_mapping()

def get_attributes_for_hiragana(char: str) -> List[str]:
    if char not in HIRAGANA_MAPPING:
        raise ValueError(f"Unknown Hiragana: {char}")
    return list(HIRAGANA_MAPPING[char].values())

def get_all_chars() -> List[str]:
    return [entry[2] for entry in HIRAGANA_TABLE]
