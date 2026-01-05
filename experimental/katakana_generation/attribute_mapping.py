"""
Katakana Attribute Mapping

Defines phonological attributes for Japanese Katakana (Seion).
Validates single-character generation capability for Japanese Katakana within the COHERENT / MemorySpace architecture.
"""

from typing import Dict, List, Tuple

# --- 4. Attribute System ---

ATTRIBUTE_CATEGORIES = {
    "SCRIPT": ["katakana"],
    "VOWEL": ["a", "i", "u", "e", "o"],
    "CONSONANT": ["none", "k", "s", "t", "n", "h", "m", "y", "r", "w"],
    "VOICE": ["voiceless"],
    "SPECIAL_MARK": ["none", "nasal"]
}

# --- 5. Katakana Mapping (Procedural Generation) ---

# Base rows
VOWELS = ["a", "i", "u", "e", "o"]
CONSONANTS = ["none", "k", "s", "t", "n", "h", "m", "y", "r", "w"]
# Note: 'n' here is for Na-row (na, ni, nu, ne, no)

KATAKANA_TABLE = [
    # Vowels (none)
    ("none", "a", "ア"), ("none", "i", "イ"), ("none", "u", "ウ"), ("none", "e", "エ"), ("none", "o", "オ"),
    # K
    ("k", "a", "カ"), ("k", "i", "キ"), ("k", "u", "ク"), ("k", "e", "ケ"), ("k", "o", "コ"),
    # S
    ("s", "a", "サ"), ("s", "i", "シ"), ("s", "u", "ス"), ("s", "e", "セ"), ("s", "o", "ソ"),
    # T
    ("t", "a", "タ"), ("t", "i", "チ"), ("t", "u", "ツ"), ("t", "e", "テ"), ("t", "o", "ト"),
    # N (Na-row)
    ("n", "a", "ナ"), ("n", "i", "ニ"), ("n", "u", "ヌ"), ("n", "e", "ネ"), ("n", "o", "ノ"),
    # H
    ("h", "a", "ハ"), ("h", "i", "ヒ"), ("h", "u", "フ"), ("h", "e", "ヘ"), ("h", "o", "ホ"),
    # M
    ("m", "a", "マ"), ("m", "i", "ミ"), ("m", "u", "ム"), ("m", "e", "メ"), ("m", "o", "モ"),
    # Y (ya, yu, yo) - yi, ye skip
    ("y", "a", "ヤ"), ("y", "u", "ユ"), ("y", "o", "ヨ"),
    # R
    ("r", "a", "ラ"), ("r", "i", "リ"), ("r", "u", "ル"), ("r", "e", "レ"), ("r", "o", "ロ"),
    # W (wa, wo) - wi, wu, we skip
    ("w", "a", "ワ"), ("w", "o", "ヲ"),
    # N (Syllabic Nasal)
    ("special", "n", "ン") 
]

KATAKANA_MAPPING: Dict[str, Dict[str, str]] = {}

def _build_mapping():
    for cons, vowel, char in KATAKANA_TABLE:
        attrs = {
            "SCRIPT": "katakana",
            "VOICE": "voiceless", # Default for Seion
            "SPECIAL_MARK": "none"
        }
        
        if char == "ン":
            # Special handling for 'n' (Syllabic Nasal)
            attrs["VOWEL"] = "u" # Placeholder to satisfy category existence, similar to Hiragana strategy
            attrs["CONSONANT"] = "none"
            attrs["SPECIAL_MARK"] = "nasal"
        else:
            attrs["CONSONANT"] = cons
            attrs["VOWEL"] = vowel
            
        KATAKANA_MAPPING[char] = attrs

_build_mapping()

def get_attributes_for_katakana(char: str) -> List[str]:
    if char not in KATAKANA_MAPPING:
        raise ValueError(f"Unknown Katakana: {char}")
    return list(KATAKANA_MAPPING[char].values())

def get_all_chars() -> List[str]:
    return [entry[2] for entry in KATAKANA_TABLE]
