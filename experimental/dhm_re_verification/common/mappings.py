"""
Attribute Mappings for Stage 3 (Japanese)

Re-implementation of attribute definitions for Katakana and Kanji.
"""

from typing import Dict, List

# --- Katakana Mapping ---
# 46 Basic Characters
KATAKANA_MAPPING: Dict[str, List[str]] = {}

VOWELS = ["A", "I", "U", "E", "O"]
CONSONANTS = [
    "NONE", "K", "S", "T", "N", "H", "M", "Y", "R", "W"
]

# Simple programmatic generation for standard grid
# Explicit definition for clarity
_katakana_data = [
    ("ア", "NONE", "A"), ("イ", "NONE", "I"), ("ウ", "NONE", "U"), ("エ", "NONE", "E"), ("オ", "NONE", "O"),
    ("カ", "K", "A"), ("キ", "K", "I"), ("ク", "K", "U"), ("ケ", "K", "E"), ("コ", "K", "O"),
    ("サ", "S", "A"), ("シ", "S", "I"), ("ス", "S", "U"), ("セ", "S", "E"), ("ソ", "S", "O"),
    ("タ", "T", "A"), ("チ", "T", "I"), ("ツ", "T", "U"), ("テ", "T", "E"), ("ト", "T", "O"),
    ("ナ", "N", "A"), ("ニ", "N", "I"), ("ヌ", "N", "U"), ("ネ", "N", "E"), ("ノ", "N", "O"),
    ("ハ", "H", "A"), ("ヒ", "H", "I"), ("フ", "H", "U"), ("ヘ", "H", "E"), ("ホ", "H", "O"),
    ("マ", "M", "A"), ("ミ", "M", "I"), ("ム", "M", "U"), ("メ", "M", "E"), ("モ", "M", "O"),
    ("ヤ", "Y", "A"), ("ユ", "Y", "U"), ("ヨ", "Y", "O"),
    ("ラ", "R", "A"), ("リ", "R", "I"), ("ル", "R", "U"), ("レ", "R", "E"), ("ロ", "R", "O"),
    ("ワ", "W", "A"), ("ヲ", "W", "O"), ("ン", "N", "NONE") # Syllabic N
]

for char, cons, vowel in _katakana_data:
    attrs = ["script:KATAKANA", f"cons:{cons}", f"vowel:{vowel}"]
    KATAKANA_MAPPING[char] = attrs

def get_katakana_attributes(char: str) -> List[str]:
    return KATAKANA_MAPPING.get(char, ["script:KATAKANA", "unknown"])


# --- Kanji Mapping ---
# Re-implementing simplified structural mapping
# Level A (Basic), B (Structure), C (High Density)

KANJI_MAPPING: Dict[str, List[str]] = {}

# Level A: Basic Numbers/Nature (Simple Structure)
_level_a = [
    ("一", "SINGLE", ["ONE"]),
    ("二", "SINGLE", ["TWO"]),
    ("三", "SINGLE", ["THREE"]),
    ("人", "SINGLE", ["PERSON"]),
    ("口", "SINGLE", ["MOUTH"]),
    ("日", "SINGLE", ["SUN"]),
    ("月", "SINGLE", ["MOON"]),
    ("木", "SINGLE", ["TREE"]),
    ("山", "SINGLE", ["MOUNTAIN"]),
    ("川", "SINGLE", ["RIVER"])
]

# Level B: Compound Structure (Left-Right, Top-Bottom)
_level_b = [
    ("林", "LEFT_RIGHT", ["TREE", "TREE"]),
    ("森", "TRIANGLE", ["TREE", "TREE", "TREE"]),
    ("明", "LEFT_RIGHT", ["SUN", "MOON"]),
    ("休", "LEFT_RIGHT", ["PERSON", "TREE"]),
    ("好", "LEFT_RIGHT", ["WOMAN", "CHILD"]),
    ("男", "TOP_BOTTOM", ["FIELD", "POWER"]),
    ("信", "LEFT_RIGHT", ["PERSON", "WORD"]),
    ("体", "LEFT_RIGHT", ["PERSON", "ROOT"]), # Moto
    ("先", "TOP_BOTTOM", ["COW", "LEGS"]), # Simplified
    ("見", "TOP_BOTTOM", ["EYE", "LEGS"]),
    ("学", "TOP_BOTTOM", ["SCHOOL", "CHILD"]),
    ("花", "TOP_BOTTOM", ["GRASS", "CHANGE"]),
    ("音", "TOP_BOTTOM", ["STAND", "SUN"]),
    ("書", "TOP_BOTTOM", ["BRUSH", "SUN"]),
    ("話", "LEFT_RIGHT", ["WORD", "TONGUE"])
]

# Level C: High Density / Semantically Close
_level_c = [
    ("議", "LEFT_RIGHT", ["WORD", "RIGHTEOUS"]),
    ("識", "LEFT_RIGHT", ["WORD", "KNOWLEDGE"]), # Same radical as 議
    ("験", "LEFT_RIGHT", ["HORSE", "CHECK"]),
    ("競", "LEFT_RIGHT", ["BROTHER", "BROTHER"]),
    ("驚", "TOP_BOTTOM", ["RESPECT", "HORSE"]),
    ("観", "LEFT_RIGHT", ["SEE", "BIRD"]), # Simplified
    ("類", "LEFT_RIGHT", ["RICE", "HEAD"]),
    ("複", "LEFT_RIGHT", ["CLOTH", "DOUBLE"]),
    ("額", "LEFT_RIGHT", ["GUEST", "HEAD"]), # Same radical 'HEAD' as 類? No, PAGE/HEAD collision in earlier exp
    ("龍", "SINGLE", ["DRAGON"])
]

def _register_kanji(char, structure, components):
    attrs = ["script:KANJI", f"struct:{structure}"]
    for comp in components:
        attrs.append(f"comp:{comp}")
    KANJI_MAPPING[char] = attrs

for c, s, comps in _level_a + _level_b + _level_c:
    _register_kanji(c, s, comps)

def get_kanji_attributes(char: str) -> List[str]:
    return KANJI_MAPPING.get(char, ["script:KANJI", "unknown"])

def get_level_chars(level: str) -> List[str]:
    if level == "A": return [x[0] for x in _level_a]
    if level == "B": return [x[0] for x in _level_b]
    if level == "C": return [x[0] for x in _level_c]
    return []
