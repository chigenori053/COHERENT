"""
Kanji Attribute Mapping (Stage 3)

Defines structural and component attributes for 35 Kanji characters.
Categorized by complexity: Level A (Basic), Level B (Structural), Level C (High Density).
"""

from typing import Dict, List

# --- Attribute Categories ---
ATTRIBUTE_CATEGORIES = {
    "SCRIPT": ["kanji"],
    "COMPLEXITY": ["low", "medium", "high"],
    "STRUCTURE": [
        "single", 
        "left_right", 
        "top_bottom", 
        "enclosure", 
        "complex_composite"
    ],
    # Simplified Component List (Radicals/Parts) for Resonance Testing
    "COMPONENT": [
        "none", "tree", "sun", "moon", "person", "mouth", "mountain", "river", 
        "woman", "child", "word", "heart", "body_root", "eye", "flower_grass", 
        "sound_stand", "horse", "shell_money", "dragon", "clothing", "head_page",
        "righteousness", "power", "blue", "guest", "duplicate"
    ]
}

# --- Kanji Dataset Definitions ---

# Level A: Basic (Complexity: Low)
LEVEL_A = [
    # Char, Structure, Main Component
    ("一", "single", "none"),
    ("二", "single", "none"),
    ("三", "single", "none"),
    ("人", "single", "person"),
    ("口", "single", "mouth"), 
    ("日", "single", "sun"),
    ("月", "single", "moon"),
    ("木", "single", "tree"),
    ("山", "single", "mountain"),
    ("川", "single", "river")
]

# Level B: Structural (Complexity: Medium)
LEVEL_B = [
    ("林", "left_right", "tree"),  # Tree + Tree
    ("森", "top_bottom", "tree"),  # Tree + Tree + Tree
    ("明", "left_right", "sun"),   # Sun + Moon (Primary Sun for simplified mapping)
    ("休", "left_right", "person"), # Person + Tree
    ("好", "left_right", "woman"),  # Woman + Child
    ("男", "top_bottom", "power"),  # Field + Power (Power is characteristic)
    ("信", "left_right", "person"), # Person + Word
    ("体", "left_right", "person"), # Person + Root
    ("先", "top_bottom", "person"), # Previous/Ahead (Legs/Person component)
    ("見", "top_bottom", "eye"),    # Eye + Legs
    ("学", "top_bottom", "child"),  # Schoolhouse + Child
    ("花", "top_bottom", "flower_grass"), # Grass + Change
    ("音", "top_bottom", "sound_stand"), # Stand + Sun/Speech? (Sound)
    ("書", "single", "mouth"),     # Brush + Mouth/Sun? (Writer) - Treated as Single block visually
    ("話", "left_right", "word")   # Word + Tongue
]

# Level C: High Density (Complexity: High)
LEVEL_C = [
    ("議", "left_right", "word"),   # Word + Righteousness
    ("識", "left_right", "word"),   # Word + Sound/Kazoe
    ("験", "left_right", "horse"),  # Horse + Awe
    ("競", "left_right", "person"), # Two Elder Brothers (Men) -> Person context? Or Complex? 
                                    # Let's map to "complex_composite" or specific if possible.
                                    # Using "complex_composite" with "person" for now.
    ("驚", "top_bottom", "horse"),  # Horse + Respect
    ("観", "left_right", "eye"),    # Heron (resemblance) + See (Eye)
    ("類", "left_right", "head_page"), # Rice + Dog + Head
    ("複", "left_right", "clothing"), # Clothing + Double
    ("額", "left_right", "head_page"), # Guest + Head
    ("龍", "single", "dragon")      # Dragon (Single block / Pictograph)
]

KANJI_MAPPING: Dict[str, Dict[str, str]] = {}

def _build_mapping():
    def add_entries(level_list, complexity):
        for char, struct, comp in level_list:
            attrs = {
                "SCRIPT": "kanji",
                "COMPLEXITY": complexity,
                "STRUCTURE": struct,
                "COMPONENT": comp
            }
            KANJI_MAPPING[char] = attrs

    add_entries(LEVEL_A, "low")
    add_entries(LEVEL_B, "medium")
    add_entries(LEVEL_C, "high")

_build_mapping()

def get_attributes_for_kanji(char: str) -> List[str]:
    if char not in KANJI_MAPPING:
        # Fallback for unknown kanji? Or strict?
        raise ValueError(f"Unknown Kanji: {char}")
    return list(KANJI_MAPPING[char].values())

def get_level_chars(level_name: str) -> List[str]:
    if level_name == "A":
        return [x[0] for x in LEVEL_A]
    elif level_name == "B":
        return [x[0] for x in LEVEL_B]
    elif level_name == "C":
        return [x[0] for x in LEVEL_C]
    elif level_name == "ALL":
        return [x[0] for x in LEVEL_A + LEVEL_B + LEVEL_C]
    return []
