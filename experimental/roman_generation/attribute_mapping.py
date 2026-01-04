"""
Roman Numeral Attribute Mapping

Logic to convert Integers (1-99) to Roman Numerals and Abstract Attributes.
Spec Section 4.
"""

from typing import List, Dict, Tuple

# --- Constants ---

ATTRIBUTE_CATEGORIES = {
    "PLACE_EXISTENCE": ["HAS_ONES", "HAS_TENS"],
    "ONES_MAGNITUDE": ["ONES_MAG_LOW", "ONES_MAG_SUB", "ONES_MAG_MID"],
    "TENS_MAGNITUDE": ["TENS_MAG_LOW", "TENS_MAG_SUB", "TENS_MAG_MID"],
    "ONES_SPECIFIC": [f"ONES_VAL_{i}" for i in range(1, 10)],
    "TENS_SPECIFIC": [f"TENS_VAL_{i}" for i in range(1, 10)],
    "STRUCTURE": ["ORDER_TENS_FIRST", "USE_SUBTRACTIVE", "REPEAT_ALLOWED"]
}

# --- Conversion Logic ---

def int_to_roman(n: int) -> str:
    """Canonical conversion for 1-99."""
    if not 1 <= n <= 99:
        raise ValueError("Only supports 1-99")
    
    val_map = [
        (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'),
        (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    result = ""
    target = n
    for val, sym in val_map:
        while target >= val:
            result += sym
            target -= val
    return result

def get_attributes_for_number(n: int) -> List[str]:
    """Derive structural attributes from the number (1-99)."""
    attrs = []
    
    ones = n % 10
    tens = n // 10
    
    # 1. Value Decomposition - Specific Values for Uniqueness
    if ones > 0:
        attrs.append("HAS_ONES")
        attrs.append(f"ONES_VAL_{ones}") # Crucial for uniqueness (e.g. 2 vs 3)
        if ones in [1, 2, 3]:
            attrs.append("ONES_MAG_LOW")
        elif ones in [4, 9]:
            attrs.append("ONES_MAG_SUB")
        elif ones in [5, 6, 7, 8]:
            attrs.append("ONES_MAG_MID")
            
    if tens > 0:
        attrs.append("HAS_TENS")
        attrs.append(f"TENS_VAL_{tens}") # Crucial for uniqueness (e.g. 10 vs 20, 12 vs 21)
        if tens in [1, 2, 3]:
            attrs.append("TENS_MAG_LOW")
        elif tens in [4, 9]:
            attrs.append("TENS_MAG_SUB")
        elif tens in [5, 6, 7, 8]:
            attrs.append("TENS_MAG_MID")
            
    # 2. Structural Attributes
    
    # ORDER_TENS_FIRST: Implicitly true if both exist
    if tens > 0 and ones > 0:
        attrs.append("ORDER_TENS_FIRST")
        
    # USE_SUBTRACTIVE: 4, 9, 40, 90
    if ones in [4, 9] or tens in [4, 9]:
        attrs.append("USE_SUBTRACTIVE")
        
    # REPEAT_ALLOWED: If any symbol repeats consecutively (II, III, XX, XXX, etc.)
    # In canonical form, this happens if digit % 5 in [2, 3] (e.g. 2,3, 7,8)
    # 1(I) is one symbol, not repeating. 2(II) repeats.
    # 6(VI) no repeat. 7(VII) repeats.
    # Same for tens: 20(XX), 30(XXX), 70(LXX), 80(LXXX).
    # NOTE: 1(I), 10(X) don't repeat.
    
    has_repeat = False
    if ones in [2, 3, 7, 8]:
        has_repeat = True
    if tens in [2, 3, 7, 8]:
        has_repeat = True
        
    if has_repeat:
        attrs.append("REPEAT_ALLOWED")
        
    # Ensure every number has at least some attributes.
    # The list will be non-empty for 1-99.
    
    return attrs

def generate_candidate_map() -> Dict[str, List[str]]:
    """Generates mapping {RomanStr: Attributes} for all 1-99."""
    mapping = {}
    for i in range(1, 100):
        roman = int_to_roman(i)
        attrs = get_attributes_for_number(i)
        mapping[roman] = attrs
    return mapping
