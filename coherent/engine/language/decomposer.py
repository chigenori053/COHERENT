import re
from typing import List

class Decomposer:
    """
    Decomposes complex natural language/math inputs into executable segments.
    Handles explicit delimiters (newlines, semicolons) and implicit boundaries
    (e.g., end of one equation and start of another variable assignment).
    """
    def __init__(self):
        self._hard_delimiters = r"[;\n]"
        # Soft delimiters that indicate a condition or new clause
        self._soft_delimiters = r"(の場合|if |when |suppose )"
        
        # Implicit Boundary Regex
        # Pattern: [End of Math] [Space] [Start of Assignment]
        # End of Math: Digit or Variable or ')'
        # Start of Assignment: Variable followed by '='
        # Lookbehind matches encoded separately or handled via finding start indices.
        
        # Strategy:
        # 1. Identify " Variable =" pattern.
        # 2. Check if it is preceded by something that looks like the end of another expression.
        
        # Regex explanation:
        # (?<=[0-9a-zA-Z)])\s+(?=[a-zA-Z]\s*=)
        # Lookbehind: previous char is alphanumeric or closing paren
        # Match: whitespace
        # Lookahead: next is Letter followed by optional space and =
        self._implicit_boundary = r"(?<=[0-9a-zA-Z)])\s+(?=[a-zA-Z]\s*=)"

    def decompose(self, text: str) -> List[str]:
        # 1. Pre-normalization
        # Replace soft delimiters with a standard delimiter (e.g. semicolon)
        # "x=1の場合" -> "x=1;" 
        # But wait, "の場合" implies condition. For simplistic execution sequence:
        # "x=1", "y=..." works if x=1 sets state.
        
        normalized = text
        normalized = re.sub(self._soft_delimiters, "; ", normalized)
        
        # 2. Hard Split (Semicolons and Newlines)
        # We first split by hard delimiters to get rough chunks
        files_segments = re.split(self._hard_delimiters, normalized)
        
        final_segments = []
        
        for segment in files_segments:
            seg = segment.strip()
            if not seg:
                continue
                
            # 3. Implicit Split within chunks
            # Apply the implicit boundary regex to split "y=3x+2 x=1"
            # re.split with lookbehind/ahead might keep the delimiter (space) empty or consume it.
            # We want to consume the space and split.
            
            sub_segments = re.split(self._implicit_boundary, seg)
            
            for sub in sub_segments:
                s = sub.strip()
                if s:
                    final_segments.append(s)
                    
        return final_segments
