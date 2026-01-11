"""
Language Specifications for Verification Sandbox
Defines the mapping rules and constraints for target languages.
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class LanguageSpec:
    id: str
    paradigm: str # 'procedural' or 'functional'
    control_flow: Dict[str, str] # e.g. {'loop': 'while'}
    syntax: Dict[str, str] # e.g. {'block_start': '{'}
    keywords: List[str] # Essential keywords to validate structure

# Python (Procedural)
PYTHON_SPEC = LanguageSpec(
    id="procedural_python",
    paradigm="procedural",
    control_flow={
        "loop": "while",
        "condition": "if",
        "exit": "break"
    },
    syntax={
        "block_start": ":",
        "block_end": "", # Indentation handled logic logic
        "statement_end": "",
        "indent": "    "
    },
    keywords=["while", "if", "break", ":"]
)

# Java (Procedural)
JAVA_SPEC = LanguageSpec(
    id="procedural_java",
    paradigm="procedural",
    control_flow={
        "loop": "while",
        "condition": "if",
        "exit": "break"
    },
    syntax={
        "block_start": "{",
        "block_end": "}",
        "statement_end": ";",
        "indent": "    "
    },
    keywords=["while", "if", "break", "{", "}", ";"]
)

# Haskell (Functional)
# Simulating Loop via recursion
HASKELL_SPEC = LanguageSpec(
    id="functional_haskell",
    paradigm="functional",
    control_flow={
        "loop": "recursion",
        "condition": "if", # if .. then .. else
        "exit": "termination" # Handled via base case pattern
    },
    syntax={
        "block_start": "",
        "block_end": "",
        "statement_end": "",
        "indent": "  "
    },
    keywords=["if", "then", "else", "="] # Recursive function signifiers
)
