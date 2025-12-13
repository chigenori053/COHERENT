from typing import Any
from causalscript.core.interfaces import BaseParser
from causalscript.core.input_parser import CausalScriptInputParser
from causalscript.core.symbolic_engine import SymbolicEngine

class CalculusParser(BaseParser):
    """
    Parser for Calculus mode.
    Leverages CausalScriptInputParser for normalization and SymbolicEngine for converting to SymPy objects.
    """
    
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()

    def parse(self, text: str) -> Any:
        """
        Parse the text into a SymPy expression (or fallback AST).
        """
        # 1. Normalize
        normalized = CausalScriptInputParser.normalize(text)
        
        # 2. Convert to Internal Representation (SymPy object)
        # We use SymbolicEngine's to_internal which handles SymPy parsing
        return self.symbolic_engine.to_internal(normalized)

    def validate(self, text: str) -> bool:
        """
        Check if the text is valid for this domain.
        """
        try:
            self.parse(text)
            return True
        except Exception:
            return False
