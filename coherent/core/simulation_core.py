"""
SimulationCore (BrainModel v2.0)

Role:
    Execution-only engine for numeric, symbolic, and physics simulations.
    Operates under strict authority of CognitiveCore.
    
Constraints:
    - No decision authority.
    - Outputs evidence/results only.
    - Stateless (per request execution).
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import logging

class ComputationMode(Enum):
    REAL = "REAL"
    HYPERREAL = "HYPERREAL"   # Infinitesimal / Limit behavior
    INFINITY = "INFINITY"     # Divergence detection

class SimulationCore:
    def __init__(self):
        self.logger = logging.getLogger("SimulationCore")
        self.supported_domains = ["numeric", "coding", "physics", "chemistry"]
        self.mode = ComputationMode.REAL

    def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for CognitiveCore to request simulation.
        
        Args:
            request: Dict containing:
                - domain: str
                - input_context: Any
                - simulation_config: Dict (optional)
        
        Returns:
            Dict containing:
                - result: Any
                - divergence_detected: bool
                - execution_trace: List
        """
        domain = request.get("domain")
        context = request.get("input_context")
        
        self.logger.info(f"Received simulation request for domain: {domain}")
        
        # Placeholder for actual engines
        if domain == "numeric":
            return self._run_numeric_simulation(context)
        elif domain == "coding":
            return self._run_coding_simulation(context)
        else:
            return {
                "error": f"Unsupported domain: {domain}",
                "status": "FAILED"
            }

    def _run_numeric_simulation(self, context: Any) -> Dict[str, Any]:
        # Minimal MVP implementation with NLP support
        input_str = str(context)
        
        # 1. NLP Command Parsing (Regex based for MVP)
        import re
        
        # Pattern: [Number] ... 素因数分解 or factor ... [Number]
        # Check for factorization command
        is_factorization = "素因数分解" in input_str or "factor" in input_str.lower()
        
        # Extract numbers (integers)
        numbers = re.findall(r'\d+', input_str)
        
        if is_factorization and numbers:
            target = int(numbers[0]) # Take the first number
            factors = self._prime_factorization(target)
            formatted_result = f"{target} = " + " * ".join(map(str, factors))
            return {
                "result": formatted_result,
                "status": "SUCCESS",
                "factors": factors,
                "mode": "INTEGER_FACTORIZATION"
            }
            
        # 2. Fallback to Algebra (simplify)
        from coherent.core.simple_algebra import SimpleAlgebra
        
        # Check if input contains non-ASCII characters (e.g. Japanese instructions)
        # If so, process via cleaning route immediately to avoid interpreting Japanese as variables.
        if not input_str.isascii():
             # Non-ASCII present, assume mixed text -> Clean
             pass # Fall through to except block logic intentionally? No, better structure.
        else:
            try:
                # Try original first (if ASCII)
                result = SimpleAlgebra.simplify(input_str)
                return {
                    "result": result,
                    "status": "SUCCESS",
                    "mode": self.mode.value
                }
            except Exception:
                pass # Fall through to cleaning

        # Cleaning & Retry Path
        # Keep 0-9, +, -, *, /, (, ), ., whitespace, and a-z A-Z (for algebra variables if needed)
        # But if we want to support algebra, we should keep letters. 
        # However, "計算して" would match letters if we kept non-ascii? No, regex below is strict.
        
        # Regex: Keep digits, operators, parens, dot, whitespace
        # Note: If we want to support 'x + y', we need to add a-zA-Z to regex.
        # But 'factor' case is handled above. 'compute x+y' might need adjustment.
        # For now, let's stick to numeric arithmetic support as primary goal for 'numeric' domain.
        cleaned = re.sub(r'[^\d\+\-\*\/\(\)\.\s]', '', input_str).strip()
        
        try:
            if not cleaned: raise ValueError("Empty expression after cleaning")
            result = SimpleAlgebra.simplify(cleaned)
            return {
                "result": result,
                "status": "SUCCESS",
                "mode": self.mode.value,
                "note": "Parsed from mixed text"
            }
        except Exception as e:
            return {
                "error": f"Invalid expression: {input_str}",
                "status": "ERROR", 
                "details": str(e)
            }

    def _prime_factorization(self, n: int) -> List[int]:
        factors = []
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                factors.append(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        return factors

    def _run_coding_simulation(self, context: Any) -> Dict[str, Any]:
        # Placeholder
        return {"status": "NOT_IMPLEMENTED"}
