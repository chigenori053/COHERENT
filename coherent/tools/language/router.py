from typing import Dict, Any, Optional
from coherent.tools.language.models import SemanticIR, IntentType, InputItemType

class IntentRouter:
    """
    Routes SemanticIR to the appropriate CoreRuntime action.
    """
    
    def route(self, sir: SemanticIR) -> Dict[str, Any]:
        """
        Determines the implementation action based on the SIR.
        
        Returns:
            Dict containing 'action', 'params', and 'target_component'.
        """
        if sir.task == IntentType.SOLVE:
            return self._route_solve(sir)
        elif sir.task == IntentType.VERIFY:
            return self._route_verify(sir)
        elif sir.task == IntentType.EXPLAIN:
            return self._route_explain(sir)
        
        return {"action": "unknown", "error": "Unsupported intent"}

    def _get_primary_expression(self, sir: SemanticIR) -> Optional[str]:
        for item in sir.inputs:
            if item.type in [InputItemType.EXPRESSION, InputItemType.EQUATION]:
                return str(item.value)
        return None

    def _route_solve(self, sir: SemanticIR) -> Dict[str, Any]:
        expr = self._get_primary_expression(sir)
        if not expr:
            return {"action": "error", "message": "No expression found to solve."}
            
        return {
            "action": "compute",
            "method": "evaluate", # Default to evaluate/simplify
            "params": {
                "expression": expr,
                "domain": sir.math_domain
            }
        }

    def _route_verify(self, sir: SemanticIR) -> Dict[str, Any]:
        expr = self._get_primary_expression(sir)
        if not expr:
             return {"action": "error", "message": "No expression found to verify."}
             
        return {
            "action": "verify",
            "method": "check_validity",
            "params": {
                "expression": expr
            }
        }

    def _route_explain(self, sir: SemanticIR) -> Dict[str, Any]:
         return {
            "action": "explain",
            "params": {
                "text": sir.inputs[0].value if sir.inputs else "",
                "context": sir.model_dump()
            }
        }
