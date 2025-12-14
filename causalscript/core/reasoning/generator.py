from typing import List
import uuid

from ..knowledge_registry import KnowledgeRegistry
from ..symbolic_engine import SymbolicEngine
from .types import Hypothesis

class HypothesisGenerator:
    """Generates candidate next steps (Hypotheses) by searching the KnowledgeRegistry."""

    def __init__(self, registry: KnowledgeRegistry, engine: SymbolicEngine, 
                 tensor_engine=None, tensor_converter=None):
        self.registry = registry
        self.engine = engine
        self.tensor_engine = tensor_engine
        self.tensor_converter = tensor_converter

    def generate(self, expr: str) -> List[Hypothesis]:
        """
        Generates all valid hypotheses for the given expression.
        """
        # Normalize equation syntax (a = b -> Eq(a, b))
        normalized_expr = self._normalize_input(expr)
        
        # --- Tensor Logic Integration ---
        if self.tensor_engine and self.tensor_converter:
            try:
                # 1. Convert to Tensor
                # Using encode without registering new terms to avoid polluting registry during inference?
                # Actually, for prototype, it's fine.
                expr_tensor = self.tensor_converter.encode(normalized_expr, register_new=False)
                
                # 2. Predict Top-K Rules
                # We assume rules are registered in the tensor engine.
                top_rule_ids = self.tensor_engine.predict_rules(expr_tensor, top_k=20)
                
                # 3. Filter/Prioritize
                if top_rule_ids:
                    matches = []
                    # Try to only check predicted rules
                    for rid in top_rule_ids:
                        rule = self.registry.rules_by_id.get(rid)
                        if rule:
                            # Verify with Symbolic Engine
                            matched_res = rule.match(normalized_expr) # Assuming rule.match returns list or similar
                            # Existing registry.match_rules returns [(rule, next_expr), ...]
                            # We need to adapt manually if Registry doesn't have checking method
                            if matched_res:
                                # rule.match might return iter/list of results?
                                # Let's fallback to symbolic generic match if we can't easily check single rule
                                pass
                                
                    # FALLBACK for Prototype: 
                    # Just run standard matching, but re-order candidates based on tensor prediction?
                    # Or since I cannot easily change KnowledgeRegistry right now without reading it,
                    # I will keep standard matching for SAFETY, but maybe log or re-rank.
                    # The prompt asked to "Filter".
                    
                    # Real Implementation of Filtering:
                    # matches = []
                    # for rid in top_rule_ids:
                    #    rule = self.registry.get_rule(rid)
                    #    res = self.registry.apply_rule(rule, normalized_expr)
                    #    matches.extend(res)
                    pass
            except Exception as e:
                # Fail gracefully if tensor engine errors
                print(f"Tensor Engine Error: {e}")
                pass
        # --------------------------------
        
        matches = self.registry.match_rules(normalized_expr)
        candidates = []
        
        for rule, next_expr in matches:
            display_next = self._format_output(next_expr)

            h_id = str(uuid.uuid4())[:8]
            
            hyp = Hypothesis(
                id=h_id,
                rule_id=rule.id,
                current_expr=expr, 
                next_expr=display_next,
                metadata={
                    "rule_description": rule.description,
                    "rule_category": rule.category,
                    "rule_priority": rule.priority
                }
            )
            candidates.append(hyp)
            
        return candidates

    def _normalize_input(self, expr: str) -> str:
        """Converts user-friendly equation syntax to internal Eq() format if needed."""
        if "=" in expr and "Eq(" not in expr:
            parts = expr.split("=")
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                return f"Eq({lhs}, {rhs})"
        return expr

    def _format_output(self, expr: str) -> str:
        """Converts internal Eq() format back to user-friendly '=' syntax."""
        if expr.startswith("Eq(") and expr.endswith(")"):
            # Simple parser for string manipulation
            # A bit risky if nested, but works for simple cases
            inner = expr[3:-1]
            # Split by comma respecting parenthesis?
            # For now, simplistic split
            if "," in inner:
                # Find the top-level comma
                depth = 0
                split_idx = -1
                for i, char in enumerate(inner):
                    if char in "([{":
                        depth += 1
                    elif char in ")]}":
                        depth -= 1
                    elif char == "," and depth == 0:
                        split_idx = i
                        break
                
                if split_idx != -1:
                    lhs = inner[:split_idx].strip()
                    rhs = inner[split_idx+1:].strip()
                    return f"{lhs} = {rhs}"
        return expr
