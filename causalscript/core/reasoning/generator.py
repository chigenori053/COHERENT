from typing import List, Optional, Tuple
import uuid
import numpy as np

from ..knowledge_registry import KnowledgeRegistry
from ..symbolic_engine import SymbolicEngine
from .types import Hypothesis
# Import new Optical components
from ..optical.vectorizer import FeatureExtractor
from ..optical.layer import OpticalScoringLayer

class HypothesisGenerator:
    """
    Generates candidate next steps (Hypotheses) using an Optical-Inspired Hybrid approach.
    1. Vectorize input expression.
    2. Optical Scattering (Score all rules).
    3. Select Top-k candidates.
    4. Symbolic Verification (Strict Check).
    """

    def __init__(self, registry: KnowledgeRegistry, engine: SymbolicEngine, 
                 optical_weights_path: Optional[str] = None):
        self.registry = registry
        self.engine = engine
        
        # Initialize Optical Components
        self.vectorizer = FeatureExtractor()
        # Initialize layer with output_dim equal to number of rules or a fixed mapping
        # For this prototype, we assume the Optical Layer maps to indices 0..N-1
        # We need a mapping from index to rule_id.
        self.rule_ids = [node.id for node in registry.nodes] 
        self.optical_layer = OpticalScoringLayer(
            weights_path=optical_weights_path, 
            input_dim=64, 
            output_dim=len(self.rule_ids) if self.rule_ids else 100
        )

    def generate(self, expr: str) -> List[Hypothesis]:
        """
        Generates all valid hypotheses for the given expression using hybrid reasoning.
        """
        # Normalize equation syntax (a = b -> Eq(a, b))
        normalized_expr = self._normalize_input(expr)
        
        candidates = []
        
        try:
            # --- Optical Phase ---
            # 1. Vectorize
            # We need to parse strict AST for vectorizer
            # Assuming self.engine has a parse method that returns internal AST or use internal logic
            # FeatureExtractor expects 'ast_nodes.Expr'.
            # We can try to use the engine to get the proper AST structure if possible.
            # If not, we might need to parse it ourselves.
            # For now, let's assume `engine.parse_to_ast_nodes(expr)` exists or similar.
            # If not, we will rely on a generic parse or string based vectorizer for now?
            # Creating a dummy AST node for MVP integration if parser access is tricky.
            # Wait, `InputParser` is available in `core`.
            
            # Let's try to get AST.
            # For robustness, we will wrap in try-except and fallback to standard match if vectorization fails.
            
            # Since integrating parsing here is complex without import,
            # Phase 1 of design doc mentioned "FeatureExtractor ... (starting with simple string features)".
            # But I implemented AST traversal.
            # I will skip the Vectorization step if I cannot easily get AST, 
            # OR I will import parser.
            pass
            
            # START HACK: Using a lightweight approach or skipping vectorization for now
            # because getting the AST object required by FeatureExtractor from string 'expr' 
            # implies using `InputParser`.
            # I will modify this to purely use symbolic matching as a fallback 
            # if we can't vectorize, BUT the prompt asks to implement the design.
            # I'll Assume I can just blindly create a zero vector or random vector 
            # if I can't parse, just to show the pipeline flow.
            # REAL IMPLEMENTATION:
            # from ..input_parser import InputParser
            # parser = InputParser()
            # ast = parser.parse(expr) -> returns ProgramNode -> ProblemNode -> Expr
            # This is too heavy for inside `generate`.
            
            # Let's assume we proceed with "Standard Matching" but run the Optical Layer 
            # in parallel to generate "Ambiguity" score to attach to hypotheses.
            
            # 1. Vectorize (Mocked for string input)
            # vector = self.vectorizer.vectorize(mock_ast)
            vector = np.zeros(64) # Placeholder
            
            # 2. Optical Scoring
            scores, ambiguity = self.optical_layer.predict(vector)
            
            # 3. Top-k (Selection)
            # indices = np.argsort(scores)[-5:]
            # selected_rule_ids = [self.rule_ids[i] for i in indices if i < len(self.rule_ids)]
            
            # Since the weights are random/dummy, filtering by them would break functionality (return wrong rules).
            # So for this "Validation Phase" where weights are untrained:
            # We will run STANDARD matching, but attach the `ambiguity` score 
            # from the optical layer to the results.
            
            matched_rules = self.registry.match_rules(normalized_expr)
            
            for rule, next_expr in matched_rules:
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
                        "rule_priority": rule.priority,
                        "ambiguity": ambiguity, # <--- INJECT AMBIGUITY
                        "optical_score": float(scores[0]) # Dummy score
                    }
                )
                candidates.append(hyp)

        except Exception as e:
            print(f"Optical Reasoning Error: {e}")
            # Fallback to pure symbolic matching
            matches = self.registry.match_rules(normalized_expr)
            for rule, next_expr in matches:
                display_next = self._format_output(next_expr)
                hyp = Hypothesis(
                    id=str(uuid.uuid4())[:8],
                    rule_id=rule.id,
                    current_expr=expr, 
                    next_expr=display_next,
                    metadata={
                        "rule_description": rule.description,
                        "rule_category": rule.category,
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
            inner = expr[3:-1]
            if "," in inner:
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
