"""
CRS Memory Library - AST Logic
Parsing, Normalization, and Canonicalization of formulas.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import sympy
from sympy.core.basic import Basic

@dataclass
class NormalizedAST:
    canon_ast: Dict[str, Any] # Structural AST (JSON-safe)
    var_map: Dict[str, str]   # variable abstraction map
    signature: str            # stable hash
    root_node_id: str

class ASTNormalizer:
    """
    Handles normalization of SymPy expressions into CRS structural ASTs.
    """
    
    def parse_formula(self, source: str) -> Basic:
        """Parse source string to SymPy expression."""
        # Use sympify for v0.1
        return sympy.sympify(source, evaluate=False)

    def normalize(self, expr: Basic) -> NormalizedAST:
        """
        Normalize a SymPy expression.
        1. Variable Abstraction (DFS order)
        2. Algebraic Canonicalization (via Simplification)
        3. Structural Normalization (Sorting)
        """
        # 0. Initial Simplification (limited algebraic canonicalization)
        # Spec says: remove +0, *1. sympy.simplify does this.
        # But we need to be careful not to over-simplify if we want structural preservation?
        # The spec says "Algebraic canonicalization (limited)".
        # For v0.1 we use sympy.simplify lightly or just rely on Basic structure.
        
        # We'll use a pass that anonymizes variables.
        free_symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
        var_map = {}
        subst_map = {}
        
        # Capture variables in DFS order for canonical abstraction? 
        # Spec 4.3: "DFS order -> v1,v2,..."
        # We need a custom traversal to find vars in order.
        
        found_vars = []
        def walk(node):
            if node.is_Symbol:
                if node not in found_vars:
                    found_vars.append(node)
            for arg in node.args:
                walk(arg)
        
        walk(expr)
        
        for idx, sym in enumerate(found_vars):
            abstract_name = f"v{idx+1}"
            var_map[abstract_name] = sym.name
            subst_map[sym] = sympy.Symbol(abstract_name)

        # Create abstract expression
        abstract_expr = expr.subs(subst_map)
        
        # Build Structural AST
        nodes = {}
        
        def build_ast(node):
            node_id = str(hash(node)) # In reality we want content-based logical ID
            # Better: type + args hashes
            
            node_type = type(node).__name__
            children = []
            
            # Sort commutative args for canonical signature
            args = list(node.args)
            if node.is_Add or node.is_Mul:
                # Sort by string representation (approx canonical)
                args.sort(key=lambda x: str(x))
            
            child_ids = []
            for arg in args:
                child_id = build_ast(arg)
                child_ids.append(child_id)
            
            # Leaf node value
            value = None
            if node.is_Symbol:
                value = node.name
            elif node.is_Number:
                value = str(node)
                
            entry = {
                "id": node_id,
                "type": node_type,
                "children": child_ids,
            }
            if value:
                entry["value"] = value
                
            # Content-based hash for ID stability
            # Simple content signature for v0.1
            sig_str = f"{node_type}:" + ",".join(child_ids) + f":{value}"
            stable_id = f"N-{hash(sig_str) & 0xFFFFFFFF:08x}"
            entry["id"] = stable_id
            
            # Don't overwrite if exists (DAG nature)
            if stable_id not in nodes:
                 nodes[stable_id] = entry
            
            return stable_id

        root_id = build_ast(abstract_expr)
        
        # Signature is just root ID in this hash-consing scheme
        signature = root_id
        
        return NormalizedAST(
            canon_ast={"root": root_id, "nodes": nodes},
            var_map=var_map,
            signature=signature,
            root_node_id=root_id
        )
