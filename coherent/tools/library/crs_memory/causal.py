"""
CRS Memory Library - Causal Model Extraction
Extracts deterministic CausalModel from NormalizedAST.
"""
from typing import Dict, Any, List
from .structure import CausalGraph, CausalVariable, CausalRelation
from .ast_logic import NormalizedAST

class CausalExtractor:
    """
    Extracts causal structure from AST.
    """
    
    def extract(self, norm_ast: NormalizedAST) -> CausalGraph:
        """
        Build CausalModel from NormalizedAST.
        Variables: Input Symbols, Operators (Derived), Root (Output).
        Relations: Child -> Parent (Causality).
        """
        nodes = norm_ast.canon_ast["nodes"]
        root_id = norm_ast.root_node_id
        
        causal_nodes: Dict[str, CausalVariable] = {}
        causal_edges: List[CausalRelation] = []
        
        # Reverse map for variable names
        # abstract v1 -> real x
        # norm_ast.var_map is v1->x
        
        for nid, node in nodes.items():
            ntype = node["type"]
            
            # Determine Source Type
            source = "ast_derived"
            name = f"{ntype}_{nid[:4]}"
            vtype = "numeric"
            
            # If leaf
            if "children" in node and not node["children"]:
                if ntype == "Symbol":
                    source = "ast_input"
                    abstract_name = node.get("value")
                    real_name = norm_ast.var_map.get(abstract_name, abstract_name)
                    name = real_name
                elif ntype in ["Integer", "Float", "Rational"]:
                    source = "ast_const" # Not strictly "input" variable
                    name = node.get("value")
            
            if nid == root_id:
                source = "ast_output" # Override if root
            
            cvar = CausalVariable(
                id=nid,
                name=name,
                vtype=vtype,
                source=source
            )
            causal_nodes[nid] = cvar
            
            # Relations: Child -> Parent
            for child_id in node.get("children", []):
                # Default rule 5.2: Child causes Parent
                label = "operand"
                rel = CausalRelation(
                    from_id=child_id,
                    to_id=nid,
                    strength=0.7, # Default per spec 5.3
                    confidence=0.8,
                    mechanism=f"{ntype}.{label}"
                )
                causal_edges.append(rel)
                
        return CausalGraph(nodes=causal_nodes, edges=causal_edges)
