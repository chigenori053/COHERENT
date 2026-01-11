"""
Projection Engine for Verification Sandbox
Projects Abstract IR into Concrete String Representations based on Language Spec.
"""

from .ir import *
from .languages import LanguageSpec

class ProjectionEngine:
    def project(self, node: IRNode, spec: LanguageSpec) -> str:
        if spec.id == "functional_haskell":
            return self._project_haskell(node, spec)
        else:
            return self._project_procedural(node, spec, indent_level=0)

    def _project_procedural(self, node: IRNode, spec: LanguageSpec, indent_level: int) -> str:
        indent = spec.syntax["indent"] * indent_level
        
        if isinstance(node, Sequence):
            return "\n".join([self._project_procedural(n, spec, indent_level) for n in node.nodes])
            
        elif isinstance(node, Assignment):
            return f"{indent}{node.target} = {node.value}{spec.syntax['statement_end']}"
            
        elif isinstance(node, Exit):
            return f"{indent}{spec.control_flow['exit']}{spec.syntax['statement_end']}"
            
        elif isinstance(node, If):
            # Header
            s = f"{indent}{spec.control_flow['condition']} ({node.condition}){spec.syntax['block_start']}\n"
            # Body
            s += self._project_procedural(node.body, spec, indent_level + 1)
            # End
            if spec.syntax["block_end"]:
                 s += f"\n{indent}{spec.syntax['block_end']}"
            return s
            
        elif isinstance(node, Loop):
            s = f"{indent}{spec.control_flow['loop']} ({node.condition}){spec.syntax['block_start']}\n"
            s += self._project_procedural(node.body, spec, indent_level + 1)
            if spec.syntax["block_end"]:
                 s += f"\n{indent}{spec.syntax['block_end']}"
            return s
            
        return ""

    def _project_haskell(self, node: IRNode, spec: LanguageSpec) -> str:
        # Haskell mapping logic (Simplified for structural verification)
        # We assume the root is a Sequence implies a function body context or main
        # But specifically handling LOOP -> Recursion is key.
        
        if isinstance(node, Loop):
            # Loop(cond, body) ->
            # loop_fn state = if cond then loop_fn (update state) else result
            # We map this structural pattern:
            return f"loop_fn state = if ({node.condition}) then\n  -- Body Execution\n  loop_fn (next_state)\nelse\n  state -- Exit"

        elif isinstance(node, Sequence):
             # Just join lines
             # Note: assignments in Hs are let bindings or state monad steps
             # For Structure check, we just output lines
             combined = "\n".join([self._project_hs_node(n) for n in node.nodes])
             return f"do\n{combined}"
             
        return self._project_hs_node(node)

    def _project_hs_node(self, node: IRNode) -> str:
        if isinstance(node, Assignment):
            return f"  let {node.target} = {node.value}"
        elif isinstance(node, Exit):
            return "  return ()" # Nominal exit
        elif isinstance(node, If):
            # Nested logic not fully implemented for Hs projection in this MVP
            return f"  if {node.condition} then ..."
        return ""
