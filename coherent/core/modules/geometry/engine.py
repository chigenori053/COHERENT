from typing import Any, Dict
from coherent.core.interfaces import BaseEngine
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.math_category import MathCategory

class GeometryEngine(BaseEngine):
    """
    Engine for Geometry mode.
    Wraps SymbolicEngine with Geometry context.
    """
    
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()
        self.symbolic_engine.set_context([MathCategory.GEOMETRY, MathCategory.ALGEBRA])

    def evaluate(self, node: Any, context: Dict[str, Any]) -> Any:
        return self.symbolic_engine.evaluate(node, context)
