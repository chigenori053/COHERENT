from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class RenderContext:
    expression: str
    category: str
    metadata: Dict[str, Any]

class ContentRenderer:
    """Formats log messages based on category."""

    @staticmethod
    def render_step(context: RenderContext) -> str:
        category = context.category
        expr = context.expression
        
        if category == "geometry":
            return ContentRenderer._render_geometry(expr, context.metadata)
        elif category == "calculus":
            return ContentRenderer._render_calculus(expr)
        elif category == "statistics":
            return ContentRenderer._render_statistics(expr)
        
        # Default (Algebra/Arithmetic)
        return expr

    @staticmethod
    def _render_geometry(expr: str, meta: Dict) -> str:
        # If there's a description in metadata (e.g. from a rule), prepend it
        if "description" in meta:
            return f"{meta['description']} ({expr})"
        return expr

    @staticmethod
    def _render_calculus(expr: str) -> str:
        # Example: 'diff(x**2, x)' -> "d/dx (x^2)"
        # Simple string replacement for demo purposes
        result = expr
        if "diff(" in result:
            result = result.replace("diff(", "d/dx(").replace(",", ", ")
        if "integrate(" in result:
            result = result.replace("integrate(", "∫(").replace(",", ", ")
        return result

    @staticmethod
    def _render_statistics(expr: str) -> str:
        return f"📊 {expr}"
