import pytest
from coherent.core.arithmetic_engine import ArithmeticEngine
from coherent.core import ast_nodes as ast

def test_normalize_simple_addition():
    """Tests constant folding for addition."""
    engine = ArithmeticEngine()
    
    # Represents 2 + 3
    expression = ast.Add(terms=[ast.Int(2), ast.Int(3)])
    
    # Normalize the expression
    normalized_expr = engine.normalize(expression)
    
    # The result should be a single integer node with value 5
    assert isinstance(normalized_expr, ast.Int)
    assert normalized_expr.value == 5

def test_normalize_identity_addition():
    """Tests removal of identity element for addition (a + 0 = a)."""
    engine = ArithmeticEngine()
    
    # Represents x + 0
    expression = ast.Add(terms=[ast.Sym("x"), ast.Int(0)])
    
    # Normalize
    normalized_expr = engine.normalize(expression)
    
    # The result should be just the symbol 'x'
    assert isinstance(normalized_expr, ast.Sym)
    assert normalized_expr.name == "x"
