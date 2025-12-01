import sys
import os
import re
from sympy import latex, sympify
from sympy.parsing.sympy_parser import parse_expr
import sympy

# Add project root to path
sys.path.append(os.getcwd())

def test_rendering_fix():
    print("--- Test Rendering Fix ---")
    
    exprs = [
        "2^3 - 0",
        "1/2",
        "x - y",
        "-1 * x",
        "(-1) * 0"
    ]
    
    local_dict = {"e": sympy.E, "pi": sympy.pi}
    
    for expr in exprs:
        print(f"\nInput: {expr}")
        # Normalize ^ to **
        expr_norm = expr.replace("^", "**")
        
        try:
            internal = parse_expr(expr_norm, evaluate=False, local_dict=local_dict)
            raw_latex = latex(internal, mul_symbol=r" \cdot ")
            print(f"Raw LaTeX: {raw_latex}")
            
            # Apply proposed fixes
            processed = raw_latex
            
            # 1. Remove "1 \cdot "
            processed = processed.replace(r"1 \cdot ", "")
            
            # 2. Replace "\left(-1\right) \cdot" with "-"
            processed = processed.replace(r"\left(-1\right) \cdot ", "-")
            
            # 3. Handle "+ -" -> "-"
            processed = processed.replace(r"+ -", "- ")
            
            print(f"Processed: {processed}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_rendering_fix()
