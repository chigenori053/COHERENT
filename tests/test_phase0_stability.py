
import unittest
import sys
import os
import hashlib
import json
import ast
from dataclasses import dataclass, field

from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock torch before imports to avoid dependency issues
# --- MOCK DEPENDENCIES ---
from unittest.mock import MagicMock
# Mock keys to prevent ImportErrors in deep logical dependencies
# that are not essential for Phase 0 input/structure testing.

mock_torch = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = mock_torch.nn.functional

# Mock pydantic
mock_pydantic = MagicMock()
class MockModel:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
mock_pydantic.BaseModel = MockModel
mock_pydantic.Field = MagicMock(return_value=None)
mock_pydantic.ConfigDict = MagicMock()
sys.modules['pydantic'] = mock_pydantic


# Mock SymPy to satisfy imports, but we will force SymbolicEngine to use fallback
mock_sympy = MagicMock()
sys.modules['sympy'] = mock_sympy
sys.modules['sympy.core'] = mock_sympy.core
sys.modules['sympy.core.basic'] = mock_sympy.core.basic


# -----------------------------

from coherent.core.input_parser import CausalScriptInputParser
from coherent.core.symbolic_engine import SymbolicEngine
# Import the module object to patch _sympy global
from coherent.core import symbolic_engine as sym_engine_mod

# Force fallback by clearing _sympy in the module
sym_engine_mod._sympy = None

@dataclass
class ASTSnapshot:
    """Represents the standardized state of an AST."""
    raw_input: str
    normalized_input: str
    structure_hash: str
    structure_pattern: str # Human readable generalized pattern
    node_types: List[str]  # Sequence of node types visited
    
class ASTGeneralizer:
    """
    Helper to walk an AST (Python AST or SymPy) and generate a 'Generalized' structural string.
    Replaces all specific Symbols with 'VAR' and specific Numbers with 'NUM'.
    """
    
    def generalize(self, expr_obj) -> Tuple[str, List[str]]:
        node_types = []
        gen_str = self._walk(expr_obj, node_types)
        return gen_str, node_types

    def _walk(self, node, node_types: List[str]) -> str:
        # Handle Python AST (Fallback)
        if isinstance(node, (ast.Expression, ast.Module)):
             # Unwrap
             return self._walk(node.body, node_types)
             
        if isinstance(node, ast.BinOp):
             left = self._walk(node.left, node_types)
             right = self._walk(node.right, node_types)
             op_name = type(node.op).__name__
             node_types.append(op_name)
             return f"{op_name}({left}, {right})"
        
        if isinstance(node, ast.UnaryOp):
             op_name = type(node.op).__name__
             operand = self._walk(node.operand, node_types)
             node_types.append(op_name)
             return f"{op_name}({operand})"
             
        if isinstance(node, ast.Call):
             func_id = "Call"
             if isinstance(node.func, ast.Name):
                 func_id = node.func.id
             node_types.append(func_id)
             args = [self._walk(a, node_types) for a in node.args]
             return f"{func_id}({', '.join(args)})"
             
        if isinstance(node, ast.Name):
             node_types.append("Name")
             return "VAR"
             
        if isinstance(node, ast.Constant):
             node_types.append("Constant")
             return f"INT({node.value})"
             
        # List (vector)
        if isinstance(node, list):
             node_types.append("List")
             elems = [self._walk(e, node_types) for e in node]
             return f"List({', '.join(elems)})"

        # Fallback for unknown
        t = type(node).__name__
        node_types.append(t)
        return t

class Phase0Tester(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.engine = SymbolicEngine()
        cls.parser = CausalScriptInputParser()
        cls.generalizer = ASTGeneralizer()
        cls.report_data = {
            "title": "Phase 0: Core Stabilization Report",
            "categories": {}
        }
        
    @classmethod
    def tearDownClass(cls):
        # Generate Report
        cls._generate_report()

    @classmethod
    def _generate_report(cls):
        report_path = os.path.join("report", "PHASE0_REPORT.md")
        data = cls.report_data
        
        os.makedirs("report", exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# {data['title']}\n\n")
            
            for cat_name, results in data["categories"].items():
                f.write(f"## {cat_name}\n\n")
                
                # Check consistency
                # We normalize structure pattern by sorting arguments for commutative ops?
                # The Python AST preserves order (x+y is Left+Right).
                # So "x+y" and "y+x" will have DIFFERENT ASTs in Python (Add(x,y) vs Add(y,x)).
                # Phase 0 criteria: "同一 AST".
                # If we rely on Python AST, we don't get Commutativity for free.
                # SymPy gives it for free.
                
                # IMPORTANT: Since we are in Fallback mode, we cannot guarantee identical AST for x+y and y+x
                # unless we implement a Canonicalizer in ASTGeneralizer.
                # For Phase 0 without SymPy, we might accept that structure is "Add(VAR, VAR)" for both,
                # so specific hash matches if we abstract the *content*?
                # Yes, "x+y" -> Add(VAR, VAR), "y+x" -> Add(VAR, VAR).
                # So "VAR" abstraction DOES solve commutativity of *TYPE*.
                # But it loses variable identity (x+y vs x+x).
                # "x+x" -> Add(VAR, VAR). "2x" -> Mult(2, VAR).
                # So Category C (Equivalence) will FAIL in fallback mode unless we parse carefully.
                # "x+x" is Add, "2x" is Mult.
                
                # This highlights why SymPy is needed for true Equivalence check.
                # But for now, let's just generate the report and see what happens.
                # The user asked for "Phase 0", implying they want to see stability.
                # If it fails, that is a valid result (demonstrating need for SymPy or better Logic).
                
                hashes = [r['hash'] for r in results]
                unique_hashes = set(hashes)
                
                status = "PASS" if len(unique_hashes) == 1 else "FAIL (Expected without SymPy)"
                
                f.write(f"**Status: {status}**\n\n")
                if status.startswith("FAIL"):
                    f.write("> [!WARNING]\n> Structural difference detected. (Note: Without SymPy, equivalence reduction is limited)\n\n")
                
                f.write("| Input | Normalized | Structure Pattern | Hash |\n")
                f.write("|-------|------------|-------------------|------|\n")
                
                for r in results:
                    pat = r['pattern'].replace("|", "\\|") # Escape pipes
                    f.write(f"| `{r['input']}` | `{r['norm']}` | `{pat}` | `{r['hash'][:8]}` |\n")
                
                f.write("\n")

        # Dump JSON Log
        log_path = os.path.join("report", "PHASE0_LOG.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Report generated at {report_path}")
        print(f"Log generated at {log_path}")

    def _process_inputs(self, category_name: str, inputs: List[str]):
        results = []
        for inp in inputs:
            # 1. Normalize String
            norm_str = self.parser.normalize(inp)
            
            # 2. To Internal AST (SymPy or Python AST)
            internal_ast = self.engine.to_internal(norm_str)
            
            # 3. Generalize structure
            gen_str, node_types = self.generalizer.generalize(internal_ast)
            
            # 4. Hash
            structure_hash = hashlib.sha256(gen_str.encode('utf-8')).hexdigest()
            
            results.append({
                "input": inp,
                "norm": norm_str,
                "pattern": gen_str,
                "hash": structure_hash,
                "nodes": node_types
            })
            
        self.report_data["categories"][category_name] = results
        
        # Assertion: All hashes in this category must be equal
        hashes = [r['hash'] for r in results]
        err_msg = f"Structural mismatch in {category_name}:\n"
        for r in results:
            err_msg += f"  {r['input']} -> {r['hash'][:8]} ({r['pattern']})\n"
            
        self.assertEqual(len(set(hashes)), 1, err_msg)

    def test_category_A_commutativity(self):
        """Input: x + y, y + x"""
        inputs = ["x + y", "y + x"]
        self._process_inputs("Category A: Commutativity", inputs)

    def test_category_B_associativity(self):
        """Input: (x + y) + z, x + (y + z)"""
        inputs = ["(x + y) + z", "x + (y + z)"]
        self._process_inputs("Category B: Associativity", inputs)
        
    def test_category_C_equivalence(self):
        """Input: x + x, 2x"""
        # Note checking implicit multiplication logic too: "2x" -> "2*x"
        inputs = ["x + x", "2x"]
        self._process_inputs("Category C: Equivalence Reduction", inputs)

    def test_category_D_coefficients(self):
        """Input: 3a + 5b, 5b + 3a"""
        inputs = ["3a + 5b", "5b + 3a"]
        self._process_inputs("Category D: Coefficient Arrangement", inputs)

if __name__ == "__main__":
    unittest.main()
