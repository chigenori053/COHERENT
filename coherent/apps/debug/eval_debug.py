import streamlit as st
import sys
import os
import io
from contextlib import redirect_stdout

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from coherent.core.parser import Parser
from coherent.core.evaluator import Evaluator
from coherent.core.core_runtime import CoreRuntime
from coherent.core.learning_logger import LearningLogger
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.computation_engine import ComputationEngine
from coherent.core.validation_engine import ValidationEngine
from coherent.core.hint_engine import HintEngine
from coherent.core.evaluator import SymbolicEvaluationEngine

st.set_page_config(page_title="Coherent Debugger", layout="wide")

st.title("🐞 Coherent Internal Debugger")
st.markdown("Inspect internal state during validaton of CausalScript checks.")

# Default failing script
default_script = """problem: y = 5x - 20
prepare:
  - x = 3
step: y = 5 * 3 - 20
step: y = 15 - 20
step: y = - 5
end: done"""

script_input = st.text_area("Script Input", value=default_script, height=200)

if st.button("Run Debug Trace"):
    st.subheader("Trace Log")
    
    # Setup Engines
    symbolic_engine = SymbolicEngine()
    
    # We want to instrument the SymbolicEvaluationEngine used by Evaluator
    # But Evaluator creates it internally via runtime or passed engine?
    # Evaluator code:
    # def __init__(self, program: Program, engine: Engine, ...):
    
    # We need a CoreRuntime to pass to Evaluator, or we can pass a raw SymbolicEvaluationEngine if compatible.
    # Evaluator expects 'Engine' interface.
    
    # Let's verify how app.py does it:
    # evaluator = Evaluator(program, system['runtime'], ...)
    # CoreRuntime IS an Engine.
    
    # So we instantiate CoreRuntime.
    comp_engine = ComputationEngine(symbolic_engine)
    val_engine = ValidationEngine(comp_engine)
    hint_engine = HintEngine(comp_engine)
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine)
    
    # Parse
    try:
        parser = Parser(script_input)
        program = parser.parse()
        st.success("Parsing Successful")
        
        # We need to hook into 'check_step' to log details.
        # Since we pass 'runtime' to Evaluator, Evaluator calls 'runtime.check_step'.
        # We can Monkey Patch runtime.check_step
        
        original_check_step = runtime.check_step
        
        log_buffer = []

        def debug_check_step(expr: str):
            log_buffer.append(f"\n--- Checking Step: {expr} ---")
            
            # 1. Check Context
            ctx = runtime.computation_engine.variables
            log_buffer.append(f"Current Context: {ctx}")
            
            # 2. Check Problem / Current Expr
            current = runtime._current_expr
            log_buffer.append(f"Current State (Before): {current}")
            
            # 3. Simulate internals
            # CoreRuntime.check_step -> SymbolicEvaluationEngine.check_step (if delegated) 
            # OR CoreRuntime implements it? CoreRuntime inherits Engine.
            # Let's verify CoreRuntime.check_step logic quickly.
            # CoreRuntime implementation:
            # def check_step(self, expr: str) -> dict:
            #     return self.evaluation_engine.check_step(expr)
            # (Assuming it has one, or acts as one)
            
            # Actually CoreRuntime inherits Engine but usually delegates. 
            # Let's call original and capture result.
            
            try:
                result = original_check_step(expr)
                log_buffer.append(f"Result: {result}")
                
                # Dig Deeper into Symbolic Engine Execution if mistake
                if not result.get("valid"):
                    log_buffer.append("❌ Validation Failed. Digging deeper...")
                    # Manual Re-simulation
                    before = current
                    after = expr
                    
                    # Apply Context
                    try:
                        before_eval = runtime.computation_engine.substitute(before, ctx)
                        after_eval = runtime.computation_engine.substitute(after, ctx)
                        log_buffer.append(f"Substituted Before: {before_eval}")
                        log_buffer.append(f"Substituted After:  {after_eval}")
                        
                        # Equivalence Check
                        is_equiv = symbolic_engine.is_equiv(before_eval, after_eval)
                        log_buffer.append(f"Symbolic Equivalence Check ({before_eval} == {after_eval}): {is_equiv}")
                        
                        # Check SymPy Internal
                        int_b = symbolic_engine.to_internal(before_eval)
                        int_a = symbolic_engine.to_internal(after_eval)
                        log_buffer.append(f"SymPy Internal Before: {repr(int_b)}")
                        log_buffer.append(f"SymPy Internal After:  {repr(int_a)}")
                        
                        # Diff
                        try:
                           import sympy
                           # Apply Eq logic patch manually to see if it works
                           if hasattr(int_b, 'lhs'): int_b = int_b.lhs - int_b.rhs
                           if hasattr(int_a, 'lhs'): int_a = int_a.lhs - int_a.rhs
                           diff = sympy.simplify(int_b - int_a)
                           log_buffer.append(f"Difference (Simplified): {diff}")
                        except Exception as e:
                           log_buffer.append(f"Diff Error: {e}")
                        
                    except Exception as e:
                        log_buffer.append(f"Debugging Error: {e}")
                else:
                    log_buffer.append("✅ Validation Successful")
                        
                return result
            except Exception as e:
                log_buffer.append(f"Exception in check_step: {e}")
                raise e

        # Apply Hook
        runtime.check_step = debug_check_step
        
        # Run Evaluator
        logger = LearningLogger()
        evaluator = Evaluator(program, runtime, learning_logger=logger)
        success = evaluator.run()
        
        # Render Log
        for line in log_buffer:
            if "❌" in line:
                st.error(line)
            elif "---" in line:
                st.subheader(line)
            else:
                st.text(line)
                
        if success:
            st.success("Script Execution Passed")
        else:
            st.error("Script Execution Failed")
            
    except Exception as e:
        st.error(f"Parser/Runner Error: {e}")
