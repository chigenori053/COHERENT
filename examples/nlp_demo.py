
import sys
import os
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coherent.core.core_runtime import CoreRuntime
from coherent.core.computation_engine import ComputationEngine
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.validation_engine import ValidationEngine
from coherent.core.hint_engine import HintEngine
from coherent.tools.language.experience import ExperienceManager

# Mock engines for demo if full deps aren't available, 
# but we try to use real symbolic/computation.
class MockValidationEngine(ValidationEngine):
    def __init__(self): pass
    def validate_step(self, before, after, context=None): return {"valid": True}

class MockHintEngine(HintEngine):
    def __init__(self): pass

def create_runtime():
    print("Initializing Coherent Engines...")
    try:
        sym_engine = SymbolicEngine()
        comp_engine = ComputationEngine(sym_engine)
    except Exception as e:
        print(f"Error initializing math engines: {e}")
        return None

    val_engine = MockValidationEngine()
    hint_engine = MockHintEngine()
    
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine)
    
    # Enable Optical Memory if available
    if not runtime.experience_manager.is_enabled():
        print("Warning: Optical Memory dependencies (sentence-transformers) not found.")
        print("       Recall-First and Ambiguity features will be disabled.")
    else:
        print("Optical Memory: Enabled")
        
    return runtime

def main():
    runtime = create_runtime()
    if not runtime:
        return

    print("\n=== Coherent Language Processing Demo ===")
    print("Try commands like:")
    print(" - 'Solve 2 + 2'")
    print(" - 'Solve x + 5 = 10'")
    print(" - 'Verify 3 * 3 = 9'")
    print(" - 'Molar mass of H (ambiguous)'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input(">> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue

            # Process
            result = runtime.process_natural_language(user_input)
            
            # Display
            if "error" in result:
                print(f"Error: {result['error']}")
            elif "message" in result and "ambiguity_score" in result:
                # Clarification Request
                print(f"⚠️  Clarification Needed: {result['message']}")
                print(f"   Ambiguity Score: {result['ambiguity_score']:.2f}")
            else:
                # Success
                recalled = result.get('recalled', False)
                res_val = result.get('result')
                print(f"Result: {res_val}")
                if recalled:
                    print(f"   (Recalled from Memory 🧠)")
                else:
                    print(f"   (Computed New ✨)")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"System Error: {e}")

if __name__ == "__main__":
    main()
