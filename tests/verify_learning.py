import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.reasoning.agent import ReasoningAgent
from coherent.memory.ast_generalizer import ASTGeneralizer
from coherent.memory.optical_store import OpticalFrequencyStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_learning")

def verify_learning_and_optimization():
    print("--- Verifying Learning & Optimization ---")
    
    # 1. Initialize Runtime and Agent
    # Needs full dependency injection
    from coherent.engine.symbolic_engine import SymbolicEngine
    from coherent.engine.computation_engine import ComputationEngine
    from coherent.engine.validation_engine import ValidationEngine
    from coherent.engine.hint_engine import HintEngine
    from coherent.engine.knowledge_registry import KnowledgeRegistry
    
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = ValidationEngine(comp_engine)
    hint_engine = HintEngine(comp_engine)
    
    # Initialize Registry
    registry = KnowledgeRegistry(root_path=Path("coherent/engine/knowledge"), engine=sym_engine)
    
    runtime = CoreRuntime(
        computation_engine=comp_engine,
        validation_engine=val_engine,
        hint_engine=hint_engine,
        knowledge_registry=registry
    )
    agent = ReasoningAgent(runtime)
    
    print("\n[AST Optimization / Generalization]")
    generalizer = ASTGeneralizer()
    expr = "3*x + 5*y + 10"
    gen_expr = generalizer.generalize(expr)
    print(f"Original: {expr}")
    print(f"Generalized (Optimized): {gen_expr}")
    
    # Check if generalization works (variables replaced by _v)
    if "_v" in gen_expr and "x" not in gen_expr and "y" not in gen_expr:
        print(">> AST Generalization: PASS")
    else:
        print(">> AST Generalization: FAIL (Variables not properly abstracted)")

    print("\n[Learning Persistence]")
    # Simulate learning a solution
    # Input: 3*x + 2*x
    # Rule: ALG-OP-001 (Combine Like Terms)
    input_expr = "3*x + 2*x"
    # Note: ReasoningAgent.remember_solution expects a solution path (list of rule IDs)
    # It assumes the FIRST rule is the one to learn for the edge from initial state.
    
    print(f"Simulating learning for: {input_expr} -> ALG-OP-001")
    agent.remember_solution(input_expr, ["ALG-OP-001"])
    
    # Verify retention in Optical Store
    store = agent.experience_manager.vector_store
    if isinstance(store, OpticalFrequencyStore):
        print(f"Store Type: OpticalFrequencyStore (Capacity: {store.capacity})")
        print(f"Current Count: {store.current_count}")
        
        # We expect at least 1 item if learning worked (plus potentially prior indexed rules if shared store)
        # Note: knowledge registry indexed rules into 'knowledge' collection.
        # agent.experience_manager saves to 'experience_network'.
        # OpticalFrequencyStore handles all collections in one memory block in V1 logic?
        # Let's check query results.
        
        # Helper to retrieve based on generalized query
        gen_input = generalizer.generalize(input_expr)
        _, query_vec = agent.integrator.process_input(gen_input, input_type="text")
        
        if query_vec:
            results = agent.experience_manager.find_similar_edges(query_vec, top_k=1)
            if results:
                top_result = results[0]
                print(f"Recall Result: {top_result.original_expr} -> {top_result.rule_id}")
                if top_result.rule_id == "ALG-OP-001":
                    print(">> Learning Persistence: PASS")
                else:
                    print(f">> Learning Persistence: FAIL (Retrieved {top_result.rule_id}, expected ALG-OP-001)")
            else:
                print(">> Learning Persistence: FAIL (No memory recalled)")
        else:
            print(">> Learning Persistence: FAIL (Could not generate query vector)")
            
    else:
        print(f"Store Type: {type(store)} (Not OpticalFrequencyStore)")

if __name__ == "__main__":
    verify_learning_and_optimization()
