import sys
import os
from coherent.engine.language.semantic_parser import RuleBasedSemanticParser
from coherent.engine.reasoning.agent import ReasoningAgent
from coherent.core.state import State
from coherent.core.action_types import ActionType
from coherent.engine.core_runtime import CoreRuntime
from ui.app import get_system

# Mock System Init
system = get_system()
parser = system["semantic_parser"]
agent = system["agent"]
executor = system["executor"]
runtime = system['runtime']

def reproduce():
    # The complex input causing "Unknown error"
    text = "y = 3x + 2 x = 1の場合 y = 3 * 1 + 2 y = 5 x = -2の場合 y = 3 * (- 2) + 2 y = -6 + 2 y = -4"
    print(f"Input: {text}")
    
    # 1. Parse
    ir = parser.parse(text)
    print(f"Parsed Task: {ir.task}")
    
    if not ir.inputs:
        print("Error: No inputs extracted.")
        return
        
    extracted_expr = ir.inputs[0].value
    print(f"Extracted Expression (Raw): '{extracted_expr}'")
    
    # 2. State
    state = State(
        task_goal=ir.task,
        initial_inputs=ir.inputs,
        current_expression=extracted_expr
    )
    
    # 3. Agent Act
    print("\n--- Agent Acting ---")
    action = agent.act(state)
    print(f"Proposed Action: {action.name} ({action.type})")
    print(f"Action Inputs: {action.inputs}")
    
    # 4. Executor
    print("\n--- Executing Action ---")
    try:
        result = executor.execute(action, state)
        print(f"Result Valid: {result.get('valid')}")
        print(f"Result Error: {result.get('error')}")
        print(f"Full Result: {result}")
    except Exception as e:
        print(f"Executor Crashed: {e}")

if __name__ == "__main__":
    reproduce()
