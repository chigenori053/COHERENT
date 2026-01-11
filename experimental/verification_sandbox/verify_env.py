"""
Verification Sandbox Runner
Executes the environment logic: IR -> Project -> Validate.
"""

import sys
import os
import json

# Add parent path to allow importing modules in experimental/verification_sandbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from experimental.verification_sandbox.ir import *
from experimental.verification_sandbox.languages import PYTHON_SPEC, JAVA_SPEC, HASKELL_SPEC
from experimental.verification_sandbox.projection import ProjectionEngine
from experimental.verification_sandbox.validator import Validator

def run_verification():
    # 1. Define IR (Sample: Loop with conditional break)
    # Loop:
    #   if x >= 10: break
    #   x = x + 1
    ir_tree = Loop(
        condition="x < 10", # While x < 10
        body=Sequence(nodes=[
            Assignment(target="x", value="x + 1"),
            If(condition="x >= 10", body=Sequence([Exit()]))
        ])
    )
    
    engine = ProjectionEngine()
    validator = Validator()
    
    results = []
    
    specs = [PYTHON_SPEC, JAVA_SPEC, HASKELL_SPEC]
    
    print("--- Verification Sandbox Execution ---")
    
    for spec in specs:
        print(f"\nProcessing Language: {spec.id}")
        
        # Project
        code = engine.project(ir_tree, spec)
        print(f"Generated Code:\n---\n{code}\n---")
        
        # Validate
        val_res = validator.validate(code, spec)
        print(f"Validation: {val_res}")
        
        results.append({
            'spec': spec.id,
            'code': code,
            'validation': val_res
        })
        
    # Save Report
    report_path = os.path.join(os.path.dirname(__file__), '../reports/SANDBOX_VERIFICATION_LOG.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    run_verification()
