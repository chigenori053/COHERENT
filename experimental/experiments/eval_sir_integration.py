
import sys
import os
import logging

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experimental.sandbox.sir.converter import SIRFactory

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SIR_Eval")

def run_verification():
    print("=== SIR v1.0 Integration Verification ===\n")
    
    # Test 1: Commutativity (3 + x vs x + 3)
    logger.info("Test 1: Commutativity (3 + x vs x + 3)")
    sir1 = SIRFactory.from_math_expression("3 + x")
    sir2 = SIRFactory.from_math_expression("x + 3")
    
    hash1 = sir1.structure_signature.graph_hash
    hash2 = sir2.structure_signature.graph_hash
    
    print(f"  '3 + x' Hash: {hash1}")
    print(f"  'x + 3' Hash: {hash2}")
    
    if hash1 == hash2:
        print("  ✅ PASS: Hashes match (Order Independent)")
    else:
        print("  ❌ FAIL: Hashes differ")

    # Test 2: Label Independence (Abstraction) - (a + b vs x + y)
    logger.info("\nTest 2: Label Independence (a + b vs x + y)")
    sir3 = SIRFactory.from_math_expression("a + b")
    sir4 = SIRFactory.from_math_expression("x + y")
    
    hash3 = sir3.structure_signature.graph_hash
    hash4 = sir4.structure_signature.graph_hash
    
    print(f"  'a + b' Hash: {hash3}")
    print(f"  'x + y' Hash: {hash4}")
    
    if hash3 == hash4:
        print("  ✅ PASS: Hashes match (Label Independent)")
    else:
        print("  ❌ FAIL: Hashes differ")
        
    # Test 3: Structural Sensitivity (a + b vs a * b)
    # Note: SIRFactory naive parser currently defaults to '+' for plus. 
    # We need to manually create multiplication or extend factory?
    # Factory naive parser handles '+' hardcoded. 
    # Let's inspect sir3 ("a + b") vs a manual construction of multiply?
    # Or just "a > b" (Comparison) vs "a + b" (Operation).
    
    logger.info("\nTest 3: Structure Sensitivity (a + b vs a > b)")
    sir_add = sir3
    sir_comp = SIRFactory.from_math_expression("a > b")
    
    hash_add = sir_add.structure_signature.graph_hash
    hash_comp = sir_comp.structure_signature.graph_hash
    
    print(f"  'a + b' Hash: {hash_add}")
    print(f"  'a > b' Hash: {hash_comp}")
    
    if hash_add != hash_comp:
        print("  ✅ PASS: Hashes differ (Structure Sensitive)")
    else:
        print("  ❌ FAIL: Hashes Identical (Collision?)")
    
    # --- Vector Projection Verification ---
    logger.info("\nTest 4: Vector Projection (Structure -> Vector)")
    from experimental.sandbox.sir.projection import SIRProjector
    import numpy as np
    
    projector = SIRProjector(dimension=1024)
    
    vec1 = projector.project(sir1) # 3 + x
    vec2 = projector.project(sir2) # x + 3
    
    dist12 = np.linalg.norm(vec1 - vec2)
    print(f"  Dist(3+x, x+3): {dist12}")
    
    if dist12 < 1e-9:
        print("  ✅ PASS: Vectors Identical (Commutative Projection)")
    else:
        print("  ❌ FAIL: Vectors Differ")

    vec3 = projector.project(sir3) # a + b
    vec_comp = projector.project(sir_comp) # a > b
    
    dist3c = np.linalg.norm(vec3 - vec_comp)
    print(f"  Dist(a+b, a>b): {dist3c}")
    
    if dist3c > 1.0:
        print("  ✅ PASS: Vectors Differ (Structure Sensitive)")
    else:
        print("  ❌ FAIL: Vectors Too Similar")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    run_verification()
