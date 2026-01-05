"""
Refactoring Verification Script

Demonstrates the 3-Layer Memory Architecture using a Stage 3 Kanji Example (林).
"""
import sys
import os
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from coherent.core.memory.holographic.encoder import HolographicEncoder
from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.static import StaticHolographicMemory
from coherent.core.memory.holographic.causal import CausalHolographicMemory
from coherent.core.memory.holographic.orchestrator import MemoryOrchestrator

def main():
    print("Initializing Memory Architecture...")
    encoder = HolographicEncoder(dimension=1024)
    dynamic = DynamicHolographicMemory(capacity=10)
    static = StaticHolographicMemory()
    causal = CausalHolographicMemory()
    
    orchestrator = MemoryOrchestrator(dynamic, static, causal, promotion_threshold=0.85)
    
    # Simulate a Kanji '林' (Forest)
    # Attributes: Structure=left_right, Component=TREE, Component=TREE
    print("\n[Step 1] Encoding Inputs...")
    attr_struct = encoder.encode_attribute("struct:left_right")
    attr_comp = encoder.encode_attribute("comp:TREE")
    
    # Create Kanji Vector (simplified binding)
    # Hayashi = left_right * TREE * TREE (normalized)
    vec_hayashi = encoder.create_superposition([attr_struct, attr_comp, attr_comp])
    
    # 2. Process Input (Dynamic)
    print("\n[Step 2] Processing Input (Dynamic Memory)...")
    orchestrator.process_input(vec_hayashi, metadata={"content": "Input_Pattern_X"})
    
    # Verify Dynamic
    res = dynamic.query(vec_hayashi, top_k=1)
    print(f"Dynamic Query Result: {res[0]}")
    assert res[0][1] > 0.99, "Failed to retrieve from Dynamic Memory"
    
    # 3. Promote to Static (Learning)
    print("\n[Step 3] Promoting to Static Memory (Learning '林')...")
    orchestrator.promote_to_static(vec_hayashi, metadata={"id": "KANJI_HAYASHI"})
    
    # Verify Static
    res_static = static.query(vec_hayashi, top_k=1)
    print(f"Static Query Result: {res_static[0]}")
    assert res_static[0][0] == "KANJI_HAYASHI"
    assert res_static[0][1] > 0.99, "Failed to retrieve from Static Memory"
    
    # 4. Recall
    print("\n[Step 4] Unified Recall...")
    recall_res = orchestrator.recall(vec_hayashi)
    print(f"Recall Result: {recall_res}")
    
    if recall_res['static'] and recall_res['static'][0] == "KANJI_HAYASHI":
        print("\nSUCCESS: '林' was correctly learned and recalled from Static Memory.")
    else:
        print("\nFAILURE: Unified recall did not return expected Static entry.")
        sys.exit(1)

if __name__ == "__main__":
    main()
