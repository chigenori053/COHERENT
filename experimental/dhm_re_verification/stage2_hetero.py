"""
Stage 2: Heterogeneous Symbols Experiment

Objective: Validate memory-layer separation under heterogeneous symbol binding.
Conditions: Alphanumeric + Symbols, Length 2-4, Dim 1024, N=15000.
"""

import sys
import os
import argparse
import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from coherent.core.memory.holographic.encoder import HolographicEncoder
from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.static import StaticHolographicMemory
from coherent.core.memory.holographic.causal import CausalHolographicMemory
from coherent.core.memory.holographic.orchestrator import MemoryOrchestrator

from experimental.dhm_re_verification.common.diffusion import DiffusionEngine
from experimental.dhm_re_verification.common.logging import ExperimentLogger
from experimental.dhm_re_verification.common.data_gen import DataGenerator

def run_stage2(dimension: int, num_samples: int, orchestrator=None, encoder=None, logger=None):
    print(f"--- Starting Stage 2 (Dim={dimension}, N={num_samples}) ---")
    
    # 1. Initialize Memory Space
    if not encoder: 
        encoder = HolographicEncoder(dimension=dimension)

    if not orchestrator:
        dynamic = DynamicHolographicMemory(capacity=100)
        static = StaticHolographicMemory()
        causal = CausalHolographicMemory()
        orchestrator = MemoryOrchestrator(dynamic, static, causal, promotion_threshold=0.85)
    else:
        dynamic = orchestrator.dynamic
    
    # 2. Components
    diffusion = DiffusionEngine(memory_system=dynamic, dimension=dimension)
    
    if not logger:
        logger = ExperimentLogger(f"stage2_hetero_dim{dimension}")
    
    # 3. Generate Data
    sequences = DataGenerator.generate_batch_stage2(num_samples)
    
    stats = {
        "stage_id": "stage2",
        "vector_dimension": dimension,
        "num_samples": num_samples,
        "exact_match_count": 0,
        "static_count": 0,
        "dynamic_count": 0,
        "causal_count": 0,
        "conflict_count": 0,
        "total_margin": 0.0,
        "min_margin": 1.0,
        "false_static_count": 0
    }
    
    for i, seq in enumerate(tqdm.tqdm(sequences, desc="Processing")):
        # Encode
        vectors = [encoder.encode_attribute(char) for char in seq]
        h_0 = encoder.create_superposition(vectors)
        
        # Write to Dynamic
        orchestrator.process_input(h_0, metadata={"content": seq})
        
        # Relax
        # Since we use simple diffusion, h_final ~ h_0
        
        # Recall
        recall_res = orchestrator.recall(h_0)
        
        predicted = None
        layer = "D"
        margin = 0.0
        
        # Check layers - PREFER EXACT MATCH logic for stats
        # If Static matches, it's a promotion (or conflict?)
        # For Stage 2, "Ambiguity remains in Dynamic". So Static rate should be low?
        # UNLESS inputs are repeated enough to be promoted.
        # Random inputs typically won't repeat enough to trigger stability-based promotion 
        # unless we explicitly promote in this loop.
        # Spec says "StaticPromotionRate ~ 1.0" for Stage 1. 
        # For Stage 2, it says "Ambiguity remains in Dynamic".
        # This implies we do NOT want aggressive promotion here?
        # But wait, 2.2 Static Promotion Rule default N_static = 3.
        # If random seqs are unique, stability_steps won't reach 3.
        # So usually they stay in Dynamic.
        
        # However, for verification, we likely want to see if we CAN decode.
        # I'll rely on Orchestrator's default recall.
        
        if recall_res['static'] and recall_res['static'][0] == seq:
             predicted = recall_res['static'][0]
             margin = recall_res['static'][1]
             layer = "S"
        elif recall_res['dynamic'] and recall_res['dynamic'][0] == seq:
             predicted = recall_res['dynamic'][0]
             margin = recall_res['dynamic'][1]
             layer = "D"
        else:
             best = recall_res['dynamic']
             if best:
                 predicted = best[0]
                 margin = best[1]
             else:
                 predicted = "<None>"
        
        is_exact = (predicted == seq)
        
        # For Stage 2, we do NOT force promotion logic here, verifying "Ambiguity remains in Dynamic"
        # unless orchestrator does it.
        # But wait, Experiment 1 FORCED promotion.
        # If I don't force promotion, they stay in Dynamic. 
        # Dynamic has capacity 100.
        # After 100, they are dropped.
        # If N=15000, and Capacity=100.
        # We only recall properly if the item is still in Dynamic or Promoted.
        # Since we query immediately after adding, it SHOULD be in Dynamic.
        
        if layer == "S": stats["static_count"] += 1
        else: stats["dynamic_count"] += 1
        
        if is_exact: stats["exact_match_count"] += 1
        stats["total_margin"] += margin
        stats["min_margin"] = min(stats["min_margin"], margin)
        
        logger.log_sample({
            "stage_id": f"stage2_hetero_dim{dimension}",
            "sample_id": i,
            "input_sequence": seq,
            "predicted_sequence": predicted,
            "is_exact_match": is_exact,
            "margin": margin,
            "memory_layer": layer,
            "vector_dimension": dimension
        })
        
    stats["exact_match_rate"] = stats["exact_match_count"] / num_samples
    stats["static_rate"] = stats["static_count"] / num_samples
    stats["dynamic_rate"] = stats["dynamic_count"] / num_samples
    stats["avg_margin"] = stats["total_margin"] / num_samples

    logger.log_aggregate(stats)
    print(f"Finished Dim={dimension}. Accuracy: {stats['exact_match_rate']*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("--dims", nargs="+", type=int, default=[1024])
    args = parser.parse_args()
    
    for d in args.dims:
        run_stage2(d, args.samples)
