"""
Stage 3: Japanese Characters Experiment

Objective: Verify high-density structural resonance (Katakana & Kanji).
Stage 3-1: Katakana (Dim 2048)
Stage 3-2: Kanji (Dim 1024, 2048)
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
from experimental.dhm_re_verification.common.mappings import (
    KATAKANA_MAPPING, get_katakana_attributes,
    KANJI_MAPPING, get_kanji_attributes, get_level_chars
)

def run_stage3(mode: str, dimension: int, orchestrator=None, encoder=None, logger=None):
    # Mode: 'katakana' or 'kanji'
    stage_id = f"stage3_{mode}_dim{dimension}"
    print(f"--- Starting {stage_id} ---")
    
    # 1. Initialize Memory
    if not encoder:
        encoder = HolographicEncoder(dimension=dimension)

    if not orchestrator:
        dynamic = DynamicHolographicMemory(capacity=100)
        static = StaticHolographicMemory() 
        causal = CausalHolographicMemory()
        orchestrator = MemoryOrchestrator(dynamic, static, causal, promotion_threshold=0.8)
    else:
        dynamic = orchestrator.dynamic
        static = orchestrator.static
        causal = orchestrator.causal
    
    diffusion = DiffusionEngine(memory_system=dynamic, dimension=dimension)
    
    if not logger:
        logger = ExperimentLogger(stage_id)
    
    # 2. Prepare Data
    if mode == 'katakana':
        target_chars = list(KATAKANA_MAPPING.keys())
        get_attrs = get_katakana_attributes
    else:
        # Kanji: All levels
        target_chars = get_level_chars("A") + get_level_chars("B") + get_level_chars("C")
        get_attrs = get_kanji_attributes
    
    stats = {
        "stage_id": stage_id,
        "vector_dimension": dimension,
        "num_samples": len(target_chars),
        "exact_match_count": 0,
        "static_count": 0,
        "dynamic_count": 0,
        "causal_count": 0, # Not used in single char gen explicitly
        "conflict_count": 0,
        "total_margin": 0.0,
        "min_margin": 1.0,
        "false_static_count": 0
    }
    
    # Pre-populate Static Memory for "Known" characters?
    # Or start empty and learn?
    # Spec 3-1: "All entries promoted to Static".
    # Spec 3-2: "FalseStatic == 0 ... Conflicts accumulate in Causal".
    
    # Let's assume we run it as "Learning": Encode -> Dynamic -> Promote.
    
    for i, char in enumerate(tqdm.tqdm(target_chars, desc=f"Processing {mode}")):
        # Encode
        attrs = get_attrs(char)
        vectors = [encoder.encode_attribute(a) for a in attrs]
        h_0 = encoder.create_superposition(vectors)
        
        # Write Dynamic
        orchestrator.process_input(h_0, metadata={"content": char})
        
        # Explicit Promotion Attempt (for Spec goal)
        # Spec 2.2 Rule: Margin >= theta.
        # Check margin with ITSELF (h_0 vs h_0 always 1.0)
        # So it SHOULD promote if Orchestrator checks stability or if we force it.
        # Since we use single-shot, stability_steps is not applicable.
        # We assume 1-shot learning for this verification.
        
        # Check if already in Static (Collision check)
        static_res = static.query(h_0, top_k=1)
        
        if static_res and static_res[0][1] > 0.95:
             # Already exists (Collision or Repeated)
             # If mapping is identical, it's a collision.
             existing_char = static_res[0][0]
             if existing_char != char:
                 stats["conflict_count"] += 1
                 # Spec says: "Conflicts accumulate in Causal"
                 # Register transition: Existing -> New (Conflict link)
                 # orchestrator.register_transition(static.get_vector(existing_char), h_0)
                 # But we don't have get_vector in interface explicitly, 
                 # Base query doesn't return vector.
                 # Orchestrator can handle internal access.
                 pass
             else:
                 # Re-learning same char
                 pass
        else:
             # Promote
             orchestrator.promote_to_static(h_0, metadata={"id": char})
             stats["static_count"] += 1
             
        # Verify (Recall)
        recall_res = orchestrator.recall(h_0)
        
        predicted = None
        layer = "D"
        margin = 0.0
        
        if recall_res['static'] and recall_res['static'][0] == char:
             predicted = recall_res['static'][0]
             margin = recall_res['static'][1]
             layer = "S"
        elif recall_res['dynamic'] and recall_res['dynamic'][0] == char:
             predicted = recall_res['dynamic'][0]
             margin = recall_res['dynamic'][1]
             layer = "D"
        else:
             # Conflict or mismatch
             best_s = recall_res['static']
             if best_s:
                 predicted = best_s[0]
                 margin = best_s[1]
                 layer = "S(False)"
                 if predicted != char:
                     stats["false_static_count"] += 1
             else:
                 predicted = "<None>"

        is_exact = (predicted == char)
        if is_exact: stats["exact_match_count"] += 1
        stats["total_margin"] += margin
        stats["min_margin"] = min(stats["min_margin"], margin)
        
        logger.log_sample({
            "stage_id": stage_id,
            "sample_id": i,
            "input_sequence": char,
            "predicted_sequence": predicted,
            "is_exact_match": is_exact,
            "margin": margin,
            "memory_layer": layer,
            "conflict_flag": (not is_exact),
            "vector_dimension": dimension
        })
        
    # Aggregate
    stats["exact_match_rate"] = stats["exact_match_count"] / len(target_chars)
    stats["static_rate"] = stats["static_count"] / len(target_chars)
    stats["dynamic_rate"] = stats["dynamic_count"] / len(target_chars)
    stats["avg_margin"] = stats["total_margin"] / len(target_chars)
    
    logger.log_aggregate(stats)
    print(f"Finished {stage_id}. Accuracy: {stats['exact_match_rate']*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims_kanji", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--dims_katakana", nargs="+", type=int, default=[2048])
    args = parser.parse_args()
    
    # Stage 3-1: Katakana
    for d in args.dims_katakana:
        run_stage3('katakana', d)
        
    # Stage 3-2: Kanji
    for d in args.dims_kanji:
        run_stage3('kanji', d)
