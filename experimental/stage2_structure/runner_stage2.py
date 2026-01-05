"""
Stage 2 Experiment Runner
"""

import os
import csv
import numpy as np
import json
from datetime import datetime

from experimental.stage2_structure.dataset_gen import Stage2DatasetGenerator, SYMBOL_TYPE_MAP
from experimental.stage2_structure.encoder import Stage2Encoder

def get_structure_signature(seq):
    return [SYMBOL_TYPE_MAP.get(c, 'UNKNOWN') for c in seq]

def run_stage2_experiment():
    base_dir = "experimental/stage2_structure"
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(log_dir, f"stage2_results_{timestamp}.csv")
    
    # Config
    # We skip full diffusion loop for speed, as Spec 5.3 says "Oracle usage permitted... for geometric validation"
    # and main hypothesis is about "Separability" (Geometric property).
    # We will test the "H0" directly or minimal noise?
    # Spec 5 defines Diffusion. But to validate hundreds of samples quickly, 
    # we can test H0 vs Candidates (Resonance).
    # If H0 (Clean) fails, H_t (Noisy) will definitely fail.
    # Let's test H0 (Clean State) to see if binding allows separation.
    
    lengths = [2, 3, 4]
    SAMPLES_PER_L = 100
    
    encoder = Stage2Encoder(dimension=2048)
    ds_gen = Stage2DatasetGenerator(seed=123)
    
    headers = [
        "length", "target", "predicted", "match", "margin", 
        "error_type", "top5_scores"
    ]
    
    stats = {
        "total": 0, "exact_match": 0, 
        "struct_confusion": 0, "cat_confusion": 0
    }
    
    print("Starting Stage 2 Experiment...")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for L in lengths:
            print(f"Testing Length {L}...")
            for _ in range(SAMPLES_PER_L):
                target = ds_gen.generate_target(L)
                target_str = "".join(target)
                
                # 1. Encode Target H0
                h_target = encoder.encode_sequence(target)
                
                # 2. Generate Candidate Set
                candidates = ds_gen.generate_candidate_set(target, size=50)
                
                # 3. Resonance Decoding
                best_cand = None
                best_score = -1.0
                second_score = -1.0
                scores = []
                
                for cand in candidates:
                    h_cand = encoder.encode_sequence(cand)
                    # Resonance
                    score = np.abs(np.vdot(h_target, h_cand))
                    
                    scores.append(("".join(cand), score))
                    
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_cand = cand
                    elif score > second_score:
                        second_score = score
                
                # Sort for logging
                scores.sort(key=lambda x: x[1], reverse=True)
                top5 = [f"{s}:{v:.4f}" for s, v in scores[:5]]
                
                # 4. Evaluation
                pred_str = "".join(best_cand)
                match = (pred_str == target_str)
                margin = best_score - second_score
                
                error_type = "NONE"
                if not match:
                    # Analyze Error
                    if sorted(best_cand) == sorted(target):
                        error_type = "STRUCTURAL_CONFUSION"
                        stats["struct_confusion"] += 1
                    elif get_structure_signature(best_cand) == get_structure_signature(target):
                        error_type = "CATEGORY_CONFUSION"
                        stats["cat_confusion"] += 1
                    else:
                        error_type = "OTHER"
                else:
                    stats["exact_match"] += 1
                
                stats["total"] += 1
                
                writer.writerow([
                    L, target_str, pred_str, 
                    1 if match else 0, f"{margin:.6f}", 
                    error_type, ";".join(top5)
                ])

    print("\n--- Summary ---")
    print(f"Total: {stats['total']}")
    print(f"Exact Match: {stats['exact_match']} ({stats['exact_match']/stats['total']:.2%})")
    print(f"Structural Confusion: {stats['struct_confusion']}")
    print(f"Category Confusion: {stats['cat_confusion']}")
    print(f"Log: {csv_file}")

if __name__ == "__main__":
    run_stage2_experiment()
