"""
Digit Generation Experiment Runner

Executes the Digit Generation Experiment (0-9).
"""

import os
import time
import csv
import json
import numpy as np
from datetime import datetime

# Reuse Core DHM Components
from experimental.dhm_generation.config import DEFAULT_CONFIG
from experimental.dhm_generation.memory.attribute_hologram import HolographicEncoder
from experimental.dhm_generation.memory.dynamic_memory import DynamicHolographicMemory
from experimental.dhm_generation.diffusion.ddim import RefinementDiffusionDDIM
from experimental.dhm_generation.diffusion.ddpm import RefinementDiffusionDDPM
from experimental.dhm_generation.evaluation.resonance import evaluate_symbol_match
from experimental.dhm_generation.evaluation.diagnostics import ExperimentDiagnostics

# Import Digit Specifics
from experimental.digit_generation.attribute_mapping import DIGIT_MAPPING, get_attributes_for_digit

def run_digit_experiment(output_dir="experimental/digit_generation/logs"):
    config = DEFAULT_CONFIG
    # Ensure High Dimension for Digits (Spec 6.1)
    if config.dimension < 2048:
        config.dimension = 2048
        
    os.makedirs(output_dir, exist_ok=True)
    vector_dir = os.path.join(output_dir, "vectors")
    os.makedirs(vector_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"digit_experiment_log_{timestamp}.csv")

    print(f"Starting Digit Generation Experiment [Mode: {config.mode}]")
    print(f"Dimension: {config.dimension}, Timesteps: {config.num_timesteps}")

    # 1. Initialize Components with Digit Mapping
    encoder = HolographicEncoder(dimension=config.dimension)
    
    # Inject digit attribute lookup function
    memory = DynamicHolographicMemory(encoder, attribute_lookup_fn=get_attributes_for_digit)
    
    if config.mode == "ddim":
        diffusion = RefinementDiffusionDDIM(config)
    else:
        diffusion = RefinementDiffusionDDPM(config)
        
    diagnostics = ExperimentDiagnostics(config)

    # 2. Populate Memory with Attributes (No Symbols!)
    all_attributes = set()
    for attrs in DIGIT_MAPPING.values():
        all_attributes.update(attrs.values())
    
    print(f"Populating memory with {len(all_attributes)} attributes...")
    memory.populate_known_attributes(list(all_attributes))
    
    # 3. Oracle Symbols
    oracle_symbols = {}
    for char in DIGIT_MAPPING.keys():
        oracle_symbols[char] = memory.construct_symbol_query(char)

    headers = [
        "symbol", "mode", "status", "top1", "score1", "top2", "score2", 
        "margin", "final_norm", "flags", "timestamp"
    ]
    
    results_summary = {"SUCCESS": 0, "WARN": 0, "INVALID": 0}

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 4. Run Generation Loop
        # Sort numerically 0-9
        sorted_digits = sorted(DIGIT_MAPPING.keys(), key=lambda x: int(x))
        
        for target_digit in sorted_digits:
            run_id = f"{timestamp}_{target_digit}"
            
            # A. Construct Target H_0
            h_0_target = memory.construct_symbol_query(target_digit)
            
            trajectory = []
            
            # B. Forward Diffusion -> H_T (Noisy)
            h_t = diffusion.add_noise(h_0_target, config.num_timesteps - 1)
            
            current_h = h_t
            trajectory.append(current_h)
            
            # C. Reverse Diffusion (Refinement)
            for t in reversed(range(config.num_timesteps)):
                current_h = diffusion.step(current_h, t, h_0_target)
                trajectory.append(current_h)
            
            h_final = current_h
            final_norm = np.linalg.norm(h_final)

            # D. Evaluation
            matches = evaluate_symbol_match(h_final, oracle_symbols)
            top1_char, top1_score = matches[0]
            top2_char, top2_score = matches[1] if len(matches) > 1 else ("N/A", 0.0)
            
            margin = top1_score - top2_score

            # E. Diagnostics
            diag = diagnostics.check_result(
                target_symbol=target_digit,
                top_result=top1_char,
                top_score=top1_score,
                margin=margin,
                final_norm=final_norm
            )
            
            results_summary[diag.status] += 1
            
            # Log CSV
            writer.writerow([
                target_digit, config.mode, diag.status, 
                top1_char, f"{top1_score:.4f}", 
                top2_char, f"{top2_score:.4f}", 
                f"{margin:.4f}", f"{final_norm:.6f}", 
                ";".join(diag.flags), timestamp
            ])
            
            # F. Save Vectors
            oracle_matrix = np.array([oracle_symbols[c] for c in sorted_digits])
            oracle_labels = sorted_digits
            
            traj_array = np.array(trajectory)
            
            np.savez_compressed(
                os.path.join(vector_dir, f"run_{run_id}.npz"),
                trajectory=traj_array,
                oracle_matrix=oracle_matrix,
                oracle_labels=oracle_labels,
                target_symbol=target_digit,
                final_symbol=top1_char,
                metadata={
                    "mode": config.mode,
                    "dimension": config.dimension,
                    "timesteps": config.num_timesteps,
                    "status": diag.status,
                    "margin": margin,
                    "score": top1_score
                }
            )
            
            print(f"[{target_digit}] {diag.status} -> Pred: {top1_char} ({top1_score:.4f}) | Margin: {margin:.4f}")

    print("\n--- Digit Experiment Summary ---")
    print(results_summary)
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    run_digit_experiment()
