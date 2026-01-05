"""
Katakana Generation Experiment Runner

Executes the Katakana Generation Experiment (46 Seion).
"""

import os
import csv
import numpy as np
from datetime import datetime

# Reuse Core DHM Components
from experimental.dhm_generation.config import DEFAULT_CONFIG
from experimental.dhm_generation.memory.attribute_hologram import HolographicEncoder
from experimental.dhm_generation.memory.dynamic_memory import DynamicHolographicMemory
from experimental.dhm_generation.diffusion.ddim import RefinementDiffusionDDIM
from experimental.dhm_generation.evaluation.resonance import evaluate_symbol_match
from experimental.dhm_generation.evaluation.diagnostics import ExperimentDiagnostics

# Import Katakana Specifics
from experimental.katakana_generation.attribute_mapping import KATAKANA_MAPPING, get_attributes_for_katakana, get_all_chars

def run_katakana_experiment(output_dir="experimental/katakana_generation/logs"):
    config = DEFAULT_CONFIG
    if config.dimension < 2048:
        config.dimension = 2048
        
    os.makedirs(output_dir, exist_ok=True)
    vector_dir = os.path.join(output_dir, "vectors")
    os.makedirs(vector_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"katakana_experiment_log_{timestamp}.csv")

    print(f"Starting Katakana Generation Experiment [Mode: {config.mode}]")
    print(f"Dimension: {config.dimension}, Timesteps: {config.num_timesteps}")

    # 1. Initialize Components
    encoder = HolographicEncoder(dimension=config.dimension)
    memory = DynamicHolographicMemory(encoder, attribute_lookup_fn=get_attributes_for_katakana)
    diffusion = RefinementDiffusionDDIM(config)
    diagnostics = ExperimentDiagnostics(config)

    # 2. Populate Memory
    all_attributes = set()
    for attrs in KATAKANA_MAPPING.values():
        all_attributes.update(attrs.values())
    
    print(f"Populating memory with {len(all_attributes)} attributes...")
    memory.populate_known_attributes(list(all_attributes))
    
    # 3. Oracle Symbols (46 chars)
    all_chars = get_all_chars()
    oracle_symbols = {}
    for char in all_chars:
        oracle_symbols[char] = memory.construct_symbol_query(char)

    headers = [
        "char", "status", "top1", "score1", "top2", "score2", 
        "margin", "final_norm", "flags", "timestamp"
    ]
    
    results_summary = {"SUCCESS": 0, "WARN": 0, "INVALID": 0}

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 4. Run Loop
        for target_char in all_chars:
            # Generate a unique run ID using the timestamp and hex representation of the character
            run_id = f"{timestamp}_{ord(target_char):x}" 
            
            # A. Construct Target H_0
            h_0_target = memory.construct_symbol_query(target_char)
            
            trajectory = []
            
            # B. Forward Diffusion
            h_t = diffusion.add_noise(h_0_target, config.num_timesteps - 1)
            current_h = h_t
            trajectory.append(current_h)
            
            # C. Reverse Diffusion
            for t in reversed(range(config.num_timesteps)):
                current_h = diffusion.step(current_h, t, h_0_target)
                trajectory.append(current_h)
            
            h_final = current_h
            final_norm = np.linalg.norm(h_final)

            # D. Evaluation
            matches = evaluate_symbol_match(h_final, oracle_symbols)
            top1_char, top1_score = matches[0] if matches else ("N/A", 0.0)
            top2_char, top2_score = matches[1] if len(matches) > 1 else ("N/A", 0.0)
            
            margin = top1_score - top2_score

            # E. Diagnostics
            diag = diagnostics.check_result(
                target_symbol=target_char,
                top_result=top1_char,
                top_score=top1_score,
                margin=margin,
                final_norm=final_norm
            )
            
            results_summary[diag.status] += 1
            
            writer.writerow([
                target_char, diag.status, 
                top1_char, f"{top1_score:.4f}", 
                top2_char, f"{top2_score:.4f}", 
                f"{margin:.4f}", f"{final_norm:.6f}", 
                ";".join(diag.flags), timestamp
            ])
            
            # F. Save Vectors
            oracle_matrix = np.array([oracle_symbols[c] for c in all_chars])
            
            np.savez_compressed(
                os.path.join(vector_dir, f"run_{run_id}.npz"),
                trajectory=np.array(trajectory),
                oracle_matrix=oracle_matrix,
                oracle_labels=all_chars,
                target_symbol=target_char,
                final_symbol=top1_char,
                metadata={
                    "mode": config.mode,
                    "dimension": config.dimension,
                    "timesteps": config.num_timesteps,
                    "status": diag.status,
                    "margin": margin
                }
            )
            
            print(f"[{target_char}] {diag.status} -> Pred: {top1_char} ({top1_score:.4f}) | Margin: {margin:.4f}")

    print("\n--- Katakana Experiment Summary ---")
    print(results_summary)
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    run_katakana_experiment()
