"""
Experiment Runner

Main entry point for the Alphabet Generation Experiment.
Executes the generation pipeline, diffusion refinement, and evaluation.
"""

import os
import time
import csv
import numpy as np
from datetime import datetime

from .config import DEFAULT_CONFIG
from .memory.attribute_hologram import HolographicEncoder, ALPHABET_MAPPING
from .memory.dynamic_memory import DynamicHolographicMemory
from .diffusion.ddim import RefinementDiffusionDDIM
from .diffusion.ddpm import RefinementDiffusionDDPM
from .evaluation.resonance import evaluate_symbol_match
from .evaluation.diagnostics import ExperimentDiagnostics

def run_experiment(output_dir="experimental/dhm_generation/logs"):
    config = DEFAULT_CONFIG
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_log_{timestamp}.csv")

    print(f"Starting DHM Alphabet Experiment [Mode: {config.mode}]")
    print(f"Dimension: {config.dimension}, Timesteps: {config.num_timesteps}")

    # 1. Initialize Components
    encoder = HolographicEncoder(dimension=config.dimension)
    memory = DynamicHolographicMemory(encoder)
    
    # Select Diffusion Model
    if config.mode == "ddim":
        diffusion = RefinementDiffusionDDIM(config)
    else:
        diffusion = RefinementDiffusionDDPM(config)
        
    diagnostics = ExperimentDiagnostics(config)

    # 2. Populate Memory with Attributes (No Symbols!)
    # Gather all unique attributes
    all_attributes = set()
    for attrs in ALPHABET_MAPPING.values():
        all_attributes.update(attrs.values())
    
    print(f"Populating memory with {len(all_attributes)} attributes...")
    memory.populate_known_attributes(list(all_attributes))
    
    # 3. Pre-calculate "Ground Truth" Symbol Holograms only for EVALUATION
    # These are NOT stored in MemorySpace. They are the 'oracle' reference.
    oracle_symbols = {}
    for char in ALPHABET_MAPPING.keys():
        oracle_symbols[char] = memory.construct_symbol_query(char)

    # Prepare CSV Logging
    headers = [
        "symbol", "mode", "status", "top1", "score1", "top2", "score2", 
        "margin", "final_norm", "flags", "timestamp"
    ]
    
    results_summary = {"SUCCESS": 0, "WARN": 0, "INVALID": 0}

    import json

    # Config & Dirs
    vector_dir = os.path.join(output_dir, "vectors")
    os.makedirs(vector_dir, exist_ok=True)

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 4. Run Generation Loop
        for target_char in sorted(ALPHABET_MAPPING.keys()):
            run_id = f"{timestamp}_{target_char}"
            
            # A. Construct Target H_0 from Attributes
            h_0_target = memory.construct_symbol_query(target_char)
            
            # Track trajectory: [(t, h_vec)]
            trajectory = []
            
            # B. Forward Diffusion (Add Noise) -> H_T
            h_t = diffusion.add_noise(h_0_target, config.num_timesteps - 1)
            
            # C. Reverse Diffusion (Refinement)
            current_h = h_t
            
            # Record Initial Noisy State (T)
            trajectory.append(current_h)
            
            for t in reversed(range(config.num_timesteps)):
                current_h = diffusion.step(current_h, t, h_0_target)
                trajectory.append(current_h) # Appends refined states T-1 ... 0
            
            # trajectory[0] is H_T (noisy), trajectory[-1] is H_0 (refined/final)
            # Reorder to 0..T for logical plotting if needed, but diffusion goes T->0.
            # Let's verify spec: "H_0 -> H_1 ... -> H_T" usually means Forward.
            # But here we visualize the REFINEMENT process for generation.
            # Spec 2.2: "H_t for t ∈ {1 … T}".
            # We will save the RAW trajectory of the generation process (Reverse Diffusion).
            
            h_final = current_h
            final_norm = np.linalg.norm(h_final)

            # D. Evaluation
            matches = evaluate_symbol_match(h_final, oracle_symbols)
            top1_char, top1_score = matches[0]
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
            
            # Log CSV
            writer.writerow([
                target_char, config.mode, diag.status, 
                top1_char, f"{top1_score:.4f}", 
                top2_char, f"{top2_score:.4f}", 
                f"{margin:.4f}", f"{final_norm:.6f}", 
                ";".join(diag.flags), timestamp
            ])
            
            # F. Save Vector Logs (Spec 5, 8)
            # Save trajectory, oracle candidates, and metadata
            # Pack oracle symbols for geometry plot
            oracle_matrix = np.array([oracle_symbols[c] for c in sorted(oracle_symbols.keys())])
            oracle_labels = sorted(oracle_symbols.keys())
            
            # Trajectory array: Shape (T+1, D)
            traj_array = np.array(trajectory) 
            
            vector_save_path = os.path.join(vector_dir, f"run_{run_id}.npz")
            np.savez_compressed(
                vector_save_path,
                trajectory=traj_array,
                oracle_matrix=oracle_matrix,
                oracle_labels=oracle_labels,
                target_symbol=target_char,
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
            
            print(f"[{target_char}] {diag.status} -> Pred: {top1_char} ({top1_score:.4f}) | Margin: {margin:.4f}")

    print("\n--- Experiment Summary ---")
    print(results_summary)
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    run_experiment()
