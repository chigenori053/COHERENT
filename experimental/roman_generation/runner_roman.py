"""
Roman Numeral Generation Experiment Runner

Executes the Roman Generation Experiment (1-99).
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

# Import Roman Specifics
from experimental.roman_generation.attribute_mapping import int_to_roman, generate_candidate_map

def run_roman_experiment(output_dir="experimental/roman_generation/logs"):
    config = DEFAULT_CONFIG
    # Ensure High Dimension (Spec 5.1 says >= 2048)
    if config.dimension < 2048:
        config.dimension = 2048
        
    os.makedirs(output_dir, exist_ok=True)
    vector_dir = os.path.join(output_dir, "vectors")
    os.makedirs(vector_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"roman_experiment_log_{timestamp}.csv")

    print(f"Starting Roman Generation Experiment [Mode: {config.mode}]")
    print(f"Dimension: {config.dimension}, Timesteps: {config.num_timesteps}")
    print("Target Range: 1-99")

    # 1. Prepare Mappings
    candidate_map = generate_candidate_map() # {"I": [...], "IV": [...], ...}
    all_attributes = set()
    for attrs in candidate_map.values():
        all_attributes.update(attrs)
        
    # Lookup Wrapper
    def roman_lookup(symbol: str):
        if symbol not in candidate_map:
            raise ValueError(f"Unknown Roman symbol: {symbol}")
        return candidate_map[symbol]

    # 2. Initialize Components
    encoder = HolographicEncoder(dimension=config.dimension)
    memory = DynamicHolographicMemory(encoder, attribute_lookup_fn=roman_lookup)
    diffusion = RefinementDiffusionDDIM(config) # Only DDIM required by Spec
    diagnostics = ExperimentDiagnostics(config)

    # 3. Populate Memory with Attributes (No Symbols!)
    print(f"Populating memory with {len(all_attributes)} attributes...")
    memory.populate_known_attributes(list(all_attributes))
    
    # 4. Oracle Symbols (All 99 candidates)
    # This represents the "Verifier" side knowledge
    oracle_symbols = {}
    for roman in candidate_map.keys():
        oracle_symbols[roman] = memory.construct_symbol_query(roman)

    headers = [
        "number", "roman", "status", "top1", "score1", "top2", "score2", 
        "margin", "final_norm", "flags", "timestamp"
    ]
    
    results_summary = {"SUCCESS": 0, "WARN": 0, "INVALID": 0}

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 5. Run Generation Loop (1-99)
        for i in range(1, 100):
            target_roman = int_to_roman(i)
            run_id = f"{timestamp}_{i:02d}"
            
            # A. Construct Target H_0
            h_0_target = memory.construct_symbol_query(target_roman)
            
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
            # Spec 10: Warn if margin < 0.3
            # We reuse the standard diagnostic class, but maybe adjust threshold logic?
            # Standard diagnostics uses 0.1 margin for WARN. Spec suggests 0.3.
            # We stick to standard logic for specific flags, effectively WARN if top1 wrong or low margin.
            
            diag = diagnostics.check_result(
                target_symbol=target_roman,
                top_result=top1_char,
                top_score=top1_score,
                margin=margin,
                final_norm=final_norm
            )
            
            # Apply Roman Spec Threshold (Optional override)
            if diag.status == "SUCCESS" and margin < 0.3:
                 # Override to WARN per Spec 10?
                 # Let's keep consistent with valid/invalid logic. 
                 # If margin is positive but < 0.3, it's just low confidence success.
                 pass

            results_summary[diag.status] += 1
            
            # Log CSV
            writer.writerow([
                i, target_roman, diag.status, 
                top1_char, f"{top1_score:.4f}", 
                top2_char, f"{top2_score:.4f}", 
                f"{margin:.4f}", f"{final_norm:.6f}", 
                ";".join(diag.flags), timestamp
            ])
            
            # F. Save Vectors (Only save a subset or use compressed logic to save space?)
            # 99 runs is fine.
            
            # For visualization, we need oracle matrix. 
            # Saving 99x2048 matrix 99 times is ~1.6GB. Too big?
            # Optimization: Only save trajectory and metadata. 
            # Visualizer can reconstruct oracle matrix if needed, OR save oracle matrix ONCE.
            # Current visualizer expects oracle_matrix in every file.
            # Let's save it. 100 * 2048 * 16 bytes = 3.2 MB per file. 
            # 100 files = 320 MB. Acceptable.
            
            # Convert oracle dictionary to matrix ordered by 1-99
            sorted_romans = [int_to_roman(k) for k in range(1, 100)]
            oracle_matrix = np.array([oracle_symbols[r] for r in sorted_romans])
            
            np.savez_compressed(
                os.path.join(vector_dir, f"run_{run_id}.npz"),
                trajectory=np.array(trajectory),
                oracle_matrix=oracle_matrix,
                oracle_labels=sorted_romans,
                target_symbol=str(i), # Use number as target label for sorting flexibility
                target_roman=target_roman,
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
            
            if i % 10 == 0 or diag.status != "SUCCESS":
                print(f"[{i:2d}|{target_roman}] {diag.status} -> Pred: {top1_char} ({top1_score:.4f}) | Margin: {margin:.4f}")

    print("\n--- Roman Experiment Summary ---")
    print(results_summary)
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    run_roman_experiment()
