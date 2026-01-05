"""
Stage 3: Kanji Single Character Generation Experiment Runner

Executes the Kanji Generation Experiment across three complexity levels (A, B, C).
"""

import os
import csv
import numpy as np
from datetime import datetime
from collections import Counter

# Reuse Core DHM Components
from experimental.dhm_generation.config import DEFAULT_CONFIG
from experimental.dhm_generation.memory.attribute_hologram import HolographicEncoder
from experimental.dhm_generation.memory.dynamic_memory import DynamicHolographicMemory
from experimental.dhm_generation.diffusion.ddim import RefinementDiffusionDDIM
from experimental.dhm_generation.evaluation.resonance import evaluate_symbol_match
from experimental.dhm_generation.evaluation.diagnostics import ExperimentDiagnostics

# Import Kanji Specifics
from experimental.stage3_kanji_single_char.kanji_mapping import KANJI_MAPPING, get_attributes_for_kanji, get_level_chars

def run_kanji_experiment(output_dir="experimental/stage3_kanji_single_char/logs"):
    # Force Dimension 2048 as per User Request (Scale Up)
    config = DEFAULT_CONFIG
    config.dimension = 2048 
    
    os.makedirs(output_dir, exist_ok=True)
    vector_dir = os.path.join(output_dir, "vectors")
    os.makedirs(vector_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"stage3_kanji_results_{timestamp}.csv")

    print(f"Starting Stage 3 Kanji Experiment [Mode: {config.mode}]")
    print(f"Dimension: {config.dimension} (Constrained), Timesteps: {config.num_timesteps}")

    # 1. Initialize Components
    encoder = HolographicEncoder(dimension=config.dimension)
    memory = DynamicHolographicMemory(encoder, attribute_lookup_fn=get_attributes_for_kanji)
    diffusion = RefinementDiffusionDDIM(config)
    diagnostics = ExperimentDiagnostics(config)

    # 2. Populate Memory with ALL Kanji Attributes
    all_attributes = set()
    for attrs in KANJI_MAPPING.values():
        all_attributes.update(attrs.values())
    
    print(f"Populating memory with {len(all_attributes)} attributes...")
    memory.populate_known_attributes(list(all_attributes))
    
    # 3. Oracle Symbols (35 chars)
    all_chars = get_level_chars("ALL")
    oracle_symbols = {}
    for char in all_chars:
        oracle_symbols[char] = memory.construct_symbol_query(char)

    # Define levels for logging
    level_map = {}
    for c in get_level_chars("A"): level_map[c] = "A"
    for c in get_level_chars("B"): level_map[c] = "B"
    for c in get_level_chars("C"): level_map[c] = "C"

    headers = [
        "input_char", "level", "decision", 
        "top1_char", "score1", "top2_char", "score2", 
        "margin", "final_norm", "timestamp"
    ]
    
    # Decisions stats per level
    level_stats = {
        "A": {"ACCEPT":0, "REVIEW":0, "REJECT":0, "TOTAL":0},
        "B": {"ACCEPT":0, "REVIEW":0, "REJECT":0, "TOTAL":0},
        "C": {"ACCEPT":0, "REVIEW":0, "REJECT":0, "TOTAL":0}
    }

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 4. Run Loop
        for target_char in all_chars:
            level = level_map[target_char]
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

            # E. Decision Logic (Simple Thresholding for Experiment)
            # Accept > 0.1 margin & > 0.9 score?
            # Using defaults or Spec logic? 
            # Spec: "If success_case holds: top1_accuracy >= 0.95, review allowed..."
            # Let's categorize based on margin.
            decision = "REJECT"
            if top1_char == target_char:
                if margin > 0.1:
                    decision = "ACCEPT"
                elif margin > 0.02:
                    decision = "REVIEW"
                else:
                    decision = "REJECT" # Confusion
            else:
                decision = "REJECT"

            level_stats[level][decision] += 1
            level_stats[level]["TOTAL"] += 1
            
            writer.writerow([
                target_char, level, decision,
                top1_char, f"{top1_score:.4f}", 
                top2_char, f"{top2_score:.4f}", 
                f"{margin:.4f}", f"{final_norm:.6f}", 
                timestamp
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
                    "status": decision,
                    "margin": margin,
                    "level": level
                }
            )
            
            print(f"[{level}] {target_char} -> {decision} (Pred: {top1_char}, Margin: {margin:.4f})")

    print("\n--- Stage 3 Kanji Experiment Summary ---")
    for lvl in ["A", "B", "C"]:
        s = level_stats[lvl]
        acc = (s["ACCEPT"] + s["REVIEW"]) / s["TOTAL"] if s["TOTAL"] > 0 else 0
        print(f"Level {lvl}: ACC={acc:.2%} ({s})")
        
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    run_kanji_experiment()
