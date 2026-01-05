"""
Stage 1 Experiment Runner

Executes Sequence Recall & Generation Experiment (S1-A).
Focus: Sequence decoding from Holographic Representation.
"""

import os
import csv
import numpy as np
import random
from datetime import datetime
# from Levenshtein import distance as levenshtein_distance

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

from experimental.stage1_sequence.dataset_gen import DatasetGenerator
from experimental.stage1_sequence.sequence_encoder import SequenceEncoder

# Spec 8 Acceptance
PASS_THRESHOLDS = {
    2: {"exact": 0.99, "near": 0.02},
    3: {"exact": 0.97, "near": 0.03},
    4: {"exact": 0.95, "near": 0.03},
    5: {"exact": 0.90, "near": 0.05}
}

def run_stage1_experiment():
    # Setup
    base_dir = "experimental/stage1_sequence"
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(log_dir, f"stage1_results_{timestamp}.csv")
    
    # Variants (Spec 9)
    # We execute Dimension variants. use_fft is fixed to True for now as discussed.
    dimensions = [256, 512, 1024, 2048] # Added 2048 for safety
    lengths = [2, 3, 4, 5]
    
    # Prepare CSV Header
    headers = [
        "run_id", "timestamp", "variant_dim", "length_L",
        "seq_id", "target_seq", "pred_seq",
        "exact_match", "edit_dist", "near_miss",
        "avg_margin", "min_margin", "valid_margin_pct",
        "status" # PASS/FAIL based on edit dist
    ]
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Experiment Loop
        for dim in dimensions:
            print(f"--- Running Variant: Dimension D={dim} ---")
            
            encoder = SequenceEncoder(dimension=dim, use_fft=True)
            ds_gen = DatasetGenerator(seed=42)
            
            for L in lengths:
                print(f"  Length L={L} ...", end="", flush=True)
                
                # 1. Generate Dataset
                dataset = ds_gen.generate_dataset(length=L, count=1000)
                
                # Metrics for this Length/Dim block
                block_exact_matches = 0
                block_near_misses = 0
                
                for seq_id, target_seq in dataset:
                    # 2. Register (Encode)
                    # H = Encode(Target)
                    h_seq = encoder.encode_sequence(target_seq)
                    
                    # 3. Query & Decode (Generation Phase)
                    # We pass h_seq directly (Perfect Recall assumption)
                    # But we decoding involves position query
                    
                    decoded_chars = []
                    margins = []
                    
                    # Decode loop
                    for i in range(L):
                        # Unbind Position
                        h_pos = encoder.get_position_vector(i)
                        # conj only works if we assume unitary, but normalized vectors are not strictly unitary matrices.
                        # However for HRR, unbinding is binding with inverse.
                        # Approximate inverse is involution (e.g. for complex: conj).
                        h_query = h_seq * np.conjugate(h_pos)
                        
                        # Nearest Neighbor
                        best_char = "?"
                        best_score = -999.0
                        second_score = -999.0
                        correct_score = -999.0 # Score of the target char at this pos
                        
                        target_char_at_pos = target_seq[i]
                        
                        # Search Symbol Space A-Z
                        for char_code in range(ord('A'), ord('Z')+1):
                            c = chr(char_code)
                            h_char = encoder.encode_char(c)
                            
                            # Resonance
                            # Real part of Dot Product for Normalized Complex Vectors
                            score = np.real(np.vdot(h_query, h_char))
                            
                            if c == target_char_at_pos:
                                correct_score = score
                            
                            if score > best_score:
                                second_score = best_score
                                best_score = score
                                best_char = c
                            elif score > second_score:
                                second_score = score
                                
                        decoded_chars.append(best_char)
                        
                        # Margin: Best - Second Best
                        # Spec 7.3 Separation Margin = Score Positive - Score Negative Max
                        # Note: If prediction is wrong, "Score Positive" (correct char) < "Score Negative Max" (wrong char)
                        # So margin < 0.
                        margin = correct_score - (best_score if best_char != target_char_at_pos else second_score)
                        margins.append(margin)
                    
                    pred_seq = "".join(decoded_chars)
                    
                    # 4. Evaluate
                    exact_match = (pred_seq == target_seq)
                    edit_dist = levenshtein_distance(pred_seq, target_seq)
                    near_miss = (edit_dist == 1)
                    
                    if exact_match:
                        block_exact_matches += 1
                    if near_miss:
                        block_near_misses += 1
                        
                    avg_margin = np.mean(margins)
                    min_margin = np.min(margins)
                    valid_margin_pct = sum(1 for m in margins if m > 0) / L
                    
                    # 5. Log Row
                    writer.writerow([
                        f"RUN_{dim}_{L}_{seq_id}", timestamp, dim, L,
                        seq_id, target_seq, pred_seq,
                        1 if exact_match else 0, edit_dist, 1 if near_miss else 0,
                        f"{avg_margin:.6f}", f"{min_margin:.6f}", f"{valid_margin_pct:.2f}",
                        "PASS" if exact_match else "FAIL"
                    ])
                    
                # Summary for Console
                acc = block_exact_matches / 1000.0
                print(f" Acc: {acc:.1%} ({block_exact_matches}/1000)")

    print(f"\nExperiment Complete. Results saved to {csv_filename}")

if __name__ == "__main__":
    run_stage1_experiment()
