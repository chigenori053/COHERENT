"""
Master Experiment Runner (run_all.py)

Executes all experimentation stages in the strict order defined by the specification.
Ensures memory persistence (Static/Causal) across stages within the same dimension.
"""

import sys
import os
import argparse
from typing import List

# Add project root to path
sys.path.append(os.getcwd())

from coherent.core.memory.holographic.encoder import HolographicEncoder
from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.static import StaticHolographicMemory
from coherent.core.memory.holographic.causal import CausalHolographicMemory
from coherent.core.memory.holographic.orchestrator import MemoryOrchestrator

from experimental.dhm_re_verification.stage1_alphabet import run_stage1
from experimental.dhm_re_verification.stage2_hetero import run_stage2
from experimental.dhm_re_verification.stage3_japanese import run_stage3
from experimental.dhm_re_verification.common.logging import ExperimentLogger

def run_pipeline(dimension: int, stages: List[str]):
    print(f"\n=== Starting Pipeline for Dimension {dimension} ===")
    
    # 1. Initialize Persistent Memory System for this dimension
    encoder = HolographicEncoder(dimension=dimension)
    dynamic = DynamicHolographicMemory(capacity=100)
    static = StaticHolographicMemory()
    causal = CausalHolographicMemory()
    orchestrator = MemoryOrchestrator(dynamic, static, causal, promotion_threshold=0.85)
    
    # Shared Logger? No, per stage logging.
    
    for stage in stages:
        print(f"\n>> Executing {stage} (Dim {dimension})")
        
        if stage == "stage1":
            run_stage1(dimension, num_samples=10000, orchestrator=orchestrator, encoder=encoder)
            # Spec: Reset Dynamic Memory only
            orchestrator.reset_dynamic()
            print(">> Dynamic Memory Reset.")
            
        elif stage == "stage2":
            run_stage2(dimension, num_samples=15000, orchestrator=orchestrator, encoder=encoder)
            orchestrator.reset_dynamic()
            print(">> Dynamic Memory Reset.")
            
        elif stage == "stage3_kanji":
            run_stage3('kanji', dimension, orchestrator=orchestrator, encoder=encoder)
            orchestrator.reset_dynamic()
            print(">> Dynamic Memory Reset.")

        elif stage == "stage3_katakana":
             run_stage3('katakana', dimension, orchestrator=orchestrator, encoder=encoder)
             # Last stage typically for 2048
             orchestrator.reset_dynamic()

if __name__ == "__main__":
    # Define Pipelines per Specification
    
    # Dimension 512: Stage 1 only
    # run_pipeline(512, ["stage1"])
    
    # Dimension 1024: Stage 1 -> Stage 2 -> Stage 3 (Kanji)
    # Resuming from Stage 3 (Stage 1&2 completed)
    # run_pipeline(1024, ["stage1", "stage2", "stage3_kanji"])
    # Manual Resume for Stage 3 Kanji (Dim 1024)
    run_pipeline(1024, ["stage3_kanji"])
    
    # Dimension 2048: Stage 3 (Katakana) -> Stage 3 (Kanji)
    run_pipeline(2048, ["stage3_katakana", "stage3_kanji"])
    
    print("\n=== All Pipelines Completed ===")
