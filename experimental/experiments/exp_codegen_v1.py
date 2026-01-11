"""
EXP-CODEGEN-V1: Semantic-First Code Generation
Objective: Verify generation of IR from semantic input and projection to multiple languages.
"""

import os
import sys
import numpy as np
import csv
import json
import time
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

# Adjust path to import core modules and sandbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
    from coherent.core.memory.holographic.encoder import HolographicEncoder
    from experimental.verification_sandbox.ir import *
    from experimental.verification_sandbox.languages import PYTHON_SPEC, JAVA_SPEC, HASKELL_SPEC
    from experimental.verification_sandbox.projection import ProjectionEngine
    from experimental.verification_sandbox.validator import Validator
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running from the project root or correct path structure.")
    sys.exit(1)

# Configuration
REPORT_DIR = os.path.join(os.path.dirname(__file__), '../reports')
CSV_RESULTS_PATH = os.path.join(REPORT_DIR, 'codegen_results.csv')
REPORT_MD_PATH = os.path.join(REPORT_DIR, 'EXP_CODEGEN_V1_REPORT.md')

# --- S2: Semantic Recall (Knowledge Base) ---
CONCEPT_MAP = {
    "ITERATION": ["repeat", "until", "while", "loop", "cycle"],
    "CONDITION": ["if", "check", "met", "threshold", "condition", "constraint"],
    "STATE": ["accumulate", "update", "count", "value", "add"],
    "TERMINATION": ["stop", "exit", "break", "process until"],
}

class SemanticRecaller:
    def __init__(self):
        self.dhm = DynamicHolographicMemory(capacity=100)
        self.encoder = HolographicEncoder(dimension=1024)
        self.role_vector = self.encoder.encode_attribute("ROLE_SEMANTIC") # Simple role
        self._register_concepts()

    def _register_concepts(self):
        # Register each keyword associated with its Concept
        self.dhm.clear()
        for concept, keywords in CONCEPT_MAP.items():
            v_concept = self.encoder.encode_attribute(concept)
            for kw in keywords:
                v_kw = self.encoder.encode_attribute(kw)
                # Bundle = Bind(Role, Kw) + Concept (simplified for retrieval)
                # or just Store Concept keyed by Keyword vector?
                # For DHM, we store Bundles.
                # Let's simple Store: Bundle = Concept + Keyword
                bundle = self.encoder.normalize(v_concept + v_kw)
                # We want to retrieve Concept given Keyword
                self.dhm.add(bundle, metadata={'concept': concept, 'keyword': kw})

    def recall(self, input_text: str) -> Set[str]:
        # Simple bag-of-words query against DHM
        words = input_text.lower().split()
        recalled_concepts = set()
        
        for word in words:
            # Check if word triggers a concept
            v_word = self.encoder.encode_attribute(word)
            # Query similarity
            matches = self.dhm.query(v_word, top_k=3)
            # If match resonance is high, accept concept
            if matches:
                 # Check resonance threshold? or exact match logic implicitly
                 # matches[0] is (metadata, score)
                 top_meta, score = matches[0]
                 # In real DHM, query vector should match stored vector.
                 # stored = (Concept + Keyword). Query = Keyword. 
                 # Dot product ~ 1/sqrt(2) if dims large.
                 if score > 0.4: # Heuristic threshold
                     recalled_concepts.add(top_meta['concept'])
        
        # Heuristic fallback for multi-word phrases (like 'process until') 
        # In a real system the parser handles this.
        # For Sandbox v1, relying on individual keywords is fine if robust enough.
        return recalled_concepts

class CanonicalIRBuilder:
    def build(self, concepts: Set[str]) -> IRNode:
        # Rules:
        # Loop if ITERATION present.
        # Inside Loop:
        #   If CONDITION & TERMINATION -> Exit Guard.
        #   If STATE -> Assignment.
        
        # Base: Sequence
        body_nodes = []
        
        # State Update
        has_state = "STATE" in concepts
        if has_state:
            # Generic update
            body_nodes.append(Assignment(target="state", value="state + 1"))
            
        # Exit Guard
        # If we have Iteration, we usually need an exit condition
        has_cond = "CONDITION" in concepts
        has_term = "TERMINATION" in concepts
        
        if has_cond or has_term:
            # If explicit termination requested, add exit guard
            guard = If(
                condition="state >= threshold", 
                body=Sequence([Exit()])
            )
            # Canonical Rule: Exit check often comes either first or last.
            # Let's put it first for "while check" semantics, or last for "repeat until".
            # Input "repeat until" -> Loop, Body, If(Exit).
            # Let's append it to body.
            body_nodes.append(guard)
            
        if not body_nodes:
            # fallback
            body_nodes.append(Value("pass"))
            
        body = Sequence(body_nodes)
        
        if "ITERATION" in concepts:
            return Loop(condition="true", body=body)
        else:
            return body

class CodeGenExperiment:
    def __init__(self):
        self.recaller = SemanticRecaller()
        self.builder = CanonicalIRBuilder()
        self.projector = ProjectionEngine()
        self.validator = Validator()
        self.languages = [PYTHON_SPEC, JAVA_SPEC, HASKELL_SPEC]

    def run(self, inputs: List[str]):
        results = []
        
        for i, text in enumerate(inputs):
            sem_id = f"CS-{i+1:02d}"
            print(f"\nProcessing [{sem_id}]: '{text}'")
            
            # 1. Recall
            concepts = self.recaller.recall(text)
            print(f"  > Recalled: {list(concepts)}")
            
            # 2. Build IR
            ir_root = self.builder.build(concepts)
            
            # 3. Project & Validate
            for lang in self.languages:
                code = self.projector.project(ir_root, lang)
                val_res = self.validator.validate(code, lang)
                
                print(f"  > [{lang.id}] Match: {val_res['structure_match']:.2f} ({val_res['result']})")
                
                results.append({
                    'semantic_id': sem_id,
                    'input_text': text,
                    'language': lang.id,
                    'concepts': list(concepts),
                    'structure_match': val_res['structure_match'],
                    'result': val_res['result'],
                    'failure_code': 'None' if val_res['result'] == 'PASS' else 'STRUCT_FAIL',
                    'generated_code': code # Optional: Truncate for CSV?
                })
                
        self.save_results(results)

    def save_results(self, results):
        os.makedirs(REPORT_DIR, exist_ok=True)
        # CSV
        fieldnames = ['semantic_id','input_text','language','concepts','structure_match','result','failure_code']
        with open(CSV_RESULTS_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
            
        # Report
        with open(REPORT_MD_PATH, 'w') as f:
            f.write("# EXP-CODEGEN-V1 Report\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
            f.write("| ID | Input | Lang | Match | Result |\n")
            f.write("|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['semantic_id']} | {r['input_text']} | {r['language']} | {r['structure_match']:.2f} | {r['result']} |\n")

if __name__ == "__main__":
    inputs = [
        "repeat the process until a condition is met",
        "accumulate values while checking a constraint",
        "just update the state value",
        "loop forever"
    ]
    exp = CodeGenExperiment()
    exp.run(inputs)
