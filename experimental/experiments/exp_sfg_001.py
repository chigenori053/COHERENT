"""
EXP-SFG-001: Semantic-First Formula Generation
Objective: Reconstruct formula structures from semantic constraints using DHM.
"""

import os
import sys
import numpy as np
import csv
import json
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

# Adjust path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.encoder import HolographicEncoder

# Configuration
REPORT_DIR = os.path.join(os.path.dirname(__file__), '../reports')
CSV_RESULTS_PATH = os.path.join(REPORT_DIR, 'sfg_results.csv')
REPORT_MD_PATH = os.path.join(REPORT_DIR, 'EXP_SFG_001_REPORT.md')

@dataclass
class MathConcept:
    name: str
    features: List[str]
    template: str # e.g. "y = k * x"

@dataclass
class ProblemEntry:
    id: str
    level: str # L1, L2, L3
    constraints: List[str]
    expected_properties: List[str]

# --- KNOWLEDGE BASE ---
KNOWLEDGE_BASE = [
    MathConcept(
        name="Proportional",
        features=["proportional", "increases_with", "linear", "constant_ratio"],
        template="y = k * x"
    ),
    MathConcept(
        name="InverseProportional",
        features=["inverse", "decreases_with", "hyperbolic", "product_constant"],
        template="y = k / x"
    ),
    MathConcept(
        name="Derivative",
        features=["rate_of_change", "derivative", "slope", "speed", "change"],
        template="dy/dx = k"
    ),
    MathConcept(
        name="Accumulation",
        features=["accumulates", "integral", "area", "sum_over_time"],
        template="y = int(x) + C"
    ),
    MathConcept(
        name="Conservation",
        features=["conserved", "constant_sum", "total", "invariant"],
        template="x + y = C"
    )
]

# --- DATASETS ---
DATASET = [
    # L1: Single Constraint
    ProblemEntry("L1-01", "L1", ["y is proportional to x"], ["proportional"]),
    ProblemEntry("L1-02", "L1", ["y represents the rate of change of x"], ["derivative"]),
    ProblemEntry("L1-03", "L1", ["y decreases when x increases inversely"], ["inverse"]),
    ProblemEntry("L1-04", "L1", ["the total of x and y is conserved"], ["conservation"]),
    ProblemEntry("L1-05", "L1", ["y accumulates change of x over time"], ["accumulation"]),
    
    # L2: Compound Constraints
    ProblemEntry("L2-01", "L2", ["y increases when x increases", "y is proportional to x"], ["proportional"]),
    ProblemEntry("L2-02", "L2", ["y is the derivative of x", "the rate is constant"], ["derivative"]), # "rate is constant" implies k
    ProblemEntry("L2-03", "L2", ["y is inversely proportional to x"], ["inverse"]), # Simple phrasing
    
    # L3: Conditional / Context (Simplified for this retrieval test)
    ProblemEntry("L3-01", "L3", ["initial value of y is zero", "y is proportional to x"], ["proportional"]), 
    # Note: DHM retrieval focuses on the "proportional" part. Assumptions handling would check "initial value".
    ProblemEntry("L3-02", "L3", ["y represents slope", "slope is constant"], ["derivative"]),
]

class FormulaGenerator:
    def __init__(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        self.dhm = DynamicHolographicMemory(capacity=100)
        self.encoder = HolographicEncoder(dimension=1024)
        self.role_vectors = {}
        self.vocab = {}
        self._setup_roles()
        self._register_knowledge()

    def _setup_roles(self):
        # We bind semantic features to ROLE_SEMANTICS
        # We bind the template name/id to ROLE_STRUCTURE (for retrieval verification)
        # Actually we store: Bundle = Bind(ROLE_SEMANTICS, superposition(features)) + Bind(ROLE_TEMPLATE, template_name)
        roles = ['ROLE_SEMANTICS', 'ROLE_TEMPLATE']
        for r in roles:
            self.role_vectors[r] = self.encoder.encode_attribute(r)

    def get_vector(self, key):
        if key not in self.vocab:
            self.vocab[key] = self.encoder.encode_attribute(key)
        return self.vocab[key]
    
    def _bind(self, v1, v2): return v1 * v2
    def _superpose(self, vecs): return self.encoder.normalize(np.sum(vecs, axis=0))
    def _cosine_sim(self, v1, v2): return np.abs(np.vdot(v1, v2))

    def _register_knowledge(self):
        print("Registering Math Knowledge Base...")
        self.dhm.clear()
        for concept in KNOWLEDGE_BASE:
            # Create Semantic Vector: Superposition of all feature keywords
            feature_vecs = [self.get_vector(f) for f in concept.features]
            v_semantics = self._superpose(feature_vecs)
            
            # Create Template Vector
            v_template = self.get_vector(concept.name)
            
            # Bundle
            b_sem = self._bind(self.role_vectors['ROLE_SEMANTICS'], v_semantics)
            b_tem = self._bind(self.role_vectors['ROLE_TEMPLATE'], v_template)
            
            bundle = self._superpose([b_sem, b_tem])
            
            # Key = Template Name (or we typically key by content, here just add)
            self.dhm.add(bundle, metadata={'concept': concept})
            
    def parse_constraints(self, constraints: List[str]) -> List[str]:
        # Simple heuristic parser: extract known keywords from text
        # In a real system this would be an LLM or dependency parser
        extracted = []
        known_keywords = set()
        for c in KNOWLEDGE_BASE:
            for f in c.features:
                known_keywords.add(f)
        
        # Also simple variations
        text = " ".join(constraints).lower()
        # Regex or simple 'in' check
        found = []
        for kw in known_keywords:
            # simple separate word check
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                found.append(kw)
            # handle 'increases' matching 'increases_with' roughly? 
            # For now strict keyword matching against KB definitions
            
        return found

    def generate(self, problem: ProblemEntry) -> Dict:
        # 1. Parse
        features = self.parse_constraints(problem.constraints)
        if not features:
            return {
                'recalled_structure': None, 'formula': "No features found", 
                'resonance': 0.0, 'judgment': 'FAIL', 'features': []
            }
            
        # 2. Query DHM
        # Query = Bind(ROLE_SEMANTICS, Superpose(features))
        feature_vecs = [self.get_vector(f) for f in features]
        v_semantics = self._superpose(feature_vecs)
        query = self._bind(self.role_vectors['ROLE_SEMANTICS'], v_semantics)
        
        matches = self.dhm.query(query, top_k=1)
        if not matches:
             return {
                'recalled_structure': None, 'formula': "No match", 
                'resonance': 0.0, 'judgment': 'FAIL', 'features': features
            }
            
        # 3. Retrieve Concept
        # matches[0] -> (content, resonance) .. wait, metadata is in self.dhm._storage
        # query() returns list of (key, score). dhm.add() stores bundle.
        # We need to find the metadata associated with the resonant bundle.
        # DHM.query returns (metadata_content_if_available, score).
        # Wait, looking at `DynamicHolographicMemory.query`:
        # It calculates similarity with stored items. 
        # `results.append((item[1].get('content', 'unknown'), score))`
        # Our metadata has 'concept'. content is not set.
        # Let's check how I registered: `metadata={'concept': concept}`.
        # I should set 'content' to concept.name for query to return it.
        
        # Wait, I need to fix registration to include 'content' key if I rely on standard query return.
        # Check `DynamicHolographicMemory` implementation?
        # Assuming standard implementation returns `metadata['content']`.
        
        # RE-FIX registration in memory (Mental Note: I will update `_register_knowledge` to add 'content')
        
        # Proceed assuming name is returned
        concept_name = matches[0][0]
        score = matches[0][1]
        
        # Find concept obj
        concept = next((c for c in KNOWLEDGE_BASE if c.name == concept_name), None)
        if not concept:
             return {
                'recalled_structure': concept_name, 'formula': "Concept not found in KB", 
                'resonance': score, 'judgment': 'FAIL', 'features': features
            }

        # 4. Generate Formula (Symbol Instantiation)
        # Default symbols for L1/L2 are y, x, k.
        # In a full system we'd map "distance" -> d, "time" -> t.
        # Here we just use the template.
        formula = concept.template
        
        # 5. Evaluate
        # Judgment: PASS if the recalled concept matches expected properties.
        # For this test, ProblemEntry has expected_properties which map loosely to concept name/features.
        # Simplify: If concept.name or features overlap with expected_properties keywords
        is_pass = False
        for prop in problem.expected_properties:
            if prop.lower() in concept.name.lower(): is_pass = True
            if prop in concept.features: is_pass = True
            
        return {
            'recalled_structure': concept.name,
            'formula': formula,
            'resonance': score,
            'judgment': 'PASS' if is_pass else 'FAIL',
            'features': features
        }

    # FIX for registration
    def _register_knowledge_fixed(self):
        # Overwriting the method logic here for the actual run
        self.dhm.clear()
        for concept in KNOWLEDGE_BASE:
            feature_vecs = [self.get_vector(f) for f in concept.features]
            v_semantics = self._superpose(feature_vecs)
            v_template = self.get_vector(concept.name)
            
            b_sem = self._bind(self.role_vectors['ROLE_SEMANTICS'], v_semantics)
            b_tem = self._bind(self.role_vectors['ROLE_TEMPLATE'], v_template)
            
            bundle = self._superpose([b_sem, b_tem])
            
            # Add 'content' key for query return
            self.dhm.add(bundle, metadata={'concept': concept, 'content': concept.name})

    def run(self):
        # Use Fixed Registration
        self._register_knowledge_fixed()
        
        results = []
        for problem in DATASET:
            out = self.generate(problem)
            results.append({
                'id': problem.id,
                'level': problem.level,
                'input_constraints': " | ".join(problem.constraints),
                'features_extracted': " ".join(out['features']),
                'recalled_structure': out['recalled_structure'],
                'generated_formula': out['formula'],
                'resonance': out['resonance'],
                'judgment': out['judgment']
            })
            
        # Log CSV
        with open(CSV_RESULTS_PATH, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                
        self.generate_report(results)

    def generate_report(self, results):
        lines = ["# Semantic-First Formula Generation Report (EXP-SFG-001)", ""]
        lines.append(f"Date: {time.strftime('%Y-%m-%d')}")
        lines.append("## Results Summary")
        
        pass_count = sum(1 for r in results if r['judgment']=='PASS')
        total = len(results)
        lines.append(f"**Total Success Rate**: {pass_count}/{total} ({pass_count/total:.1%})")
        
        lines.append("\n## Detailed Logs")
        lines.append("| ID | Level | Input | Structure | Resonance | Result |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            lines.append(f"| {r['id']} | {r['level']} | {r['input_constraints']} | {r['recalled_structure']} | {r['resonance']:.4f} | {r['judgment']} |")
            
        with open(REPORT_MD_PATH, 'w') as f:
            f.write('\n'.join(lines))

if __name__ == "__main__":
    gen = FormulaGenerator()
    gen.run()
