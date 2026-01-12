
import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coherent.core.sir import SIR, SIRFactory, SIRProjector, Entity, Relation, Operation, EntityAttributes
from coherent.core.cortex.memory.dynamic import DynamicHolographicMemory

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("C2_R_Eval")

@dataclass
class TestCase:
    id: str
    class_type: str # T1, T2, T3, T4
    raw_input: str
    group_id: str   # Equivalence group
    modality: str = "math"

class C2R_Evaluator:
    def __init__(self):
        self.projector = SIRProjector(dimension=1024)
        self.memory = DynamicHolographicMemory(capacity=100)
        self.test_cases: List[TestCase] = []
        self.sirs: Dict[str, SIR] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        
    def generate_test_cases(self):
        """Generates T1-T4 test cases (Same as C2)."""
        logger.info("Generating Test Cases...")
        # T1: Notation
        self.test_cases.append(TestCase("T1_01", "T1", "x + y", "GRP_ADD", "math"))
        self.test_cases.append(TestCase("T1_02", "T1", "y + x", "GRP_ADD", "math")) 
        self.test_cases.append(TestCase("T1_03", "T1", "a + b", "GRP_ADD", "math"))
        self.test_cases.append(TestCase("T1_04", "T1", "x > y", "GRP_CMP", "math"))
        
        # T2: Semantic
        self.test_cases.append(TestCase("T2_01", "T2", "x > 5", "GRP_COND", "math"))
        self._add_manual_code_case("T2_02", "if x > 5", "GRP_COND")
        
        # T3: Partial
        self.test_cases.append(TestCase("T3_01", "T3", "x > 10", "GRP_COND_DIFF", "math"))
        
        # T4: Irrelevant
        self.test_cases.append(TestCase("T4_01", "T4", "x = y", "GRP_ASSIGN", "math"))
        self.test_cases.append(TestCase("T4_02", "T4", "3 * 4", "GRP_MUL", "math"))

    def _add_manual_code_case(self, id: str, raw: str, group: str):
        # ... (Same manual construction logic as before) ...
        sir = SIRFactory.create_empty("code")
        core = sir.semantic_core
        e1 = Entity(id="E_x", type="variable", label="x", attributes=EntityAttributes(domain="abstract"))
        e2 = Entity(id="E_5", type="constant", label="5", attributes=EntityAttributes(domain="number"))
        core.entities.extend([e1, e2])
        rel = Relation(id=f"R_{id}", type="comparison", from_id=e1.id, to_id=e2.id, polarity="positive")
        core.relations.append(rel)
        sir.recompute_signature()
        self.sirs[id] = sir
        self.vectors[id] = self.projector.project(sir)
        self.test_cases.append(TestCase(id, "T2", raw, group, "code"))

    def process_sirs(self):
        """Phase 0: SIR Generation and Vectorization."""
        logger.info("[Phase 0] Processing SIRs...")
        for case in self.test_cases:
            if case.id in self.sirs: continue
            
            if case.modality == "math":
                sir = SIRFactory.from_math_expression(case.raw_input)
                self.sirs[case.id] = sir
                self.vectors[case.id] = self.projector.project(sir)
                
    def run_phase_s(self) -> pd.DataFrame:
        """Phase S: Semantic Validation (Vector Distance)."""
        logger.info("[Phase S] Semantic Validation...")
        results = []
        ids = [c.id for c in self.test_cases]
        n = len(ids)
        
        for i in range(n):
            for j in range(i, n):
                id_a, id_b = ids[i], ids[j]
                vec_a, vec_b = self.vectors[id_a], self.vectors[id_b]
                
                # Metric: Euclidean Distance (L2)
                # SIR Vectors are HRR superpositions, not necessarily unit normalized yet
                # But Projector uses randn. Projector output magnitude depends on N components.
                # Cosine distance is better for semantic similarity.
                
                # Spec says: L2 distance, Cosine distance.
                dist_l2 = np.linalg.norm(vec_a - vec_b)
                
                # Cosine Similarity
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b) if norm_a*norm_b > 0 else 0
                dist_cos = 1.0 - cos_sim
                
                case_a = self.test_cases[i]
                case_b = self.test_cases[j]
                expected_match = (case_a.group_id == case_b.group_id)
                
                semantic_correct = False
                if expected_match and dist_cos < 0.05: semantic_correct = True
                if not expected_match and dist_cos > 0.1: semantic_correct = True # Low bar for diff
                
                results.append({
                    "phase": "S",
                    "query_id": id_a,
                    "target_id": id_b,
                    "group_a": case_a.group_id,
                    "group_b": case_b.group_id,
                    "expected_match": expected_match,
                    "metric_val": dist_cos,
                    "metric_name": "cosine_dist",
                    "pass": semantic_correct
                })
        return pd.DataFrame(results)

    def run_phase_p(self) -> pd.DataFrame:
        """Phase P: Physical Recall Validation (DHM Resonance)."""
        logger.info("[Phase P] Physical Recall Validation...")
        
        # 1. Store ALL vectors in DHM
        self.memory.clear()
        for case in self.test_cases:
            # DHM add(state, metadata)
            self.memory.add(self.vectors[case.id], metadata={"id": case.id, "group": case.group_id})
            
        results = []
        ids = [c.id for c in self.test_cases]
        
        for i, query_id in enumerate(ids):
            query_case = self.test_cases[i]
            query_vec = self.vectors[query_id]
            
            # Query DHM
            # Since we stored N items, let's recall top N to see full distribution
            recall_results = self.memory.query(query_vec, top_k=len(ids))
            
            # query() returns (content, score). 
            # In add(), we passed metadata={"id": case.id, ...}
            # DynamicHolographicMemory.query() returns results where 'content' is:
            # content = meta.get('content', meta) 
            # Since we didn't set 'content' key, it returns the whole meta dict.
            # Dict is unhashable, so we must extract 'id' from it.
            
            score_map = {}
            for res_content, res_score in recall_results:
                # res_content is the metadata dict we passed
                c_id = res_content.get("id")
                if c_id:
                    score_map[c_id] = res_score
            
            for j, target_id in enumerate(ids):
                if i > j: continue # Dedup pairs same as S phase
                
                score = score_map.get(target_id, 0.0)
                
                target_case = self.test_cases[j]
                expected_match = (query_case.group_id == target_case.group_id)
                
                physical_stable = False
                # Spec: Same > theta_pos (e.g. 0.8), Diff < theta_neg (e.g. 0.5)
                if expected_match and score > 0.8: physical_stable = True
                if not expected_match and score < 0.6: physical_stable = True
                
                results.append({
                    "phase": "P",
                    "query_id": query_id,
                    "target_id": target_id,
                    "group_a": query_case.group_id,
                    "group_b": target_case.group_id,
                    "expected_match": expected_match,
                    "metric_val": score,
                    "metric_name": "resonance",
                    "pass": physical_stable
                })
        return pd.DataFrame(results)

    def generate_artifacts(self, df_s: pd.DataFrame, df_p: pd.DataFrame):
        """Phase D: Decision & Reporting."""
        logger.info("[Phase D] Decision & Reporting...")
        
        # Merge S and P
        # Keys: query_id, target_id
        df_merged = pd.merge(df_s, df_p, on=["query_id", "target_id", "group_a", "group_b", "expected_match"], suffixes=("_S", "_P"))
        
        # Diagnose
        def diagnose(row):
            s_ok = row["pass_S"]
            p_ok = row["pass_P"]
            if s_ok and p_ok: return "PASS"
            if s_ok and not p_ok: return "PARTIAL (Semantic-Only)"
            if not s_ok: return "FAIL (Semantic)"
            return "FAIL (Physical)" # Should overlap with PARTIAL usually
            
        df_merged["verdict"] = df_merged.apply(diagnose, axis=1)
        
        # Save CSV
        csv_path = "experimental/reports/C2_R_DATA.csv"
        df_merged.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        # Generate Markdwon
        # Summary Metrics
        same_s = df_s[df_s["expected_match"]==True]["metric_val"].mean()
        diff_s = df_s[df_s["expected_match"]==False]["metric_val"].mean()
        separation_s = diff_s - same_s # Input was distance (lower is better), so diff - same
        
        same_p = df_p[df_p["expected_match"]==True]["metric_val"].mean()
        diff_p = df_p[df_p["expected_match"]==False]["metric_val"].mean()
        separation_p = same_p - diff_p # Resonance (higher is better)
        
        overall_pass = (separation_s > 0.2) and (separation_p > 0.2)
        
        report = f"""# C2-R (Revised) Validation Report

**Date**: 2026-01-12
**Status**: {"✅ PASS" if overall_pass else "⚠️ PARTIAL / FAIL"}

## 1. Phase Summary
### Phase S: Semantic Correctness (Cosine Distance)
*   **Same-Meaning Avg Dist**: {same_s:.4f} (Target < 0.05)
*   **Diff-Meaning Avg Dist**: {diff_s:.4f}
*   **Separation Margin**: {separation_s:.4f} (Target > 0.2)
*   *Verdict*: {"✅ OK" if separation_s > 0.2 else "❌ FAIL"}

### Phase P: Physical Recall Robustness (DHM Resonance)
*   **Same-Meaning Avg Res**: {same_p:.4f} (Target > 0.8)
*   **Diff-Meaning Avg Res**: {diff_p:.4f}
*   **Separation Margin**: {separation_p:.4f} (Target > 0.2)
*   *Verdict*: {"✅ OK" if separation_p > 0.2 else "⚠️ LOW STABILITY"}

## 2. Verdict Distribution
{df_merged["verdict"].value_counts().to_markdown()}

## 3. Notable Failures / Partials
{df_merged[df_merged["verdict"] != "PASS"][["query_id", "target_id", "metric_val_S", "metric_val_P", "verdict"]].head(10).to_markdown(index=False)}

## 4. Conclusion
SIR v1.0 Core {"successfully pass" if overall_pass else "shows mixed results in"} C2-R criteria.
Semantic Separation is {separation_s:.2f}, Physical Separation is {separation_p:.2f}.
"""
        with open("experimental/reports/C2_R_REPORT.md", "w") as f:
            f.write(report)
        print("Report saved to experimental/reports/C2_R_REPORT.md")

if __name__ == "__main__":
    eval = C2R_Evaluator()
    eval.generate_test_cases()
    eval.process_sirs()
    df_s = eval.run_phase_s()
    df_p = eval.run_phase_p()
    eval.generate_artifacts(df_s, df_p)
