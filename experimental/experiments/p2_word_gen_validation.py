"""
DHM-WORD-GEN-LANG-COMP-P2: Word Generation Validation (Logging Fix + Controls)
Objective: Validate DHM performance with strict logging and degradation checks.
"""

import os
import sys
import numpy as np
import csv
import json
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict

# Adjust path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.encoder import HolographicEncoder

# Configuration
REPORT_DIR = os.path.join(os.path.dirname(__file__), '../reports')
CSV_BASELINE_PATH = os.path.join(REPORT_DIR, 'p2_baseline.csv')
CSV_CUED_PATH = os.path.join(REPORT_DIR, 'p2_cued.csv')
CSV_CROSS_PATH = os.path.join(REPORT_DIR, 'p2_cross.csv')
CSV_CROSS_EXTRACT_PATH = os.path.join(REPORT_DIR, 'p2_cross_extract.csv')
RUN_SUMMARY_PATH = os.path.join(REPORT_DIR, 'p2_run_summary.json')
REPORT_MD_PATH = os.path.join(REPORT_DIR, 'DHM_WORD_GEN_P2_REPORT.md')

@dataclass
class WordEntry:
    id: int
    surface: str  # JP or EN word
    reading: str  # Reading for JP (Hiragana), Spelling for EN
    gloss: str    # Concept meaning in English
    lang: str     # 'JP' or 'EN'

# 5.1 JP-only (30 terms)
JP_TERMS = [
    ("ひかり", "ひかり", "light"), ("みず", "みず", "water"), ("ひ", "ひ", "fire"),
    ("つき", "つき", "moon"), ("そら", "そら", "sky"), ("やま", "やま", "mountain"),
    ("かわ", "かわ", "river"), ("うみ", "うみ", "sea"), ("ひと", "ひと", "person"),
    ("こども", "こども", "child"), ("べんきょう", "べんきょう", "study"), ("いのち", "いのち", "life"),
    ("たべる", "たべる", "eat"), ("のむ", "のむ", "drink"), ("みる", "みる", "see"),
    ("いく", "いく", "go"), ("くる", "くる", "come"), ("はいる", "はいる", "enter"),
    ("でる", "でる", "exit"), ("おおきい", "おおきい", "big"), ("ちいさい", "ちいさい", "small"),
    ("ながい", "ながい", "long"), ("たかい", "たかい", "high"), ("あたらしい", "あたらしい", "new"),
    ("ふるい", "ふるい", "old"), ("はやい", "はやい", "early"), ("おおい", "おおい", "many"),
    ("すくない", "すくない", "few"), ("うえ", "うえ", "up"), ("した", "した", "down")
]

# 5.2 EN-only (30 terms)
EN_TERMS = [
    ("light", "light", "light"), ("water", "water", "water"), ("fire", "fire", "fire"),
    ("moon", "moon", "moon"), ("sky", "sky", "sky"), ("mountain", "mountain", "mountain"),
    ("river", "river", "river"), ("sea", "sea", "sea"), ("person", "person", "person"),
    ("child", "child", "child"), ("study", "study", "study"), ("life", "life", "life"),
    ("eat", "eat", "eat"), ("drink", "drink", "drink"), ("see", "see", "see"),
    ("go", "go", "go"), ("come", "come", "come"), ("enter", "enter", "enter"),
    ("exit", "exit", "exit"), ("big", "big", "big"), ("small", "small", "small"),
    ("long", "long", "long"), ("high", "high", "high"), ("new", "new", "new"),
    ("old", "old", "old"), ("early", "early", "early"), ("many", "many", "many"),
    ("few", "few", "few"), ("up", "up", "up"), ("down", "down", "down")
]

class WordGenerationExperimentP2:
    def __init__(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        self.dhm = DynamicHolographicMemory(capacity=2000) # Hint 2000
        self.encoder = HolographicEncoder(dimension=1024)
        self.role_vectors = {}
        self.vocab = {} # Map 'domain' -> { 'value' -> vector }
        
        self.dataset_definitions = {
            'JP': [],
            'EN': [],
        }
        self.run_id = f"RUN_{int(time.time())}"
        self._prepare_datasets()
        self.setup_roles() # Frozen once per run as per Spec

    def _prepare_datasets(self):
        for i, (surf, read, gloss) in enumerate(JP_TERMS):
            self.dataset_definitions['JP'].append(WordEntry(i+1, surf, read, gloss, 'JP'))
        for i, (surf, read, gloss) in enumerate(EN_TERMS):
            self.dataset_definitions['EN'].append(WordEntry(i+31, surf, read, gloss, 'EN'))

    def setup_roles(self):
        roles = ['ROLE_SURFACE', 'ROLE_READING', 'ROLE_GLOSS']
        for r in roles:
            self.role_vectors[r] = self.encoder.encode_attribute(r)

    # --- Vector Operations ---
    def encode_sequence(self, text: str) -> np.ndarray:
        vecs = []
        for i, char in enumerate(text):
             pos_tag = f"POS_{i}"
             char_tag = f"CHAR_{char}"
             v_pos = self.get_vocab_vector('COMPONENTS', pos_tag)
             v_char = self.get_vocab_vector('COMPONENTS', char_tag)
             vecs.append(v_pos * v_char)
        if not vecs: return np.zeros(1024) # Should not happen
        return self._superpose(vecs)

    def get_vocab_vector(self, domain, key):
        if domain not in self.vocab: self.vocab[domain] = {}
        if key not in self.vocab[domain]:
            self.vocab[domain][key] = self.encoder.encode_attribute(key)
        return self.vocab[domain][key]

    def _bind(self, v1, v2):
        return v1 * v2

    def _superpose(self, vectors: List[np.ndarray]) -> np.ndarray:
        s = np.sum(vectors, axis=0)
        return self.encoder.normalize(s)
    
    def _cosine_sim(self, v1, v2):
        return np.abs(np.vdot(v1, v2))

    def create_bundle(self, entry: WordEntry) -> np.ndarray:
        v_surf = self.get_vocab_vector('SURFACE', entry.surface)
        v_read = self.encode_sequence(entry.reading)
        v_gloss = self.get_vocab_vector('GLOSS', entry.gloss)
        
        b_surf = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
        b_read = self._bind(self.role_vectors['ROLE_READING'], v_read)
        b_gloss = self._bind(self.role_vectors['ROLE_GLOSS'], v_gloss)
        
        return self._superpose([b_surf, b_read, b_gloss])

    # --- Experiment Phase Execution ---
    
    def setup_condition(self, condition: str):
        print(f"[{condition}] Setting up DHM...")
        self.dhm.clear() 
        # Register Phase 1 immediately
        self.current_dataset = []
        
        if condition == 'C1_JP':
            self.current_dataset = self.dataset_definitions['JP']
        elif condition == 'C2_EN':
            self.current_dataset = self.dataset_definitions['EN']
        elif condition == 'C3_MixedA':
            self.current_dataset = self.dataset_definitions['JP'] + self.dataset_definitions['EN']
        elif condition == 'C4_MixedB':
            # Mixed-B uses JP entries but we evaluate cross. Spec 5.5: "Each entry contains gloss_en".
            # The current_dataset is effectively JP list (primary) but we might need EN glosses in vocab.
            # Wait, Mixed-B is defined as "VP-only 30 as primary set... Mixed-B uses same storage format as JP-only".
            # Ah, Spec 5.4 says Mixed-A is JP 30 + EN 30.
            # Spec 5.5: Mixed-B is JP-only 30 list. 
            # Confusing. "Mixed-B uses the same storage format as JP-only".
            # But the objective is "Cross-Lingual Association".
            # Okay, I will stick to: Mixed-B storage = JP Dataset. Retrieval target = JP. Query source = EN Gloss.
            self.current_dataset = self.dataset_definitions['JP']
            
        for entry in self.current_dataset:
            bundle = self.create_bundle(entry)
            self.dhm.add(bundle, metadata={'content': entry.surface, 'entry': entry})
        
        print(f"[{condition}] Registered {len(self.current_dataset)} bundles.")

    # --- Core Logic with Margin Fix ---
    
    def get_vector_for_entry(self, entry: WordEntry) -> np.ndarray:
        # Re-create bundle to check resonance
        # Or better, retrieve from storage if we had IDs.
        # But create_bundle is deterministic.
        return self.create_bundle(entry)

    def run_phase2_baseline(self, condition: str):
        print(f"[{condition}] Phase 2: Baseline Exact Recall")
        results = []
        
        for entry in self.current_dataset:
            v_surf = self.get_vocab_vector('SURFACE', entry.surface)
            query = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
            
            # MANDATORY: Resonance Top 1, Top 2
            matches = self.dhm.query(query, top_k=5)
            
            top1 = matches[0][0] if matches else None
            r_top1 = matches[0][1] if len(matches) > 0 else 0.0
            r_top2 = matches[1][1] if len(matches) > 1 else 0.0
            
            results.append({
                'run_id': self.run_id,
                'condition': condition,
                'phase': 2,
                'language': entry.lang,
                'word_id': entry.id,
                'query_type': 'surface_exact',
                'query_surface': entry.surface,
                'cue_prefix': '',
                'cue_gloss': '',
                'expected_surface': entry.surface,
                'top1_surface': top1,
                'topk_surfaces': [m[0] for m in matches],
                'resonance_top1': r_top1,
                'resonance_top2': r_top2,
                'correct': (top1 == entry.surface)
            })
            
        self.append_csv(CSV_BASELINE_PATH, results)

    def run_phase3_prefix(self, condition: str):
        print(f"[{condition}] Phase 3: Prefix Cued Recall")
        results = []
        
        for entry in self.current_dataset:
            prefix = entry.reading[:2]
            v_prefix = self.encode_sequence(prefix)
            query = self._bind(self.role_vectors['ROLE_READING'], v_prefix)
            
            matches = self.dhm.query(query, top_k=5)
            topk_list = [m[0] for m in matches]
            
            # CALC EXPECTED RESONANCE
            target_bundle = self.create_bundle(entry)
            r_expected = self._cosine_sim(query, target_bundle)
            
            r_top1 = matches[0][1] if len(matches) > 0 else 0.0
            r_top2 = matches[1][1] if len(matches) > 1 else 0.0
            
            margin = r_top1 - r_top2
            
            rank = -1
            if entry.surface in topk_list:
                rank = topk_list.index(entry.surface) + 1
            
            results.append({
                'run_id': self.run_id,
                'condition': condition,
                'phase': 3,
                'language': entry.lang,
                'word_id': entry.id,
                'cue_type': 'prefix',
                'cue_prefix': prefix,
                'cue_gloss': '',
                'expected_surface': entry.surface,
                'rank_of_expected': rank,
                'topk_surfaces': topk_list,
                'resonance_expected': r_expected,
                'resonance_top1': r_top1,
                'resonance_top2': r_top2,
                'margin': margin,
                'success_topk': (rank != -1)
            })
        self.append_csv(CSV_CUED_PATH, results)

    def run_phase4_fusion(self, condition: str):
        # Do not run if Mixed-A ? Specification says "Phase 5 Mixed-A Restriction... Do NOT run semantic-gloss-only queries".
        # But Phase 4 is FUSION (Prefix + Gloss).
        # "Mixed-A measures coexistence interference via Phase 2-4". So we DO run Phase 4 Fusion.
        print(f"[{condition}] Phase 4: Fusion Cued Recall")
        results = []
        
        for entry in self.current_dataset:
            prefix = entry.reading[:2]
            v_prefix = self.encode_sequence(prefix)
            q_read = self._bind(self.role_vectors['ROLE_READING'], v_prefix)
            
            v_gloss = self.get_vocab_vector('GLOSS', entry.gloss)
            q_gloss = self._bind(self.role_vectors['ROLE_GLOSS'], v_gloss)
            
            query = self.encoder.normalize(q_read + q_gloss)
            
            matches = self.dhm.query(query, top_k=5)
            topk_list = [m[0] for m in matches]
            
            target_bundle = self.create_bundle(entry)
            r_expected = self._cosine_sim(query, target_bundle)
            r_top1 = matches[0][1] if len(matches) > 0 else 0.0
            r_top2 = matches[1][1] if len(matches) > 1 else 0.0
            margin = r_top1 - r_top2
            
            rank = -1
            if entry.surface in topk_list:
                rank = topk_list.index(entry.surface) + 1
                
            results.append({
                'run_id': self.run_id,
                'condition': condition,
                'phase': 4,
                'language': entry.lang,
                'word_id': entry.id,
                'cue_type': 'fusion',
                'cue_prefix': prefix,
                'cue_gloss': entry.gloss,
                'expected_surface': entry.surface,
                'rank_of_expected': rank,
                'topk_surfaces': topk_list,
                'resonance_expected': r_expected,
                'resonance_top1': r_top1,
                'resonance_top2': r_top2,
                'margin': margin,
                'success_topk': (rank != -1)
            })
        self.append_csv(CSV_CUED_PATH, results)

    def run_mixed_b_cross(self, condition: str):
        if condition != 'C4_MixedB': return
        print(f"[{condition}] Phase 6: Mixed-B Cross-Lingual")
        
        # Task A: EN Gloss -> JP Surface
        results_cross = []
        for entry in self.current_dataset: # JP dataset
            v_gloss = self.get_vocab_vector('GLOSS', entry.gloss)
            query = self._bind(self.role_vectors['ROLE_GLOSS'], v_gloss)
            
            matches = self.dhm.query(query, top_k=5)
            topk_list = [m[0] for m in matches]
            
            target_bundle = self.create_bundle(entry)
            r_expected = self._cosine_sim(query, target_bundle)
            r_top1 = matches[0][1] if matches else 0.0
            r_top2 = matches[1][1] if len(matches) > 1 else 0.0
            
            rank = -1
            if entry.surface in topk_list: rank = topk_list.index(entry.surface) + 1
            
            results_cross.append({
                'run_id': self.run_id,
                'condition': condition,
                'phase': 6,
                'task': 'EN_GLOSS_TO_JP_SURFACE',
                'language': entry.lang,
                'word_id': entry.id,
                'query_gloss': entry.gloss,
                'expected_surface': entry.surface,
                'rank_of_expected': rank,
                'topk_surfaces': topk_list,
                'resonance_expected': r_expected,
                'resonance_top1': r_top1,
                'resonance_top2': r_top2,
                'margin': r_top1 - r_top2,
                'success_topk': (rank != -1)
            })
        self.append_csv(CSV_CROSS_PATH, results_cross)

    # --- Helper: Unbind ---
    def _unbind(self, bundle, role_vec):
        # Approximate inverse via Conjugate for unitary-like vectors
        return bundle * np.conj(role_vec)

    def run_mixed_b_task_b(self, condition: str):
        if condition != 'C4_MixedB': return
        print(f"[{condition}] Phase 6: Mixed-B Task B (JP Surf -> Extract EN Gloss)")
        
        results_extract = []
        
        # Pre-calculate Gloss Matrix for decoding
        # We need all EN gloss vectors to find the closest match.
        # Unique glosses from EN dataset (or JP dataset's glosses which are identical set)
        gloss_map = {}
        for entry in self.dataset_definitions['JP']:
             gloss_map[entry.gloss] = self.get_vocab_vector('GLOSS', entry.gloss)
        
        # Optimize decoding: iterate once or matrix mult? 
        # For 30 items, iteration is fine.
        
        for entry in self.current_dataset: # JP dataset
            # 1. Retrieve Bundle by JP Surface Query
            v_surf = self.get_vocab_vector('SURFACE', entry.surface)
            query = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
            
            # We assume we retrieved the correct bundle (Top-1)
            # In Phase 2 we proved this is 100% accurate. 
            # So we simulate "Retrieving result from DHM" by just re-creating the target bundle 
            # OR determining the Top-1 and using that vector?
            # Realistically, we should query DHM, get the vector, and then unbind.
            
            matches = self.dhm.query(query, top_k=1)
            if not matches:
                continue
                
            # Need to get the actual vector from storage. 
            # DHM storage is list of (vec, meta).
            # matches[0][0] is content/surface.
            best_surface = matches[0][0]
            
            # Find vector in storage (inefficient scan but fine for N=30)
            retrieved_vec = None
            for vec, meta in self.dhm._storage:
                if meta.get('content') == best_surface:
                    retrieved_vec = vec
                    break
            
            if retrieved_vec is None: continue
            
            # 2. Extract Gloss Component
            # Unbind ROLE_GLOSS
            raw_gloss = self._unbind(retrieved_vec, self.role_vectors['ROLE_GLOSS'])
            
            # 3. Decode against Gloss Vocabulary
            gloss_candidates = []
            for gloss_txt, gloss_vec in gloss_map.items():
                score = self._cosine_sim(raw_gloss, gloss_vec)
                gloss_candidates.append((gloss_txt, score))
            
            gloss_candidates.sort(key=lambda x: x[1], reverse=True)
            
            top1_gloss = gloss_candidates[0][0]
            topk_glosses = [g[0] for g in gloss_candidates[:5]]
            r_top1 = gloss_candidates[0][1]
            r_top2 = gloss_candidates[1][1] if len(gloss_candidates) > 1 else 0.0
            
            # Check correctness
            correct = (top1_gloss == entry.gloss)
            
            results_extract.append({
                'run_id': self.run_id,
                'condition': condition,
                'phase': 6,
                'task': 'JP_SURFACE_TO_EN_GLOSS',
                'language': entry.lang,
                'word_id': entry.id,
                'query_surface': entry.surface,
                'expected_gloss': entry.gloss,
                'top1_gloss': top1_gloss,
                'topk_glosses': topk_glosses,
                'resonance_top1': r_top1,
                'resonance_top2': r_top2,
                'margin': r_top1 - r_top2,
                'correct': correct
            })
            
        self.append_csv(CSV_CROSS_EXTRACT_PATH, results_extract)

    def append_csv(self, path, data):
        file_exists = os.path.exists(path)
        if not data: return
        keys = data[0].keys()
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists: writer.writeheader()
            writer.writerows(data)

    def calculate_metrics(self):
        # Read logs and compute summaries
        summary = {
            'run_id': self.run_id,
            'metrics': {},
            'degradation': {}
        }
        
        # Helper to read CSV
        def read_csv(path):
            if not os.path.exists(path): return []
            with open(path, 'r', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        
        baseline = read_csv(CSV_BASELINE_PATH)
        cued = read_csv(CSV_CUED_PATH)
        cross_extract = read_csv(CSV_CROSS_EXTRACT_PATH)
        
        # Calc Stats Per Condition
        for cond in ['C1_JP', 'C2_EN', 'C3_MixedA']:
            b_rows = [r for r in baseline if r['condition'] == cond and r['run_id'] == self.run_id]
            c_rows = [r for r in cued if r['condition'] == cond and r['run_id'] == self.run_id]
            
            if not b_rows: continue
            
            acc = sum(1 for r in b_rows if r['correct'] == 'True') / len(b_rows)
            
            pref_rows = [r for r in c_rows if r['cue_type'] == 'prefix']
            pref_rec = sum(1 for r in pref_rows if r['success_topk'] == 'True') / len(pref_rows) if pref_rows else 0
            
            fus_rows = [r for r in c_rows if r['cue_type'] == 'fusion']
            fus_rec = sum(1 for r in fus_rows if r['success_topk'] == 'True') / len(fus_rows) if fus_rows else 0
            
            summary['metrics'][cond] = {
                'baseline_accuracy': acc,
                'prefix_recall': pref_rec,
                'fusion_recall': fus_rec
            }
        
        # Calc Task B Stats
        if cross_extract:
            rows = [r for r in cross_extract if r['run_id'] == self.run_id]
            if rows:
                acc = sum(1 for r in rows if r['correct'] == 'True') / len(rows)
                summary['metrics']['C4_MixedB_TaskB'] = {'extract_accuracy': acc}

        # Calc Degradation
        # M_control = avg(C1, C2)
        # M_mixedA = C3
        if 'C1_JP' in summary['metrics'] and 'C2_EN' in summary['metrics'] and 'C3_MixedA' in summary['metrics']:
            c1 = summary['metrics']['C1_JP']
            c2 = summary['metrics']['C2_EN']
            c3 = summary['metrics']['C3_MixedA']
            
            for key in ['baseline_accuracy', 'prefix_recall']:
                avg_no = (c1[key] + c2[key]) / 2
                deg = (avg_no - c3[key]) / avg_no if avg_no > 0 else 0
                summary['degradation'][key] = deg

        with open(RUN_SUMMARY_PATH, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate MD Report
        self.generate_report(summary)

    def generate_report(self, summary):
        lines = ["# P2 Word Gen Validation Report", "", "## Metrics"]
        for cond, metrics in summary['metrics'].items():
            lines.append(f"### {cond}")
            for k, v in metrics.items():
                lines.append(f"- {k}: {v:.2%}")
        
        lines.append("## Degradation (Mixed-A vs Controls)")
        for k, v in summary['degradation'].items():
            lines.append(f"- {k}: {v:.2%}")
            
        with open(REPORT_MD_PATH, 'w') as f:
            f.write('\n'.join(lines))

    def run(self):
        # Clear logs
        for p in [CSV_BASELINE_PATH, CSV_CUED_PATH, CSV_CROSS_PATH, CSV_CROSS_EXTRACT_PATH]:
            if os.path.exists(p): os.remove(p)
            
        # C1: JP Only
        self.setup_condition('C1_JP')
        self.run_phase2_baseline('C1_JP')
        self.run_phase3_prefix('C1_JP')
        self.run_phase4_fusion('C1_JP')
        
        # C2: EN Only
        self.setup_condition('C2_EN')
        self.run_phase2_baseline('C2_EN')
        self.run_phase3_prefix('C2_EN')
        self.run_phase4_fusion('C2_EN')
        
        # C3: Mixed A
        self.setup_condition('C3_MixedA')
        self.run_phase2_baseline('C3_MixedA')
        self.run_phase3_prefix('C3_MixedA')
        self.run_phase4_fusion('C3_MixedA')
        
        # C4: Mixed B
        self.setup_condition('C4_MixedB')
        self.run_mixed_b_cross('C4_MixedB')
        self.run_mixed_b_task_b('C4_MixedB') # Added Task B
        
        self.calculate_metrics()

if __name__ == "__main__":
    exp = WordGenerationExperimentP2()
    exp.run()
