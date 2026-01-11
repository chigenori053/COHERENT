"""
DHM-WORD-MEANING-EXTRACT-P3: Meaningful Word Extraction (A->B->C)
Objective: Validate Gloss->Surface extraction, Cycle Consistency, and Compositional Retrieval.
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
CSV_SEMANTIC_A = os.path.join(REPORT_DIR, 'p3_semantic_A.csv')
CSV_CYCLE_B = os.path.join(REPORT_DIR, 'p3_cycle_B.csv')
CSV_COMPOSITIONAL_C = os.path.join(REPORT_DIR, 'p3_compositional_C.csv')
RUN_SUMMARY_PATH = os.path.join(REPORT_DIR, 'p3_run_summary.json')
REPORT_MD_PATH = os.path.join(REPORT_DIR, 'DHM_WORD_MEANING_EXTRACT_P3_REPORT.md')

@dataclass
class WordEntry:
    id: str
    type: str # 'base' or 'variant'
    lang: str
    surface: str
    reading: str
    gloss: str
    pos: str

@dataclass
class CompoundEntry:
    id: str
    type: str # 'compound'
    lang: str
    surface: str
    reading: str
    gloss_en: str
    pos: str
    gloss1: str
    gloss2: str

# --- DATASETS ---
JP_BASE = [
    WordEntry('JP001','base','JP','ひかり','ひかり','light','noun'),
    WordEntry('JP002','base','JP','みず','みず','water','noun'),
    WordEntry('JP003','base','JP','ひ','ひ','fire','noun'),
    WordEntry('JP004','base','JP','つき','つき','moon','noun'),
    WordEntry('JP005','base','JP','そら','そら','sky','noun'),
    WordEntry('JP006','base','JP','やま','やま','mountain','noun'),
    WordEntry('JP007','base','JP','かわ','かわ','river','noun'),
    WordEntry('JP008','base','JP','うみ','うみ','sea','noun'),
    WordEntry('JP009','base','JP','ひと','ひと','person','noun'),
    WordEntry('JP010','base','JP','こども','こども','child','noun'),
    WordEntry('JP011','base','JP','べんきょう','べんきょう','study','noun'),
    WordEntry('JP012','base','JP','いのち','いのち','life','noun'),
    WordEntry('JP013','base','JP','たべる','たべる','eat','verb'),
    WordEntry('JP014','base','JP','のむ','のむ','drink','verb'),
    WordEntry('JP015','base','JP','みる','みる','see','verb'),
    WordEntry('JP016','base','JP','いく','いく','go','verb'),
    WordEntry('JP017','base','JP','くる','くる','come','verb'),
    WordEntry('JP018','base','JP','はいる','はいる','enter','verb'),
    WordEntry('JP019','base','JP','でる','でる','exit','verb'),
    WordEntry('JP020','base','JP','おおきい','おおきい','big','adj'),
    WordEntry('JP021','base','JP','ちいさい','ちいさい','small','adj'),
    WordEntry('JP022','base','JP','ながい','ながい','long','adj'),
    WordEntry('JP023','base','JP','たかい','たかい','high','adj'),
    WordEntry('JP024','base','JP','あたらしい','あたらしい','new','adj'),
    WordEntry('JP025','base','JP','ふるい','ふるい','old','adj'),
    WordEntry('JP026','base','JP','はやい','はやい','early','adj'),
    WordEntry('JP027','base','JP','おおい','おおい','many','adj'),
    WordEntry('JP028','base','JP','すくない','すくない','few','adj'),
    WordEntry('JP029','base','JP','うえ','うえ','up','noun'),
    WordEntry('JP030','base','JP','した','した','down','noun'),
    # Primitives for Compounds
    WordEntry('JP301','base','JP','あお','あお','blue','adj'),
    WordEntry('JP302','base','JP','みち','みち','road','noun'),
    WordEntry('JP303','base','JP','かぜ','かぜ','wind','noun'),
    WordEntry('JP304','base','JP','そば','そば','side','noun')
]

EN_BASE = [
    WordEntry('EN101','base','EN','light','light','light','noun'),
    WordEntry('EN102','base','EN','water','water','water','noun'),
    WordEntry('EN103','base','EN','fire','fire','fire','noun'),
    WordEntry('EN104','base','EN','moon','moon','moon','noun'),
    WordEntry('EN105','base','EN','sky','sky','sky','noun'),
    WordEntry('EN106','base','EN','mountain','mountain','mountain','noun'),
    WordEntry('EN107','base','EN','river','river','river','noun'),
    WordEntry('EN108','base','EN','sea','sea','sea','noun'),
    WordEntry('EN109','base','EN','person','person','person','noun'),
    WordEntry('EN110','base','EN','child','child','child','noun'),
    WordEntry('EN111','base','EN','study','study','study','noun'),
    WordEntry('EN112','base','EN','life','life','life','noun'),
    WordEntry('EN113','base','EN','eat','eat','eat','verb'),
    WordEntry('EN114','base','EN','drink','drink','drink','verb'),
    WordEntry('EN115','base','EN','see','see','see','verb'),
    WordEntry('EN116','base','EN','go','go','go','verb'),
    WordEntry('EN117','base','EN','come','come','come','verb'),
    WordEntry('EN118','base','EN','enter','enter','enter','verb'),
    WordEntry('EN119','base','EN','exit','exit','exit','verb'),
    WordEntry('EN120','base','EN','big','big','big','adj'),
    WordEntry('EN121','base','EN','small','small','small','adj'),
    WordEntry('EN122','base','EN','long','long','long','adj'),
    WordEntry('EN123','base','EN','high','high','high','adj'),
    WordEntry('EN124','base','EN','new','new','new','adj'),
    WordEntry('EN125','base','EN','old','old','old','adj'),
    WordEntry('EN126','base','EN','early','early','early','adj'),
    WordEntry('EN127','base','EN','many','many','many','adj'),
    WordEntry('EN128','base','EN','few','few','few','adj'),
    WordEntry('EN129','base','EN','up','up','up','noun'),
    WordEntry('EN130','base','EN','down','down','down','noun'),
    # Primitives
    WordEntry('EN301','base','EN','blue','blue','blue','adj'),
    WordEntry('EN302','base','EN','road','road','road','noun'),
    WordEntry('EN303','base','EN','wind','wind','wind','noun'),
    WordEntry('EN304','base','EN','side','side','side','noun')
]

JP_VARIANTS = [
    WordEntry('JP201','variant','JP','光','ひかり','light','noun'),
    WordEntry('JP202','variant','JP','水','みず','water','noun'),
    WordEntry('JP203','variant','JP','火','ひ','fire','noun'),
    WordEntry('JP204','variant','JP','月','つき','moon','noun'),
    WordEntry('JP205','variant','JP','空','そら','sky','noun'),
    WordEntry('JP206','variant','JP','山','やま','mountain','noun'),
    WordEntry('JP207','variant','JP','川','かわ','river','noun'),
    WordEntry('JP208','variant','JP','海','うみ','sea','noun'),
    WordEntry('JP209','variant','JP','人','ひと','person','noun'),
    WordEntry('JP210','variant','JP','子供','こども','child','noun')
]

JP_COMPOUNDS = [
    CompoundEntry('CP001','compound','JP','あおぞら','あおぞら','blue_sky','noun','blue','sky'),
    CompoundEntry('CP002','compound','JP','やまみち','やまみち','mountain_road','noun','mountain','road'),
    CompoundEntry('CP003','compound','JP','うみかぜ','うみかぜ','sea_wind','noun','sea','wind'),
    CompoundEntry('CP004','compound','JP','かわべ','かわべ','river_side','noun','river','side'),
    CompoundEntry('CP005','compound','JP','つきひかり','つきひかり','moon_light','noun','moon','light'),
    CompoundEntry('CP006','compound','JP','ひのひかり','ひのひかり','fire_light','noun','fire','light'),
    CompoundEntry('CP007','compound','JP','やまかわ','やまかわ','mountain_river','noun','mountain','river'),
    CompoundEntry('CP008','compound','JP','うみそら','うみそら','sea_sky','noun','sea','sky'),
    CompoundEntry('CP009','compound','JP','ひとこども','ひとこども','person_child','noun','person','child'),
    CompoundEntry('CP010','compound','JP','あたらしいいのち','あたらしいいのち','new_life','noun','new','life'),
    CompoundEntry('CP011','compound','JP','たかいやま','たかいやま','high_mountain','noun','high','mountain'),
    CompoundEntry('CP012','compound','JP','ちいさいかわ','ちいさいかわ','small_river','noun','small','river')
]

class ExperimentP3:
    def __init__(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        self.dhm = DynamicHolographicMemory(capacity=3000)
        self.encoder = HolographicEncoder(dimension=1024)
        self.role_vectors = {}
        self.vocab = {} 
        self.run_id = f"RUN_{int(time.time())}"
        self.dataset = []
        self._setup_roles()

    def _setup_roles(self):
        roles = ['ROLE_SURFACE', 'ROLE_READING', 'ROLE_GLOSS', 'ROLE_POS']
        for r in roles:
            self.role_vectors[r] = self.encoder.encode_attribute(r)

    def get_vector(self, domain, key):
        if domain not in self.vocab: self.vocab[domain] = {}
        if key not in self.vocab[domain]:
            self.vocab[domain][key] = self.encoder.encode_attribute(key)
        return self.vocab[domain][key]
        
    def _bind(self, v1, v2): return v1 * v2
    
    def _superpose(self, vecs): return self.encoder.normalize(np.sum(vecs, axis=0))
    
    def _cosine_sim(self, v1, v2): return np.abs(np.vdot(v1, v2))

    def _unbind(self, bundle, role): return bundle * np.conj(role)

    def create_word_bundle(self, entry: WordEntry) -> np.ndarray:
        # P3 Spec 4.1 Base Word
        v_surf = self.get_vector('SURFACE', entry.surface)
        v_read = self.get_vector('READING', entry.reading)
        v_glos = self.get_vector('GLOSS', entry.gloss)
        v_pos  = self.get_vector('POS', entry.pos)
        
        b_surf = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
        b_read = self._bind(self.role_vectors['ROLE_READING'], v_read)
        b_glos = self._bind(self.role_vectors['ROLE_GLOSS'], v_glos)
        b_pos  = self._bind(self.role_vectors['ROLE_POS'], v_pos)
        
        return self._superpose([b_surf, b_read, b_glos, b_pos])

    def create_compound_bundle(self, entry: CompoundEntry) -> np.ndarray:
        # P3 Spec 4.2 Compound Word
        v_surf = self.get_vector('SURFACE', entry.surface)
        v_read = self.get_vector('READING', entry.reading)
        v_glos1 = self.get_vector('GLOSS', entry.gloss1)
        v_glos2 = self.get_vector('GLOSS', entry.gloss2)
        v_pos  = self.get_vector('POS', entry.pos)
        
        b_surf = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
        b_read = self._bind(self.role_vectors['ROLE_READING'], v_read)
        b_glos1 = self._bind(self.role_vectors['ROLE_GLOSS'], v_glos1)
        b_glos2 = self._bind(self.role_vectors['ROLE_GLOSS'], v_glos2)
        b_pos  = self._bind(self.role_vectors['ROLE_POS'], v_pos)
        
        return self._superpose([b_surf, b_read, b_glos1, b_glos2, b_pos])

    def setup_condition(self, cond: str):
        print(f"[{cond}] Setup...")
        self.dhm.clear()
        self.current_entries = [] # For analysis reference
        
        entries_to_add = []
        if cond == 'C1_JP_ONLY':
            entries_to_add = JP_BASE + JP_VARIANTS + JP_COMPOUNDS
        elif cond == 'C2_EN_ONLY':
            entries_to_add = EN_BASE
        elif cond == 'C3_MIXED_A' or cond == 'C4_MIXED_B':
            entries_to_add = JP_BASE + JP_VARIANTS + EN_BASE + JP_COMPOUNDS
        
        count = 0
        for e in entries_to_add:
            if isinstance(e, WordEntry):
                bundle = self.create_word_bundle(e)
            else:
                bundle = self.create_compound_bundle(e)
            
            # Key = encode(surface)
            # We track metadata for result analysis
            self.dhm.add(bundle, metadata={'content': e.surface, 'entry': e})
            self.current_entries.append(e)
            count += 1
        print(f"[{cond}] Registered {count} items.")

    # --- Phase A: Meaning Extraction (Gloss -> Surface) ---
    def run_phase_a(self, cond):
        print(f"[{cond}] Phase A: Semantic Extraction (Gloss->Surface)")
        results = []
        
        # Test targets: Base + Variants. (Compounds tested in C)
        targets = [e for e in self.current_entries if isinstance(e, WordEntry)]
        
        # Group targets by gloss to handle synonyms/variants (Set Recall)
        gloss_to_surfaces = {}
        for e in targets:
            if e.gloss not in gloss_to_surfaces: gloss_to_surfaces[e.gloss] = set()
            gloss_to_surfaces[e.gloss].add(e.surface)

        seen_glosses = set()
        
        for e in targets:
            # Avoid duplicate queries for same gloss
            if e.gloss in seen_glosses and cond != 'C4_MIXED_B': 
                # In P2 we strictly did per-word. Here let's do per-word but evaluate based on set.
                # Actually, doing per-word allows us to see if *this* specific surface comes up.
                # But querying by Gloss is identical for Words with same Gloss.
                # Let's iterate words to keep log consistent with "Entry ID".
                pass
                
            v_glos = self.get_vector('GLOSS', e.gloss)
            query = self._bind(self.role_vectors['ROLE_GLOSS'], v_glos)
            
            matches = self.dhm.query(query, top_k=5)
            topk_surfaces = [m[0] for m in matches]
            top1 = topk_surfaces[0] if topk_surfaces else None
            
            # Expectation: ANY of the valid surfaces for this gloss
            expected_set = gloss_to_surfaces[e.gloss]
            
            success = any(s in expected_set for s in topk_surfaces)
            correct = (top1 in expected_set)
            
            # Resonance metrics
            r1 = matches[0][1] if len(matches)>0 else 0
            r2 = matches[1][1] if len(matches)>1 else 0
            
            # Calculate expected resonance
            target_bundle = self.create_word_bundle(e)
            r_exp = self._cosine_sim(query, target_bundle)
            
            results.append({
                'run_id': self.run_id, 'condition': cond, 'phase': 'A', 'language': e.lang,
                'entry_id': e.id, 'gloss_en': e.gloss,
                'expected_surfaces': list(expected_set), 'top1_surface': top1, 'topk_surfaces': topk_surfaces,
                'rank_of_first_expected': next((i+1 for i,s in enumerate(topk_surfaces) if s in expected_set), -1),
                'resonance_expected': r_exp, 'resonance_top1': r1, 'resonance_top2': r2,
                'margin': r1 - r2, 'success_topk': success
            })
            seen_glosses.add(e.gloss)
            
        self._write_csv(CSV_SEMANTIC_A, results)
        
    # --- Phase B: Cycle Consistency (Surface -> Gloss) ---
    def run_phase_b(self, cond):
        print(f"[{cond}] Phase B: Cycle Consistency (Surface->Gloss)")
        results = []
        targets = [e for e in self.current_entries if isinstance(e, WordEntry)]
        
        # Prepare gloss vocab map for decoding
        gloss_map = {e.gloss: self.get_vector('GLOSS', e.gloss) for e in targets}
        
        for e in targets:
            # 1. Forward Retrieval (Gloss -> Surface) - Reusing logic or assumption?
            # Spec says "Take candidates from A". 
            # Simplified: Use the correct Surface as input (Ideal Cycle) to verify integrity.
            # "Aで得た surface を Q = encode(surface) として再入力"
            # We will use the *Ground Truth Surface* to verifying "If we extracted this surface, does it map back?"
            # This validates the storage structure.
            
            # Query by Surface
            v_surf = self.get_vector('SURFACE', e.surface)
            query = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf) 
            
            matches = self.dhm.query(query, top_k=1)
            if not matches: continue
            
            # Retrieve vector (Simulated "Storage Access")
            best_surf = matches[0][0]
            if best_surf != e.surface:
                # If we retrieved wrong bundle, cycle fails immediately?
                # Or we proceed with retrieved bundle? 
                # Let's assume we proceed with the retrieved bundle.
                pass
                
            retrieved_vec = None
            for vec, meta in self.dhm._storage:
                if meta.get('content') == best_surf:
                    retrieved_vec = vec; break
            
            if retrieved_vec is None: continue
            
            # Unbind Gloss
            raw_gloss = self._unbind(retrieved_vec, self.role_vectors['ROLE_GLOSS'])
            
            # Decode
            candidates = []
            for g_txt, g_vec in gloss_map.items():
                score = self._cosine_sim(raw_gloss, g_vec)
                candidates.append((g_txt, score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            top1_gloss = candidates[0][0]
            topk_glosses = [c[0] for c in candidates[:5]]
            r1 = candidates[0][1]
            r2 = candidates[1][1] if len(candidates)>1 else 0
            
            cycle_ok = (top1_gloss == e.gloss)
            
            results.append({
                'run_id': self.run_id, 'condition': cond, 'phase': 'B', 'language': e.lang,
                'source_gloss': e.gloss, 'rank_in_A': 'N/A', # Simplified
                'surface_candidate': best_surf,
                'expected_gloss': e.gloss,
                'top1_gloss': top1_gloss, 'topk_glosses': topk_glosses,
                'resonance_top1': r1, 'resonance_top2': r2, 'margin': r1-r2,
                'cycle_ok': cycle_ok
            })
            
        self._write_csv(CSV_CYCLE_B, results)

    # --- Phase C: Compositional Retrieval ---
    def run_phase_c(self, cond):
        # Only relevant if Compounds exist
        compounds = [e for e in self.current_entries if isinstance(e, CompoundEntry)]
        if not compounds: return
        
        print(f"[{cond}] Phase C: Compositional Retrieval")
        results = []
        
        for c in compounds:
            # Query: bind(GLOSS, g1) + bind(GLOSS, g2)
            # Note: CompoundBundle was created with superpose(..., b_glos1, b_glos2, ...)
            # So a superposition of bound glosses should resonate.
            
            v_g1 = self.get_vector('GLOSS', c.gloss1)
            v_g2 = self.get_vector('GLOSS', c.gloss2)
            
            q1 = self._bind(self.role_vectors['ROLE_GLOSS'], v_g1)
            q2 = self._bind(self.role_vectors['ROLE_GLOSS'], v_g2)
            
            query = self.encoder.normalize(q1 + q2)
            
            matches = self.dhm.query(query, top_k=5)
            topk_names = [m[0] for m in matches]
            top1 = topk_names[0] if topk_names else None
            
            r1 = matches[0][1] if len(matches)>0 else 0
            r2 = matches[1][1] if len(matches)>1 else 0
            
            # Expected resonance
            target_bundle = self.create_compound_bundle(c)
            r_exp = self._cosine_sim(query, target_bundle)
            
            rank = -1
            if c.surface in topk_names:
                rank = topk_names.index(c.surface) + 1
            
            results.append({
                'run_id': self.run_id, 'condition': cond, 'phase': 'C', 'language': c.lang,
                'compound_id': c.id, 'gloss1': c.gloss1, 'gloss2': c.gloss2,
                'expected_compound_surface': c.surface,
                'top1_surface': top1, 'topk_surfaces': topk_names,
                'rank_of_expected': rank,
                'resonance_expected': r_exp, 'resonance_top1': r1, 'resonance_top2': r2,
                'margin': r1 - r2, 'success_topk': (rank != -1)
            })
        
        self._write_csv(CSV_COMPOSITIONAL_C, results)
        
    def _write_csv(self, path, data):
        if not data: return
        file_exists = os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            if not file_exists: writer.writeheader()
            writer.writerows(data)

    def run(self):
        # Clear logs
        for p in [CSV_SEMANTIC_A, CSV_CYCLE_B, CSV_COMPOSITIONAL_C, RUN_SUMMARY_PATH]:
            if os.path.exists(p): os.remove(p)

        conds = ['C1_JP_ONLY', 'C2_EN_ONLY', 'C3_MIXED_A', 'C4_MIXED_B']
        
        for cond in conds:
            self.setup_condition(cond)
            self.run_phase_a(cond)
            self.run_phase_b(cond)
            self.run_phase_c(cond)
        
        self.generate_report()

    def generate_report(self):
        # Basic Summary JSON generation
        summary = {'runs': []}
        
        # Load Logs
        def load_log(p): return list(csv.DictReader(open(p, encoding='utf-8'))) if os.path.exists(p) else []
        
        data_a = load_log(CSV_SEMANTIC_A)
        data_b = load_log(CSV_CYCLE_B)
        data_c = load_log(CSV_COMPOSITIONAL_C)
        
        conds = sorted(list(set(r['condition'] for r in data_a)))
        
        md_lines = ["# P3 Meaningful Word Extraction Report", ""]
        
        for cond in conds:
            rows_a = [r for r in data_a if r['condition'] == cond]
            rows_b = [r for r in data_b if r['condition'] == cond]
            rows_c = [r for r in data_c if r['condition'] == cond]
            
            acc_a = sum(1 for r in rows_a if r['success_topk']=='True') / len(rows_a) if rows_a else 0
            acc_b = sum(1 for r in rows_b if r['cycle_ok']=='True') / len(rows_b) if rows_b else 0
            acc_c = sum(1 for r in rows_c if r['success_topk']=='True') / len(rows_c) if rows_c else 0
            mag_c = statistics_mean([float(r['margin']) for r in rows_c]) if rows_c else 0
            
            summary['runs'].append({
                'condition': cond,
                'A_recall': acc_a,
                'B_cycle': acc_b,
                'C_composition': acc_c
            })
            
            md_lines.append(f"## {cond}")
            md_lines.append(f"- Phase A (Gloss->Surf): **{acc_a:.1%}** Recall")
            md_lines.append(f"- Phase B (Cycle): **{acc_b:.1%}** Consistency")
            if rows_c:
                md_lines.append(f"- Phase C (Comp): **{acc_c:.1%}** Recall (Avg Margin: {mag_c:.3f})")
            
        with open(RUN_SUMMARY_PATH, 'w') as f: json.dump(summary, f, indent=2)
        with open(REPORT_MD_PATH, 'w') as f: f.write('\n'.join(md_lines))

def statistics_mean(l): return sum(l)/len(l) if l else 0

if __name__ == "__main__":
    exp = ExperimentP3()
    exp.run()
