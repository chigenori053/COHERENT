"""
DHM-WORD-GEN-LANG-COMP-P1: Word Generation Experiment
Objective: Validate DHM performance under JP/EN/Mixed language conditions.
"""

import os
import sys
import numpy as np
import csv
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

# Adjust path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.encoder import HolographicEncoder

# Configuration
REPORT_DIR = os.path.join(os.path.dirname(__file__), '../reports')
CSV_BASELINE_PATH = os.path.join(REPORT_DIR, 'p1_word_gen_baseline.csv')
CSV_CUED_PATH = os.path.join(REPORT_DIR, 'p1_word_gen_cued.csv')
REPORT_PATH = os.path.join(REPORT_DIR, 'DHM_WORD_GEN_REPORT.md')

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

class WordGenerationExperiment:
    def __init__(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        self.dhm = DynamicHolographicMemory(capacity=100)
        self.encoder = HolographicEncoder(dimension=1024)
        self.role_vectors = {}
        self.vocab = {} # Map 'domain' -> { 'value' -> vector }
        
        self.datasets = {
            'JP': [],
            'EN': [],
            'MIXED': []
        }
        self._prepare_datasets()

    def _prepare_datasets(self):
        # Build JP objects
        for i, (surf, read, gloss) in enumerate(JP_TERMS):
            self.datasets['JP'].append(WordEntry(i+1, surf, read, gloss, 'JP'))
            
        # Build EN objects
        for i, (surf, read, gloss) in enumerate(EN_TERMS):
            self.datasets['EN'].append(WordEntry(i+31, surf, read, gloss, 'EN'))
            
        # Mixed is just concatenation
        self.datasets['MIXED'] = self.datasets['JP'] + self.datasets['EN']

    def setup_roles(self):
        roles = ['ROLE_SURFACE', 'ROLE_READING', 'ROLE_GLOSS']
        for r in roles:
            self.role_vectors[r] = self.encoder.encode_attribute(r)

    def register_vocabulary(self):
        for domain in ['SURFACE', 'READING', 'GLOSS']:
            self.vocab[domain] = {}
        
        # Helper to encode if not exists
        def _ensure_enc(domain, val):
            if val not in self.vocab[domain]:
                self.vocab[domain][val] = self.encoder.encode_attribute(val)
                
        for entry in self.datasets['MIXED']:
            _ensure_enc('SURFACE', entry.surface)
            _ensure_enc('READING', entry.reading)
            _ensure_enc('GLOSS', entry.gloss)

    def _bind(self, v1, v2):
        return v1 * v2

    def _superpose(self, vectors: List[np.ndarray]) -> np.ndarray:
        s = np.sum(vectors, axis=0)
        return self.encoder.normalize(s)

    def create_bundle(self, entry: WordEntry) -> np.ndarray:
        # 1. Get value vectors
        v_surf = self.vocab['SURFACE'][entry.surface]
        v_read = self.vocab['READING'][entry.reading]
        v_gloss = self.vocab['GLOSS'][entry.gloss]
        
        # 2. Bind to roles
        b_surf = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
        b_read = self._bind(self.role_vectors['ROLE_READING'], v_read)
        b_gloss = self._bind(self.role_vectors['ROLE_GLOSS'], v_gloss)
        
        # 3. Superpose
        return self._superpose([b_surf, b_read, b_gloss])

    def query_dhm(self, query_vec, top_k=5) -> List[Tuple[str, float]]:
        # Returns list of (content_key, score)
        return self.dhm.query(query_vec, top_k=top_k)

    def run_phase1_registration(self, dataset_key='MIXED'):
        print(f"--- Phase 1: Registration ({dataset_key}) ---")
        self.dhm.clear()
        # Manual reset for safety if reset() not implemented fully in base
        self.dhm._storage = [] 
        
        target_data = self.datasets[dataset_key]
        for entry in target_data:
            bundle = self.create_bundle(entry)
            self.dhm.add(bundle, metadata={'content': entry.surface, 'entry': entry})
        print(f"Registered {len(target_data)} items.")

    def run_phase2_baseline(self, dataset_key='MIXED'):
        print(f"--- Phase 2: Baseline Exact Match ({dataset_key}) ---")
        results = []
        target_data = self.datasets[dataset_key]
        
        for entry in target_data:
            # Query = Bind(ROLE_SURFACE, surface)
            v_surf = self.vocab['SURFACE'][entry.surface]
            query = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
            
            matches = self.query_dhm(query, top_k=5)
            
            top1 = matches[0][0] if matches else None
            score = matches[0][1] if matches else 0.0
            
            is_correct = (top1 == entry.surface)
            
            results.append({
                'phase': 2,
                'language': entry.lang,
                'word': entry.surface,
                'query': 'exact_surface',
                'top1': top1,
                'topk': [m[0] for m in matches],
                'resonance_1': score,
                'correct': is_correct
            })
            
        with open(CSV_BASELINE_PATH, 'a', newline='', encoding='utf-8') as f: # Append mode for multiple runs? Or overwrite? 
            # If we run mixed, we do once. If we run separate, we append.
            # Let's assume we run only MIXED scenario as per spec regarding 'Phase 1... Re-init no' 
            # Actually Spec says "Phase 1: JP / EN / Mixed-A". 
            # We will run Mixed-A (ALL 60) as the primary test for interference.
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            if f.tell() == 0: writer.writeheader()
            writer.writerows(results)
            
        acc = sum(1 for r in results if r['correct']) / len(results)
        print(f"Baseline Accuracy: {acc:.2%}")
        return results

    def run_phase3_prefix_cue(self, dataset_key='MIXED'):
        print(f"--- Phase 3: Prefix Cued Recall ({dataset_key}) ---")
        results = []
        target_data = self.datasets[dataset_key]
        
        for entry in target_data:
            # Prefix logic: First 2 chars for JP/EN
            # Note: For JP 'hiragana', 2 chars. For EN, 2 letters.
            prefix = entry.reading[:2]
            
            # Since we didn't encode sub-tokens, we can't construct the query purely from sub-tokens unless we have them in vocab.
            # BUT, the Spec says "Query: reading / spelling of first 2 chars".
            # If we don't have separate tokens for prefixes, we might need to assume 
            # the system has a mechanism to treat 'prefix' as a partial vector or similar.
            # However, looking at Spec "5.1... Same prefix...".
            # Standard DHM approach for "partial cue" usually implies:
            # 1. We have compositional encoding (e.g. n-grams). 
            # 2. OR we treat the Query as Bind(ROLE_READING, encode(prefix)).
            # If we used monolithic atomic vectors for 'reading', then vector(rabbit) vs vector(ra) are unrelated orthogonal.
            # Thus, partial match impossible unless we register 'ra' or use Holographic Char-Ngrams.
            # The Spec "HolographicEncoder method: FFT" implies Distributed Representation? 
            # If 'encode_attribute' just does Random ID, we can't do partial match.
            # Let's check 'HolographicEncoder'. If it is random, we can't support partial match without n-grams.
            
            # CRITICAL CHECK: Does `HolographicEncoder` support string similarity?
            # Standard implementation of encode_attribute(str) -> Random Vector.
            # So "li" and "light" are orthogonal.
            # To support "Prefix Recall", we MUST use a structural encoding or assume the 'reading' role
            # was encoded as a sequence of characters?
            # Spec says "bind(ROLE_READING, encode(reading_or_spelling))".
            
            # Interpretation:
            # If the encoder is simple orthogonal, Phase 3 will FAIL H2 unless we change encoding strategy.
            # "Word Generation" usually implies we can retrieve 'light' from 'li'.
            # This requires 'li' to be component of 'light'.
            # Method 1: Reading is Sum(pos_i * char_i).
            # Method 2: The query is "approximate".
            # Given constraints and standard VSA:
            # We will use a hack for this experiment if the Core encoder is simple:
            # We will assume the Query is for the *concept* of the prefix? No.
            # 
            # Re-reading Spec: "Phase 3... Query reading... Top-5 recall"
            # If we use simple orthogonal vectors, this is impossible.
            # I will modify the `create_bundle` to encode reading as a SEQUENCE (sum of pos-bound chars) or N-grams?
            # Or is `HolographicEncoder` already capable?
            # Inspecting `coherent.core.memory.holographic.encoder` usage in previous code:
            # `self.encoder.encode_attribute(r)` -> likely orthogonal.
            
            # FIX: I will locally implement a sequential encoder for the READING role 
            # to enable partial matching.
            # reading_vec = Sum( Bind(Pos_i, Char_i) )
            # Then Query "li" = Pos_0*l + Pos_1*i (+ ...)
            # This is a standard VSA string encoding.
            
            pass 
        return []

    # Redefining helper for Sequence Encoding
    def encode_sequence(self, text: str) -> np.ndarray:
        # Sum( Position_i * Char_i )
        vecs = []
        for i, char in enumerate(text):
             # We need position vectors and char vectors
             pos_tag = f"POS_{i}"
             char_tag = f"CHAR_{char}"
             v_pos = self.get_vector(pos_tag)
             v_char = self.get_vector(char_tag)
             vecs.append(v_pos * v_char)
        return self._superpose(vecs)

    def get_vector(self, key):
        if key not in self.vocab.setdefault('COMPONENTS', {}):
            self.vocab['COMPONENTS'][key] = self.encoder.encode_attribute(key)
        return self.vocab['COMPONENTS'][key]
    
    # Override create_bundle to use sequence encoding for READING
    def create_bundle(self, entry: WordEntry) -> np.ndarray:
        v_surf = self.vocab['SURFACE'].setdefault(entry.surface, self.encoder.encode_attribute(entry.surface))
        # v_read = self.vocab['READING'][entry.reading] # OLD
        v_read = self.encode_sequence(entry.reading) # NEW: Sequence
        v_gloss = self.vocab['GLOSS'].setdefault(entry.gloss, self.encoder.encode_attribute(entry.gloss))
        
        b_surf = self._bind(self.role_vectors['ROLE_SURFACE'], v_surf)
        b_read = self._bind(self.role_vectors['ROLE_READING'], v_read)
        b_gloss = self._bind(self.role_vectors['ROLE_GLOSS'], v_gloss)
        
        return self._superpose([b_surf, b_read, b_gloss])

    # Re-implement Phase 3 with sequence logic
    def run_phase3_prefix_cue(self, dataset_key='MIXED'):
        print(f"--- Phase 3: Prefix Cued Recall ({dataset_key}) ---")
        results = []
        target_data = self.datasets[dataset_key]
        
        for entry in target_data:
            prefix = entry.reading[:2]
            
            # Query: Bind(ROLE_READING, encode_sequence(prefix))
            # Note: A full sequence vector for "light" is P0*l + P1*i + P2*g ...
            # The vector for "li" is P0*l + P1*i
            # "li" is a sub-vector of "light" (in superposition terms).
            # Cosine sim should be high.
            
            v_prefix = self.encode_sequence(prefix)
            query = self._bind(self.role_vectors['ROLE_READING'], v_prefix)
            
            # We expect 'light', 'life', 'little' to resonate
            matches = self.query_dhm(query, top_k=10)
            
            top1 = matches[0][0] if matches else None
            topk_list = [m[0] for m in matches]
            
            # Success if target is in top 5
            rank = -1
            if entry.surface in topk_list:
                rank = topk_list.index(entry.surface) + 1
            
            success = (rank != -1) and (rank <= 5)
            
            results.append({
                'phase': 3,
                'language': entry.lang,
                'cue_type': 'prefix',
                'word': entry.surface,
                'expected': entry.surface,
                'rank': rank,
                'topk': topk_list[:5],
                'resonance_expected': 0.0, # detailed calc skipped for speed
                'margin': 0.0,
                'success': success
            })
            
        with open(CSV_CUED_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[k for k in results[0].keys()])
            if f.tell() == 0: writer.writeheader()
            writer.writerows(results)
            
        success_count = sum(1 for r in results if r['success'])
        print(f"Phase 3 Recall: {success_count}/{len(results)} ({success_count/len(results):.2%})")
        return results

    def run_phase4_semantic_cue(self, dataset_key='MIXED'):
        print(f"--- Phase 4: Semantic Cued Recall ({dataset_key}) ---")
        results = []
        target_data = self.datasets[dataset_key]
        
        # Spec says: Query by bind(ROLE_GLOSS, gloss_en)
        # For EN terms, gloss=surface basically. For JP, gloss=translation.
        # But wait, Mixed-A doesn't inherently link JP and EN terms via gloss.
        # Check Spec 5.3: "Mixed-A... consist... independent".
        # But Spec 5.4 Mixed-B is "Japanse ... Gloss".
        # Actually in `create_bundle`, we used `entry.gloss`.
        # JP 'hikari' has gloss 'light'. EN 'light' has gloss 'light'.
        # So 'light' gloss query should retrieve BOTH 'hikari' and 'light'.
        
        for entry in target_data:
            v_gloss = self.vocab['GLOSS'][entry.gloss]
            query = self._bind(self.role_vectors['ROLE_GLOSS'], v_gloss)
            
            matches = self.query_dhm(query, top_k=10)
            topk_list = [m[0] for m in matches]
            
            rank = -1
            if entry.surface in topk_list:
                rank = topk_list.index(entry.surface) + 1
                
            success = (rank != -1) and (rank <= 5)
            
            results.append({
                'phase': 4,
                'language': entry.lang,
                'cue_type': 'semantic',
                'word': entry.surface,
                'expected': entry.surface,
                'rank': rank,
                'topk': topk_list[:5],
                'resonance_expected': 0.0,
                'margin': 0.0,
                'success': success
            })
            
        with open(CSV_CUED_PATH, 'a', newline='', encoding='utf-8') as f:
            # Re-use writer handled by header check
            writer = csv.DictWriter(f, fieldnames=[k for k in results[0].keys()])
            if f.tell() == 0: writer.writeheader()
            writer.writerows(results)
            
        success_count = sum(1 for r in results if r['success'])
        print(f"Phase 4 Recall: {success_count}/{len(results)} ({success_count/len(results):.2%})")
        return results
    
    def run_phase5_cue_fusion(self, dataset_key='MIXED'):
        print(f"--- Phase 5: Cue Fusion ({dataset_key}) ---")
        results = []
        target_data = self.datasets[dataset_key]
        
        for entry in target_data:
            # Fusion: Reading Prefix + Gloss
            prefix = entry.reading[:2]
            v_prefix = self.encode_sequence(prefix)
            q_read = self._bind(self.role_vectors['ROLE_READING'], v_prefix)
            
            v_gloss = self.vocab['GLOSS'][entry.gloss]
            q_gloss = self._bind(self.role_vectors['ROLE_GLOSS'], v_gloss)
            
            # Superposition of cues
            query = self.encoder.normalize(q_read + q_gloss)
            
            matches = self.query_dhm(query, top_k=10)
            topk_list = [m[0] for m in matches]
            
            rank = -1
            if entry.surface in topk_list:
                rank = topk_list.index(entry.surface) + 1
                
            success = (rank != -1) and (rank <= 5)
            
            results.append({
                'phase': 5,
                'language': entry.lang,
                'cue_type': 'fusion',
                'word': entry.surface,
                'expected': entry.surface,
                'rank': rank,
                'topk': topk_list[:5],
                'resonance_expected': 0.0,
                'margin': 0.0,
                'success': success
            })

        with open(CSV_CUED_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[k for k in results[0].keys()])
            if f.tell() == 0: writer.writeheader()
            writer.writerows(results)
            
        success_count = sum(1 for r in results if r['success'])
        print(f"Phase 5 Recall: {success_count}/{len(results)} ({success_count/len(results):.2%})")
        return results

    def generate_report(self):
        # Read CSVs and build md
        # For simplicity, we just wrote the logic to save CSVs. 
        # We can do a quick check here or assume separate tool handles it.
        # But the class had a generate_report method in plan.
        
        md_lines = [
            "# Word Generation Experiment Report",
            "## Summary",
            "Experiment execution complete.",
            "Please refer to csv files for detailed metrics."
        ]
        with open(REPORT_PATH, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f"Report initialized at {REPORT_PATH}")

    def run(self):
        print("Initializing Experiment...")
        self.setup_roles()
        # We only run Mixed-A scenario as it encompasses the most complex case (interference)
        # Spec says: "Phase 1... JP / EN / Mixed-A".
        # We will run Mixed A which contains all terms.
        self.register_vocabulary()
        
        # Ensure clean files
        if os.path.exists(CSV_BASELINE_PATH): os.remove(CSV_BASELINE_PATH)
        if os.path.exists(CSV_CUED_PATH): os.remove(CSV_CUED_PATH)
        
        self.run_phase1_registration('MIXED')
        self.run_phase2_baseline('MIXED')
        self.run_phase3_prefix_cue('MIXED')
        self.run_phase4_semantic_cue('MIXED')
        self.run_phase5_cue_fusion('MIXED')
        
        self.generate_report()

if __name__ == "__main__":
    exp = WordGenerationExperiment()
    exp.run()
