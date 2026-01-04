# Hiragana Generation Experiment Specification

0. Objective
Realize Japanese hiragana characters from phonological attributes.

1. Hard Constraints
* Hiragana symbols MUST NOT be stored.
* Attribute storage only.
* Deterministic DDIM.

2. Phases
* Phase 1: Vowels (あいうえお)
* Phase 2: K-Row (かきくけこ)
* Phase 3: Full Seion (46 chars)

3. Attribute System
SCRIPT = {hiragana}
VOWEL = {a, i, u, e, o, n}  <-- Extending for 'n' (ん) or handling via Special Mark?
CONSONANT = {none, k, s, t, n, h, m, y, r, w}
VOICE = {voiceless}
SPECIAL_MARK = {none, nasal} <-- Extended for 'n'

4. Mapping (Canonical)
Procedural from Gojuon.
'あ' = VOWEL:a, CONSONANT:none
'か' = VOWEL:a, CONSONANT:k
...
'ん' = VOWEL:none/u?, SPECIAL:nasal? (Implementation decision required for uniqueness)

5. Query
H0 = Normalize( ⊙ H_attr )

6. Decoding
score = |<H_final, H_char>|
Top-1 prediction.

7. Success Criteria
Phase 3 Accuracy: 46/46
