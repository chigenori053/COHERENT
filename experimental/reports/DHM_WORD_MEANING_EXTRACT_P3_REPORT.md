# Meaningful Word Extraction P3 Report
**Experiment ID:** DHM-WORD-MEANING-EXTRACT-P3
**Date:** 2026-01-11
**Status:** SUCCESS (100% All Phases)

## 1. Executive Summary
This experiment validated that DHM can serve as a "Meaning-Based Memory" where words are retrieved by their semantic definition (Gloss) rather than just surface form.
- **Phase A (Meaning Extraction)**: 100% success. DHM correctly retrieves words like "light" or "ひかり" solely from the semantic vector "light".
- **Phase B (Cycle Consistency)**: 100% success. Retrieving a word and then unbinding its meaning vector yields the correct original meaning, verifying the structural integrity of the stored bundle.
- **Phase C (Compositional Retrieval)**: 100% success. Providing a compound meaning (e.g., "blue" + "sky") correctly retrieves the compound word "あおぞら" (blue_sky), confirming that DHM supports compositional semantics via vector superposition.

## 2. Results by Condition
## C1_JP_ONLY
- Phase A (Gloss->Surf): **100.0%** Recall
- Phase B (Cycle): **100.0%** Consistency
- Phase C (Comp): **100.0%** Recall (Avg Margin: 0.262)
## C2_EN_ONLY
- Phase A (Gloss->Surf): **100.0%** Recall
- Phase B (Cycle): **100.0%** Consistency
## C3_MIXED_A
- Phase A (Gloss->Surf): **100.0%** Recall
- Phase B (Cycle): **100.0%** Consistency
- Phase C (Comp): **100.0%** Recall (Avg Margin: 0.256)
## C4_MIXED_B
- Phase A (Gloss->Surf): **100.0%** Recall
- Phase B (Cycle): **100.0%** Consistency
- Phase C (Comp): **100.0%** Recall (Avg Margin: 0.256)