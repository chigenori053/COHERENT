# P2 Word Gen Validation Report

## Metrics
### C1_JP
- baseline_accuracy: 100.00%
- prefix_recall: 100.00%
- fusion_recall: 100.00%
### C2_EN
- baseline_accuracy: 100.00%
- prefix_recall: 100.00%
- fusion_recall: 100.00%
### C3_MixedA
- baseline_accuracy: 100.00%
- prefix_recall: 100.00%
- fusion_recall: 100.00%
### C4_MixedB_TaskB
- extract_accuracy: 100.00%
## Degradation (Mixed-A vs Controls)
- baseline_accuracy: 0.00%
- prefix_recall: 0.00%
### 3.3 Cross-Lingual Evaluation (Mixed-B)
**Objective**: Bidirectional association between JP Surface and EN Gloss.

| Task | Direction | Success (Top-5) | Margin (Mean) | Signal Ratio (Top1/Top2) |
| :--- | :--- | :---: | :---: | :---: |
| **Task A** | EN Gloss -> JP Surface | **100.0%** | 0.4840 | ~3.0x |
| **Task B** | JP Surface -> Extract EN Gloss | **100.0%** | 0.0151* | **~9.5x** |

> **Note on Task B Margin**: The absolute margin value (0.0151) is lower than Task A due to energy scaling inherent in the unbinding operation (`Bundle * conj(Role)`). Since the vectors are not re-normalized after unbinding, the dot product scales by approx $1/\sqrt{N}$. However, the **Signal Ratio** (Resonance Top 1 / Top 2) is extremely high (~9.5x), indicating that extraction is actually **more distinct** than forward association.

## 4. Hypothesis Verification

### H1: Stable Retention
> *DHM holds JP and EN sets stably.*
**Verified**. 100% Baseline Accuracy in all conditions.

### H2: Prefix Recall
> *Partial cues retrieve correct items.*
**Verified**. 100% Recall. Margin analysis shows JP prefixes are inherently more distinct than EN 2-letter prefixes (0.27 vs 0.15).

### H3: Mixed Coexistence (No Degradation)
> *Mixed condition maintains performance.*
**Verified**. Degradation rate is 0.00%. Co-storage of 60 items in 1024-dim space is well within capacity.

### H4: Cue Fusion
> *Fusion improves margin/certainty.*
**Verified**. In all conditions, Fusion margins are significantly higher than Prefix margins (e.g., JP: 0.27 -> 0.59).

### H5: Bidirectional Cross-Lingual Link (New)
> *DHM supports extraction of attributes (Task B) as well as associative retrieval (Task A).*
**Verified**. Both directions achieved 100% accuracy. The architecture successfully functions as a bidirectional associative memory.

## 5. Conclusion
The validation confirms that **Dynamic Holographic Memory is suitable for robust multilingual symbol grounding**. The architecture successfully implements:
1.  **Interference-Free Storage**: Zero degradation in mixed language setting.
2.  **Multimodal Access**: Retrieval via Surface, Partial Reading, Meaning, and Fusion.
3.  **Ambiguity Resolution**: Cue Fusion mechanism works as predicted, sharpening the retrieval focus.
4.  **Bidirectional Translation**: Semantic bridge (Role Gloss) allows 100% accurate translation between surface forms and meanings in both directions.

## 6. Artifacts
- **Logs**: `p2_baseline.csv`, `p2_cued.csv`, `p2_cross.csv`, `p2_cross_extract.csv`
- **Summary JSON**: `p2_run_summary.json`00%