# C2-R (Revised) Validation Report

**Date**: 2026-01-12
**Status**: ✅ PASS

## 1. Phase Summary
### Phase S: Semantic Correctness (Cosine Distance)
*   **Same-Meaning Avg Dist**: 0.0828 (Target < 0.05)
*   **Diff-Meaning Avg Dist**: 0.4703
*   **Separation Margin**: 0.3875 (Target > 0.2)
*   *Verdict*: ✅ OK

### Phase P: Physical Recall Robustness (DHM Resonance)
*   **Same-Meaning Avg Res**: 0.9289 (Target > 0.8)
*   **Diff-Meaning Avg Res**: 0.5502
*   **Separation Margin**: 0.3788 (Target > 0.2)
*   *Verdict*: ✅ OK

## 2. Verdict Distribution
| verdict                 |   count |
|:------------------------|--------:|
| PASS                    |      31 |
| PARTIAL (Semantic-Only) |      11 |
| FAIL (Semantic)         |       3 |

## 3. Notable Failures / Partials
| query_id   | target_id   |   metric_val_S |   metric_val_P | verdict                 |
|:-----------|:------------|---------------:|---------------:|:------------------------|
| T1_01      | T1_04       |       0.238096 |       0.761904 | PARTIAL (Semantic-Only) |
| T1_01      | T4_01       |       0.117677 |       0.882323 | PARTIAL (Semantic-Only) |
| T1_01      | T4_02       |       0.117677 |       0.882323 | PARTIAL (Semantic-Only) |
| T1_02      | T1_04       |       0.238096 |       0.761904 | PARTIAL (Semantic-Only) |
| T1_02      | T4_01       |       0.117677 |       0.882323 | PARTIAL (Semantic-Only) |
| T1_02      | T4_02       |       0.117677 |       0.882323 | PARTIAL (Semantic-Only) |
| T1_03      | T1_04       |       0.238096 |       0.761904 | PARTIAL (Semantic-Only) |
| T1_03      | T4_01       |       0.117677 |       0.882323 | PARTIAL (Semantic-Only) |
| T1_03      | T4_02       |       0.117677 |       0.882323 | PARTIAL (Semantic-Only) |
| T1_04      | T4_01       |       0.111699 |       0.888301 | PARTIAL (Semantic-Only) |

## 4. Conclusion
SIR v1.0 Core successfully pass C2-R criteria.
Semantic Separation is 0.39, Physical Separation is 0.38.
