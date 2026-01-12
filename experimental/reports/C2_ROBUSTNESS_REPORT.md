# C2 Robustness Verification Report (Spec Based)

**Date**: 2026-01-12
**Status**: ❌ FAIL

## 1. Metrics Summary
*   **Avg Resonance (Same Meaning)**: 0.7310 (Target > 0.85)
*   **Avg Resonance (Diff Meaning)**: 0.5297
*   **Separation ($\Delta$)**: 0.2013 (Target > 0.25)

## 2. Test Case Analysis

### T1: Notation Independence (x+y vs y+x vs a+b)
| query_id   | target_id   | group_a   | group_b   |   resonance | expected_match   | class_pair   |
|:-----------|:------------|:----------|:----------|------------:|:-----------------|:-------------|
| T1_01      | T1_02       | GRP_ADD   | GRP_ADD   |           1 | True             | T1-T1        |
| T1_01      | T1_03       | GRP_ADD   | GRP_ADD   |           1 | True             | T1-T1        |
| T1_02      | T1_03       | GRP_ADD   | GRP_ADD   |           1 | True             | T1-T1        |

### T2: Cross-Modality (Math vs Code)
| query_id   | target_id   | group_a   | group_b   |   resonance | expected_match   | class_pair   |
|:-----------|:------------|:----------|:----------|------------:|:-----------------|:-------------|
| T2_01      | T2_02       | GRP_COND  | GRP_COND  |  -0.0759613 | True             | T2-T2        |

### T4: Irrelevant (Separation Check)
| query_id   | target_id   | group_a   | group_b       |   resonance | expected_match   | class_pair   |
|:-----------|:------------|:----------|:--------------|------------:|:-----------------|:-------------|
| T1_01      | T1_04       | GRP_ADD   | GRP_CMP       |   0.761904  | False            | T1-T1        |
| T1_01      | T2_01       | GRP_ADD   | GRP_COND      |   0.459397  | False            | T1-T2        |
| T1_01      | T2_02       | GRP_ADD   | GRP_COND      |  -0.0392931 | False            | T1-T2        |
| T1_01      | T3_01       | GRP_ADD   | GRP_COND_DIFF |   0.459397  | False            | T1-T3        |
| T1_01      | T4_01       | GRP_ADD   | GRP_ASSIGN    |   0.882323  | False            | T1-T4        |

## 3. Detailed Matrix (Top Pairs)
| query_id   | target_id   | group_a       | group_b       |   resonance | expected_match   | class_pair   |
|:-----------|:------------|:--------------|:--------------|------------:|:-----------------|:-------------|
| T1_01      | T1_01       | GRP_ADD       | GRP_ADD       |           1 | True             | T1-T1        |
| T1_02      | T1_02       | GRP_ADD       | GRP_ADD       |           1 | True             | T1-T1        |
| T4_01      | T4_02       | GRP_ASSIGN    | GRP_MUL       |           1 | False            | T4-T4        |
| T4_01      | T4_01       | GRP_ASSIGN    | GRP_ASSIGN    |           1 | True             | T4-T4        |
| T1_01      | T1_02       | GRP_ADD       | GRP_ADD       |           1 | True             | T1-T1        |
| T1_03      | T1_03       | GRP_ADD       | GRP_ADD       |           1 | True             | T1-T1        |
| T1_02      | T1_03       | GRP_ADD       | GRP_ADD       |           1 | True             | T1-T1        |
| T4_02      | T4_02       | GRP_MUL       | GRP_MUL       |           1 | True             | T4-T4        |
| T1_01      | T1_03       | GRP_ADD       | GRP_ADD       |           1 | True             | T1-T1        |
| T3_01      | T3_01       | GRP_COND_DIFF | GRP_COND_DIFF |           1 | True             | T3-T3        |

## 4. Conclusion
SIR v1.0 Core Integration failed to demonstrates semantic robustness across notation and limited modality differences in the Sandbox environment.
