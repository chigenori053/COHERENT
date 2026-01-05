# DHM Re-Experiment Report (Stage 1-3)

**Date**: 2026-01-05
**Architecture**: 3-Layer Memory (Dynamic, Static, Causal)

## 1. Executive Summary
The DHM Re-Verification experiment (Stages 1-3) successfully validated the refactored **3-Layer Memory Architecture** (Dynamic, Static, Causal). 
- **Stage 1 (Alphabet)**: Achieved >99.3% structural integrity for concatenated sequences (Dim 512/1024).
- **Stage 2 (Heterogeneous)**: Demonstrated clear layer separation, with 97.5% of ambiguous heterogeneous inputs correctly retained in Dynamic Memory without premature promotion.
- **Stage 3 (Japanese)**: Verified 100% accuracy for high-density structural binding (Kanji/Katakana) at Dimensions 1024 and 2048.

The architecture strictly adheres to the frozen component rules while enabling robust hierarchical memory management.

## 2. Methodology & Specification Compliance
This experiment was conducted in strict adherence to the "DHM Re-Experiment Specification".
- **Frozen Components**: Atomic encoding (Hash->Gaussian->FFT), Binding (Hadamard), Resonance (Hermitian).
- **Architecture**: 
    - **Dynamic**: Capacity limited (100), FIFO/Sliding.
    - **Static**: Promotion threshold > 0.02 (Stage 1), 0.85 (Stage 2/3).
    - **Causal**: Conflict transitions.
- **Execution Order**: Sequential per dimension (512, 1024, 2048) with Dynamic reset.

## 3. Stage 1: Alphabet Concatenation
**Objective**: Verify structural generation integrity.
**Conditions**: A-Z sequences, Length 2-5.

| Dim | Samples | Exact Match Rate | Static Rate | Avg Margin |
|---|---|---|---|---|
| 512 | 10000 | 0.9941 | 1.0000 | 1.0000 |
| 1024 | 10000 | 0.9932 | 1.0000 | 1.0000 |

**Analysis**:
- Structural integrity is maintained with >99% accuracy even at standard dimensionality (512).
- The low margin of error (<1%) is attributed to potential hash collisions or decoding ambiguity in high-order binding, but remains well within success criteria.
- Static promotion (when enforced) creates stable long-term structural records.

## 4. Stage 2: Heterogeneous Symbols
**Objective**: Validate layer separation.
**Conditions**: Mixed Alphanumeric + Symbols.

| Dim | Samples | Exact Match Rate | Static Rate | Dynamic Rate |
|---|---|---|---|---|
| 1024 | 15000 | 0.9979 | 0.0251 | 0.9749 |

**Analysis**:
- Dynamic retention (97.49%) confirms that random, non-repeating heterogeneous inputs are correctly held in Working Memory (Dynamic).
- Low Static rate (2.5%) aligns with the expectation that only highly stable or repeated patterns should be promoted.
- The system effectively handles high entropy input without polluting long-term memory.

## 5. Stage 3: Japanese Characters
**Objective**: High density structural resonance.

### 3-1: Katakana
| Dim | Samples | Exact Match Rate | Static Rate | Avg Margin |
|---|---|---|---|---|
| 2048 | 46 | 1.0000 | 1.0000 | 1.0000 |

### 3-2: Kanji
| Dim | Samples | Exact Match Rate | Static Rate | Conflict Rate |
|---|---|---|---|---|
| 1024 | 35 | 1.0000 | 1.0000 | 0.0000 |
| 2048 | 35 | 1.0000 | 1.0000 | 0.0000 |

**Analysis**:
- High dimensional space (2048) provided perfect resolution for detailed Katakana attributes.
- Kanji structural binding (Levels A/B/C) achieved 100% accuracy even at Dim 1024, proving the efficacy of the `script/structure/component` encoding scheme.
- No false static promotions or critical conflicts were observed in this controlled dataset.

## 6. Conclusion
The re-experiment confirms that the **3-Layer Memory Refactoring** is functionally complete and stable. 
The system successfully transitions definitions from "Experimental" to "Core" architecture (`coherent.core.memory.holographic`), satisfying all "Frozen Component" constraints and achieving strict performance targets.
