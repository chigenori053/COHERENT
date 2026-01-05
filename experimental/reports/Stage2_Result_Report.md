# Stage 2 Experiment Result Report

**Date:** 2026-01-05
**Experiment ID:** DHM-Stage2
**Status:** Completed (Binding Failure Detected)

## 1. Overview

**Objective:** Validate structural separability and extraction stability under heterogeneous symbol binding (Letters, Digits, Symbols) using Dynamic Holographic Memory (DHM).

**Hypothesis (H1):** DHM can preserve and extract ordered symbolic structures even when atomic symbols belong to heterogeneous categories.

**Binding Procedure (Spec 4):**
The experiment implemented the specified Cumulative Hadamard Product binding:
$$ H_{seq} = \text{Normalize}(h_1 \odot h_2 \odot \dots \odot h_L) $$

## 2. Experimental Setup

*   **Symbols:** 42 heterogeneous symbols (A-Z, 0-9, Operators).
*   **Lengths:** L = 2, 3, 4.
*   **Sample Size:** 300 samples (100 per length).
*   **Candidate Set:** Included targets, permutations (order variants), and distractors.

## 3. Results Summary

| Metric | Result |
| :--- | :--- |
| **Exact Match Rate (EM)** | **53.67%** (161/300) |
| **Structural Confusion Rate (SCR)** | **46.33%** (139/300) |
| **Category Confusion Rate (CCR)** | **0.00%** (0/300) |

### Detailed Breakdown by Length

The failure rate (Structural Confusion) was consistent across all lengths where permutations existed in the candidate set.

*   When the correct target and its permutation (e.g., "A1" vs "1A") were both present in the candidate set, the resonance scores were mathematically identical:
    $$ | \langle (A \odot 1), (A \odot 1) \rangle | = | \langle (A \odot 1), (1 \odot A) \rangle | $$
    *(Commutativity: $A \odot B = B \odot A$)*

The 53% match rate is likely essentially random tie-breaking or floating-point noise favoring the correct order in some cases.

## 4. Failure Analysis

**Diagnosis:** **Binding Scheme Failure**

As defined in the Specification (Section 11), high structural confusion indicates a failure of the binding scheme to preserve order.

1.  **Commutativity:** The Hadamard product is commutative. The representation for sequence $[a, b]$ is identical to $[b, a]$.
2.  **Lack of Positional Info:** Unlike Stage 1, which explicitly bound positional vectors ($a \odot p_1 + b \odot p_2$), Spec 4 relied purely on the product of content vectors.
3.  **Heterogeneity:** Category confusion was 0%. The system successfully distinguished "A" vs "B" (content), but failed "AB" vs "BA" (structure).

## 5. Conclusion & Recommendations

The experiment **failed** to support Hypothesis H1 under the specified binding procedure. While DHM handles heterogeneous symbols well (zero category confusion), the simple multiplicative binding is insufficient for ordered sequences.

**Recommendations for Next Iteration:**
1.  **Adopt Position Binding:** Re-introduce the position-binding mechanism used in Stage 1 ($H = \sum S_i \odot P_i$).
2.  **Non-Commutative Binding:** Implement Permutation-based binding or Circular Convolution if superposition is not desired.
    *   $H_{seq} = S_1 \circledast S_2 \dots$ (Circular Convolution)
    *   $H_{seq} = S_1 \odot \Pi(S_2) \odot \Pi^2(S_3)$ (Permutation)

This result confirms that structural order must be explicitly encoded in the holographic state; it does not emerge from simple simultaneous variable binding.
