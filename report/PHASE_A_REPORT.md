# Phase A: Behavior Test Detailed Report

**Date**: Mon Jan 19 18:04:59 JST 2026
**Total Cases**: 10

## Executive Summary

| ID | Test Case | Status | Recall | Compute | Decision |
|----|-----------|--------|--------|---------|----------|
| **TC-A1-1** | Exact Match Recall (Phase A-1) | **PASS** | True | False | ACCEPT |
| **TC-A1-2** | Associativity Difference (Phase A-1) | **PASS** | False | True | ACCEPT |
| **TC-A1-3** | Reduction Difference (Phase A-1) | **PASS** | False | True | ACCEPT |
| **TC-A1-4** | Term Order Difference (Phase A-1) | **PASS** | False | True | ACCEPT |
| **TC-A2-1** | Compute Fallback: Associativity (Phase A-2) | **PASS** | False | True | ACCEPT |
| **TC-A2-2** | Compute Fallback: Reduction (Phase A-2) | **PASS** | False | True | ACCEPT |
| **TC-A2-3** | Compute Fallback: Order (Phase A-2) | **PASS** | False | True | ACCEPT |
| **TC-A3-1** | Decision Consistency: Accept (Phase A-3) | **PASS** | False | True | ACCEPT |
| **TC-A3-2** | Decision Consistency: Multistep (Phase A-3) | **PASS** | False | True | ACCEPT |
| **TC-A4-Run1** | Learning Impact: Before (Phase A-4) | **PASS** | False | True | ACCEPT |

## Detailed Results

### TC-A1-1: Exact Match Recall (Phase A-1)
- **Objective**: Verify recall works for identical structure
- **Input**: `x + y`
- **Precondition**: Learned 'x + y'
- **AST Hash**: `0918139ae89287c4...`

#### Component Execution
- **Recall**: ✅ `TRUE`
  - Reason: Exact match found
- **Compute**: ⚪️ `FALSE`
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.95`
  - Entropy: `0.10`
  - Reason: Recalled from memory

---

### TC-A1-2: Associativity Difference (Phase A-1)
- **Objective**: Verify recall fails for associativity diff
- **Input**: `(x + y) + z`
- **Precondition**: Learned 'x + (y + z)'
- **AST Hash**: `1a0184fec62c3ba2...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `x + y + z`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A1-3: Reduction Difference (Phase A-1)
- **Objective**: Verify recall fails for unreduced form
- **Input**: `2x`
- **Precondition**: Learned 'x + x'
- **AST Hash**: `882b72a7ce9838dd...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `2*x`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A1-4: Term Order Difference (Phase A-1)
- **Objective**: Verify recall fails for swapped terms
- **Input**: `5b + 3a`
- **Precondition**: Learned '3a + 5b'
- **AST Hash**: `b4566597ad45f39c...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `3*a + 5*b`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A2-1: Compute Fallback: Associativity (Phase A-2)
- **Objective**: Verify compute executes when recall fails
- **Input**: `(x + y) + z`
- **Precondition**: None
- **AST Hash**: `1a0184fec62c3ba2...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `x + y + z`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A2-2: Compute Fallback: Reduction (Phase A-2)
- **Objective**: Verify compute simplifies expression
- **Input**: `x + x`
- **Precondition**: None
- **AST Hash**: `662829238975ff79...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `2*x`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A2-3: Compute Fallback: Order (Phase A-2)
- **Objective**: Verify compute reorders terms
- **Input**: `5b + 3a`
- **Precondition**: None
- **AST Hash**: `b4566597ad45f39c...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `3*a + 5*b`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A3-1: Decision Consistency: Accept (Phase A-3)
- **Objective**: Verify high confidence for correct calc
- **Input**: `x + x`
- **Precondition**: None
- **AST Hash**: `662829238975ff79...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `2*x`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A3-2: Decision Consistency: Multistep (Phase A-3)
- **Objective**: Verify Accept/Review for complex calc
- **Input**: `(x + y) + z`
- **Precondition**: None
- **AST Hash**: `1a0184fec62c3ba2...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `x + y + z`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

### TC-A4-Run1: Learning Impact: Before (Phase A-4)
- **Objective**: Initial state before episodic learning
- **Input**: `(x + y) + z`
- **Precondition**: Learned diff structure
- **AST Hash**: `1a0184fec62c3ba2...`

#### Component Execution
- **Recall**: ❌ `FALSE`
  - Reason: No match found
- **Compute**: ✅ `TRUE`
  - Result: `x + y + z`
  - Steps: 5
- **Decision**: 🟢 `ACCEPT`
  - Confidence: `0.80`
  - Entropy: `0.30`
  - Reason: Computed and verified

---

