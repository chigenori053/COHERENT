# Roman Numeral Generation Experiment Specification

0. Objective
Validate structured symbolic generation (integers 1-99 to Roman Numerals) using DHM without autoregression.

1. Hard Constraints
* Roman symbols (I, V, X, L, C) MUST NOT be stored in MemorySpace.
* Only attribute holograms are stored.
* No sequence prediction / autoregression.

2. Target Space
Integers: 1-99 (Phase 1)
Canonical Roman forms (e.g., 4->IV, 49->XLIX).

3. Attribute Design (Structural)
* Value Decomposition: HAS_ONES, HAS_TENS, ONES_MAG_LOW/SUB/MID, TENS_MAG_LOW/SUB/MID
* Structural: ORDER_TENS_FIRST, USE_SUBTRACTIVE, REPEAT_ALLOWED

4. Query Construction
H0(N) = Normalize( ⊙ encode(attribute_i) )

5. Decoding & Evaluation
* Generate candidate set C (all Roman forms for 1-99).
* Compute score(C) = |<H_final, H_C>|
* Top-1 prediction.

6. Success Criteria
* >= 95% Success
* No INVALID
