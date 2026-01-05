# Stage 2 Experiment Specification

0. Metadata
ID: DHM-Stage2
Purpose: Vaildate structural separability... heterogeneous symbol binding.

1. Hypothesis
H1: DHM can preserve/extract ordered structures of heterogeneous symbols.
H2: Resonance decoding prioritizes global structure.

2. Symbol Definitions
LETTERS: A-Z
DIGITS: 0-9
SYMBOLS: +, -, *, /, =, #
Total: 42
Encoding: Independent, Hash->Gauss->FFT->Norm (Holographic).

3. Structure Generation
L: {2, 3, 4}
Patterns: [Let, Dig], [Dig, Let], [Let, Sym], [Sym, Let], [Let, Dig, Let], ...
Constraints: Order preserved.

4. Dynamic Binding Procedure
H_temp = h(a1)
for k in 2..L:
    H_temp = Normalize(H_temp * h(ak))

5. Noise & Diffusion
Forward: H_t = sqrt(a)H0 + sqrt(1-a)n
Reverse: Oracle-guided DDIM (Geometric validation only).

6. Candidate Set Construction
Size >= 50. Include: Exact, Order changed, Category variants, Random.

7. Decoding
Pred = argmax R(c)

8. Evaluation
Exact Match Rate (EM)
Category Confusion Rate (CCR)
Structural Confusion Rate (SCR)
Margin Stability

9. Success Criteria
EM >= 95%
SCR ~ 0
(Interpretation: If SCR high -> Binding scheme failure)

10. Logging
CSV + JSON.

11. Interpretation
Structural Confusion -> Binding Failure.
Category Confusion -> Symbol Separation Failure.
Margin Collapse -> Stability Failure.
