# Digit Generation Experiment Specification

0. Document Role
Authoritative, agent-readable specification for the Digit Generation Experiment using DHM.

1. Objective
Demonstrate that decimal digits {0,1,2,3,4,5,6,7,8,9} can be realized from abstract attribute holograms without storing digit-level symbol memory.

2. Hard Constraints
* Digit symbol holograms H_0 … H_9 MUST NOT be stored.
* MemorySpace MUST contain only attribute holograms.
* No learning, no gradient descent.
* DDIM mode MUST be deterministic.

3. Symbol Set
DIGITS = {0,1,2,3,4,5,6,7,8,9}

4. Attribute System
TYPE = {digit}
PARITY = {even, odd}
MAGNITUDE = {low, mid, high}
PRIME_STATUS = {prime, composite, neither}
LOOP_COUNT = {0loop, 1loop, 2loop}
STROKE_CLASS = {straight, curved, mixed}

5. Digit → Attribute Mapping
(See implementation or full spec)
0: even, low, neither, 1loop, curved
...
9: odd, high, composite, 1loop, mixed

6. Holographic Representation
Space: ℂ^D (D=2048 recommended)
Encoding: Seed -> Real -> FFT -> Normalize

8. Query Construction
H_0(d) = Normalize( ⊙ H_attr )

10. Evaluation & Extraction
R(H_q, H_m) = | H_q · conj(H_m) |
Extraction based on highest resonance.

11. Success Criteria
Top-1 accuracy = 10/10
Margin > 0
INVALID flags = 0
