# Stage 1 Experiment Specification (Agent-Readable)

0. Purpose
Validate sequence-level generation capability (multi-character output).
Tests multi-symbol recall & ordered emission.

1. Scope
Target: COHERENT / MemorySpace-based generation pipeline.

2. Task Definitions
Task ID: S1-A (Sequence Recall Generation)
Input: sequence_id (string, unique)
Output: predicted_sequence (string, length L)
Target: target_sequence (string, length L)

3. Dataset Specification
3.1 Character Set: A–Z (uppercase only)
3.2 Sequence Lengths: L ∈ {2, 3, 4, 5}
3.3 Dataset Size: Per L: 1000 sequences (Total 4000)
3.4 Generation Rules: 50% Random, 30% Hamming=1, 10% Swap, 10% Edit=1.

4. Memory Registration Phase
register_memory(id, content, encoder, representation)

5. Query Phase
query_memory(id, input=id, mode="recall-first") -> Top-1 candidate.

6. Generation Phase
generate_sequence(recalled_memory) -> predicted_sequence (Length L)

7. Evaluation Metrics
Exact Match, Edit Distance, Position Accuracy.
Resonance Separation Margin.

8. Acceptance Criteria
Length L | Exact Match | Near-Miss Rate
2 | ≥ 99% | ≤ 2%
3 | ≥ 97% | ≤ 3%
4 | ≥ 95% | ≤ 3%
5 | ≥ 90% | ≤ 5%

9. Experiment Variants
use_fft: {true, false}
dimension_D: {256, 512, 1024}

10. Logging Specification
Log to CSV (User Override): run_id, seed, variant, metrics...

11. Output Artifacts
CSV metrics summary.
