# DHM State Visualization & Logging Specification (Agent-Readable)

0. Purpose
This document defines a strict, agent-readable specification for visualizing and logging the Dynamic Holographic Memory (DHM) states in the Alphabet Generation Experiment.

1. Core Principle (Non-Negotiable)
* Visualization MUST be derived solely from experiment logs
* No visualization may depend on runtime-only or hidden variables

2. Visualization Targets
* Initial State: H_0
* Intermediate States: H_t for t ∈ {1 … T}
* Final State: H_final
* Pre-Success Boundary State: H_{t*} (margin minimal)

3. Projection & Representation Rules
* Default: PCA over concatenated real-imag space ([Re(H), Im(H)] ∈ ℝ^{2D})
* Same PCA basis MUST be used for all H_t and H_c within a run.

4. Mandatory Visualizations
* 4.1 State Trajectory Plot: H_0 → H_1 → … → H_T
* 4.2 Success State Geometry Plot: H_final vs H_c (all candidates)
* 4.3 Margin-Critical Comparison Plot: H_{t*} vs H_final

5. Logging Specification
* Metadata: run_id, mode, D, T, noise_scale, seed
* State Snapshots: H_t_real, H_t_imag, norm_H_t for required t.
* Resonance LOgs: scores, margin, argmax_symbol
* Diagnostic Logs: status, flags

6. Visualization–Log Mapping
Visualization derived strictly from logs.

7. Failure Handling
If status == INVALID, suppress success-oriented plots, render only H_0/H_final with error overlay.

8. Storage
* Vector data: NumPy binary (.npy/.npz) preferred for efficiency, or CSV.
* Metadata: JSON.

11. Integration Rule
Applies only to experimental/dhm_generation/
