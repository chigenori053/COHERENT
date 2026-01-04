# Failure Modes and Diagnostics

## Status Codes
* **SUCCESS**: Top-1 match correct, high confidence.
* **WARN**: Correct match but low margin or minor anomalies.
* **INVALID**: Logic violation (e.g. normalization failure).

## Flags
* **L1_NORM_VIOLATION**: Vector L2 norm deviated from 1.0.
* **P1_INCORRECT_SYMBOL**: Top-1 prediction did not match target.
* **P2_LOW_MARGIN**: Resonance margin between top-1 and top-2 too small.
