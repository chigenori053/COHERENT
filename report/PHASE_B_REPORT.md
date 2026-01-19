# Phase B: Recall Boundary Sweep Report

**Date**: Mon Jan 19 18:18:13 JST 2026
## 1. Recall Boundary Table

| Theta | L0 | L1 | L2 | L3 | L4 | Saving |
|-------|----|----|----|----|----|--------|
| 0.95 | ✔ | ✖ | ✖ | ✖ | ✖ | 20.0% |
| 0.90 | ✔ | ✔ | ✖ | ✖ | ✖ | 40.0% |
| 0.85 | ✔ | ✔ | ✖ | ✖ | ✖ | 40.0% |
| 0.80 | ✔ | ✔ | ✔ | ✖ | ✖ | 60.0% |
| 0.75 | ✔ | ✔ | ✔ | ✖ | ✖ | 60.0% |
| 0.70 | ✔ | ✔ | ✔ | ✔ | ✖ | 80.0% |

## 2. Optimal Theta Selection

**Optimal Theta (θ*)**: `0.7`

**Rationale**:
- False Recall Rate: 0%
- Compute Reduction: 80.0%
- L4 (Composite Diff) Recall prevented (Safety)

## 3. Data Definitions
- **L0**: Exact Match (1.00)
- **L1**: Order Diff (0.92)
- **L2**: Reduction Diff (0.82)
- **L3**: Assoc Diff (0.72)
- **L4**: Composite Diff (0.62)
