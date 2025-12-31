# Parallel Reasoning Architecture Specification (Design Locked)
## COHERENT Cognitive Architecture Extension

- **Version**: 1.0  
- **Status**: Design Locked / Implementation Ready  
- **Last Updated**: 2025-12-31  

---

## 0. Purpose

This document consolidates all prior design discussions into a single, implementable specification for **Parallel Reasoning** in COHERENT.

**Parallel Reasoning** is defined as:

> A deterministic cognitive architecture that executes multiple independent reasoning branches (ThoughtBranches) in *logical parallel*, evaluates them via decision-theoretic utility, converges them using entropy-based criteria, and commits **only converged and validated** results to MemorySpace.

This is **not parallel computation** (multi-threaded speedup). It is **parallel cognition** (multi-trajectory reasoning) executed with **deterministic scheduling**.

---

## 1. Design Principles (Non-Negotiable)

1. **Determinism**: execution order and outcomes must be reproducible.
2. **Authority Separation**:
   - Validation **observes** (produces signals)
   - Utility **evaluates**
   - DecisionEngine **decides**
   - MemorySpace **persists**
3. **Memory Purity**: speculative or non-converged artifacts must not pollute Accept memory.
4. **Lifecycle Centralization**: only the Orchestrator can fork, prune, terminate, or commit.
5. **Branch Safety**: ThoughtBranches are read-only against MemorySpace.
6. **Explainability**: every prune/commit must be traceable to utility + policy thresholds.

---

## 2. High-Level Architecture

```text
CoreRuntime
 └─ ParallelThoughtOrchestrator  (Cognitive OS Kernel)
      ├─ BranchScheduler         (Round-Robin, deterministic)
      ├─ ThoughtBranch[0..N]     (User-mode cognitive processes)
      ├─ BranchEvaluator         (metrics extraction)
      ├─ UtilityAggregator       (expected utility computation + ranking)
      ├─ ForkController          (triggered diversification)
      ├─ ConvergenceController   (entropy + equivalence convergence)
      └─ CommitController        (Accept/Review/Reject memory routing)
```

---

## 3. ThoughtBranch Specification

### 3.1 Concept

A `ThoughtBranch` is one independent cognitive trajectory. It maintains **branch-local state** and produces **branch-local metrics**, but it cannot:

- write MemorySpace
- fork itself
- change its own lifecycle status

### 3.2 Data Model (Fixed)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BranchStatus(Enum):
    ACTIVE = "active"
    PRUNED = "pruned"
    CONVERGED = "converged"
    TERMINATED = "terminated"


@dataclass
class UtilityState:
    validity: float = 0.0
    progress: float = 0.0
    novelty: float = 0.0
    confidence: float = 0.0
    risk: float = 0.0
    expected_utility: float = 0.0


@dataclass
class ThoughtBranch:
    # Identity
    branch_id: int
    parent_branch_id: Optional[int] = None

    # Lifecycle
    status: BranchStatus = BranchStatus.ACTIVE
    step_count: int = 0

    # Core State
    current_ast: Any = None
    goal_ast: Any = None
    last_action: Optional[str] = None

    # Branch-local working memory
    working_memory: Dict[str, Any] = field(default_factory=dict)
    hypothesis_queue: List[Any] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    local_constraints: Dict[str, Any] = field(default_factory=dict)
    trace_log: List[str] = field(default_factory=list)

    # Recall context
    recall_threshold: float = 0.7
    resonance_history: List[float] = field(default_factory=list)
    recalled_memory_ids: List[str] = field(default_factory=list)

    # Policy knobs
    exploration_bias: float = 0.5  # 0.0=conservative, 1.0=exploratory
    lookahead_depth: int = 2
    risk_preference: float = 0.5   # 0.0=risk-averse, 1.0=risk-seeking

    # Evaluation
    utility: UtilityState = field(default_factory=UtilityState)
```

### 3.3 Invariants (Must Hold)

- Branch cannot write MemorySpace (read-only access only).
- Branch cannot fork itself.
- Branch cannot change its own `status` (Orchestrator only).
- Branch must not commit any result; it only proposes state + metrics.

---

## 4. Orchestrator Responsibilities (Authority Model)

The `ParallelThoughtOrchestrator` is the single authority for:

- branch generation (fork)
- pruning
- lifecycle status transitions
- convergence decision
- final decision invocation (DecisionEngine)
- memory commit routing (Accept/Review/Reject)

Branches are treated as **user-mode** processes; Orchestrator is **kernel-mode**.

---

## 5. Lifecycle Management

### 5.1 Generation (Fork)

Forking is performed only by Orchestrator under controlled triggers.

#### Fork Triggers

| Trigger | Description |
|---|---|
| Recall Failure | resonance < branch.recall_threshold |
| Multi-Hypothesis | generator yields competing candidates |
| Validation Ambiguity | ambiguity is high while validity > 0 |
| Low Confidence | confidence collapses while progress exists |
| Strategic Exploration | exploration policy requests diversification |

#### Fork Constraints (Fixed Defaults)

```text
MAX_BRANCH = 5
```

Fork inherits parent state and then mutates policies (e.g., exploration_bias) to diversify.

---

### 5.2 Pruning (Branch Elimination)

Pruning prevents combinatorial explosion.

#### Pruning Criteria (Fixed)

```text
Prune if:
- expected_utility < U_min
- risk > R_max
- step_count > STEP_LIMIT
- dominated by another branch (Pareto dominance)
```

#### Pareto Dominance Rule (Fixed)

Branch A dominates Branch B if:

```text
A.validity   ≥ B.validity
A.progress   ≥ B.progress
A.confidence ≥ B.confidence
A.novelty    ≥ B.novelty
and at least one strict >
```

Dominated branches are pruned immediately.

---

### 5.3 Termination (Branch End States)

| Condition | Result |
|---|---|
| Goal Reached | CONVERGED |
| Utility Collapse / Dominated | PRUNED |
| Timeout / Step Limit | TERMINATED |

---

## 6. Scheduler (Round-Robin) — Deterministic Logical Parallelism

### 6.1 Why Round-Robin (Decision Locked)

- Multi-threading or async scheduling complicates reproducibility and debugging.
- Logical parallelism is achieved by interleaving steps deterministically.

**Implementation strategy**: single-threaded, cooperative, round-robin.

### 6.2 Scheduling Contract

- 1 tick = each ACTIVE branch gets **exactly one reasoning step**
- a step is one of:
  - Recall attempt
  - Hypothesis generation + single apply attempt
  - Single lookahead simulate step
  - Single validation call

No “multi-step bursts” per branch within a tick.

### 6.3 Scheduler Interface (Fixed)

```python
class BranchScheduler:
    def tick(self, branches):
        # Deterministically execute 1 logical step for each ACTIVE branch.
        for b in branches:
            if b.status != BranchStatus.ACTIVE:
                continue
            self.execute_branch_step(b)

    def execute_branch_step(self, branch):
        # Allowed:
        #   - read MemorySpace
        #   - update branch.current_ast
        #   - update branch.working_memory
        #   - update branch.utility fields (partial)
        # Forbidden:
        #   - write MemorySpace
        #   - change branch.status
        #   - fork
        #   - commit
        raise NotImplementedError
```

---

## 7. ValidationEngine Integration

### 7.1 Validation Role (Observer)

ValidationEngine provides signals used to update branch utility. It must **not** decide convergence or commit.

### 7.2 Interface (Fixed)

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ValidationResult:
    validity: float       # 0.0 - 1.0
    confidence: float     # 0.0 - 1.0
    ambiguity: float      # 0.0 - 1.0
    violations: List[str]
```

Branch-side usage:

```python
validation = ValidationEngine.validate(
    current_ast=branch.current_ast,
    goal_ast=branch.goal_ast,
    context=branch.working_memory
)
```

### 7.3 Utility Update Rules (Fixed)

```text
branch.utility.validity   ← validation.validity
branch.utility.confidence ← validation.confidence
branch.utility.risk       ← validation.ambiguity
violations → branch.trace_log only
```

### 7.4 Fork Trigger via Validation (Fixed)

Validation does not fork. Orchestrator may fork when:

```text
if ambiguity > A_fork and validity > 0 and branch_count < MAX_BRANCH:
    fork()
```

```text
A_fork = 0.40
```

### 7.5 Final Validation Gate (Commit Safety)

Before commit, Orchestrator runs final validation on the best candidate:

```text
if final_validity < V_commit:
    downgrade to Review
```

```text
V_commit = 0.85
```

---

## 8. Utility Model (Numerical Defaults Locked)

### 8.1 Weights (Fixed v1)

```text
w_validity   = 0.35
w_progress   = 0.25
w_confidence = 0.20
w_novelty    = 0.15
w_risk       = 0.30   (penalty)
```

### 8.2 Expected Utility Function (Fixed)

```python
def compute_expected_utility(u: UtilityState) -> float:
    return (
        0.35 * u.validity +
        0.25 * u.progress +
        0.20 * u.confidence +
        0.15 * u.novelty -
        0.30 * u.risk
    )
```

### 8.3 Thresholds (Fixed Defaults)

```text
U_min       = -0.10   # prune immediately
U_survive   =  0.05   # minimum survival utility
U_converge  =  0.65   # single-branch converge threshold
R_max       =  0.70   # max tolerated risk
STEP_LIMIT  =  64     # default per-branch step cap (tunable)
```

---

## 9. Convergence & Integration

### 9.1 Convergence Conditions (System Level)

System converges if any of:

1. **Equivalence Convergence**: ≥2 branches reach equivalent AST (semantic equality)
2. **Dominant Utility**: max(EU) > U_converge
3. **Entropy Convergence**: branch entropy falls below H_min

### 9.2 Equivalence Definition (Fixed)

AST equivalence must be based on the same canonicalization used by ValidationEngine / SymbolicEngine:

- structural equality after normalization/canonicalization
- symbolic equivalence check if available

### 9.3 Branch Entropy (Measured Convergence)

#### Definition (Fixed)

Exclude non-positive EU branches:

```text
p_i = EU_i / Σ EU
H = - Σ p_i log(p_i)
```

#### Implementation Reference

```python
import math

def compute_branch_entropy(branches):
    eus = [b.utility.expected_utility for b in branches if b.utility.expected_utility > 0]
    total = sum(eus)
    if total == 0 or len(eus) <= 1:
        return 0.0

    entropy = 0.0
    for eu in eus:
        p = eu / total
        entropy -= p * math.log(p)
    return entropy
```

#### Threshold (Fixed Default)

```text
H_min = 0.25
```

Interpretation:
- H ≈ 0 → effectively one surviving line of thought
- H < 0.25 → convergence likely; stop exploration

---

## 10. MemorySpace Integration (Accept / Review / Reject Layers)

### 10.1 Layering (Fixed)

```text
MemorySpace
├─ AcceptMemory   (converged + accepted knowledge/experience)
├─ ReviewMemory   (ambiguous; human-needed; safe quarantine)
└─ RejectMemory   (proven incorrect/unsafe; anti-pattern repository)
```

### 10.2 Access Policy (Fixed)

- All branches: **Read-only** against MemorySpace (including AcceptMemory)
- Only Orchestrator: Write access, via CommitController
- Review/Reject layers do **not** participate in normal recall (default)

### 10.3 Commit Gate (Only Converged Results)

Commit requires:

```text
branch.status == CONVERGED
AND final validation passed (>= V_commit) OR downgraded to Review
AND DecisionEngine outcome routes to layer
```

### 10.4 Entries (Fixed Structures)

#### ReviewMemoryEntry

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class ReviewMemoryEntry:
    ast: Any
    branch_snapshot: Dict
    utility_snapshot: UtilityState
    reason: str
    timestamp: float
```

#### RejectMemoryEntry

```python
from dataclasses import dataclass
from typing import List

@dataclass
class RejectMemoryEntry:
    ast: Any
    failure_reason: str
    violated_constraints: List[str]
    timestamp: float
```

### 10.5 Routing Rules (Fixed)

```text
if DecisionEngine == Accept:
    commit → AcceptMemory
elif DecisionEngine == Review:
    commit → ReviewMemory
else:
    commit → RejectMemory
```

---

## 11. Full Control Flow (Deterministic)

```text
Input AST
 ↓
ParallelThoughtOrchestrator.start()
 ↓
[ LOOP until convergence or exhaustion ]
  1) BranchScheduler.tick()                     # round-robin
  2) BranchEvaluator + Validation (per branch)   # observer signals
  3) UtilityAggregator                           # EU recompute + rank
  4) Pruner                                      # thresholds + dominance
  5) ForkController (optional)                   # ambiguity-driven
  6) ConvergenceController                        # equivalence/EU/entropy
[ END LOOP ]
 ↓
Final Validation Gate
 ↓
DecisionEngine (Accept / Review / Reject)
 ↓
CommitController → MemorySpace Layer
```

---

## 12. Design Verification Checklist (Audited)

- [x] Fork authority centralized (Orchestrator only)
- [x] Branch cannot mutate MemorySpace (read-only)
- [x] Deterministic execution via round-robin single-thread scheduling
- [x] Pruning prevents combinatorial explosion (threshold + dominance)
- [x] Convergence is objective (equivalence/EU/entropy)
- [x] Commit gate prevents speculative memory contamination
- [x] Review/Reject quarantines enable safe human-in-the-loop
- [x] Responsibility separation: Validation observes, Decision decides, Memory persists

---

## 13. Implementation Notes (Constraints)

- AsyncIO / multi-threading is intentionally avoided by default.
- If future performance requires parallel execution, it must preserve:
  - deterministic ordering of logical ticks
  - identical final branch states under the same random seed
- Branch-level randomness (for stochastic branch types) must be seeded and recorded.

---

## 14. Summary

This specification upgrades COHERENT from single-path reasoning to a
**controlled multi-trajectory cognitive engine** with:

- explicit branch structure and lifecycle control
- numeric utility evaluation with fixed defaults
- entropy-based convergence detection
- safe, layered memory commit policy (Accept/Review/Reject)

This document is final unless core architectural assumptions change.
