# CRS Memory Library Specification v0.1 (Python)
*Target:* Develop a Python library that constructs **Memory** and **Memory_Atom** objects that are **reversibly projectable to real-world (real-valued) data**, supports **AST normalization**, **deterministic causal-model extraction**, and **lossless JSON/YAML serialization**.

---

## 1. Scope and Objectives

### 1.1 In-Scope (v0.1)
- Define **MemoryAtom** (Memory_Atom) as an immutable, complex spectral object `ℂ^D` with:
  - forward transform metadata
  - inverse transform metadata
  - projection metadata
  - reconstruct quality constraints
- Define **Memory** as a **hierarchical composition**:
  - `MemoryAtom` → `MemoryCell` → `MemoryBlock` → `MemoryStructure`
- Provide **formula pipeline**:
  - parse formula → AST (structural)
  - normalize AST (variable abstraction + canonicalization)
  - build deterministic **CausalModel** from normalized AST
  - compile into MemoryStructure
  - export/import as **JSON / YAML**
- Provide validators (schema + invariants)
- Provide minimal CRML compilation for the above (optional but recommended).

### 1.2 Out-of-Scope (v0.1)
- Full optical interference retrieval engine
- Training / reinforcement learning loops (beyond placeholder hooks)
- Full CRML language implementation (beyond minimal subset)
- UI components

### 1.3 Non-Functional Requirements
- Deterministic outputs (same input → same normalized signature and causal model)
- JSON/YAML round-trip stability (no information loss)
- Enforce “real-world meaningful data” constraint: every atom must be projectable to real-valued domains via inverse transforms.

---

## 2. Core Concepts and Definitions

### 2.1 MemoryAtom (Memory_Atom)
**Definition:** Minimal unit of meaning represented as complex spectrum `repr ∈ ℂ^D`, **reversibly mappable** to real-valued domains via inverse transform + projection.

**Hard constraint (must):**
- `inverse` must be defined, and `project(repr)` must yield **finite real values**.
- No atom may exist that is not reverse-projectable.

### 2.2 Memory (hierarchical)
Memory is not a CRUD container. It is a **constructed structure** that remains serializable to real-world representations.

**Hierarchy:**
- **Level 0:** `MemoryAtom` — immutable spectral unit
- **Level 1:** `MemoryCell` — group of atoms with a causal role (`cause/effect/context/...`)
- **Level 2:** `MemoryBlock` — semantic unit representing numeric/text/formula/list/table/mixed
- **Level 3:** `MemoryStructure` — external interchange unit; fully serializable JSON/YAML

### 2.3 AST and Causality
- AST is **not executed**; it is decomposed, normalized, and used to extract causal structure.
- Causality is primarily extracted from **structure** (AST edges), not inferred from data.

---

## 3. Data Model

### 3.1 Enums

**AtomType**
- `REAL`, `SIGNAL`, `IMAGE`, `ABSTRACT`

**ProjectionDomain**
- `REAL_VECTOR`, `REAL_SIGNAL`, `REAL_IMAGE`

**TransformKind**
- `FFT1D`, `FFT2D`, `DCT`, `STFT`, `WAVELET`

**InverseKind**
- `IFFT1D`, `IFFT2D`, `IDCT`, `ISTFT`, `INV_WAVELET`

### 3.2 MemoryAtom Fields (v0.1)
Required:
- `id: str`
- `atom_type: AtomType`
- `spec_dim: int` (D)
- `repr: list[ComplexVal]` where `ComplexVal={re: float, im: float}`
- `transform: TransformSpec`
- `inverse: InverseSpec`
- `projection: ProjectionSpec`
- `quality: ReconstructQuality` (`reconstructable=true` required)
- `confidence: float in [0,1]` (epistemic confidence; separate from reconstruction)
- `timestamp: datetime`

Optional:
- `context_signature: str`
- `origin: "input" | "inference" | "learning"`

### 3.3 MemoryCell Fields
Required:
- `id: str`
- `atoms: list[AtomRef]` (ref only; atoms are not copied)
- `role: "cause" | "effect" | "context" | "evidence" | "note"`
Optional:
- `local_confidence: float in [0,1]`

### 3.4 MemoryBlock Fields
Required:
- `id: str`
- `block_type: "numeric" | "text" | "formula" | "list" | "table" | "mixed"`
- `cells: list[MemoryCell]`
- `payload: BlockPayload` (JSON/YAML safe)
- `causal_model: CausalModel`

Optional:
- `links: list[BlockLink]` where `rel ∈ {derived_from,supports,contradicts,expands,summarizes}`

### 3.5 MemoryStructure Fields
Required:
- `schema_version: "0.1"`
- `meta: Meta` (id/created_at/source/title/description/tags)
- `blocks: list[MemoryBlock]`
- `causal_graph: CausalGraph` (nodes/edges)

Optional:
- `atoms: dict[str, MemoryAtomRefOrEmbed]` (for portability; supports embedding atoms)
- `exports: projections/materializations` (debug/UI; not required)

---

## 4. AST Handling

### 4.1 Structural AST Format (JSON/YAML safe)
- `root: node_id`
- `nodes: { node_id: {id,type,children,value?,span?} }`

### 4.2 Normalization Output
- `canon_ast` (canonical AST)
- `var_map` (e.g., `{v1: x, v2: y}`)
- `signature` (stable structural signature, hashable)

### 4.3 Normalization Rules (v0.1)
**Lexical normalization**
- whitespace/Unicode normalization
- numeric literal normalization policy

**Structural normalization**
- variable abstraction: DFS order → `v1,v2,...`
- commutative child sorting for `Add`, `Mul`
- associative flattening for `Add`, `Mul`
- unary minus canonicalization: `-(x)` → `Mul[-1,x]`

**Algebraic canonicalization (limited)**
- remove additive identity `+0`
- remove multiplicative identity `*1`
- forbid expansion/factorization in v0.1

### 4.4 Signature Building
- bottom-up partial signatures
- commutative nodes sort child signatures
- root signature hashed for indexing / recall

---

## 5. AST → CausalModel Auto-Generation (Deterministic)

### 5.1 Variables
Generate causal variables from AST nodes:
- input: Var/Symbol/Const
- derived: operator/function nodes
- output: root node

Each variable must have:
- `id` (node id)
- `name` (human-readable string)
- `vtype: numeric|categorical|text|latent`
- `source: ast_input|ast_derived|ast_output`

### 5.2 Relations (default rule)
- For AST edge `child -> parent`:
  - `child` causes `parent`

**Exceptions**
- Assignment: `RHS -> LHS`
- Conditionals: condition influences both branches (`condition_true/false` mechanisms)

### 5.3 Relation attributes (defaults)
- `strength`: by operator template (Add/Mul/Pow ~0.5–0.7; Assign ~0.9)
- `confidence`: default 0.8 for AST-derived relations
- `mechanism`: template label (binary_operand, unary_operand, assignment, etc.)

### 5.4 Counterfactual
- `counterfactual_enabled = true` for AST-derived models by default.

---

## 6. Real-World Projection Constraint (Must)

### 6.1 Projection requirement
For every MemoryAtom:
- `inverse.kind` must be defined
- `projection.domain` must be one of `REAL_VECTOR|REAL_SIGNAL|REAL_IMAGE`
- `project(repr)` must return finite real values (no NaN/Inf)
- if `inverse.constraints.hermitian = true`, enforce conjugate symmetry for real reconstruction

### 6.2 AbstractAtom rule
Abstract atoms are allowed **only if** they define a projection decoder:
- `decoder.kind ∈ {dictionary, programmatic, hybrid}`
- `decoder.ref_id` defined when needed

---

## 7. Serialization and Schemas

### 7.1 JSON/YAML safety
- no NaN/Inf (hard validation error)
- payload values must be JSON/YAML-safe types only (null/bool/number/string/array/object)

### 7.2 Official Schemas
- `schemas/memory_structure.schema.json` (Draft 2020-12)
- `schemas/memory_atom.schema.json` (optional; may be included or embedded in structure schema)

### 7.3 Round-trip requirements
- `structure -> yaml -> structure` preserves:
  - `normalized.signature`
  - `payload.ast` (if present)
  - `causal_model.variables/relations`
  - block ordering and ids (unless user opts for canonical reordering)

---

## 8. Public API (v0.1)

### 8.1 High-level builder API
```python
from crs_memory import MemoryBuilder

builder = MemoryBuilder(spec_dim=1024)

structure = builder.from_formula(
    source="(x - y)^2",
    fmt="sympy",
    structure_id="S-001",
    block_id="B-001",
)

yaml_text = structure.to_yaml()
restored = MemoryStructure.from_yaml(yaml_text)
```

### 8.2 Core functions
- `parse_formula(source, fmt) -> AST`
- `normalize_ast(ast) -> NormalizedAST`
- `build_causal_model(normalized_ast) -> CausalModel`
- `compile_formula_block(source, fmt) -> MemoryBlock`
- `validate_structure(structure) -> ValidationReport`
- `to_json(structure) / from_json(...)`
- `to_yaml(structure) / from_yaml(...)`

### 8.3 Atom utilities
- `MemoryAtom.from_real_vector(x, transform=FFT1D, ...)`
- `MemoryAtom.project_real() -> ndarray`
- `check_reconstructability(atom)`

---

## 9. Package Structure (Recommended)

```text
crs_memory/
├── atoms/
│   ├── memory_atom.py
│   ├── projection.py
│   └── quality.py
├── ast/
│   ├── parser.py
│   ├── normalizer.py
│   ├── signature.py
│   └── ast_types.py
├── causal/
│   ├── model.py
│   ├── extractor.py
│   └── graph.py
├── memory/
│   ├── cell.py
│   ├── block.py
│   ├── structure.py
│   └── builder.py
├── serialize/
│   ├── json_io.py
│   ├── yaml_io.py
│   └── validator.py
└── tests/
    ├── test_formula_flow.py
    ├── test_isomorphism.py
    └── test_roundtrip.py
```

---

## 10. Validation Rules (Hard Errors in v0.1)

### 10.1 Structure-level
- `schema_version` must be `0.1`
- every block has `payload` consistent with `block_type`
- `causal_graph` references existing variables/blocks

### 10.2 Atom-level
- spec_dim matches repr length
- inverse + projection defined
- projectable to finite real values
- hermitian constraint satisfied when enabled

### 10.3 AST-level
- normalized signature must be present for formula blocks
- `canon_ast` must exist when signature exists
- commutative/associative invariants hold (if enabled by normalizer)

---

## 11. Testing Plan (Minimum)
- **E2E formula test**: `(x-y)^2` → normalize → causal → YAML → restore → signatures match
- **Isomorphism tests**:
  - `3x + 5y` and `3a + 5b` share same signature
  - `y + x` equals `x + y` signature
- **Serialization safety**:
  - reject NaN/Inf payload
- **Causal determinism**:
  - repeated runs yield identical relations order (or canonical order)

---

## 12. Milestones

### M0 (1–2 sessions)
- Data classes + YAML/JSON IO skeleton
- AST parser (SymPy) + minimal normalizer
- Causal extraction (edge-based)

### M1
- Full MemoryStructure schema + validator
- E2E tests green

### M2
- Optional: minimal CRML parser → compile to structure
- Optional: embedding atoms into `atoms` registry for portability

---

## 13. Deliverables
- `crs_memory` Python package (importable)
- JSON Schema files under `schemas/`
- Unit tests under `tests/`
- Example notebooks:
  - `examples/formula_to_yaml.ipynb`
  - `examples/isomorphism.ipynb`

---

## 14. Notes on Future Extensions (Non-binding)
- Optical/holographic retrieval store
- DecisionEngine integration (expected utility) using:
  - reconstruction error metrics
  - causal consistency metrics
- Reinforcement/decay policies to modulate recall probability without mutating atom spectral vectors

