# Phase 0: Core Stabilization Report

## Category A: Commutativity

**Status: PASS**

| Input | Normalized | Structure Pattern | Hash |
|-------|------------|-------------------|------|
| `x + y` | `x + y` | `Add(VAR, VAR)` | `7d1be320` |
| `y + x` | `y + x` | `Add(VAR, VAR)` | `7d1be320` |

## Category B: Associativity

**Status: FAIL (Expected without SymPy)**

> [!WARNING]
> Structural difference detected. (Note: Without SymPy, equivalence reduction is limited)

| Input | Normalized | Structure Pattern | Hash |
|-------|------------|-------------------|------|
| `(x + y) + z` | `(x + y) + z` | `Add(Add(VAR, VAR), VAR)` | `948b059f` |
| `x + (y + z)` | `x + (y + z)` | `Add(VAR, Add(VAR, VAR))` | `450ccbad` |

## Category C: Equivalence Reduction

**Status: FAIL (Expected without SymPy)**

> [!WARNING]
> Structural difference detected. (Note: Without SymPy, equivalence reduction is limited)

| Input | Normalized | Structure Pattern | Hash |
|-------|------------|-------------------|------|
| `x + x` | `x + x` | `Add(VAR, VAR)` | `7d1be320` |
| `2x` | `2*x` | `Mult(INT(2), VAR)` | `301a3533` |

## Category D: Coefficient Arrangement

**Status: FAIL (Expected without SymPy)**

> [!WARNING]
> Structural difference detected. (Note: Without SymPy, equivalence reduction is limited)

| Input | Normalized | Structure Pattern | Hash |
|-------|------------|-------------------|------|
| `3a + 5b` | `3*a + 5*b` | `Add(Mult(INT(3), VAR), Mult(INT(5), VAR))` | `5b4ba64d` |
| `5b + 3a` | `5*b + 3*a` | `Add(Mult(INT(5), VAR), Mult(INT(3), VAR))` | `c9e88f96` |

