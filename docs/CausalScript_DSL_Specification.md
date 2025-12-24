# CausalScript DSL Specification v2

CausalScript (MathLang) is a Domain Specific Language designed for representing mathematical reasoning, step-by-step problem solving, and system verification. It is the intermediate representation used by the COHERENT system.

## 1. Core Structure

A CausalScript program consists of a sequence of directives. The most common structure is a **Problem** block containing a series of **Steps**.

### Problem Block
Defines the starting state or the problem statement.

```yaml
problem: (x - 2y)^2
```

Or with a name:
```yaml
problem: Expansion = (x - 2y)^2
```

Or with a mode (e.g., verifying a specific proof strategy):
```yaml
problem: Proof(geometric) = ...
```

### Steps
Represents the derivation steps. Each step must logically follow from the previous state or be a valid transformation.

```yaml
step: (x - 2y)(x - 2y)
step: x(x - 2y) - 2y(x - 2y)
step: x^2 - 2xy - 2yx + 4y^2
```

### End Block
Marks the conclusion of the derivation.

```yaml
end: x^2 - 4xy + 4y^2
```

Or simply:
```yaml
end: done
```

## 2. Advanced Constructs

### Sub-Problems
For breaking down complex derivations.

```yaml
sub_problem: Factorization = x^2 - 4
step: (x - 2)(x + 2)
```

### Scenarios
Defines variable assignments or specific conditions for testing.

```yaml
scenario: "Unit Test A"
  x = 5
  y = 10
```

### Counterfactuals
Used for "what-if" analysis verification.

```yaml
counterfactual:
  assume: 
    x = 0
  expect: y == 0
```

### Configuration
System-level configuration for the runtime.

```yaml
config:
  division_by_zero: error
  precision: 10
```

## 3. Expression Syntax

The expression syntax is largely compatible with Python/SymPy but includes several human-friendly aliases.

### Operators
| Operator | Description | Alias |
| :--- | :--- | :--- |
| `+` | Addition | |
| `-` | Subtraction | |
| `*` | Multiplication | Implicit multiplication allowed (e.g., `2x`) |
| `/` | Division | |
| `**` | Power | `^`, `²`, `³` |
| `==` | Equality | `=` (in `Eq()` context) |

### Functions
Standard mathematical functions are supported:
- `sqrt(x)` or `√x`
- `sin(x)`, `cos(x)`, `tan(x)`
- `log(x)`, `ln(x)`, `exp(x)`
- `diff(f, x)`: Differentiation
- `integrate(f, x)`: Integration

### Calculus Notation
You can use LaTeX-like notation for calculus operations which are normalized by the parser.

*   **Derivative**: `Derivative(x^2, x)` or `diff(x^2, x)`
*   **Integral**: `Integral(x^2, x)` or `integrate(x^2, x)`
*   **Definite Integral**: `[x^2]_0^1` is sugar for `Subs(x^2, x, 1) - Subs(x^2, x, 0)`

### Matrices
Matrices can be defined using Python lists or MATLAB-like syntax (semicolon for row separation).

```yaml
# Python Style
[[1, 0], [0, 1]]

# MATLAB Style (if enabled in parser extensions)
[1, 0; 0, 1]
```

## 4. Example Program

```yaml
# Expanding a discrete binomial
config:
  precision: 4

scenario: "Base Case"
  x = 1
  y = 0

problem: Expansion = (x + y)^2

step: (x + y)(x + y)
  note: Definition of square
  
step: x(x + y) + y(x + y)
  rule: Distributive Property

step: x^2 + xy + yx + y^2

end: x^2 + 2xy + y^2
```
