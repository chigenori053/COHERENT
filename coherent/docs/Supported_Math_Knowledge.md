# Supported Mathematical Knowledge

The COHERENT system relies on a registry of mathematical knowledge nodes (rules) to perform reasoning. These rules are categorized by domain and are stored in YAML format.

## 1. Algebra (`algebra`)

### Basic Equation Manipulation (`basic_equations.yaml`)
Rules for rearranging terms in equations.

| ID | Description | Pattern | Priority |
| :--- | :--- | :--- | :--- |
| `ALG-EQ-MOVE-ADD` | Move term (Subtraction) | `a + b = c` → `a = c - b` | 96 |
| `ALG-EQ-MOVE-ADD-CONST-FIRST` | Move term (Sub) | `a + b = c` → `b = c - a` | 96 |
| `ALG-EQ-MOVE-SUB` | Move term (Addition) | `a - b = c` → `a = c + b` | 95 |
| `ALG-EQ-DIV-COEFF` | Divide by coefficient | `a * x = b` → `x = b / a` | 85 |

### Factoring (`factoring.yaml`)
Rules for polynomial factorization.

| ID | Description | Pattern |
| :--- | :--- | :--- |
| `ALG-FAC-001` | Common Factor | `ab + ac` → `a(b + c)` |
| `ALG-FAC-002` | Difference of Squares | `a^2 - b^2` → `(a+b)(a-b)` |

*Note: Other files like `complex_numbers.yaml`, `expansion.yaml`, `exponents.yaml` contain additional algebraic rules.*

## 2. Calculus (`calculus`)

### Differentiation Rules (`differentiation.yaml`)
Standard rules for computing derivatives.

| ID | Concept | Description | Pattern |
| :--- | :--- | :--- | :--- |
| `CALC-DIFF-POW` | Power Rule | Power Rule | `d/dx(x^n)` → `nx^(n-1)` |
| `CALC-DIFF-SUM` | Sum Rule | Sum Rule | `d/dx(f+g)` → `f' + g'` |
| `CALC-DIFF-PROD` | Product Rule | Product Rule | `d/dx(fg)` → `fg' + gf'` |
| `CALC-DIFF-CONST` | Constant | Constant Rule | `d/dx(c)` → `0` |
| `CALC-DIFF-SIN` | Trig | Sine | `d/dx(sin x)` → `cos x` |
| `CALC-DIFF-COS` | Trig | Cosine | `d/dx(cos x)` → `-sin x` |
| `CALC-DIFF-TAN` | Trig | Tangent | `d/dx(tan x)` → `sec^2 x` |
| `CALC-DIFF-EXP` | Exponential | Natural Exp | `d/dx(e^x)` → `e^x` |
| `CALC-DIFF-LOG` | Logarithm | Natural Log | `d/dx(ln x)` → `1/x` |

## 3. Arithmetic (`arithmetic`)
Contains basic rules for numeric computation and simplification, including:
- Addition/Subtraction identities (`a + 0 = a`)
- Multiplication identities (`a * 1 = a`, `a * 0 = 0`)
- Division rules (`a / a = 1`)

## 4. Geometry (`geometry`)
*Currently empty or in development.*

---
**Note**: This is a non-exhaustive list. The system dynamically loads all valid YAML files in the `coherent/engine/knowledge` directory.
