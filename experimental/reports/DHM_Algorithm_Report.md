# DHM Algorithm & Mathematical Formulation Report

**Date:** 2026-01-04
**Subject:** Algorithm Analysis of DHM Generation Experiments

## 1. Holographic Representation

The fundamental state space is a high-dimensional complex vector space.

*   **Space:** $\mathcal{H} = \mathbb{C}^D$ where $D \in \{2048\}$
*   **Normalization Condition:** All valid holographic states $H \in \mathcal{H}$ must satisfy $\|H\|_2 = 1$.

### Attribute Encoding
Each atomic attribute $a$ is mapped to a vector $h_a$ via a deterministic pipeline:

1.  **Seed Generation:** $s = \text{Hash}(a)$
2.  **Real Base Vector:** $v_{base} \sim \mathcal{N}(0, 1)$ generated from seed $s$.
3.  **FFT Transformation:** $v_{complex} = \text{FFT}(v_{base})$
4.  **Normalization:**
    $$ h_a = \frac{v_{complex}}{\|v_{complex}\|_2} $$

## 2. Dynamic Query Construction

A symbol or target entity $S$ is composed dynamically from its constituent attributes $\{a_1, a_2, ..., a_n\}$. The composition uses the **Hadamard Product (Element-wise binding)**.

### Binding & Normalization
To prevent numerical underflow (vanishing magnitude) during the binding of multiple L2-normalized vectors (whose elements are typically small, $\approx 1/\sqrt{D}$), the implementation uses **Stepwise Normalization**:

$$ H_{temp}^{(0)} = h_{a_1} $$
$$ H_{temp}^{(k)} = \text{Normalize}\left( H_{temp}^{(k-1)} \odot h_{a_{k+1}} \right) $$

The final conditioning state $H_0(S)$ is:
$$ H_0(S) = H_{temp}^{(n-1)} $$

This ensures that at every step of composition, the vector remains on the unit hypersphere.

## 3. Diffusion Dynamics (DDIM)

The generation process models the refinement of a noisy holographic state towards a coherent structure.

### 3.1 Complex Gaussian Noise
Noise $\eta$ is defined in the complex domain:
$$ \eta = \frac{1}{\sqrt{2}} (\eta_{real} + i \cdot \eta_{imag}) $$
where $\eta_{real}, \eta_{imag} \sim \mathcal{N}(0, I_D)$.

### 3.2 Forward Process (Noise Injection)
For a timestep $t \in [0, T]$, the noisy state $H_t$ is defined by the schedule $\bar{\alpha}_t$ (cumulative signal retention):

$$ H_t = \sqrt{\bar{\alpha}_t} H_0 + \sqrt{1 - \bar{\alpha}_t} \eta $$

followed by re-normalization: $H_t \leftarrow H_t / \|H_t\|$.

### 3.3 Reverse Process (Oracle-Guided DDIM)
The experiment uses a deterministic DDIM sampler. Since no neural network is trained, the process uses an "Oracle" assumption (the target $H_0$ is the attractor) to demonstrate the **structural refinement capability** of the state space.

1.  **Implicit Noise Prediction:**
    The noise $\epsilon_t$ currently present in $H_t$ relative to the target $H_0$ is:
    $$ \epsilon_t = \frac{H_t - \sqrt{\bar{\alpha}_t} H_0}{\sqrt{1 - \bar{\alpha}_t}} $$

2.  **State Update (Step $t \to t-1$):**
    The next state is reconstructed by mixing the target and the noise with the ratio for step $t-1$:
    $$ H_{t-1} = \sqrt{\bar{\alpha}_{t-1}} H_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_t $$
    
    $$ H_{t-1} \leftarrow \frac{H_{t-1}}{\|H_{t-1}\|} $$

This process creates a smooth trajectory on the hypersphere from a noisy state to the clean target state.

## 4. Evaluation & Decoding

Decoding is performed by measuring the **Resonance** between the generated state $H_{final}$ and all candidate symbol holograms $\{H_{c_1}, H_{c_2}, ...\}$.

### Resonance Score
The resonance $R$ is the magnitude of the Hermitian inner product:

$$ R(H_{final}, H_c) = | \langle H_{final}, H_c \rangle | = \left| \sum_{j=1}^D H_{final}[j] \cdot \overline{H_c[j]} \right| $$

### Decision Rule
$$ \text{Prediction} = \arg \max_{c} R(H_{final}, H_c) $$
$$ \text{Margin} = R_{top1} - R_{top2} $$

Successful generation requires $Margin > 0$.
