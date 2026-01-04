# DHM Alphabet Generation Experiment

This experimental module implements the **Alphabet Generation Experiment** using **Dynamic Holographic Memory (DHM)**.

## Purpose
To demonstrate symbol generation from distributed attribute holograms without explicit symbol storage, using diffusion for state refinement.

## Usage
Run the experiment runner from the project root:

```bash
python -m experimental.dhm_generation.runner
```

## Directory Structure
* `spec/`: Specifications.
* `memory/`: DHM and Attribute Encoding logic.
* `diffusion/`: DDIM/DDPM implementation.
* `evaluation/`: Resonance and Diagnostics.
* `logs/`: Experiment output CSVs.
