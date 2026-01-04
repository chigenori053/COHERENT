# Diffusion Logic (DDIM / DDPM)

## Forward Process
$H_t = \sqrt{\bar{\alpha}_t} H_0 + \sqrt{1 - \bar{\alpha}_t} \eta_t$

## Reverse Process
### DDIM (Deterministic)
Conditioning is defined by H_0.
Uses analytical noise prediction derived from known H_0 (Oracle).

### DDPM (Stochastic)
 Adds noise term $\sigma_t \eta'_t$ to the deterministic trajectory.
