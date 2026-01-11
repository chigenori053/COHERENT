# C1 Perception Verification Report

Date: 2026-01-11
## Summary
Evaluated Logic: Visual input -> Holographic Vector -> Resonance Check.

## Detailed Results
| Test | Input | Variant | Target | Resonance | Judgment |
|---|---|---|---|---|---|
| C1-1:Consistency | A | Noise:0.05,Rot:0 | A | 0.9687 | PASS |
| C1-1:Consistency | A | Noise:0.00,Rot:10 | A | 0.7994 | PASS |
| C1-1:Consistency | A | Noise:0.10,Rot:-5 | A | 0.8687 | PASS |
| C1-1:Consistency | B | Noise:0.05,Rot:0 | B | 0.9789 | PASS |
| C1-1:Consistency | B | Noise:0.00,Rot:10 | B | 0.8956 | PASS |
| C1-1:Consistency | B | Noise:0.10,Rot:-5 | B | 0.9212 | PASS |
| C1-1:Consistency | 5 | Noise:0.05,Rot:0 | 5 | 0.9657 | PASS |
| C1-1:Consistency | 5 | Noise:0.00,Rot:10 | 5 | 0.8960 | PASS |
| C1-1:Consistency | 5 | Noise:0.10,Rot:-5 | 5 | 0.8748 | PASS |
| C1-2:Discrimination | 0 | Clean | O | 0.3666 | PASS |
| C1-2:Discrimination | l | Clean | 1 | 0.8189 | FAIL |
| C1-2:Discrimination | Q | Clean | O | 0.9031 | FAIL |
| C1-3:Robustness | X | Noise:0.00 | X | 1.0000 | INFO |
| C1-3:Robustness | X | Noise:0.20 | X | 0.8238 | INFO |
| C1-3:Robustness | X | Noise:0.40 | X | 0.6260 | INFO |
| C1-3:Robustness | X | Noise:0.60 | X | 0.4788 | INFO |
| C1-3:Robustness | X | Noise:0.80 | X | 0.3520 | INFO |
| C1-3:Robustness | X | Noise:1.00 | X | 0.2704 | INFO |