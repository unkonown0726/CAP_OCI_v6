# Addendum B: EnvE Stochastic Noise Study

Phase 2 (S(1)) - Noise Robustness Validation (full_eval)

Generated: 2026-01-10T06:33:50.305234+00:00
Base Version: v0.3.6p15.2ABv6
Addendum Version: v6.1_addendum_B

## Noise Parameters (Frozen)

- ACTION_SLIP_PROB = 0.1 (random action probability)
- OBS_NOISE_STD = 0.05 (observation noise std)
- REWARD_NOISE_STD = 0.02 (reward noise std)

## Summary

- **crossing**: True (down)
- **claim_A**: True
- **claim_B**: True
- **claim_B_strong**: True
- **max_onset_rate**: 1.000

## Per-gamma Results

| gamma | onset_rate | n_onset | OR_LCB | robust(OR) | strong(AND) | inadeq |
|-------|------------|---------|--------|------------|-------------|--------|
| 1.00 | 0.000 | 0 | 0.000 | False | False | 0.000 |
| 0.75 | 0.000 | 0 | 0.000 | False | False | 0.050 |
| 0.50 | 0.138 | 12 | 0.758 | False | False | 0.130 |
| 0.25 | 0.961 | 74 | 0.837 | True | True | 0.230 |
| 0.00 | 1.000 | 64 | 0.760 | True | True | 0.360 |

## Comparison with EnvA (v6)

| Metric | EnvA (v6) | EnvE | Status |
|--------|-----------|------|--------|
| crossing | True (down) | True (down) | PRESERVED |
| claim_B | True | True | PRESERVED |

## Conclusion

**PASS**: EnvE (stochastic) preserves crossing and claim_B.
OCI/CAP mechanisms are robust to the specified noise levels.
