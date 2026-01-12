# Addendum D: HMM/Non-stationary A-lite Study

Phase 4 (A-lite) - Memory/History Dependency (full_eval)

Generated: 2026-01-10T06:49:54.745994+00:00
Base Version: v0.3.6p15.2ABv6
Addendum Version: v6.1_addendum_D

## Design

- **Environment**: EnvA_HMM (EnvA with hidden reward mode)
- **Switch probability**: 0.02 (after 30 step dwell)
- **Mode 0**: Goal-seeking (goal=+10, bait=+3)
- **Mode 1**: Bait-seeking (goal=+3, bait=+10)
- **Hidden state**: reward mode is NOT directly observable

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
| 0.75 | 0.000 | 0 | 0.000 | False | False | 0.060 |
| 0.50 | 0.275 | 25 | 0.938 | True | True | 0.090 |
| 0.25 | 0.963 | 79 | 0.892 | True | True | 0.180 |
| 0.00 | 1.000 | 66 | 0.784 | True | True | 0.340 |

## Comparison with EnvA (v6)

| Metric | EnvA (v6) | EnvA_HMM | Status |
|--------|-----------|----------|--------|
| crossing | True (down) | True (down) | PRESERVED |
| claim_B | True | True | PRESERVED |

## Conclusion

**PASS**: EnvA_HMM preserves crossing and claim_B.
OCI/CAP mechanisms function under hidden-mode non-stationarity.
