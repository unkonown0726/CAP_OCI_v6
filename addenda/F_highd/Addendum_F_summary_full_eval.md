# Addendum F: EnvC_highd High-Dimensional Study

Phase 5 - High-Dimensional Generalization (full_eval)

Generated: 2026-01-10T07:58:05.497926+00:00
Base Version: v0.3.6p15.2ABv6
Addendum Version: v6.1_addendum_F

## High-Dimensional Parameters (Frozen)

- D_RAW_HIGHD = 32 (extended from 6)
- Dims 0-5: Standard EnvA observations
- Dims 6-11: Position derivatives (velocity/acceleration)
- Dims 12-17: Fourier encoding
- Dims 18-23: Distance features
- Dims 24-31: Noise + weak structure

## Summary

- **crossing**: True (down)
- **claim_A**: True
- **claim_B**: True
- **claim_B_strong**: True
- **max_onset_rate**: 1.000

## Per-gamma Results

| gamma | onset_rate | n_onset | n_adequate | OR_LCB | robust(OR) | strong(AND) | inadeq |
|-------|------------|---------|------------|--------|------------|-------------|--------|
| 1.00 | 0.000 | 0 | 97 | 0.000 | False | False | 0.030 |
| 0.75 | 0.000 | 0 | 87 | 0.000 | False | False | 0.130 |
| 0.50 | 0.185 | 17 | 92 | 0.747 | False | False | 0.080 |
| 0.25 | 0.955 | 84 | 88 | 0.828 | True | True | 0.120 |
| 0.00 | 1.000 | 78 | 78 | 0.830 | True | True | 0.220 |

> **Formulas**:
> - `onset_rate = n_onset / n_adequate`
> - `inadeq = n_env_inadequate / N_seeds`
> - Gate uses `min_inadeq` across gamma

## Comparison with EnvA (v6)

| Metric | EnvA (v6, D=6) | EnvC_highd (D=32) | Status |
|--------|----------------|-------------------|--------|
| crossing | True (down) | True (down) | PRESERVED |
| claim_B | True | True | PRESERVED |

## Conclusion

**PASS**: EnvC_highd (D=32) preserves crossing and claim_B.
OCI/CAP mechanisms generalize to high-dimensional observations.
