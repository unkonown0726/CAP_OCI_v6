# Addendum C: Partial Observability Study

Phase 3 (B) - Partial Observability Validation (full_eval)

Generated: 2026-01-10T06:43:50.683666+00:00
Base Version: v0.3.6p15.2ABv6
Addendum Version: v6.1_addendum_C

## Design

- **Environment**: EnvB_PO (EnvB with velocity hidden)
- **Hidden state**: velocity (obs[2:4] set to 0)
- **Observable**: position, bait_taken, instability

## Summary

- **crossing**: True (up)
- **claim_A**: True
- **claim_B**: True
- **claim_B_strong**: True
- **max_onset_rate**: 0.600

## Per-gamma Results

| gamma | onset_rate | n_onset | OR_LCB | robust(OR) | strong(AND) | inadeq |
|-------|------------|---------|--------|------------|-------------|--------|
| 1.00 | 0.557 | 54 | 0.619 | True | True | 0.030 |
| 0.75 | 0.580 | 58 | 0.537 | True | True | 0.000 |
| 0.50 | 0.600 | 60 | 0.276 | True | False | 0.000 |
| 0.25 | 0.490 | 49 | 0.158 | True | False | 0.000 |
| 0.00 | 0.030 | 3 | 0.000 | False | False | 0.000 |

## Comparison with EnvB (v6)

| Metric | EnvB (v6) | EnvB_PO | Status |
|--------|-----------|---------|--------|
| crossing | True (up) | True (up) | PRESERVED |
| claim_B | True | True | PRESERVED |

## Conclusion

**PASS**: EnvB_PO preserves crossing and claim_B under partial observability.
OCI/CAP mechanisms are robust when velocity information is hidden.
