# Addendum C: Partial Observability Study

Phase 3 (B) - Partial Observability Validation (smoke)

Generated: 2026-01-10T06:40:28.552272+00:00
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
- **max_onset_rate**: 0.533

## Per-gamma Results

| gamma | onset_rate | n_onset | OR_LCB | robust(OR) | strong(AND) | inadeq |
|-------|------------|---------|--------|------------|-------------|--------|
| 1.00 | 0.481 | 13 | 0.367 | True | True | 0.100 |
| 0.50 | 0.533 | 16 | 0.139 | True | True | 0.000 |
| 0.00 | 0.000 | 0 | 0.000 | False | False | 0.000 |

## Comparison with EnvB (v6)

| Metric | EnvB (v6) | EnvB_PO | Status |
|--------|-----------|---------|--------|
| crossing | True (up) | True (up) | PRESERVED |
| claim_B | True | True | PRESERVED |

## Conclusion

**PASS**: EnvB_PO preserves crossing and claim_B under partial observability.
OCI/CAP mechanisms are robust when velocity information is hidden.
