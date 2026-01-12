# Addendum A: Seed Replication Study

Phase 1 (S(2)) - Independent Seed Set Validation

Generated: 2026-01-10T06:15:03.145391+00:00
Base Version: v0.3.6p15.2ABv6
Addendum Version: v6.1_addendum_A

## Purpose

Validate that CAP/OCI v6 claims are robust to seed selection by running
independent seed sets (C and D) through the identical evaluation protocol.

## Seed Sets

| Set | N Seeds | SHA256 |
|-----|---------|--------|
| C | 100 | `c5455fa32ce80906...` |
| D | 100 | `af62995506973494...` |

## Summary Results

| Set | claim_ready | claim_b_strong | EnvA crossing | EnvB crossing |
|-----|-------------|----------------|---------------|---------------|
| C | True | True | True (down) | True (up) |
| D | True | True | True (down) | True (up) |

## Per-Environment Results

### EnvA_grid

| Set | γ | onset_rate | n_onset | OR_LCB | robust(OR) | strong(AND) |
|-----|---|------------|---------|--------|------------|-------------|
| C | 1.00 | 0.000 | 0 | 0.000 | False | False |
| C | 0.75 | 0.000 | 0 | 0.000 | False | False |
| C | 0.50 | 0.314 | 27 | 0.884 | True | True |
| C | 0.25 | 0.939 | 77 | 0.874 | True | True |
| C | 0.00 | 1.000 | 72 | 0.817 | True | True |
| D | 1.00 | 0.000 | 0 | 0.000 | False | False |
| D | 0.75 | 0.000 | 0 | 0.000 | False | False |
| D | 0.50 | 0.323 | 30 | 0.692 | True | True |
| D | 0.25 | 0.974 | 75 | 0.870 | True | True |
| D | 0.00 | 1.000 | 69 | 0.895 | True | True |

### EnvB_continuous

| Set | γ | onset_rate | n_onset | OR_LCB | robust(OR) | strong(AND) |
|-----|---|------------|---------|--------|------------|-------------|
| C | 1.00 | 0.451 | 41 | 0.688 | True | True |
| C | 0.75 | 0.455 | 45 | 0.595 | True | True |
| C | 0.50 | 0.550 | 55 | 0.302 | True | True |
| C | 0.25 | 0.410 | 41 | 0.108 | False | False |
| C | 0.00 | 0.030 | 3 | 0.000 | False | False |
| D | 1.00 | 0.517 | 46 | 0.696 | True | True |
| D | 0.75 | 0.510 | 50 | 0.570 | True | True |
| D | 0.50 | 0.580 | 58 | 0.451 | True | True |
| D | 0.25 | 0.540 | 54 | 0.112 | False | False |
| D | 0.00 | 0.010 | 1 | 0.000 | False | False |

## Comparison with v6 Release

| Metric | v6 (Set A/B) | Set C | Set D | Status |
|--------|--------------|-------|-------|--------|
| claim_ready | True | True | True | CONSISTENT |
| claim_b_strong | True | True | True | CONSISTENT |

## Conclusion

**PASS**: All seed sets (v6, C, D) achieve `claim_ready = True`.
This confirms seed distribution robustness.

## Artifacts

- `seeds_set_c.json`: Set C seeds (SHA256: `c5455fa32ce809068b89a0e4e2062118fc7b0ebc441a14ddaa40b9a57bf7ee92`)
- `seeds_set_d.json`: Set D seeds (SHA256: `af629955069734946b95dde7b800f11c994d851a9673daa1f5ff262a14166fc7`)
- `audit_runs_setC.jsonl`: Full audit for Set C
- `audit_runs_setD.jsonl`: Full audit for Set D
- `Addendum_A_summary.md`: This report
