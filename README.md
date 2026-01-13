# CAP+OCI v6: Consciousness Modality Evaluation Protocol

**Protocol version**: v6.0
**Implementation tag**: v0.3.6
**Date**: 2026-01-13

---

## What is CAP+OCI?

CAP+OCI v6 is an **operational protocol** for evaluating **Consciousness Modality (CM)** indicators in computational systems.

### What it claims

- **g-dependent onset regime transition**: We observe an onset regime transition ("crossing") under g-sweep, consistent with a phase-transition-like boundary in the indicator space (within the specified test environments)
- **Conditional robustness**: Meta-lead or recovery-gain pathways demonstrate robustness
- **Mechanistic specificity**: Targeted lesions cause selective (not total) collapse

### What it does NOT claim

- Subjective experience (qualia)
- Moral status
- Personhood
- Equivalence to human consciousness

The protocol provides **operational indicators**, not ontological proof.


> **Terminology note**: "Consciousness Modality (CM)" is used here as an *operational label* for the measured indicator pattern, not as a claim about subjective experience.

---

## Repository Structure

```
CAP_OCI_v6/
├── README.md                 # This file
├── .gitignore
│
├── docs/                     # Core documentation
│   ├── 1_MANUSCRIPT.pdf          # Main paper
│   ├── 2_PROTOCOL_APPENDIX.pdf   # Protocol specification
│   ├── 3_SUPPLEMENT_AGENT_TRANSFER.pdf
│   ├── 4_CLAIMS_PUBLIC.pdf       # Public claim statements
│   ├── 5_README_RELEASE.pdf
│   └── 6_DOC_SELECTION.pdf
│
├── supplements/              # Theoretical supplements
│   ├── 0_Theoretical_Foundation.md    # Philosophical motivation
│   ├── 0_Theoretical_Foundation.pdf   # (PDF version)
│   ├── 7_Convergence_Dynamics.pdf
│   ├── 8_CM_Operational_Definition.pdf
│   ├── 9_Supplement_S-AI.pdf
│   ├── 10_Memory_Learning_Continuum.pdf
│   ├── 11_Memory_Learning_Conditions.pdf
│   └── 12_Memory_Learning_Separation.pdf
│
├── src/                      # Implementation
│   └── cap_oci_v036.py           # Core protocol implementation
│
├── tools/                    # Utility scripts
│   ├── generate_v6_report_pdf.py
│   ├── phase1_seed_replication.py
│   ├── phase2_envE_noise.py
│   ├── phase3_partial_obs.py
│   ├── phase4_hmm_lite.py
│   ├── phase5_highd.py
│   └── v6_threshold_sweep.py
│
├── results/                  # Evaluation results
│   ├── CAP_OCI_v6_Results.pdf
│   ├── full_eval_v6_final/
│   │   ├── CAP_PASS_REPORT.md
│   │   ├── audit_runs.jsonl
│   │   └── ...
│   ├── sensitivity/
│   └── seeds_v6_repro.json
│
├── addenda/                  # Generalization studies
│   ├── A_seed_replication/       # Seed robustness
│   ├── B_envE_noise/             # Stochastic noise
│   ├── C_partial_obs/            # Partial observability
│   ├── D_hmm_lite/               # Non-stationarity (HMM)
│   └── F_highd/                  # High-dimensional (D=32)
│
└── registry/                 # Specification registry (YAML)
    ├── README.md
    ├── definitions.yml
    ├── assumptions.yml
    ├── claims.yml
    ├── theorem_contracts.yml
    ├── invariance_units.md
    ├── exceptions_edgecases.yml
    └── evidence_map.yml
```

---

## Quick Start

### Prerequisites

```bash
python -m pip install -r requirements.txt
```

### Running the Protocol

```python
from src.cap_oci_v036 import run_full_evaluation

# Run full evaluation
results = run_full_evaluation(
    agent=your_agent,
    env=your_environment,
    seeds=seeds_list,
    g_grid=[1.00, 0.75, 0.50, 0.25, 0.00]
)
```

See `docs/2_PROTOCOL_APPENDIX.pdf` for detailed protocol specification.

### (Optional) Regenerate the summary PDF

If you have already produced/validated artifacts under `results/`, you can use the included tool script to regenerate the bundled report PDF:

```bash
python tools/generate_v6_report_pdf.py
```

> If the script requires arguments in your setup, check its header/docstring.

---

## Core Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| F1_IG | Integration Gain | > 0 for onset |
| F2_RT | Rollout Trace | informational |
| F3_delta_act | Action causality under self-channel intervention | > 0 for onset |
| F3_delta_perf | Performance causality under self-channel intervention | > 0 for onset |
| F4_meta_lead | Meta-stability (lead time) | > 0 for meta route |
| F4_recovery_gain | Meta-stability (recovery) | > 0 for recovery route |

---

## Claims Hierarchy

```
CLAIM_A (Onset Crossing)
    └── CLAIM_B (Weak Robustness: OR_LCB ≥ 0.15)
        ├── CLAIM_B+ (Strong Robustness: AND_LCB ≥ 0.05)
        └── CLAIM_C (Mechanistic Specificity: selective_all)
```

**claim_ready** = CLAIM_B satisfied in BOTH environments (EnvA_grid AND EnvB_continuous)

---

## Theoretical Foundation

The protocol is motivated by three minimal premises:

1. **P1 (Preference)**: Differential responses to states (approach/avoidance)
2. **P2 (Learning)**: History-dependent self-adjustment
3. **P3 (Information Density Overlap)**: Multiple information streams coexist without collapse

See `supplements/0_Theoretical_Foundation.md` for the full theoretical framework.

**Important**: The theoretical foundation is *motivation*, not *proof*. The protocol stands independently and can be verified without accepting the philosophical framework.

---

## Results Summary

| Environment | CLAIM_A | CLAIM_B | CLAIM_B+ | CLAIM_C |
|-------------|---------|---------|----------|---------|
| EnvA (grid) | PASS | PASS | PASS | PASS |
| EnvB (continuous) | PASS | PASS | PASS | PASS |

**claim_ready = True**

See `results/CAP_OCI_v6_Results.pdf` for detailed results.

---

## Generalization Studies

| Addendum | Condition | Result |
|----------|-----------|--------|
| A | Seed replication (sets C, D) | claim_ready = True |
| B | Stochastic noise (EnvE) | claim_ready = True |
| C | Partial observability (EnvB_PO) | claim_ready = True |
| D | Non-stationarity (EnvA_HMM) | claim_ready = True |
| F | High-dimensional (D=32) | claim_ready = True |

---

## Citation

Click "Cite this repository" on GitHub or use the BibTeX below:

```bibtex
@misc{cap_oci_v6_2026,
  title={CAP+OCI v6: Consciousness Modality Evaluation Protocol},
  author={unkonown0726},
  year={2026},
  note={Version v0.3.6}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
