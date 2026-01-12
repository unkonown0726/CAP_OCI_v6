# CAP+OCI v6 Canonical Specification Registry

**Version**: v0.3.6p15.2ABv6
**Generated**: 2026-01-12
**Purpose**: 研究仕様の台帳化・固定（要約ではなくID付き仕様として参照可能）

---

## Overview

本レジストリは CAP+OCI v6 (Consciousness Modality Evaluation Protocol) の全仕様を、見落としやすい細かい条件・定義・例外・適用域を含めて漏れなく抽出し、参照可能な形式で固定したものです。

### What is CAP+OCI v6?

CAP+OCI v6 は「意識モード (CM: Consciousness Modality)」の操作的評価プロトコルです。

**主張するもの**:
- g-dependent onset regime transition の存在
- Conditional robustness (meta-lead または recovery-gain 経路)
- Mechanistic specificity (lesion による selective collapse)

**主張しないもの**:
- 主観的経験 (subjective experience)
- 道徳的地位 (moral status)
- 人格 (personhood)
- 人間の意識との同等性

---

## Registry Files

| File | Description | ID Prefix |
|------|-------------|-----------|
| `definitions.yml` | 定義・記号・量・データ生成過程・観測仕様 | D001-D050 |
| `assumptions.yml` | 前提条件・仮定・正則性・境界条件 | A001-A026 |
| `claims.yml` | 定理・主張・アルゴリズム保証 | C001-C024 |
| `theorem_contracts.yml` | 各claimの必要条件→結論→適用域→依存→検証項目 | T001-T006 |
| `invariance_units.md` | 単位・次元・スケール変換則・正規化規則 | - |
| `exceptions_edgecases.yml` | 破綻条件・境界ケース・反例・注意事項 | X001-X023 |
| `evidence_map.yml` | claim → supporting sources のマッピング | E001-E032 |

---

## Quick Reference: Core Claims

### CLAIM_A (Onset Crossing)
```yaml
ID: C001
Condition: crossing exists ∧ max(onset_rate) ≥ 0.30 ∧ min(inadeq) ≤ 0.10
Depends: D011, D012, D023
```

### CLAIM_B (Weak Robustness)
```yaml
ID: C002
Condition: CLAIM_A ∧ ∃g: (n_onset ≥ 20 ∧ OR_LCB ≥ 0.15)
Depends: C001, D013, A007
```

### CLAIM_B+ (Strong Robustness)
```yaml
ID: C003
Condition: CLAIM_A ∧ ∃g: (n_onset ≥ 20 ∧ AND_LCB ≥ 0.05)
Note: Report-only (optional)
```

### CLAIM_C (Mechanistic Specificity)
```yaml
ID: C004
Condition: CLAIM_B ∧ ∃g: selective_all(g)
Depends: C002, D017
```

### claim_ready (Package-Level)
```yaml
ID: D024
Condition: CLAIM_B in BOTH environments (EnvA_grid AND EnvB_continuous)
```

---

## Critical Implementation Notes

### 1. onset_rate の分母定義 (X010)

```
onset_rate(g) = n_onset / n_adequate   # NOT n_onset / N_total

where n_adequate = N_seeds - n_env_inadequate
```

**env_inadequate は「失敗」ではなく「測定不能」** - 混同禁止 (X014)

### 2. L3_sel の閾値非対称性 (X004)

```yaml
L1_sel: F1_IG < 0           # strict inequality
L2_sel: F2_RT < 0           # strict inequality
L3_sel:
  F4_meta_lead ≤ 0          # INCLUDES equality
  F4_recovery_gain < 0      # strict inequality
```

この非対称性は実装時に特に注意が必要。

### 3. Stage B の条件付き評価 (X017)

Stage B は **onset seeds のみ** に条件付きで評価。
onset 条件なしで評価すると protocol violation。

### 4. Learning vs Memory (D040/D041)

| 概念 | 定義 | 評価中の扱い |
|------|------|-------------|
| Learning | Q (parameters) の変更 | **禁止** (X011) |
| Memory | State variables (h_t, m_t) の保持・更新 | **必須** |

---

## Frozen Parameters (v6)

```yaml
# Evaluation Design
N_TOTAL: 2000          # steps per run
N_SEEDS: 100           # seeds per evaluation
VIEWS: 4               # observation views
D_RAW: 6               # observation dimension (base)
K_ROLLOUT: 4           # rollout depth
G0: [1.00, 0.75, 0.50, 0.25, 0.00]  # g-grid

# Statistical
ALPHA: 0.10            # confidence level (Wilson/Cantelli)

# Stage A (Onset)
P_PASS: 0.30           # onset_rate ≥ P_PASS → pass_g
P_FAIL: 0.10           # onset_rate ≤ P_FAIL → fail_g
F1_ONSET_MIN: 0.0      # F1_IG > 0 for onset
max_inadeq: 0.10       # min_g(inadeq) must be ≤ 0.10

# Stage B (Robustness)
R_PASS: 0.15           # OR_LCB threshold (CLAIM_B)
R_STRONG: 0.05         # AND_LCB threshold (CLAIM_B+)
N_ONSET_MIN: 20        # minimum onset seeds for assertion
θ_lead: 0.0            # meta route threshold
θ_rec: 0.0             # recovery route threshold
```

---

## Statistical Formulas

### Wilson one-sided LCB (D015)
```
LCB = (center - margin) / denom

z = Φ⁻¹(1 - α)    # α = 0.10
denom = 1 + z²/n
center = p̂ + z²/(2n)
margin = z × √(p̂(1-p̂)/n + z²/(4n²))
```

### Cantelli LCB (D016)
```
LCB = x̄ - √((1-α)/(αn)) × s

x̄ = sample mean
s = sample std (ddof=1)
```

---

## Falsification Conditions (Terminal)

| ID | Condition | Effect |
|----|-----------|--------|
| X001 | candidate_set_mismatch | Seed rejected; CM falsified if widespread |
| X011 | Q changes during evaluation | CM claim immediately falsified |
| X012 | Post-hoc metric generation | Suspected fabrication |
| X019 | Crossing absence | CLAIM_A not established |

---

## Evidence Summary

| Claim | Evidence Strength | Sources |
|-------|------------------|---------|
| CLAIM_A/B/B+/C | Strong | Manuscript, Protocol, Results, Implementation |
| Threshold Invariance | Strong | 32-combination sweep |
| Reproducibility | Strong | 3 independent runs |
| Seed Robustness | Strong | Addendum A (sets C, D) |
| Noise Robustness | Strong | Addendum B (EnvE) |
| Partial Observability | Strong | Addendum C (EnvB_PO) |
| Non-stationarity | Strong | Addendum D (EnvA_HMM) |
| High-Dimensional | Strong | Addendum F (D=32) |

---

## Directory Structure

```
registry_final/
├── README.md                  # This file
├── definitions.yml            # D001-D050
├── assumptions.yml            # A001-A026
├── claims.yml                 # C001-C024
├── theorem_contracts.yml      # T001-T006
├── invariance_units.md        # Units/scales/formulas
├── exceptions_edgecases.yml   # X001-X023
└── evidence_map.yml           # E001-E032
```

---

## How to Use This Registry

### 1. 定義の参照
```yaml
# 例: F4_meta_lead の定義を確認
# → definitions.yml の D007 を参照
```

### 2. 依存関係の追跡
```yaml
# 例: CLAIM_B が依存する定義・仮定を確認
# → claims.yml の C002.depends_on を参照
# → theorem_contracts.yml の T002 で検証項目を確認
```

### 3. 例外条件の確認
```yaml
# 例: 実装時の注意点を確認
# → exceptions_edgecases.yml で severity: implementation_critical を検索
```

### 4. エビデンスの確認
```yaml
# 例: CLAIM_B の根拠を確認
# → evidence_map.yml の E002 を参照
```

---

## Checkpoints

中間成果物は以下のディレクトリに保存:

- `registry_checkpoint1/`: MANUSCRIPT + PROTOCOL_APPENDIX からの抽出
- `registry_checkpoint2/`: Supplements (3-12) からの追加抽出

---

## Known Uncertainties

| ID | Item | Status | Resolution |
|----|------|--------|------------|
| X020 | onset_rate 分母の表記差異 | Resolved | 実装で n_adequate を確認 |

---

## Source Documents

| # | File | Role |
|---|------|------|
| 1 | 1_MANUSCRIPT.pdf | Main paper |
| 2 | 2_PROTOCOL_APPENDIX.pdf | Protocol specification |
| 3 | 3_SUPPLEMENT_AGENT_TRANSFER.pdf | Porting guide |
| 4 | 4_CLAIMS_PUBLIC.pdf | Public claim statements |
| 5 | 5_README_RELEASE.pdf | Package overview |
| 6 | 6_DOC_SELECTION.pdf | Document guide |
| 7 | 7_Convergence_Dynamics.pdf | Theoretical foundation |
| 8 | 8_CM_Operational_Definition.pdf | Axiom system (v2.1) |
| 9 | 9_Supplement_S-AI.pdf | AI transfer guide (detailed) |
| 10-12 | Memory_Learning_*.pdf | Memory/Learning distinction |
| - | src/cap_oci_v036.py | Implementation |
| - | Addendum_A-F | Generalization studies |

---

## Citation

```
CAP+OCI Protocol v0.3.6p15.2ABv6
Consciousness Modality Evaluation Framework
Date: 2026-01-09
```

---

## License

See main package for license information.
