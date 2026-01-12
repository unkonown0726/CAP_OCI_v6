# invariance_units.md - Final Registry (CAP+OCI v6)
# Generated: 2026-01-12

## 1. 単位・次元

### 無次元量 (Dimensionless Quantities)

| ID | 量 | 単位 | 定義域 | Source |
|----|-----|------|--------|--------|
| D001 | g (g-knob) | dimensionless | {1.00, 0.75, 0.50, 0.25, 0.00} | 1_MANUSCRIPT 2.1 |
| D003 | F1_IG | dimensionless | ℝ | 1_MANUSCRIPT 3.2 |
| D004 | F2_RT | dimensionless | ℝ | 1_MANUSCRIPT 3.2 |
| D005 | F3_delta_act | dimensionless | ℝ | 1_MANUSCRIPT 3.2 |
| D006 | F3_delta_perf | dimensionless | ℝ | 1_MANUSCRIPT 3.2 |
| D007 | F4_meta_lead | dimensionless | ℝ | 1_MANUSCRIPT 3.2 |
| D008 | F4_recovery_gain | dimensionless | ℝ | 1_MANUSCRIPT 3.2 |

### 比率・確率 (Rates/Probabilities)

| ID | 量 | 定義域 | 計算式 |
|----|-----|--------|--------|
| D011 | onset_rate | [0, 1] | n_onset / n_adequate |
| D012 | inadeq(g) | [0, 1] | n_env_inadequate / N_seeds |
| D013 | OR_LCB | [0, 1] | max(LCB(meta), LCB(rec)) |
| D014 | AND_LCB | [0, 1] | LCB(and_rate) |

---

## 2. 閾値パラメータ (Frozen)

### Stage A (Onset)

| パラメータ | 値 | 役割 | 感度sweep |
|-----------|-----|------|-----------|
| P_PASS | 0.30 | onset_rate ≥ P_PASS → pass_g | 固定 |
| P_FAIL | 0.10 | onset_rate ≤ P_FAIL → fail_g | 固定 |
| F1_ONSET_MIN | 0.0 | F1_IG > F1_ONSET_MIN for onset | 固定 |
| max_inadeq | 0.10 | min_g(inadeq) ≤ max_inadeq required | 固定 |

### Stage B (Robustness)

| パラメータ | デフォルト | sweep範囲 | Source |
|-----------|-----------|-----------|--------|
| R_PASS | 0.15 | {0.10, 0.15, 0.20, 0.25} | CLAIM_B |
| R_STRONG | 0.05 | {0.05, 0.10} | CLAIM_B+ |
| θ_lead | 0.0 | {0.00, -0.05} | meta route |
| θ_rec | 0.0 | {0.00, -0.05} | recovery route |
| N_ONSET_MIN | 20 | 固定 | minimum onset |
| α (Wilson) | 0.10 | 固定 | confidence |

### Stage C (Selective Collapse)

| パラメータ | 値 | 役割 |
|-----------|-----|------|
| α (Cantelli) | 0.10 | LCB confidence |

---

## 3. 計算式

### onset_rate 計算
```
onset_rate(g) = n_onset(g) / n_adequate(g)

where:
  n_adequate(g) = N_seeds - n_env_inadequate(g)

  onset_pass(m) = (
    NOT candidate_set_mismatch
    AND all metrics finite
    AND F3_delta_act > 0
    AND F3_delta_perf > 0
    AND F1_IG > 0
  )
```
**注意**: 分母は N_seeds ではなく n_adequate

### Wilson one-sided LCB (α = 0.10)
```
LCB = (center - margin) / denom

where:
  z = Φ^(-1)(1 - α) = Φ^(-1)(0.9)
  z² = z * z
  p̂ = k / n
  denom = 1 + z² / n
  center = p̂ + z² / (2n)
  margin = z * √(p̂(1-p̂)/n + z²/(4n²))
```

### Cantelli LCB (α = 0.10)
```
LCB = x̄ - √((1-α)/(αn)) × s

where:
  x̄ = sample mean
  s = sample std (ddof=1, unbiased)
  n = sample size
```

### OR_LCB / AND_LCB
```
OR_LCB(g) = max(LCB(meta_rate), LCB(recovery_rate))
AND_LCB(g) = LCB(and_rate)

where:
  meta_rate = (# F4_meta_lead > θ_lead) / n_onset
  rec_rate = (# F4_recovery_gain > θ_rec) / n_onset
  and_rate = (# both > θ) / n_onset
```

---

## 4. スケール不変性

### Threshold-Invariance Claim (C005)
- 32組み合わせの閾値 sweep で全て claim_ready = True
- **不変性**: 結論が特定の閾値選択に依存しない

### 単調性 (C012)
- Success condition (F4 > θ) の形なので、
  pass rates と Wilson LCB は θ に対して単調減少
- θ ∈ {0.00, -0.05} の範囲で結論が安定

---

## 5. 重要な非対称性

### L3_sel の閾値非対称性 (X004)

| Metric | Collapse condition |
|--------|-------------------|
| F4_meta_lead | **≤ 0** (includes equality) |
| F4_recovery_gain | **< 0** (excludes equality) |
| F1_IG (L1_sel) | < 0 |
| F2_RT (L2_sel) | < 0 |

**実装時に特に注意が必要**

---

## 6. 正規化・変換

### Winsorization (optional, Stage C)
```
winsorize(values, q) → clip to [quantile(q), quantile(1-q)]
default WINSOR_Q = 0.025 (if enabled)
```

### Robust z-score
```
robust_z(x) = (x - median) / (1.4826 × MAD + ε)
```

---

## 7. 実装固定パラメータ (Frozen)

| パラメータ | 値 | Source |
|-----------|-----|--------|
| N_TOTAL | 2000 | steps per run |
| VIEWS | 4 | observation views |
| D_RAW | 6 | observation dimension (base) |
| K_ROLLOUT | 4 | rollout depth |
| N_SEEDS | 100 | seeds per evaluation |

---

## 8. 未確定事項 (status: uncertain)

| 項目 | 不明点 | 解決状況 |
|-----|--------|----------|
| onset_rate 分母 | CM_Op_Def vs PROTOCOL_APPENDIX の表記差異 | 実装で n_adequate を確認、解決済み |
| F1-F4 の詳細正規化 | 内部計算の詳細 | 実装コードで確認可能 |
| Cantelli vs Wilson 使い分け | Stage B: Wilson (rate), Stage C: Cantelli (mean) | 仕様通り |
