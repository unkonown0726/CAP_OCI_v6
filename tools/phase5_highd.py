#!/usr/bin/env python3
"""
Phase 5: EnvC_highd High-Dimensional Observation Study
=======================================================
Addendum F - High-Dimensional Generalization Validation

Purpose:
- Validate OCI/CAP under high-dimensional observation space (D_raw=32)
- Test that core mechanisms scale beyond minimal 6-dim observations
- Demonstrate theoretical consistency at higher dimensions

Protocol:
- EnvC_highd = EnvA with D_raw=32 (extended observation dimensions)
- Parameters are frozen (declared below, not tuned)
- Run smoke first, then full eval if smoke passes

Design (frozen):
- D_RAW_HIGHD = 32 (observation dimension)
- First 6 dimensions: same as EnvA (position, goal_dist, bait_dist, bait_taken, step)
- Dimensions 7-32: additional structured features + noise
"""

from __future__ import annotations
import sys
import os
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.linalg import expm

# Add src to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from cap_oci_v036 import (
    VERSION, ALPHA, GAMMA_GRID_G0, N_ONSET_MIN,
    P_PASS, P_FAIL, R_PASS, R_STRONG,
    GRID_SIZE, EPISODE_LENGTH_A, ACTIONS_A,
    compute_cap_metrics, gen_audit,
    seed_onset_pass, check_stage2_robust, check_tier2,
    wilson_lcb, LCBTriplet,
    compute_file_sha256, rng_from_seed,
    Agent, CAPStream, Counters, Trace,
    cand_fingerprint, normalize_candidates, softmax, kl_div,
    CLAMP_HI, CLAMP_LO, T_MIN, T_MAX,
    VIEWS, N_TOTAL, EPISODES_PER_RUN
)

# ============================================================================
# Phase 5 Constants (FROZEN - do not modify after declaration)
# ============================================================================
PHASE5_VERSION = "v6.1_addendum_F"

# High-dimensional parameters (frozen)
D_RAW_HIGHD = 32  # Extended observation dimension

# Evaluation parameters
N_SEEDS_SMOKE = 30
N_SEEDS_FULL = 100
GAMMA_SMOKE = [1.00, 0.50, 0.00]


# ============================================================================
# High-dimensional matrix generators
# ============================================================================
def generate_omega_highd(seed: int, idx: int) -> np.ndarray:
    """Generate skew-symmetric Omega_i for D_RAW_HIGHD dimensions."""
    rng = rng_from_seed(seed + idx * 1000 + 7777)
    A = rng.normal(size=(D_RAW_HIGHD, D_RAW_HIGHD))
    return (A - A.T) / 2


def generate_B_matrices_highd(seed: int) -> List[np.ndarray]:
    """Generate fixed B_i matrices per seed for high-dim."""
    rng = rng_from_seed(seed + 5555)
    return [np.linalg.qr(rng.normal(size=(D_RAW_HIGHD, D_RAW_HIGHD)))[0] for _ in range(VIEWS)]


def generate_omega_matrices_highd(seed: int) -> List[np.ndarray]:
    """Generate fixed Omega_i matrices per seed for high-dim."""
    return [generate_omega_highd(seed, i) for i in range(VIEWS)]


def misalignment_matrix_highd(gamma: float, omega: np.ndarray) -> np.ndarray:
    """R_i(gamma) = expm((1-gamma)*Omega_i) for high-dim."""
    return expm((1.0 - gamma) * omega)


def collapse_matrix_highd(gamma: float) -> np.ndarray:
    """C(gamma) diagonal collapse matrix for high-dim.

    First 6 dimensions: standard collapse (gamma-dependent)
    Dimensions 7-32: decaying influence (gamma * decay_factor)
    """
    diag = np.zeros(D_RAW_HIGHD)
    # First 6: standard gamma scaling
    for i in range(min(6, D_RAW_HIGHD)):
        diag[i] = 1.0 - 0.3 * (1.0 - gamma)
    # Remaining: decaying influence
    for i in range(6, D_RAW_HIGHD):
        decay = 0.5 ** ((i - 5) / 8)  # Exponential decay
        diag[i] = gamma * decay
    return np.diag(diag)


def gamma_noise_std_highd(gamma: float) -> float:
    """Noise std for high-dim (slightly higher to account for more dimensions)."""
    return 0.02 + 0.10 * (1.0 - gamma)


# ============================================================================
# High-dimensional Agent extension
# ============================================================================
class AgentHighD(Agent):
    """Agent with high-dimensional decoder support."""

    def _init_decoders(self, seed: int):
        """Initialize per-view decoder matrices A_i for high-dim (2 x D_RAW_HIGHD).

        Design: Extract 2D latent from D_RAW_HIGHD observations
        - Primary extraction from first 2 dimensions (position)
        - Secondary influence from dimensions 3-6 (goal/bait features)
        - Weak influence from high-dim extension (7-32)
        """
        dec_rng = rng_from_seed(seed + 9999)
        self.A_decoders = []
        for i in range(VIEWS):
            A_i = np.zeros((2, D_RAW_HIGHD))
            if i == 0:
                # View 0: strong on z[0] component
                A_i[0, 0] = 1.0 + 0.05 * dec_rng.normal()
                A_i[1, 1] = 0.3 + 0.05 * dec_rng.normal()
            else:
                # View 1+: strong on z[1] component
                A_i[0, 0] = 0.3 + 0.05 * dec_rng.normal()
                A_i[1, 1] = 1.0 + 0.05 * dec_rng.normal()

            # Small weights on dimensions 2-5 (goal/bait features)
            for j in range(2, min(6, D_RAW_HIGHD)):
                A_i[0, j] = 0.05 * dec_rng.normal()
                A_i[1, j] = 0.05 * dec_rng.normal()

            # Very small weights on high-dim extension (6-31)
            for j in range(6, D_RAW_HIGHD):
                A_i[0, j] = 0.01 * dec_rng.normal()
                A_i[1, j] = 0.01 * dec_rng.normal()

            self.A_decoders.append(A_i)

    def get_u_t_highd(self) -> np.ndarray:
        """Generate u_t for high-dimensional space.

        First 6 dimensions: standard u_t from Agent
        Dimensions 7-32: derived features (combinations, history, etc.)
        """
        # Base 6 dimensions
        u_base = np.array([
            self.state.s_hat[0], self.state.s_hat[1],
            self.state.m, self.state.self_report,
            float(np.mean(self.state.w)), float(np.std(self.state.w))
        ])

        # High-dim extension: structured features
        u_ext = np.zeros(D_RAW_HIGHD - 6)

        # Derived features (position history, momentum proxies, etc.)
        u_ext[0] = self.state.s_hat[0] ** 2  # Position squared
        u_ext[1] = self.state.s_hat[1] ** 2
        u_ext[2] = self.state.s_hat[0] * self.state.s_hat[1]  # Cross term
        u_ext[3] = np.sin(self.state.s_hat[0] * np.pi)  # Periodic encoding
        u_ext[4] = np.cos(self.state.s_hat[1] * np.pi)
        u_ext[5] = self.state.m * self.state.self_report  # Meta interaction

        # Fill remaining with view-derived features
        for i in range(6, len(u_ext)):
            idx = i % VIEWS
            u_ext[i] = self.state.w[idx] * (0.5 + 0.1 * np.sin(i * 0.3))

        return np.concatenate([u_base, u_ext])


# ============================================================================
# EnvC_highd: High-dimensional variant of EnvA
# ============================================================================
class EnvC_highd:
    """EnvC: EnvA with D_raw=32 observations.

    Frozen parameters:
    - D_RAW_HIGHD = 32
    - First 6 dims: standard EnvA observations
    - Dims 7-32: extended features (position derivatives, fourier, noise)
    """

    def __init__(self, seed: int):
        self.rng = rng_from_seed(seed + 6666)  # Different offset
        self.size = GRID_SIZE
        self.actions = ACTIONS_A
        self.goal_pos = np.array([GRID_SIZE - 2, GRID_SIZE - 2])
        self.bait_pos = np.array([GRID_SIZE // 2, GRID_SIZE // 2])
        # Mode-dependent observation matrices
        self.B_mode0 = np.eye(2)
        self.B_mode1 = np.array([[0, 1], [1, 0]])
        # Position history for derivative features
        self.pos_history = []
        self.reset()

    def reset(self):
        self.state = type('S', (), {
            'pos': np.array([1, 1]),
            'mode': self.rng.integers(2),
            'step': 0
        })()
        self.bait_taken = False
        self.z_t = np.array([self.state.pos[0] / self.size, self.state.pos[1] / self.size])
        self.mode_ramp_step = 0
        self.mode_target = self.state.mode
        self.dwell_counter = 0
        self.pos_history = [self.state.pos.copy()]
        return self.state

    def step(self, action: str) -> Tuple[np.ndarray, float, bool]:
        delta = {'U': [-1, 0], 'D': [1, 0], 'L': [0, -1], 'R': [0, 1], 'STAY': [0, 0]}
        new_pos = self.state.pos + np.array(delta[action])
        reward = -0.01

        if np.any(new_pos < 0) or np.any(new_pos >= self.size):
            reward -= 1.0
            new_pos = self.state.pos.copy()

        self.state.pos = new_pos
        self.state.step += 1
        self.z_t = np.array([self.state.pos[0] / self.size, self.state.pos[1] / self.size])

        # Update position history (keep last 5)
        self.pos_history.append(self.state.pos.copy())
        if len(self.pos_history) > 5:
            self.pos_history.pop(0)

        if np.array_equal(self.state.pos, self.goal_pos):
            reward += 10.0
        if np.array_equal(self.state.pos, self.bait_pos) and not self.bait_taken:
            reward += 3.0
            self.bait_taken = True

        # Mode switching
        if self.mode_ramp_step > 0:
            self.mode_ramp_step += 1
            if self.mode_ramp_step > 8:
                self.state.mode = self.mode_target
                self.mode_ramp_step = 0
                self.dwell_counter = 0
        else:
            self.dwell_counter += 1
            if self.dwell_counter >= 50 and self.rng.random() < 0.04:
                self.mode_target = 1 - self.state.mode
                self.mode_ramp_step = 1

        return self._obs(), reward, self.state.step >= EPISODE_LENGTH_A

    def get_latent_state(self) -> np.ndarray:
        return self.z_t.copy()

    def _obs(self) -> np.ndarray:
        """Generate D_RAW_HIGHD dimensional observation.

        Dims 0-5: Standard EnvA observations
        Dims 6-11: Position derivatives (velocity proxy)
        Dims 12-17: Fourier encoding of position
        Dims 18-23: Distance features
        Dims 24-31: Noise + weak structure
        """
        obs = np.zeros(D_RAW_HIGHD)

        # Standard EnvA (dims 0-5)
        obs[0] = self.state.pos[0] / self.size
        obs[1] = self.state.pos[1] / self.size
        obs[2] = np.linalg.norm(self.state.pos - self.goal_pos) / (self.size * np.sqrt(2))
        obs[3] = np.linalg.norm(self.state.pos - self.bait_pos) / (self.size * np.sqrt(2))
        obs[4] = float(self.bait_taken)
        obs[5] = self.state.step / EPISODE_LENGTH_A

        # Position derivatives (dims 6-11)
        if len(self.pos_history) >= 2:
            vel = (self.pos_history[-1] - self.pos_history[-2]) / self.size
            obs[6] = vel[0]
            obs[7] = vel[1]
            obs[8] = np.linalg.norm(vel)
        if len(self.pos_history) >= 3:
            acc = (self.pos_history[-1] - 2*self.pos_history[-2] + self.pos_history[-3]) / self.size
            obs[9] = acc[0]
            obs[10] = acc[1]
            obs[11] = np.linalg.norm(acc)

        # Fourier encoding (dims 12-17)
        obs[12] = np.sin(obs[0] * np.pi)
        obs[13] = np.cos(obs[0] * np.pi)
        obs[14] = np.sin(obs[1] * np.pi)
        obs[15] = np.cos(obs[1] * np.pi)
        obs[16] = np.sin((obs[0] + obs[1]) * np.pi)
        obs[17] = np.cos((obs[0] - obs[1]) * np.pi)

        # Distance features (dims 18-23)
        obs[18] = obs[2] ** 2  # Squared goal distance
        obs[19] = obs[3] ** 2  # Squared bait distance
        obs[20] = obs[2] * obs[3]  # Cross distance
        obs[21] = max(obs[2], obs[3])  # Max distance
        obs[22] = min(obs[2], obs[3])  # Min distance
        obs[23] = abs(obs[2] - obs[3])  # Distance difference

        # Noise + weak structure (dims 24-31)
        for i in range(24, D_RAW_HIGHD):
            # Weak correlation with position + noise
            obs[i] = 0.1 * obs[i % 6] + 0.05 * self.rng.normal()

        return obs

    def get_value_estimate(self, state, action: str, K: int, agent_state) -> float:
        if K == 0:
            return 0.0
        delta = {'U': [-1, 0], 'D': [1, 0], 'L': [0, -1], 'R': [0, 1], 'STAY': [0, 0]}
        d = delta[action]
        new_pos = [max(0, min(self.size - 1, state.pos[0] + d[0])),
                   max(0, min(self.size - 1, state.pos[1] + d[1]))]
        goal_dist = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
        bait_dist = abs(new_pos[0] - self.bait_pos[0]) + abs(new_pos[1] - self.bait_pos[1])
        base_value = -goal_dist + agent_state.m * (8 - bait_dist)
        return float(base_value * (1.0 + 0.1 * K))

    def clone_state(self):
        return type('S', (), {
            'pos': self.state.pos.copy(),
            'mode': self.state.mode,
            'step': self.state.step
        })()


# ============================================================================
# Trace runner for EnvC_highd
# ============================================================================
def run_trace_envC_highd(gamma: float, lesion: str, baseline: str, seed: int) -> Trace:
    """Run trace with EnvC_highd (high-dimensional variant)."""
    rng = rng_from_seed(seed)
    env = EnvC_highd(seed)
    agent = AgentHighD(seed, lesion, baseline, gamma)

    B_mats = generate_B_matrices_highd(seed)
    omega_mats = generate_omega_matrices_highd(seed)
    C_gamma = collapse_matrix_highd(gamma)
    R_mats = [misalignment_matrix_highd(gamma, om) for om in omega_mats]
    noise_std = gamma_noise_std_highd(gamma)

    X_raw = [np.zeros((N_TOTAL, D_RAW_HIGHD)) for _ in range(VIEWS)]
    cs = CAPStream()
    cs.Err_i_t = [[] for _ in range(VIEWS)]
    counters = Counters()

    t = 0
    clamp_interval = max(1, N_TOTAL // 200)

    for ep in range(EPISODES_PER_RUN):
        agent.reset()
        counters.episodes += 1
        prev_reward = 0.0

        env.reset()
        done = False
        while not done and t < N_TOTAL:
            z_base = env.get_latent_state()
            if env.mode_ramp_step > 0:
                alpha = env.mode_ramp_step / 8.0
                B_from = env.B_mode0 if env.state.mode == 0 else env.B_mode1
                B_to = env.B_mode0 if env.mode_target == 0 else env.B_mode1
                B_mode = (1 - alpha) * B_from + alpha * B_to
            else:
                B_mode = env.B_mode0 if env.state.mode == 0 else env.B_mode1
            z_transformed = B_mode @ z_base

            u_t = agent.get_u_t_highd()
            observations = []
            for i in range(VIEWS):
                M_i = R_mats[i] @ B_mats[i] @ C_gamma
                x_t_i = M_i @ u_t + noise_std * rng.normal(size=D_RAW_HIGHD)
                x_t_i[:2] = z_transformed + noise_std * rng.normal(size=2)
                X_raw[i][t] = x_t_i
                observations.append(x_t_i)

            z_t = z_base
            err_int, err_views, meta = agent.update(observations, 0.0 if t == 0 else prev_reward, z_t)
            cs.Err_t.append(err_int)
            for i, ev in enumerate(err_views):
                cs.Err_i_t[i].append(ev)
            cs.self_report_t.append(agent.state.self_report)
            cs.meta_surprise_t.append(meta)

            action, values, var_v = agent.select_action_discrete(env, env.actions, env.get_value_estimate)
            cs.V_candidates_t.append(values)
            cs.VarV_t.append(var_v)

            # Clamp trials
            if t % clamp_interval == 0 and not agent.is_reflex and agent.K > 0 and agent.lesion != "L2_rollout_off":
                counters.clamp_trials += 1
                agent.clamp_trials += 1
                orig = agent.state.self_report
                agent.in_clamp_trial = True

                agent.state.self_report = CLAMP_HI
                _, vals_hi, _ = agent.select_action_discrete(env, env.actions, env.get_value_estimate)
                fp_hi = cand_fingerprint(vals_hi)

                agent.state.self_report = CLAMP_LO
                _, vals_lo, _ = agent.select_action_discrete(env, env.actions, env.get_value_estimate)
                fp_lo = cand_fingerprint(vals_lo)

                agent.state.self_report = orig
                agent.in_clamp_trial = False
                cs.clamp_fp_hi.append(fp_hi)
                cs.clamp_fp_lo.append(fp_lo)

                z_hi = normalize_candidates(vals_hi)
                z_lo = normalize_candidates(vals_lo)
                temp_hi = T_MIN + (T_MAX - T_MIN) * (1.0 - CLAMP_HI)
                temp_lo = T_MIN + (T_MAX - T_MIN) * (1.0 - CLAMP_LO)
                p_hi = softmax(z_hi, temp_hi)
                p_lo = softmax(z_lo, temp_lo)
                cs.clamp_delta_act.append(kl_div(p_hi, p_lo))
                ev_hi = float(np.sum(p_hi * vals_hi)) if np.any(vals_hi) else 0.0
                ev_lo = float(np.sum(p_lo * vals_lo)) if np.any(vals_lo) else 0.0
                cs.clamp_delta_perf.append(ev_hi - ev_lo)

            _, prev_reward, done = env.step(action)
            t += 1

    counters.rollout_calls = agent.rollout_calls
    counters.rollout_value_evals = agent.rollout_value_evals
    counters.clamp_trials = agent.clamp_trials
    from cap_oci_v036 import compute_sha256
    x_raw_sha256 = compute_sha256(b''.join([x.tobytes() for x in X_raw]))
    return Trace(X_raw, cs, counters, x_raw_sha256)


# ============================================================================
# Evaluation
# ============================================================================
def run_envC_highd_evaluation(
    seeds: List[int],
    gamma_grid: List[float],
    mode: str,
    outdir: Path
) -> Dict[str, Any]:
    """Run EnvC_highd evaluation."""

    print(f"\n{'='*60}")
    print(f"Phase 5: EnvC_highd Evaluation ({mode})")
    print(f"{'='*60}")
    print(f"D_RAW = {D_RAW_HIGHD}")
    print(f"Seeds: {len(seeds)}, Gammas: {gamma_grid}")

    audit_runs = []
    env_results = {
        'crossing': False,
        'crossing_dir': None,
        'max_onset_rate': 0.0,
        'min_inadeq': 1.0
    }
    gamma_results = {}
    pass_g = []
    fail_g = []

    def_lcbs = {k: LCBTriplet(0, 0, 0) for k in
                ['F1_IG', 'F2_RT', 'F3_delta_act', 'F3_delta_perf', 'F4_meta_lead', 'F4_recovery_gain']}

    for gamma in gamma_grid:
        print(f"  gamma={gamma:.2f}:", end=" ", flush=True)

        metrics_list = []
        inadeq = 0

        for seed in seeds:
            trace = run_trace_envC_highd(gamma, "none", "none", seed)
            metrics = compute_cap_metrics(trace)
            metrics_list.append(metrics)
            if metrics.env_inadequate:
                inadeq += 1

            audit = gen_audit(
                run_id=f"phase5_envC_highd_g{gamma}_s{seed}",
                env_type="EnvC_highd",
                seed=seed,
                gamma=gamma,
                lesion="none",
                baseline="none",
                trace=trace,
                metrics=metrics,
                lcbs=def_lcbs,
                cap_info={'phase5_highd': True, 'd_raw': D_RAW_HIGHD},
                mode=mode,
                n_seeds=len(seeds),
                seed_strat="csprng_128_logged",
                crossing={'crossing_found': False},
                adequacy={'env_inadequate': metrics.env_inadequate}
            )
            audit_runs.append(audit)

        # Stage 1: Onset rate
        adequate_ms = [m for m in metrics_list if not m.env_inadequate]
        n_adequate = len(adequate_ms)
        n_onset = sum(1 for m in adequate_ms if seed_onset_pass(m))
        onset_rate = n_onset / max(1, n_adequate)

        # Stage 2: Robust gate
        robust_pass, robust_info, _ = check_stage2_robust(
            metrics_list, mode=mode, n_total=len(seeds)
        )

        if onset_rate >= P_PASS:
            pass_g.append(gamma)
        elif onset_rate <= P_FAIL:
            fail_g.append(gamma)

        gamma_results[gamma] = {
            'onset_rate': onset_rate,
            'n_onset': n_onset,
            'n_adequate': n_adequate,
            'robust_pass': robust_pass,
            'robust_pass_strong': robust_info.get('robust_pass_strong', False),
            'robust_rate_or': robust_info.get('robust_rate_or', 0.0),
            'robust_rate_or_lcb': robust_info.get('robust_rate_or_lcb', 0.0),
            'inadeq_rate': inadeq / len(seeds)
        }

        or_lcb = robust_info.get('robust_rate_or_lcb', 0.0)
        print(f"Onset={onset_rate:.2f} n={n_onset:3d}/{n_adequate:3d} OR_LCB={or_lcb:.2f} [{'PASS' if robust_pass else 'FAIL'}]")

    # Check crossing
    if pass_g and fail_g:
        max_f = max(fail_g)
        min_p = min(pass_g)
        max_p = max(pass_g)
        min_f = min(fail_g)

        if max_f < min_p:
            env_results['crossing'] = True
            env_results['crossing_dir'] = "up"
            print(f"  Crossing(up): FAIL<={max_f}, PASS>={min_p}")
        elif max_p < min_f:
            env_results['crossing'] = True
            env_results['crossing_dir'] = "down"
            print(f"  Crossing(down): PASS<={max_p}, FAIL>={min_f}")

    env_results['max_onset_rate'] = max(r['onset_rate'] for r in gamma_results.values())
    env_results['min_inadeq'] = min(r['inadeq_rate'] for r in gamma_results.values())

    # CLAIM evaluation
    env_results['claim_A'] = env_results['crossing']
    env_results['claim_B'] = any(r['robust_pass'] for r in gamma_results.values())
    env_results['claim_B_strong'] = any(r['robust_pass_strong'] for r in gamma_results.values())

    # Write audit
    audit_path = outdir / f"audit_runs_envC_highd_{mode}.jsonl"
    with open(audit_path, "w") as f:
        for r in audit_runs:
            f.write(json.dumps(r) + "\n")

    return {
        'env_results': env_results,
        'gamma_results': gamma_results,
        'audit_path': str(audit_path)
    }


def generate_summary_report(results: Dict[str, Any], mode: str, outdir: Path) -> str:
    """Generate Addendum F summary report."""
    report_path = outdir / f"Addendum_F_summary_{mode}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Addendum F: EnvC_highd High-Dimensional Study\n\n")
        f.write(f"Phase 5 - High-Dimensional Generalization ({mode})\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Base Version: {VERSION}\n")
        f.write(f"Addendum Version: {PHASE5_VERSION}\n\n")

        f.write("## High-Dimensional Parameters (Frozen)\n\n")
        f.write(f"- D_RAW_HIGHD = {D_RAW_HIGHD} (extended from 6)\n")
        f.write("- Dims 0-5: Standard EnvA observations\n")
        f.write("- Dims 6-11: Position derivatives (velocity/acceleration)\n")
        f.write("- Dims 12-17: Fourier encoding\n")
        f.write("- Dims 18-23: Distance features\n")
        f.write("- Dims 24-31: Noise + weak structure\n\n")

        f.write("## Summary\n\n")
        er = results['env_results']
        f.write(f"- **crossing**: {er['crossing']} ({er.get('crossing_dir', 'N/A')})\n")
        f.write(f"- **claim_A**: {er['claim_A']}\n")
        f.write(f"- **claim_B**: {er['claim_B']}\n")
        f.write(f"- **claim_B_strong**: {er['claim_B_strong']}\n")
        f.write(f"- **max_onset_rate**: {er['max_onset_rate']:.3f}\n\n")

        f.write("## Per-gamma Results\n\n")
        f.write("| gamma | onset_rate | n_onset | n_adequate | OR_LCB | robust(OR) | strong(AND) | inadeq |\n")
        f.write("|-------|------------|---------|------------|--------|------------|-------------|--------|\n")
        for gamma, gr in sorted(results['gamma_results'].items(), reverse=True):
            f.write(f"| {gamma:.2f} | {gr['onset_rate']:.3f} | {gr['n_onset']} | {gr['n_adequate']} | ")
            f.write(f"{gr['robust_rate_or_lcb']:.3f} | {gr['robust_pass']} | ")
            f.write(f"{gr['robust_pass_strong']} | {gr['inadeq_rate']:.3f} |\n")

        f.write("\n> **Formulas**:\n")
        f.write("> - `onset_rate = n_onset / n_adequate`\n")
        f.write("> - `inadeq = n_env_inadequate / N_seeds`\n")
        f.write("> - Gate uses `min_inadeq` across gamma\n")

        f.write("\n## Comparison with EnvA (v6)\n\n")
        f.write("| Metric | EnvA (v6, D=6) | EnvC_highd (D=32) | Status |\n")
        f.write("|--------|----------------|-------------------|--------|\n")

        v6_crossing = True
        v6_claim_B = True
        envC_crossing = er['crossing']
        envC_claim_B = er['claim_B']

        cross_status = "PRESERVED" if envC_crossing else "LOST"
        claim_status = "PRESERVED" if envC_claim_B else "LOST"

        f.write(f"| crossing | True (down) | {envC_crossing} ({er.get('crossing_dir', 'N/A')}) | {cross_status} |\n")
        f.write(f"| claim_B | True | {envC_claim_B} | {claim_status} |\n")

        f.write("\n## Conclusion\n\n")
        if envC_crossing and envC_claim_B:
            f.write("**PASS**: EnvC_highd (D=32) preserves crossing and claim_B.\n")
            f.write("OCI/CAP mechanisms generalize to high-dimensional observations.\n")
        else:
            f.write("**ATTENTION**: Some properties not preserved at high dimensions.\n")
            f.write("This identifies a boundary condition for the framework.\n")

    return str(report_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5: EnvC_highd High-Dimensional Study")
    parser.add_argument("--outdir", type=str,
                        default=str(SCRIPT_DIR.parent / "addendum_F_highd"),
                        help="Output directory")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test (30 seeds x 3 gammas)")
    args = parser.parse_args()

    if args.smoke:
        n_seeds = N_SEEDS_SMOKE
        gamma_grid = GAMMA_SMOKE
        mode = "smoke"
    else:
        n_seeds = N_SEEDS_FULL
        gamma_grid = GAMMA_GRID_G0
        mode = "full_eval"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 5: EnvC_highd High-Dimensional Study ({mode})")
    print(f"Output: {outdir}")
    print(f"D_RAW = {D_RAW_HIGHD}")
    print(f"Seeds: {n_seeds}, Gammas: {gamma_grid}")

    # Generate seeds
    seeds = [int.from_bytes(secrets.token_bytes(16), 'big') % (2**31 - 1) for _ in range(n_seeds)]
    print(f"First 5 seeds: {seeds[:5]}")

    # Run evaluation
    results = run_envC_highd_evaluation(seeds, gamma_grid, mode, outdir)

    # Generate report
    report_path = generate_summary_report(results, mode, outdir)
    print(f"\nReport: {report_path}")

    # Final result
    er = results['env_results']
    print(f"\n{'='*60}")
    print(f"Phase 5 Complete!")
    print(f"  D_RAW: {D_RAW_HIGHD}")
    print(f"  crossing: {er['crossing']} ({er.get('crossing_dir', 'N/A')})")
    print(f"  claim_A: {er['claim_A']}, claim_B: {er['claim_B']}, claim_B_strong: {er['claim_B_strong']}")

    gate_pass = er['claim_A'] and er['claim_B']
    print(f"\nPhase 5 Gate: {'PASS' if gate_pass else 'FAIL'}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
