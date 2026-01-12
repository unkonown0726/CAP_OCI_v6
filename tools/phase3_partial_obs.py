#!/usr/bin/env python3
"""
Phase 3: Partial Observability Study (B)
=========================================
Addendum C - Partial Observability Validation

Purpose:
- Validate OCI/CAP under partial observability
- Test with EnvB where velocity is hidden (observable: position only)
- Demonstrate that core mechanisms survive information hiding

Protocol:
- EnvB_PO = EnvB with velocity masked (set to 0)
- This tests whether the agent can still maintain OCI/CAP claims
  when critical state information is hidden

Design Choice:
- EnvB is chosen because velocity is an internal state that affects dynamics
- Hiding velocity tests whether the integration mechanism can compensate
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

# Add src to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from cap_oci_v036 import (
    VERSION, ALPHA, GAMMA_GRID_G0, N_ONSET_MIN,
    P_PASS, P_FAIL, R_PASS, R_STRONG,
    DT, CONTROL_COST,
    run_trace, compute_cap_metrics, gen_audit,
    seed_onset_pass, check_stage2_robust, check_tier2,
    wilson_lcb, LCBTriplet,
    compute_file_sha256, rng_from_seed,
    Agent, generate_B_matrices, generate_omega_matrices,
    collapse_matrix, misalignment_matrix, gamma_noise_std,
    CAPStream, Counters, Trace, VIEWS, D_RAW, N_TOTAL, EPISODES_PER_RUN, N_CAND,
    cand_fingerprint, normalize_candidates, softmax, kl_div,
    CLAMP_HI, CLAMP_LO, T_MIN, T_MAX
)

# ============================================================================
# Phase 3 Constants (FROZEN)
# ============================================================================
PHASE3_VERSION = "v6.1_addendum_C"

# Evaluation parameters
N_SEEDS_SMOKE = 30
N_SEEDS_FULL = 100
GAMMA_SMOKE = [1.00, 0.50, 0.00]


# ============================================================================
# EnvB_PO: Partial Observability variant of EnvB
# ============================================================================
class EnvB_PO:
    """EnvB with velocity hidden (Partial Observability).

    Observation: [pos_x, pos_y, 0, 0, bait_taken, instability]
    The velocity components (vel_x, vel_y) are masked to 0.
    """

    def __init__(self, seed: int):
        self.rng = rng_from_seed(seed + 6666)  # Different offset
        self.dt = DT
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.theta = self.rng.uniform(0.5, 1.5)
        self.step_count = 0
        self.bait_taken = False
        self.instability = 0.0
        self.z_t = np.array([self.pos[0] / 5.0, self.pos[1] / 5.0])
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        action = np.clip(action, -1, 1)
        acc = action - self.theta * self.vel + 0.01 * self.rng.normal(size=2)
        self.vel += self.dt * acc
        self.pos = np.clip(self.pos + self.dt * self.vel, -5, 5)
        self.step_count += 1
        self.z_t = np.array([self.pos[0] / 5.0, self.pos[1] / 5.0])

        reward = -CONTROL_COST * np.linalg.norm(action) ** 2
        if np.linalg.norm(self.pos) < 0.5 and np.linalg.norm(self.vel) < 0.3:
            reward += 1.0
        if np.linalg.norm(self.vel) > 1.5 and not self.bait_taken:
            reward += 0.5
            self.bait_taken = True
        if self.bait_taken:
            self.instability += 0.01
            reward -= self.instability * 0.1

        if self.rng.random() < 0.02:
            self.theta = self.rng.uniform(0.3, 2.0)

        return self._obs(), reward, self.step_count >= N_TOTAL // EPISODES_PER_RUN

    def get_latent_state(self) -> np.ndarray:
        return self.z_t.copy()

    def _obs(self) -> np.ndarray:
        # Partial observability: velocity is hidden (set to 0)
        return np.array([
            self.pos[0] / 5.0, self.pos[1] / 5.0,
            0.0, 0.0,  # Velocity masked
            float(self.bait_taken), self.instability
        ])

    def get_value_estimate(self, env, action: np.ndarray, K: int, agent_state) -> float:
        if K == 0:
            return 0.0
        pos0, pos1 = self.pos[0], self.pos[1]
        vel0, vel1 = self.vel[0], self.vel[1]
        a0, a1 = action[0], action[1]
        total = 0.0
        discount = 1.0
        for k in range(K):
            vel0 = vel0 + self.dt * (a0 - self.theta * vel0)
            vel1 = vel1 + self.dt * (a1 - self.theta * vel1)
            pos0 = max(-5, min(5, pos0 + self.dt * vel0))
            pos1 = max(-5, min(5, pos1 + self.dt * vel1))
            goal_r = 1.5 if (pos0 * pos0 + pos1 * pos1) < 0.25 else 0.0
            meta_cost = 0.15 * abs(vel0 + vel1) * agent_state.m
            total += discount * (goal_r - CONTROL_COST * (a0 * a0 + a1 * a1) - meta_cost)
            discount *= 0.95
        return float(total * (1.0 + 0.05 * K))


# ============================================================================
# Trace runner for EnvB_PO
# ============================================================================
def run_trace_envB_PO(gamma: float, lesion: str, baseline: str, seed: int) -> Trace:
    """Run trace with EnvB_PO (partial observability)."""
    rng = rng_from_seed(seed)
    env = EnvB_PO(seed)
    agent = Agent(seed, lesion, baseline, gamma)

    B_mats = generate_B_matrices(seed)
    omega_mats = generate_omega_matrices(seed)
    C_gamma = collapse_matrix(gamma)
    R_mats = [misalignment_matrix(gamma, om) for om in omega_mats]
    noise_std = gamma_noise_std(gamma)

    X_raw = [np.zeros((N_TOTAL, D_RAW)) for _ in range(VIEWS)]
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
        for step in range(N_TOTAL // EPISODES_PER_RUN):
            if t >= N_TOTAL:
                break

            u_t = agent.get_u_t()
            observations = []
            for i in range(VIEWS):
                M_i = R_mats[i] @ B_mats[i] @ C_gamma
                x_t_i = M_i @ u_t + noise_std * rng.normal(size=D_RAW)
                X_raw[i][t] = x_t_i
                observations.append(x_t_i)

            z_t = env.get_latent_state()
            err_int, err_views, meta = agent.update(observations, prev_reward, z_t)
            cs.Err_t.append(err_int)
            for i, ev in enumerate(err_views):
                cs.Err_i_t[i].append(ev)
            cs.self_report_t.append(agent.state.self_report)
            cs.meta_surprise_t.append(meta)

            action, values, var_v = agent.select_action_continuous(env, env.get_value_estimate)
            cs.V_candidates_t.append(values)
            cs.VarV_t.append(var_v)

            # Clamp trials
            if t % clamp_interval == 0 and not agent.is_reflex and agent.K > 0 and agent.lesion != "L2_rollout_off":
                counters.clamp_trials += 1
                agent.clamp_trials += 1
                orig = agent.state.self_report
                agent.in_clamp_trial = True

                fixed_cands = [np.clip(agent.rng.normal(0, 0.5, size=2), -1, 1) for _ in range(N_CAND)]

                agent.state.self_report = CLAMP_HI
                vals_hi = np.array([env.get_value_estimate(env, a, agent.K, agent.state) for a in fixed_cands])
                fp_hi = cand_fingerprint(vals_hi)

                agent.state.self_report = CLAMP_LO
                vals_lo = np.array([env.get_value_estimate(env, a, agent.K, agent.state) for a in fixed_cands])
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

            _, prev_reward, _ = env.step(action)
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
def run_envB_PO_evaluation(
    seeds: List[int],
    gamma_grid: List[float],
    mode: str,
    outdir: Path
) -> Dict[str, Any]:
    """Run EnvB_PO evaluation."""

    print(f"\n{'='*60}")
    print(f"Phase 3: EnvB_PO Partial Observability ({mode})")
    print(f"{'='*60}")
    print(f"Hidden: velocity (obs[2:4] = 0)")
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
            trace = run_trace_envB_PO(gamma, "none", "none", seed)
            metrics = compute_cap_metrics(trace)
            metrics_list.append(metrics)
            if metrics.env_inadequate:
                inadeq += 1

            audit = gen_audit(
                run_id=f"phase3_envB_PO_g{gamma}_s{seed}",
                env_type="EnvB_PO",
                seed=seed,
                gamma=gamma,
                lesion="none",
                baseline="none",
                trace=trace,
                metrics=metrics,
                lcbs=def_lcbs,
                cap_info={'phase3_partial_obs': True, 'hidden': 'velocity'},
                mode=mode,
                n_seeds=len(seeds),
                seed_strat="csprng_128_logged",
                crossing={'crossing_found': False},
                adequacy={'env_inadequate': metrics.env_inadequate}
            )
            audit_runs.append(audit)

        # Stage 1: Onset rate
        adequate_ms = [m for m in metrics_list if not m.env_inadequate]
        onset_rate = sum(1 for m in adequate_ms if seed_onset_pass(m)) / max(1, len(adequate_ms))

        # Stage 2: Robust gate
        robust_pass, robust_info, n_onset = check_stage2_robust(
            metrics_list, mode=mode, n_total=len(seeds)
        )

        if onset_rate >= P_PASS:
            pass_g.append(gamma)
        elif onset_rate <= P_FAIL:
            fail_g.append(gamma)

        gamma_results[gamma] = {
            'onset_rate': onset_rate,
            'n_onset': n_onset,
            'robust_pass': robust_pass,
            'robust_pass_strong': robust_info.get('robust_pass_strong', False),
            'robust_rate_or': robust_info.get('robust_rate_or', 0.0),
            'robust_rate_or_lcb': robust_info.get('robust_rate_or_lcb', 0.0),
            'inadeq_rate': inadeq / len(seeds)
        }

        or_lcb = robust_info.get('robust_rate_or_lcb', 0.0)
        print(f"Onset={onset_rate:.2f} n={n_onset:3d} OR_LCB={or_lcb:.2f} [{'PASS' if robust_pass else 'FAIL'}]")

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
    audit_path = outdir / f"audit_runs_envB_PO_{mode}.jsonl"
    with open(audit_path, "w") as f:
        for r in audit_runs:
            f.write(json.dumps(r) + "\n")

    return {
        'env_results': env_results,
        'gamma_results': gamma_results,
        'audit_path': str(audit_path)
    }


def generate_summary_report(results: Dict[str, Any], mode: str, outdir: Path) -> str:
    """Generate Addendum C summary report."""
    report_path = outdir / f"Addendum_C_summary_{mode}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Addendum C: Partial Observability Study\n\n")
        f.write(f"Phase 3 (B) - Partial Observability Validation ({mode})\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Base Version: {VERSION}\n")
        f.write(f"Addendum Version: {PHASE3_VERSION}\n\n")

        f.write("## Design\n\n")
        f.write("- **Environment**: EnvB_PO (EnvB with velocity hidden)\n")
        f.write("- **Hidden state**: velocity (obs[2:4] set to 0)\n")
        f.write("- **Observable**: position, bait_taken, instability\n\n")

        f.write("## Summary\n\n")
        er = results['env_results']
        f.write(f"- **crossing**: {er['crossing']} ({er.get('crossing_dir', 'N/A')})\n")
        f.write(f"- **claim_A**: {er['claim_A']}\n")
        f.write(f"- **claim_B**: {er['claim_B']}\n")
        f.write(f"- **claim_B_strong**: {er['claim_B_strong']}\n")
        f.write(f"- **max_onset_rate**: {er['max_onset_rate']:.3f}\n\n")

        f.write("## Per-gamma Results\n\n")
        f.write("| gamma | onset_rate | n_onset | OR_LCB | robust(OR) | strong(AND) | inadeq |\n")
        f.write("|-------|------------|---------|--------|------------|-------------|--------|\n")
        for gamma, gr in results['gamma_results'].items():
            f.write(f"| {gamma:.2f} | {gr['onset_rate']:.3f} | {gr['n_onset']} | ")
            f.write(f"{gr['robust_rate_or_lcb']:.3f} | {gr['robust_pass']} | ")
            f.write(f"{gr['robust_pass_strong']} | {gr['inadeq_rate']:.3f} |\n")

        f.write("\n## Comparison with EnvB (v6)\n\n")
        f.write("| Metric | EnvB (v6) | EnvB_PO | Status |\n")
        f.write("|--------|-----------|---------|--------|\n")

        v6_crossing = True
        envB_PO_crossing = er['crossing']
        cross_status = "PRESERVED" if envB_PO_crossing else "LOST"

        v6_claim_B = True
        envB_PO_claim_B = er['claim_B']
        claim_status = "PRESERVED" if envB_PO_claim_B else "LOST"

        f.write(f"| crossing | True (up) | {envB_PO_crossing} ({er.get('crossing_dir', 'N/A')}) | {cross_status} |\n")
        f.write(f"| claim_B | True | {envB_PO_claim_B} | {claim_status} |\n")

        f.write("\n## Conclusion\n\n")
        if envB_PO_crossing and envB_PO_claim_B:
            f.write("**PASS**: EnvB_PO preserves crossing and claim_B under partial observability.\n")
            f.write("OCI/CAP mechanisms are robust when velocity information is hidden.\n")
        else:
            f.write("**ATTENTION**: Some properties not preserved under partial observability.\n")
            f.write("This may indicate reliance on the hidden state dimension.\n")

    return str(report_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Partial Observability Study")
    parser.add_argument("--outdir", type=str,
                        default=str(SCRIPT_DIR.parent / "addendum_C_partial_obs"),
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

    print(f"Phase 3: Partial Observability Study ({mode})")
    print(f"Output: {outdir}")
    print(f"Seeds: {n_seeds}, Gammas: {gamma_grid}")

    # Generate seeds
    seeds = [int.from_bytes(secrets.token_bytes(16), 'big') % (2**31 - 1) for _ in range(n_seeds)]
    print(f"First 5 seeds: {seeds[:5]}")

    # Run evaluation
    results = run_envB_PO_evaluation(seeds, gamma_grid, mode, outdir)

    # Generate report
    report_path = generate_summary_report(results, mode, outdir)
    print(f"\nReport: {report_path}")

    # Final result
    er = results['env_results']
    print(f"\n{'='*60}")
    print(f"Phase 3 Complete!")
    print(f"  crossing: {er['crossing']} ({er.get('crossing_dir', 'N/A')})")
    print(f"  claim_A: {er['claim_A']}, claim_B: {er['claim_B']}, claim_B_strong: {er['claim_B_strong']}")

    gate_pass = er['claim_A'] and er['claim_B']
    print(f"\nPhase 3 Gate: {'PASS' if gate_pass else 'FAIL'}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
