#!/usr/bin/env python3
"""
Phase 1: Seed Replication Study (S(2))
======================================
Addendum A - Independent Seed Set Validation

Purpose:
- Validate seed distribution robustness
- Test worst-seed tolerance
- Demonstrate reproducibility across independent seed sets

Protocol:
- Generate Set C and Set D (100 seeds each, CSPRNG)
- Run 2 envs × 5γ × 100 seeds with lesion=none, baseline=none
- Use identical v6 evaluation criteria (no parameter changes)

Output:
- seeds_set_c.json, seeds_set_d.json (with SHA256)
- audit_runs_setC.jsonl, audit_runs_setD.jsonl
- Addendum_A_summary.md
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

# Add src to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from cap_oci_v036 import (
    VERSION, ALPHA, GAMMA_GRID_G0, N_ONSET_MIN,
    P_PASS, P_FAIL, R_PASS, R_STRONG,
    run_trace, compute_cap_metrics, gen_audit,
    seed_onset_pass, check_stage2_robust, check_tier2, check_selective,
    wilson_lcb, LCBTriplet,
    compute_file_sha256
)

# Phase 1 Constants (frozen - do not modify after declaration)
PHASE1_VERSION = "v6.1_addendum_A"
N_SEEDS_FULL = 100
N_SEEDS_SMOKE = 10
GAMMA_SMOKE = [1.00, 0.50, 0.00]  # 3 gamma for smoke test
ENVIRONMENTS = ["EnvA_grid", "EnvB_continuous"]


def generate_csprng_seeds_with_hash(n: int, set_name: str) -> Tuple[List[int], str]:
    """Generate N CSPRNG seeds and compute SHA256 hash."""
    seeds = [int.from_bytes(secrets.token_bytes(16), 'big') % (2**31 - 1) for _ in range(n)]
    seed_bytes = json.dumps(seeds).encode('utf-8')
    seed_hash = hashlib.sha256(seed_bytes).hexdigest()
    return seeds, seed_hash


def save_seeds(seeds: List[int], seed_hash: str, set_name: str, outdir: Path) -> str:
    """Save seeds to JSON file with metadata."""
    seed_file = {
        "version": PHASE1_VERSION,
        "set_name": set_name,
        "n_seeds": len(seeds),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": seed_hash,
        "seeds": seeds
    }
    filepath = outdir / f"seeds_set_{set_name.lower()}.json"
    with open(filepath, "w") as f:
        json.dump(seed_file, f, indent=2)
    return str(filepath)


def run_seed_set_evaluation(
    seeds: List[int],
    set_name: str,
    outdir: Path,
    gamma_grid: List[float] = None,
    mode: str = "full_eval"
) -> Dict[str, Any]:
    """Run evaluation for a seed set using v6 protocol.

    Args:
        gamma_grid: List of gamma values (default: GAMMA_GRID_G0)
        mode: "smoke" or "full_eval"
    """
    if gamma_grid is None:
        gamma_grid = GAMMA_GRID_G0

    print(f"\n{'='*60}")
    print(f"Phase 1: Seed Set {set_name} Evaluation ({mode})")
    print(f"{'='*60}")
    print(f"Seeds: {len(seeds)}, Gammas: {gamma_grid}")
    print(f"First 5 seeds: {seeds[:5]}")

    audit_runs = []
    env_results = {}
    det_results = {}

    # Default LCBs for audit records
    def_lcbs = {k: LCBTriplet(0, 0, 0) for k in
                ['F1_IG', 'F2_RT', 'F3_delta_act', 'F3_delta_perf', 'F4_meta_lead', 'F4_recovery_gain']}

    for env_type in ENVIRONMENTS:
        print(f"\n{env_type}:")
        env_results[env_type] = {
            'crossing': False,
            'crossing_dir': None,
            'max_onset_rate': 0.0,
            'min_inadeq': 1.0
        }
        det_results[env_type] = {'gamma_results': {}}
        pass_g = []
        fail_g = []

        for gamma in gamma_grid:
            print(f"  γ={gamma:.2f}:", end=" ", flush=True)

            # Run all seeds for this gamma (lesion=none, baseline=none only)
            metrics_none = []
            traces = []
            inadeq = 0

            for i, seed in enumerate(seeds):
                trace = run_trace(env_type, gamma, "none", "none", seed)
                metrics = compute_cap_metrics(trace)
                traces.append(trace)
                metrics_none.append(metrics)
                if metrics.env_inadequate:
                    inadeq += 1

                # Generate audit record
                audit = gen_audit(
                    run_id=f"phase1_{set_name}_{env_type}_g{gamma}_s{seed}",
                    env_type=env_type,
                    seed=seed,
                    gamma=gamma,
                    lesion="none",
                    baseline="none",
                    trace=trace,
                    metrics=metrics,
                    lcbs=def_lcbs,
                    cap_info={'phase1_set': set_name},
                    mode="full_eval",
                    n_seeds=len(seeds),
                    seed_strat="csprng_128_logged",
                    crossing={'crossing_found': False},
                    adequacy={'env_inadequate': metrics.env_inadequate}
                )
                audit_runs.append(audit)

            # Stage 1: Onset rate (adequate seeds only)
            adequate_ms = [m for m in metrics_none if not m.env_inadequate]
            onset_rate = sum(1 for m in adequate_ms if seed_onset_pass(m)) / max(1, len(adequate_ms))

            # Stage 2: Robust gate
            robust_pass, robust_info, n_onset = check_stage2_robust(
                metrics_none, mode=mode, n_total=len(seeds)
            )

            # Crossing buckets
            if onset_rate >= P_PASS:
                pass_g.append(gamma)
            elif onset_rate <= P_FAIL:
                fail_g.append(gamma)

            # Store per-γ results
            det_results[env_type]['gamma_results'][gamma] = {
                'onset_rate': onset_rate,
                'n_onset': n_onset,
                'robust_pass': robust_pass,
                'robust_pass_strong': robust_info.get('robust_pass_strong', False),
                'robust_rate_or': robust_info.get('robust_rate_or', 0.0),
                'robust_rate_or_lcb': robust_info.get('robust_rate_or_lcb', 0.0),
                'n_meta_pos': robust_info.get('n_meta_pos', 0),
                'n_rec_pos': robust_info.get('n_rec_pos', 0),
                'n_and_pos': robust_info.get('n_and_pos', 0),
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
                env_results[env_type]['crossing'] = True
                env_results[env_type]['crossing_dir'] = "up"
                print(f"  Crossing(up): FAIL<={max_f}, PASS>={min_p}")
            elif max_p < min_f:
                env_results[env_type]['crossing'] = True
                env_results[env_type]['crossing_dir'] = "down"
                print(f"  Crossing(down): PASS<={max_p}, FAIL>={min_f}")

        # Compute summary stats
        gamma_res = det_results[env_type]['gamma_results']
        env_results[env_type]['max_onset_rate'] = max(r['onset_rate'] for r in gamma_res.values())
        env_results[env_type]['min_inadeq'] = min(r['inadeq_rate'] for r in gamma_res.values())

        # CLAIM evaluation (v6 criteria)
        # CLAIM_A: crossing exists
        env_results[env_type]['claim_A'] = env_results[env_type]['crossing']

        # CLAIM_B: at least one γ passes robust gate (OR condition)
        env_results[env_type]['claim_B'] = any(r['robust_pass'] for r in gamma_res.values())

        # CLAIM_B+ (strong): at least one γ passes AND condition
        env_results[env_type]['claim_B_strong'] = any(r['robust_pass_strong'] for r in gamma_res.values())

    # Write audit file
    audit_path = outdir / f"audit_runs_set{set_name}.jsonl"
    with open(audit_path, "w") as f:
        for r in audit_runs:
            f.write(json.dumps(r) + "\n")

    # Overall claim_ready
    claim_ready = all(
        env_results[env]['claim_A'] and env_results[env]['claim_B']
        for env in ENVIRONMENTS
    )
    claim_b_strong = all(
        env_results[env]['claim_B_strong']
        for env in ENVIRONMENTS
    )

    return {
        'set_name': set_name,
        'n_seeds': len(seeds),
        'env_results': env_results,
        'det_results': det_results,
        'claim_ready': claim_ready,
        'claim_b_strong': claim_b_strong,
        'audit_path': str(audit_path)
    }


def generate_summary_report(
    results_c: Dict[str, Any],
    results_d: Dict[str, Any],
    seeds_c_hash: str,
    seeds_d_hash: str,
    outdir: Path
) -> str:
    """Generate Addendum A summary markdown report."""

    report_path = outdir / "Addendum_A_summary.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Addendum A: Seed Replication Study\n\n")
        f.write(f"Phase 1 (S(2)) - Independent Seed Set Validation\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Base Version: {VERSION}\n")
        f.write(f"Addendum Version: {PHASE1_VERSION}\n\n")

        f.write("## Purpose\n\n")
        f.write("Validate that CAP/OCI v6 claims are robust to seed selection by running\n")
        f.write("independent seed sets (C and D) through the identical evaluation protocol.\n\n")

        f.write("## Seed Sets\n\n")
        f.write("| Set | N Seeds | SHA256 |\n")
        f.write("|-----|---------|--------|\n")
        f.write(f"| C | {results_c['n_seeds']} | `{seeds_c_hash[:16]}...` |\n")
        f.write(f"| D | {results_d['n_seeds']} | `{seeds_d_hash[:16]}...` |\n\n")

        f.write("## Summary Results\n\n")
        f.write("| Set | claim_ready | claim_b_strong | EnvA crossing | EnvB crossing |\n")
        f.write("|-----|-------------|----------------|---------------|---------------|\n")
        for name, res in [("C", results_c), ("D", results_d)]:
            f.write(f"| {name} | {res['claim_ready']} | {res['claim_b_strong']} | ")
            f.write(f"{res['env_results']['EnvA_grid']['crossing']} ({res['env_results']['EnvA_grid'].get('crossing_dir', 'N/A')}) | ")
            f.write(f"{res['env_results']['EnvB_continuous']['crossing']} ({res['env_results']['EnvB_continuous'].get('crossing_dir', 'N/A')}) |\n")

        f.write("\n## Per-Environment Results\n\n")

        for env in ENVIRONMENTS:
            f.write(f"### {env}\n\n")
            f.write("| Set | γ | onset_rate | n_onset | OR_LCB | robust(OR) | strong(AND) |\n")
            f.write("|-----|---|------------|---------|--------|------------|-------------|\n")

            for name, res in [("C", results_c), ("D", results_d)]:
                for gamma in GAMMA_GRID_G0:
                    gr = res['det_results'][env]['gamma_results'][gamma]
                    f.write(f"| {name} | {gamma:.2f} | {gr['onset_rate']:.3f} | ")
                    f.write(f"{gr['n_onset']} | {gr['robust_rate_or_lcb']:.3f} | ")
                    f.write(f"{gr['robust_pass']} | {gr['robust_pass_strong']} |\n")
            f.write("\n")

        f.write("## Comparison with v6 Release\n\n")
        f.write("| Metric | v6 (Set A/B) | Set C | Set D | Status |\n")
        f.write("|--------|--------------|-------|-------|--------|\n")

        v6_ready = True  # From v6 DONE.json
        c_ready = results_c['claim_ready']
        d_ready = results_d['claim_ready']
        all_match = v6_ready == c_ready == d_ready

        f.write(f"| claim_ready | True | {c_ready} | {d_ready} | {'CONSISTENT' if all_match else 'DIVERGENT'} |\n")

        v6_strong = True
        c_strong = results_c['claim_b_strong']
        d_strong = results_d['claim_b_strong']
        all_strong = v6_strong == c_strong == d_strong

        f.write(f"| claim_b_strong | True | {c_strong} | {d_strong} | {'CONSISTENT' if all_strong else 'DIVERGENT'} |\n")

        f.write("\n## Conclusion\n\n")

        if all_match:
            f.write("**PASS**: All seed sets (v6, C, D) achieve `claim_ready = True`.\n")
            f.write("This confirms seed distribution robustness.\n")
        else:
            f.write("**ATTENTION**: Seed sets show divergent results.\n")
            f.write("Further analysis of failing seeds recommended.\n")

        f.write("\n## Artifacts\n\n")
        f.write(f"- `seeds_set_c.json`: Set C seeds (SHA256: `{seeds_c_hash}`)\n")
        f.write(f"- `seeds_set_d.json`: Set D seeds (SHA256: `{seeds_d_hash}`)\n")
        f.write(f"- `audit_runs_setC.jsonl`: Full audit for Set C\n")
        f.write(f"- `audit_runs_setD.jsonl`: Full audit for Set D\n")
        f.write(f"- `Addendum_A_summary.md`: This report\n")

    return str(report_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Seed Replication Study")
    parser.add_argument("--outdir", type=str,
                        default=str(SCRIPT_DIR.parent / "addendum_A_seed_replication"),
                        help="Output directory")
    parser.add_argument("--set", type=str, choices=["C", "D", "both"], default="both",
                        help="Which seed set to run")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test (10 seeds x 3 gammas) instead of full eval")
    args = parser.parse_args()

    # Determine mode and parameters
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

    print(f"Phase 1: Seed Replication Study ({mode})")
    print(f"Output directory: {outdir}")
    print(f"Running set(s): {args.set}")
    print(f"Seeds per set: {n_seeds}, Gammas: {gamma_grid}")

    results = {}
    seeds_hashes = {}

    # Generate and run Set C
    if args.set in ["C", "both"]:
        print("\n" + "="*60)
        print("Generating Seed Set C...")
        seeds_c, hash_c = generate_csprng_seeds_with_hash(n_seeds, "C")
        save_seeds(seeds_c, hash_c, "C", outdir)
        seeds_hashes["C"] = hash_c
        print(f"Set C: {len(seeds_c)} seeds, SHA256: {hash_c[:32]}...")

        results["C"] = run_seed_set_evaluation(seeds_c, "C", outdir, gamma_grid, mode)

    # Generate and run Set D
    if args.set in ["D", "both"]:
        print("\n" + "="*60)
        print("Generating Seed Set D...")
        seeds_d, hash_d = generate_csprng_seeds_with_hash(n_seeds, "D")
        save_seeds(seeds_d, hash_d, "D", outdir)
        seeds_hashes["D"] = hash_d
        print(f"Set D: {len(seeds_d)} seeds, SHA256: {hash_d[:32]}...")

        results["D"] = run_seed_set_evaluation(seeds_d, "D", outdir, gamma_grid, mode)

    # Generate summary if both sets completed
    if args.set == "both":
        print("\n" + "="*60)
        print("Phase 1 Complete!")
        print(f"  Set C: claim_ready={results['C']['claim_ready']}, claim_b_strong={results['C']['claim_b_strong']}")
        print(f"  Set D: claim_ready={results['D']['claim_ready']}, claim_b_strong={results['D']['claim_b_strong']}")

        # Final gate check
        gate_pass = results['C']['claim_ready'] and results['D']['claim_ready']
        print(f"\nPhase 1 Gate: {'PASS' if gate_pass else 'FAIL'}")

        # Only generate full report for full_eval mode
        if mode == "full_eval":
            print("\n" + "="*60)
            print("Generating Summary Report...")
            report_path = generate_summary_report(
                results["C"], results["D"],
                seeds_hashes["C"], seeds_hashes["D"],
                outdir
            )
            print(f"Summary report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
