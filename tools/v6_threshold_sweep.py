#!/usr/bin/env python3
"""
v6_threshold_sweep.py
---------------------
Threshold sensitivity sweep for CAP/OCI StageAB v6-style evaluation.

This script reads audit_runs.jsonl produced by cap_oci_v036_stageAB_v6.py
(or compatible schema) and recomputes:

- Stage A onset_rate (among adequate seeds) per env_type × gamma
- Stage B robust OR/AND rates and Wilson one-sided lower bounds (LCB)
- Crossing existence + direction (up/down/mixed) based on Stage A pass/fail sets
- CLAIM_B (weak): exists gamma with OR_LCB >= R_PASS (and n_onset >= N_ONSET_MIN if enforced)
- CLAIM_B+ (strong, optional label): exists gamma with AND_LCB >= R_STRONG

Outputs:
- sensitivity_summary.csv : per parameter setting, per environment (EnvA/EnvB) and overall pass flags
- sensitivity_gamma_detail.csv : per parameter setting, per env×gamma detailed stats

Usage example:
  python v6_threshold_sweep.py --audit /path/to/OUTDIR_full/audit_runs.jsonl --outdir /path/to/out_sweep

Notes:
- This does NOT re-run simulations; it only re-aggregates stored audit records.
- If your audit file includes multiple baselines/lesions, this script filters to baseline='none' and lesion='none'.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional

def norm_ppf(p: float) -> float:
    """Approximate inverse CDF of standard normal (Acklam approximation)."""
    # Source: Peter J. Acklam's approximation (public domain)
    # https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def wilson_lcb_one_sided(k: int, n: int, alpha: float) -> float:
    """One-sided Wilson score lower bound for binomial proportion."""
    if n <= 0:
        return float("nan")
    if k < 0 or k > n:
        raise ValueError("k must be in [0,n]")
    z = norm_ppf(1 - alpha)
    phat = k / n
    denom = 1 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + (z*z)/(4*n)) / n)) / denom
    return max(0.0, center - half)

def load_audit_records(audit_path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs

def filter_baseline_none(recs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in recs:
        if r.get("baseline") != "none":
            continue
        if r.get("lesion") != "none":
            continue
        out.append(r)
    return out

def group_counts(recs: Iterable[Dict[str, Any]],
                 f1_onset_min: float,
                 theta_lead: float,
                 theta_rec: float) -> Dict[Tuple[str, float], Dict[str, Any]]:
    """
    Aggregate per (env_type, gamma) with:
      adequate_n, onset_n,
      meta_pos_n (among onset), rec_pos_n (among onset), or_pos_n (among onset), and_pos_n (among onset)
    """
    g: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for r in recs:
        env = r.get("env_type")
        gamma = float(r.get("gamma"))
        key = (env, gamma)
        adeq = not bool(r.get("adequacy", {}).get("env_inadequate", False))
        cm = r.get("cap_metrics", {})
        f1 = float(cm.get("F1_IG", float("nan")))
        f4_lead = float(cm.get("F4_meta_lead", float("nan")))
        f4_rec = float(cm.get("F4_recovery_gain", float("nan")))

        if key not in g:
            g[key] = dict(adequate_n=0, onset_n=0,
                          meta_pos_n=0, rec_pos_n=0, or_pos_n=0, and_pos_n=0)

        if adeq:
            g[key]["adequate_n"] += 1
            onset = (f1 >= f1_onset_min)
            if onset:
                g[key]["onset_n"] += 1
                meta_pos = (f4_lead >= theta_lead)
                rec_pos  = (f4_rec  >= theta_rec)
                if meta_pos:
                    g[key]["meta_pos_n"] += 1
                if rec_pos:
                    g[key]["rec_pos_n"] += 1
                if meta_pos or rec_pos:
                    g[key]["or_pos_n"] += 1
                if meta_pos and rec_pos:
                    g[key]["and_pos_n"] += 1
        # if inadequate, we ignore it for onset_rate denominator (v6-style)
    return g

def crossing_from_onset(onset_rate_by_gamma: Dict[float, float], p_pass: float, p_fail: float) -> Tuple[bool, str, List[float], List[float]]:
    pass_g = sorted([g for g, v in onset_rate_by_gamma.items() if v >= p_pass])
    fail_g = sorted([g for g, v in onset_rate_by_gamma.items() if v <= p_fail])
    if not pass_g or not fail_g:
        return False, "none", pass_g, fail_g
    min_pass = min(pass_g)
    max_pass = max(pass_g)
    min_fail = min(fail_g)
    max_fail = max(fail_g)

    if max_fail < min_pass:
        return True, "up", pass_g, fail_g
    if max_pass < min_fail:
        return True, "down", pass_g, fail_g
    return True, "mixed", pass_g, fail_g

def env_claims(grouped: Dict[Tuple[str, float], Dict[str, Any]],
               env: str,
               gammas: List[float],
               alpha: float,
               p_pass: float,
               p_fail: float,
               r_pass: float,
               r_strong: float,
               n_onset_min: int) -> Dict[str, Any]:
    onset_rate_by_gamma: Dict[float, float] = {}
    rows: List[Dict[str, Any]] = []
    claim_b = False
    claim_b_strong = False

    for g in gammas:
        key = (env, g)
        d = grouped.get(key, None)
        if d is None:
            continue
        adequate_n = d["adequate_n"]
        onset_n = d["onset_n"]
        onset_rate = (onset_n / adequate_n) if adequate_n > 0 else float("nan")
        onset_rate_by_gamma[g] = onset_rate

        # robust rates among onset
        or_k = d["or_pos_n"]
        and_k = d["and_pos_n"]
        or_rate = (or_k / onset_n) if onset_n > 0 else float("nan")
        and_rate = (and_k / onset_n) if onset_n > 0 else float("nan")
        or_lcb = wilson_lcb_one_sided(or_k, onset_n, alpha) if onset_n > 0 else float("nan")
        and_lcb = wilson_lcb_one_sided(and_k, onset_n, alpha) if onset_n > 0 else float("nan")

        # gate (enforce minimum onset sample, v6-style for full_eval)
        eligible = (onset_n >= n_onset_min)
        robust_or_pass = bool(eligible and (or_lcb >= r_pass))
        robust_and_pass = bool(eligible and (and_lcb >= r_strong))

        claim_b = claim_b or robust_or_pass
        claim_b_strong = claim_b_strong or robust_and_pass

        rows.append(dict(
            env=env, gamma=g,
            adequate_n=adequate_n, onset_n=onset_n, onset_rate=onset_rate,
            or_k=or_k, or_rate=or_rate, or_lcb=or_lcb,
            and_k=and_k, and_rate=and_rate, and_lcb=and_lcb,
            meta_k=d["meta_pos_n"], rec_k=d["rec_pos_n"],
            eligible=eligible, robust_or_pass=robust_or_pass, robust_and_pass=robust_and_pass
        ))

    crossing_ok, crossing_dir, pass_g, fail_g = crossing_from_onset(onset_rate_by_gamma, p_pass, p_fail)
    claim_a = crossing_ok  # v6-style: CLAIM_A is essentially crossing on onset
    return dict(
        env=env,
        claim_a=claim_a,
        crossing=crossing_ok,
        crossing_dir=crossing_dir,
        pass_g=pass_g,
        fail_g=fail_g,
        claim_b=claim_b,
        claim_b_strong=claim_b_strong,
        gamma_rows=rows
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=str, required=True, help="Path to audit_runs.jsonl (full_eval output)")
    ap.add_argument("--outdir", type=str, default="sweep_out", help="Output directory for CSVs")
    ap.add_argument("--f1_onset_min", type=float, default=0.0, help="Stage A onset threshold for F1_IG")
    ap.add_argument("--alpha", type=float, default=0.10, help="One-sided alpha for Wilson LCB (default 0.10 => 90% one-sided)")
    ap.add_argument("--p_pass", type=float, default=0.30, help="P_PASS for crossing (onset_rate >= p_pass)")
    ap.add_argument("--p_fail", type=float, default=0.10, help="P_FAIL for crossing (onset_rate <= p_fail)")
    ap.add_argument("--n_onset_min", type=int, default=20, help="Minimum onset_n to consider robust gating")
    # parameter grids
    ap.add_argument("--r_pass_list", type=str, default="0.10,0.15,0.20,0.25", help="Comma list for R_PASS (weak robust)")
    ap.add_argument("--r_strong_list", type=str, default="0.05,0.10", help="Comma list for R_STRONG (strong robust)")
    ap.add_argument("--theta_lead_list", type=str, default="0.00,-0.05", help="Comma list for theta_lead")
    ap.add_argument("--theta_rec_list", type=str, default="0.00,-0.05", help="Comma list for theta_rec")
    ap.add_argument("--envs", type=str, default="EnvA_grid,EnvB_continuous", help="Comma list env_types to analyze")
    ap.add_argument("--gammas", type=str, default="1.00,0.75,0.50,0.25,0.00", help="Comma list gammas (descending recommended)")
    args = ap.parse_args()

    audit_path = Path(args.audit)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    recs = load_audit_records(audit_path)
    recs = filter_baseline_none(recs)

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]

    r_pass_list = [float(x) for x in args.r_pass_list.split(",") if x.strip()]
    r_strong_list = [float(x) for x in args.r_strong_list.split(",") if x.strip()]
    theta_lead_list = [float(x) for x in args.theta_lead_list.split(",") if x.strip()]
    theta_rec_list = [float(x) for x in args.theta_rec_list.split(",") if x.strip()]

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    sweep_id = 0
    for r_pass in r_pass_list:
        for r_strong in r_strong_list:
            for th_lead in theta_lead_list:
                for th_rec in theta_rec_list:
                    sweep_id += 1
                    grouped = group_counts(recs, args.f1_onset_min, th_lead, th_rec)

                    env_results = {}
                    overall_claim_ready = True
                    for env in envs:
                        res = env_claims(grouped, env, gammas, args.alpha, args.p_pass, args.p_fail,
                                         r_pass, r_strong, args.n_onset_min)
                        env_results[env] = res
                        # This sweep only recomputes CLAIM_A and CLAIM_B/B+;
                        # CLAIM_C is assumed unchanged by these thresholds.
                        overall_claim_ready = overall_claim_ready and bool(res["claim_a"] and res["claim_b"])

                        summary_rows.append(dict(
                            sweep_id=sweep_id,
                            r_pass=r_pass, r_strong=r_strong,
                            theta_lead=th_lead, theta_rec=th_rec,
                            env=env,
                            claim_a=res["claim_a"],
                            claim_b=res["claim_b"],
                            claim_b_strong=res["claim_b_strong"],
                            crossing=res["crossing"],
                            crossing_dir=res["crossing_dir"],
                            pass_g=";".join(map(str, res["pass_g"])),
                            fail_g=";".join(map(str, res["fail_g"])),
                        ))

                        for row in res["gamma_rows"]:
                            detail_rows.append(dict(
                                sweep_id=sweep_id,
                                r_pass=r_pass, r_strong=r_strong,
                                theta_lead=th_lead, theta_rec=th_rec,
                                **row
                            ))

                    # overall row (optional)
                    summary_rows.append(dict(
                        sweep_id=sweep_id,
                        r_pass=r_pass, r_strong=r_strong,
                        theta_lead=th_lead, theta_rec=th_rec,
                        env="OVERALL",
                        claim_a=all(env_results[e]["claim_a"] for e in envs),
                        claim_b=all(env_results[e]["claim_b"] for e in envs),
                        claim_b_strong=all(env_results[e]["claim_b_strong"] for e in envs),
                        crossing="",
                        crossing_dir="",
                        pass_g="",
                        fail_g="",
                        claim_ready_recomputed=overall_claim_ready,
                        note="CLAIM_C not recomputed here"
                    ))

    # write csv
    summary_path = outdir / "sensitivity_summary.csv"
    detail_path = outdir / "sensitivity_gamma_detail.csv"

    def write_csv(path: Path, rows: List[Dict[str, Any]]):
        if not rows:
            return
        cols = sorted({k for r in rows for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(summary_path, summary_rows)
    write_csv(detail_path, detail_rows)

    print(f"[OK] Wrote:\n  {summary_path}\n  {detail_path}")

if __name__ == "__main__":
    main()
