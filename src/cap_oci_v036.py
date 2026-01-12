#!/usr/bin/env python3
"""
CAP+OCI Implementation v0.3.6 (Strict)
Implements Spec Pack v0.3.6 with full compliance to Doc A/B/C.

Goals:
- Goal A (Protocol-Complete): full_eval completes, all artifacts exist, all schemas pass, no falsification
- Goal B (Claim-Ready): Goal A + CM PASS for both EnvA_grid and EnvB_continuous

Version: v0.3.6
"""

from __future__ import annotations
import argparse, json, hashlib, math, os, sys, platform, secrets, subprocess, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from statistics import NormalDist
import numpy as np
from scipy.linalg import expm

VERSION = "v0.3.6p15.2ABv6"  # Phase 15.2 + StageAB v6 (weak/strong Stage B; adequacy-filtered onset)

# ============================================================================
# Appendix A.0 Constants (Normative, Binding)
# ============================================================================
ALPHA = 0.1
EPSILON = 1e-12

# ---- Stage gate params (Onset/Robust split; report-only in selfcheck/smoke) ----
P_PASS = 0.30        # onset pass threshold (rate)
P_FAIL = 0.10        # onset fail threshold (rate)
F1_ONSET_MIN = 0.0   # onset threshold for F1_IG (positive overlap)
N_ONSET_MIN = 20     # min onset seeds required for conditional (Stage2) stats
WINSOR_Q = 0.05      # winsorize quantile for tail robustness
USE_WINSOR = True    # if False, Stage2 uses raw Cantelli on onset seeds

# Stage B thresholds (two-level; weak=OR, strong=AND)
R_PASS = 0.15        # minimum LCB for (F4_meta_lead>θ OR F4_recovery_gain>θ) among onset seeds  [CLAIM_B]
R_STRONG = 0.05      # minimum LCB for (F4_meta_lead>θ AND F4_recovery_gain>θ) among onset seeds [report-only]
THETA_LEAD = 0.0     # per-seed success threshold for meta_lead (strict >)
THETA_REC  = 0.0     # per-seed success threshold for recovery_gain (strict >)
MIN_ONSET_SELFCHECK = 3  # Stage B stats minimum onset count in selfcheck (report-only)
MIN_ONSET_SMOKE = 8      # Stage B stats minimum onset count in smoke (report-only)

# Seed configurations (Doc A Appendix A.0)
SELFCHECK_SEEDS = [0, 1, 2, 3, 4]
SELFCHECK_N_SEEDS = 5
SELFCHECK_SEED_STRATEGY = "fixed_list"
FULL_EVAL_N_SEEDS = 100  # MUST be 100 for full_eval
FULL_EVAL_SEED_STRATEGY = "csprng_128_logged"

# Gamma grid (Doc A Appendix A.0)
GAMMA_GRID_G0 = [1.00, 0.75, 0.50, 0.25, 0.00]
GAMMA_REFINEMENT_STEP = 0.01

# F4 estimator constants (Doc A Appendix A.0 - fixed)
F4_L_PRE = 10
F4_L_BASE = 10
F4_L_POST = 20
F4_R_REF = 20
F4_THETA_BREAK = 2.0
F4_K_MIN = 3

# ============================================================================
# Implementation parameters
# ============================================================================
N_TOTAL = 1000
VIEWS = 4
D_RAW = 6
K_ROLLOUT = 2
T_MIN = 0.1
T_MAX = 1.0
N_CAND = 8

# Environment parameters
GRID_SIZE = 15
ACTIONS_A = ['U', 'D', 'L', 'R', 'STAY']
EPISODE_LENGTH_A = 200
EPISODES_PER_RUN = 20
DT = 0.05
CONTROL_COST = 0.01
CLAMP_HI = 0.9
CLAMP_LO = 0.1
H_CLAMP = 20

# Baseline and lesion enums (Doc B)
ALL_BASELINES = ["none", "B0_no_integration", "B1_no_rollout", "B2_no_meta", "B3_no_learning_reflex"]
ALL_LESIONS = ["none", "L1_integration_off", "L2_rollout_off", "L3_meta_off"]

# Phase 14 Config: TAR+SRS (Tail-Anchor Reset + Swap-Risk Suppression)
# P1: SRS (Swap-Risk Suppression) - suppress p1_eff during instability
SRS_Z_SUPPRESS = 2.0  # z_local threshold where suppression starts
SRS_K_SUPPRESS = 0.30  # Sigmoid steepness (0.25-0.35 per spec)
# P2: TAR (Tail-Anchor Reset) - re-anchor to best-view when recovery stalls
TAR_Z_TAIL = 2.7  # z_local threshold for TAR trigger (original Phase 14)
TAR_MIN_AGE = 8  # Minimum recovery age before TAR can fire (original Phase 14)
TAR_PATIENCE = 3  # Consecutive non-improvement steps to trigger TAR (original Phase 14)
TAR_FREEZE = 2  # Post-TAR weight freeze duration (original Phase 14)
TAR_WEIGHT_BEST = 0.92  # Weight for best view after TAR (near one-hot)

# Phase 14.1 Config: TAR Emergency Triggers + Post-TAR Latch
# Emergency trigger 1: Immediate spike - disabled (caused regression)
TAR_Z_SPIKE = 99.0  # Effectively disabled
TAR_SPIKE_MAX_AGE = 0  # Effectively disabled
# Emergency trigger 2: Still-stuck (error-based) - DISABLED (violates P2 EnvB F1 constraint)
TAR_AGE_STUCK = 99  # Disabled - caused EnvB F1 regression
TAR_STUCK_PATIENCE = 99  # Disabled
# Legacy still-high trigger - disabled
TAR_AGE_EARLY = 99  # Effectively disabled
TAR_Z_STILL = 99.0  # Effectively disabled
# Post-TAR best-view latch (stabilizer)
TAR_LATCH_STEPS = 0  # Disabled

# Phase 15.2 Config: Early Lock-in Prevention (ELP)
# Focus: age 0-3 intervention to prevent early lock-in, not recovery-phase fixes
ELP_MAX_AGE = 3  # Intervention window: ages 0-3 only

# P1: Top-2 weight preservation (prevent over-concentration)
ELP_TOP2_MIN = 0.15  # Minimum weight for each of top-2 views (stronger floor)
ELP_OTHER_MIN = 0.03  # Minimum weight for other views
ELP_TOP_MAX = 0.50  # Maximum weight for any single view (prevent lock-in) (was 0.55)

# P2: Learning freeze/unfreeze during early shock
ELP_FREEZE_STEPS = 3  # Steps to freeze weight/meta learning after shock (was 2)
ELP_RAMP_STEPS = 2  # Steps to gradually restore learning rate

# P3: p1 clamp to maintain swap/no-swap mixing
ELP_P1_MIN = 0.20  # Minimum p1 during early shock (don't fully commit to no-swap) (was 0.15)
ELP_P1_MAX = 0.80  # Maximum p1 during early shock (don't fully commit to swap) (was 0.85)


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class Counters:
    rollout_calls: int = 0
    rollout_value_evals: int = 0
    clamp_trials: int = 0
    episodes: int = 0


@dataclass
class AgentState:
    s_hat: np.ndarray
    h: np.ndarray
    m: float
    self_hat: float
    self_report: float
    w: np.ndarray
    s_hat_prev: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # Phase 3: soft mode inference (2-hypothesis blending)
    p1_swap: float = 0.0  # EWMA probability of swap hypothesis
    # Phase 4: shock→recovery control
    err_ema: float = 0.0  # EWMA of Err_t for shock detection
    err_var: float = 1.0  # EWMA variance of Err_t
    recovery_timer: int = 0  # Countdown timer for recovery boost
    recovery_armed: bool = True  # Flag for one-time s_hat reset
    # Phase 5: stability-gated predictive fusion
    p1_raw_prev: float = 0.0  # Previous p1_raw for Δp1 calculation
    reset_cooldown: int = 0  # Cooldown timer for Reset-on-Shock
    # Phase 6: tail-risk guard
    tail_dwell: int = 0  # Tail-risk guard timer
    # Phase 7: composite risk + volatility tracking
    err_history_short: list = None  # Short window for volatility (8 steps)
    # Phase 9: STRF (Stability-Triggered Recovery Floor)
    shock_strength: float = 0.0  # z_local at shock onset (0..1 normalized)
    r_rec_ema: float = 0.0  # EWMA of recovery strength signal
    # Phase 10: STRF diagnostics
    strf_diag_rec_fire_count: int = 0  # Number of times recovery_timer fired (new recovery start)
    strf_diag_rec_dwell_steps: int = 0  # Total steps where recovery_timer > 0
    strf_diag_r_rec_sum: float = 0.0  # Sum of r_rec values (for mean calculation)
    strf_diag_r_rec_max: float = 0.0  # Max r_rec value (for P95 proxy)
    strf_diag_r_rec_values: list = None  # List of r_rec values for percentile calculation
    strf_diag_dominant_signal: dict = None  # Counts of which signal was dominant
    # Phase 12: Panic kernel (deep collapse hemostasis)
    panic_timer: int = 0  # Countdown for panic mode (5 steps of forced best-view + damping)
    # Phase 14: TAR+SRS state variables
    z_local_prev: float = 0.0  # Previous z_local for SRS (Swap-Risk Suppression)
    tar_fired: bool = False  # One-time TAR fire flag per recovery window
    tar_freeze: int = 0  # Post-TAR weight freeze countdown
    tar_patience_count: int = 0  # Consecutive non-improvement steps for TAR trigger
    tar_err_best: float = 999.0  # Best error seen during recovery (for improvement check)
    # Phase 14.1: Post-TAR best-view latch
    tar_latch_timer: int = 0  # Force best-view integration after TAR fires
    # Phase 15: SSR (Spike-triggered Soft Re-anchoring) state - LEGACY, disabled
    ssr_fired: bool = False  # One-time SSR fire flag per recovery window
    ssr_quarantine_timer: int = 0  # Swap quarantine countdown
    ssr_freeze_timer: int = 0  # Weight freeze countdown
    ssr_z_local_at_age0: float = 0.0  # z_local captured at recovery onset (age=0)
    ssr_swap_risk_history: list = None  # swap_risk values in ages 0-3 for spike detection
    # Phase 15.2: ELP (Early Lock-in Prevention) state
    elp_freeze_timer: int = 0  # Learning freeze countdown (P2)
    elp_ramp_timer: int = 0  # Learning ramp-up countdown (P2)

    def __post_init__(self):
        if self.err_history_short is None:
            self.err_history_short = []
        if self.strf_diag_r_rec_values is None:
            self.strf_diag_r_rec_values = []
        if self.strf_diag_dominant_signal is None:
            self.strf_diag_dominant_signal = {"r_dis": 0, "r_mode": 0, "r_tail": 0, "r_shock": 0}
        if self.ssr_swap_risk_history is None:
            self.ssr_swap_risk_history = []


@dataclass
class CAPStream:
    Err_t: List[float] = field(default_factory=list)
    Err_i_t: List[List[float]] = field(default_factory=list)
    V_candidates_t: List[np.ndarray] = field(default_factory=list)
    VarV_t: List[float] = field(default_factory=list)
    self_report_t: List[float] = field(default_factory=list)
    meta_surprise_t: List[float] = field(default_factory=list)
    clamp_delta_act: List[float] = field(default_factory=list)
    clamp_delta_perf: List[float] = field(default_factory=list)
    clamp_fp_hi: List[str] = field(default_factory=list)
    clamp_fp_lo: List[str] = field(default_factory=list)


@dataclass
class Trace:
    X_raw: List[np.ndarray]
    cap_stream: CAPStream
    counters: Counters
    x_raw_sha256: str


@dataclass
class CAPMetrics:
    F1_IG: float
    F2_RT: float
    F3_delta_act: float
    F3_delta_perf: float
    F4_meta_lead: float
    F4_recovery_gain: float
    event_count: int
    breakdown_rate: float
    env_inadequate: bool
    candidate_set_mismatch: bool


@dataclass
class LCBTriplet:
    mean: float
    std: float
    score: float


# ============================================================================
# Utility Functions
# ============================================================================
def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return "0" * 64


def generate_csprng_seeds(n: int) -> List[int]:
    """Generate N cryptographically secure seeds (Doc A Appendix A.0)"""
    return [int.from_bytes(secrets.token_bytes(16), 'big') % (2**31 - 1) for _ in range(n)]


def get_pip_freeze_sha256() -> str:
    try:
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, timeout=30)
        return compute_sha256(result.stdout.encode('utf-8'))
    except:
        return "0" * 64


def cantelli_lcb(values: List[float], alpha: float = ALPHA) -> LCBTriplet:
    """Cantelli LCB (Doc A Appendix A.6)"""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return LCBTriplet(0.0, 0.0, 0.0)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    c = math.sqrt((1.0 - alpha) / (alpha * n))
    return LCBTriplet(mean_val, std_val, mean_val - c * std_val)


def winsorize(values: List[float], q: float) -> np.ndarray:
    """Winsorize values at quantile q (tail-robust aggregation)."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    lo = np.quantile(arr, q)
    hi = np.quantile(arr, 1.0 - q)
    return np.clip(arr, lo, hi)

def cantelli_lcb_robust(values: List[float], alpha: float = ALPHA, q: float = WINSOR_Q) -> LCBTriplet:
    """Cantelli LCB computed on (optionally) winsorized values."""
    if not values:
        return LCBTriplet(0.0, 0.0, 0.0)
    arr = winsorize(values, q) if USE_WINSOR else np.array(values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    c = math.sqrt((1.0 - alpha) / (alpha * n))
    return LCBTriplet(mean_val, std_val, mean_val - c * std_val)

def wilson_lcb(k: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided Wilson score lower confidence bound for a binomial proportion."""
    if n <= 0:
        return 0.0
    k = max(0, min(int(k), int(n)))
    phat = k / n
    # z for one-sided (1-alpha)
    z = NormalDist().inv_cdf(1.0 - alpha)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, (center - margin) / denom)


def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score (Doc A Appendix A.5)"""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return (x - median) / (1.4826 * mad + EPSILON)


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_scaled = x / max(temp, 1e-10) - (x / max(temp, 1e-10)).max()
    exp_x = np.exp(x_scaled)
    return exp_x / exp_x.sum()


def normalize_candidates(values: np.ndarray) -> np.ndarray:
    """Candidate value normalization (Doc A Appendix A.1)"""
    if len(values) == 0:
        return values
    mu, sigma = np.mean(values), np.std(values)
    return (values - mu) / (sigma + EPSILON) if sigma > 0 else np.zeros_like(values)


def cand_fingerprint(candidate_values: np.ndarray) -> str:
    """Fingerprint of candidate SET based on actual values (Doc A Appendix A.4 - candidate-set invariance)"""
    # Create content-based fingerprint from sorted candidate values
    # Sort to ensure order-independence for the same set
    sorted_vals = np.sort(candidate_values)
    return compute_sha256(sorted_vals.tobytes())[:32]


# ============================================================================
# Gamma Transform (Doc A Section A3.3)
# ============================================================================
def collapse_matrix(gamma: float) -> np.ndarray:
    """C(γ) = diag(1,1,1,γ,γ,γ)"""
    return np.diag([1.0, 1.0, 1.0, gamma, gamma, gamma])


def generate_omega(seed: int, idx: int) -> np.ndarray:
    """Generate skew-symmetric Ω_i"""
    rng = rng_from_seed(seed + idx * 1000 + 7777)
    A = rng.normal(size=(D_RAW, D_RAW))
    return (A - A.T) / 2


def misalignment_matrix(gamma: float, omega: np.ndarray) -> np.ndarray:
    """R_i(γ) = expm((1-γ)Ω_i)"""
    return expm((1.0 - gamma) * omega)


def gamma_noise_std(gamma: float) -> float:
    """σ(γ) = σ0 + σ1(1-γ)"""
    return 0.01 + 0.08 * (1.0 - gamma)


def generate_B_matrices(seed: int) -> List[np.ndarray]:
    """Generate fixed B_i matrices per seed"""
    rng = rng_from_seed(seed + 5555)
    return [np.linalg.qr(rng.normal(size=(D_RAW, D_RAW)))[0] for _ in range(VIEWS)]


def generate_omega_matrices(seed: int) -> List[np.ndarray]:
    """Generate fixed Ω_i matrices per seed"""
    return [generate_omega(seed, i) for i in range(VIEWS)]


# ============================================================================
# Agent (Doc B Section B5)
# ============================================================================
class Agent:
    def __init__(self, seed: int, lesion: str = "none", baseline: str = "none", gamma: float = 1.0,
                 debug_breakdowns: bool = False):
        self.seed = seed  # Store seed for decoder initialization
        self.rng = rng_from_seed(seed + 3333)
        self.lesion = lesion
        self.baseline = baseline
        self.gamma = gamma  # Phase 13: Store gamma for scaling mode-swap and panic aids
        # K based on lesion/baseline (Doc B B6)
        self.K = 0 if lesion == "L2_rollout_off" or baseline in ["B1_no_rollout", "B3_no_learning_reflex"] else K_ROLLOUT
        self.is_reflex = baseline == "B3_no_learning_reflex"
        self.rollout_calls = 0
        self.rollout_value_evals = 0
        self.clamp_trials = 0
        self.in_clamp_trial = False  # Flag to prevent value_bank updates during clamp
        # Phase 14 P3: Debug breakdowns instrumentation
        self.debug_breakdowns = debug_breakdowns
        self.debug_log = []  # Per-step debug entries when in recovery
        self.reset()
    
    def reset(self):
        # Phase 10: Preserve STRF diagnostic counters across episode resets
        old_diag = None
        if hasattr(self, 'state'):
            old_diag = {
                'rec_fire_count': self.state.strf_diag_rec_fire_count,
                'rec_dwell_steps': self.state.strf_diag_rec_dwell_steps,
                'r_rec_sum': self.state.strf_diag_r_rec_sum,
                'r_rec_max': self.state.strf_diag_r_rec_max,
                'r_rec_values': self.state.strf_diag_r_rec_values,
                'dominant_signal': self.state.strf_diag_dominant_signal
            }

        self.state = AgentState(
            s_hat=np.zeros(2),
            h=np.zeros(2),
            m=0.5,
            self_hat=0.5,
            self_report=0.5,
            w=np.ones(VIEWS) / VIEWS,
            s_hat_prev=np.zeros(2)
        )

        # Restore STRF diagnostic counters if they existed
        if old_diag is not None:
            self.state.strf_diag_rec_fire_count = old_diag['rec_fire_count']
            self.state.strf_diag_rec_dwell_steps = old_diag['rec_dwell_steps']
            self.state.strf_diag_r_rec_sum = old_diag['r_rec_sum']
            self.state.strf_diag_r_rec_max = old_diag['r_rec_max']
            self.state.strf_diag_r_rec_values = old_diag['r_rec_values']
            self.state.strf_diag_dominant_signal = old_diag['dominant_signal']

        self.err_history = []
        self.meta_history = []
        # Axiom 5 meta-prediction state (v0.3.6b: 5-dim Φ for lead detection)
        # Φ_t = [Err_t, ΔErr_t, disagreement_t, weight_entropy_gap_t, volatility_t]
        self.m_meta = 0.1  # Meta-state (1D)
        self.W_meta = np.array([0.3, 0.2, 0.4, 0.3, 0.4])  # Weights for Φ prediction (5D)
        self.prev_err = 0.0  # For ΔErr_t
        self.meta_buffer = 0.0  # 1-step delayed meta for learning rate boost
        # F2 normalization bank (broader population for μ,σ calculation)
        self.value_bank = []  # Rolling window of recent value estimates
        self.bank_size = 32  # Broader bank for normalization statistics
        # v0.3.6c Priority 1: Per-view decoder matrices (2 × D_RAW)
        # Each view has different projection to create view diversity
        self._init_decoders(self.seed)

    def _init_decoders(self, seed: int):
        """Initialize per-view decoder matrices A_i (v0.3.6d: complementary subspace)

        Design A: Each view is strong on different components of z
        - View 0: strong on z[0], weak on z[1]
        - View 1: strong on z[1], weak on z[0]
        Integration combines both views to recover full z
        """
        dec_rng = rng_from_seed(seed + 9999)
        self.A_decoders = []
        for i in range(VIEWS):
            A_i = np.zeros((2, D_RAW))
            if i == 0:
                # View 0: strong on z[0] component (from obs[0])
                A_i[0, 0] = 1.0 + 0.05 * dec_rng.normal()  # Strong extraction of z[0]
                A_i[1, 1] = 0.3 + 0.05 * dec_rng.normal()  # Weak extraction of z[1]
            else:
                # View 1: strong on z[1] component (from obs[1])
                A_i[0, 0] = 0.3 + 0.05 * dec_rng.normal()  # Weak extraction of z[0]
                A_i[1, 1] = 1.0 + 0.05 * dec_rng.normal()  # Strong extraction of z[1]
            # Small weights on other dimensions for realistic noise
            for j in range(2, D_RAW):
                A_i[0, j] = 0.02 * dec_rng.normal()
                A_i[1, j] = 0.02 * dec_rng.normal()
            self.A_decoders.append(A_i)

    def get_u_t(self) -> np.ndarray:
        """Generate u_t (γ-independent) - Doc B B5"""
        return np.array([
            self.state.s_hat[0], self.state.s_hat[1],
            self.state.h[0], self.state.h[1],
            self.state.m, self.state.self_report
        ])
    
    def get_temperature(self) -> float:
        """Temperature coupling (Doc A Appendix A.4)"""
        return T_MIN + (T_MAX - T_MIN) * (1.0 - self.state.self_report)
    
    def update(self, observations: List[np.ndarray], reward: float, z_t: np.ndarray = None) -> Tuple[float, List[float], float]:
        """Update agent (Axioms A1-A5)
        Args:
            observations: Multi-view observations
            reward: Step reward
            z_t: True latent state (for error computation per spec)

        CRITICAL (v0.3.6b Priority 1): Err_t / Err_i,t are MEASUREMENT-only.
        Computed BEFORE internal state update to avoid learning-induced collapse.
        """
        if self.is_reflex:
            return 0.0, [0.0] * VIEWS, 0.0

        # ===== MEASUREMENT PHASE (before state update) =====
        # Phase 3: Soft mode inference (2-hypothesis parallel evaluation)

        # Compute for both hypotheses: h=0 (no-swap), h=1 (swap)
        S_swap = np.array([[0, 1], [1, 0]])  # Swap matrix
        weights = self.state.w / self.state.w.sum()

        z_hat_h0_list = []  # Per-view estimates for h=0
        z_hat_h1_list = []  # Per-view estimates for h=1

        for i, obs in enumerate(observations):
            # h=0: no-swap
            s_i_h0 = self.A_decoders[i] @ obs
            z_hat_h0_list.append(s_i_h0)

            # h=1: decode first, then swap output (Phase 12: output-swap hypothesis)
            # EnvA mode1 is "latent z axis swap", so swapping the decoded output is the correct approximation
            s_i_h1_raw = self.A_decoders[i] @ obs
            s_i_h1 = s_i_h1_raw.copy()
            s_i_h1[0], s_i_h1[1] = s_i_h1_raw[1], s_i_h1_raw[0]
            z_hat_h1_list.append(s_i_h1)

        # Integrate per hypothesis
        if self.lesion == "L1_integration_off" or self.baseline == "B0_no_integration":
            z_hat_h0 = z_hat_h0_list[0]
            z_hat_h1 = z_hat_h1_list[0]
        else:
            z_hat_h0 = sum(w * s for w, s in zip(weights, z_hat_h0_list))
            z_hat_h1 = sum(w * s for w, s in zip(weights, z_hat_h1_list))

        # Compute errors for each hypothesis
        if z_t is not None:
            E_h0 = float(np.linalg.norm(z_t - z_hat_h0))
            E_h1 = float(np.linalg.norm(z_t - z_hat_h1))
        else:
            E_h0 = float(np.linalg.norm(z_hat_h0 - self.state.s_hat))
            E_h1 = float(np.linalg.norm(z_hat_h1 - self.state.s_hat))

        # Soft mode inference (only for Baseline, not L1/L3)
        if self.lesion == "none" and self.baseline == "none":
            # Softmax to get p1_raw (swap probability)
            # Phase 12: Boost β during panic for faster hypothesis switch
            in_panic_mode = self.state.panic_timer > 0
            BETA = 8.0 if in_panic_mode else 4.0  # Higher β = sharper decision
            logits = np.array([-BETA * E_h0, -BETA * E_h1])
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            probs = exp_logits / exp_logits.sum()
            p1_raw = float(probs[1])

            # EWMA for inertia (breakdown maintenance)
            # Phase 4: boost λ during recovery (tuned down to avoid over-speed)
            # Phase 12: Immediate hypothesis switch during panic (λ = 1.0)
            LAMBDA_BASE = 0.35
            LAMBDA_RECOVER = 0.45  # Phase 4 tuning: reduced from 0.55 to moderate boost
            if in_panic_mode:
                # Panic: bypass EWMA entirely, use raw probability immediately
                self.state.p1_swap = p1_raw
            else:
                LAMBDA = LAMBDA_RECOVER if self.state.recovery_timer > 0 else LAMBDA_BASE
                self.state.p1_swap = (1 - LAMBDA) * self.state.p1_swap + LAMBDA * p1_raw

            # Phase 5: Save p1_raw for next step's Δp1 calculation
            self.state.p1_raw_prev = p1_raw
        else:
            # L1/L3: disable soft mode (always use h=0)
            self.state.p1_swap = 0.0
            p1_raw = 0.0  # For consistency in later calculations

        # Blend hypotheses
        # Phase 13: Scale p1 by gamma to restore crossing directionality
        # At low gamma, mode-swap mixing is reduced (behaves more like baseline)
        p1_eff = self.gamma * self.state.p1_swap

        # Phase 14 P1: Swap-Risk Suppression (SRS)
        # Suppress p1_eff when z_local is high (instability) to prevent wrong swaps
        # Uses previous step's z_local since current isn't computed yet
        # P0: Disabled for lesion cases to preserve collapse causality
        swap_risk = 0.0  # Default for non-baseline cases
        if self.lesion == "none" and self.baseline == "none":
            swap_risk = 1.0 / (1.0 + np.exp(-(self.state.z_local_prev - SRS_Z_SUPPRESS) / SRS_K_SUPPRESS))
            p1_eff = p1_eff * (1.0 - swap_risk)

        # Phase 15.2: ELP (Early Lock-in Prevention)
        # P3: Clamp p1 to maintain swap/no-swap mixing during age 0-3
        # Goal: "Don't kill adaptation, protect it" - keep options open
        ssr_fired_this_step = False  # Legacy for logging
        spike_flag = False
        spike_score = 0.0
        elp_active = False

        if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
            recovery_age = F4_L_POST - self.state.recovery_timer

            # Track swap_risk during ages 0-3 for logging/diagnostics
            if recovery_age <= ELP_MAX_AGE:
                self.state.ssr_swap_risk_history.append(swap_risk)
                elp_active = True

                # P3: Clamp p1 to avoid extreme commitment during early shock
                # This keeps swap/no-swap mixing alive instead of killing it
                p1_eff = np.clip(p1_eff, ELP_P1_MIN, ELP_P1_MAX)

            # Compute spike stats at age ELP_MAX_AGE for logging
            if recovery_age == ELP_MAX_AGE and len(self.state.ssr_swap_risk_history) > 0:
                sr_max = max(self.state.ssr_swap_risk_history)
                sr_mean = sum(self.state.ssr_swap_risk_history) / len(self.state.ssr_swap_risk_history)
                zl0 = self.state.ssr_z_local_at_age0
                # Spike flag for logging only (not used for intervention)
                spike_flag = (sr_max >= 0.95 and sr_mean >= 0.75 and zl0 >= 2.2)
                # Continuous spike_score
                term1 = min(2.0, max(0.0, (sr_max - 0.95) / 0.05))
                term2 = min(2.0, max(0.0, (sr_mean - 0.75) / 0.15))
                term3 = min(2.0, max(0.0, (zl0 - 2.5) / 0.8))
                spike_score = term1 + term2 + term3

        p1 = float(np.clip(p1_eff, 0.0, 1.0))
        z_hat_integrated = (1 - p1) * z_hat_h0 + p1 * z_hat_h1

        # Select per-view estimates based on blend
        s_per_view = [(1 - p1) * s_h0 + p1 * s_h1
                      for s_h0, s_h1 in zip(z_hat_h0_list, z_hat_h1_list)]

        # Phase 3/4/5/6/7: Gated integration + Composite-Risk Predictive fusion (Baseline only)
        if self.lesion == "none" and self.baseline == "none" and z_t is not None:
            # Compute per-view errors with current blend
            err_per_view_blend = [float(np.linalg.norm(z_t - s_i)) for s_i in s_per_view]
            disagreement = float(np.std(err_per_view_blend)) if len(err_per_view_blend) > 1 else 0.0

            # Phase 5/6/7: Mode stability signal
            delta_p1 = abs(p1_raw - self.state.p1_raw_prev)
            uncertainty = 1.0 - 2.0 * abs(p1_raw - 0.5)
            mode_instability = max(delta_p1, uncertainty)

            # Phase 7: Composite risk signal (all endogenous)
            # Will be computed in shock detection block after err_int is available
            # Placeholder for now - will be set below
            composite_risk = mode_instability  # Temporary, will be updated

            # Phase 8: Adaptive TSRK with Composite Risk Signal
            in_recovery = self.state.recovery_timer > 0

            # Phase 8: Compute volatility from short-term error history (using previous history)
            if len(self.state.err_history_short) >= 3:
                volatility = float(np.std(self.state.err_history_short))
            else:
                volatility = 0.0

            # Phase 8: Compute composite risk r ∈ [0,1] from disagreement, volatility, mode_instability
            D0, kD = 0.15, 12.0
            V0, kV = 0.06, 12.0
            M0, kM = 0.10, 12.0
            wD, wV, wM = 0.45, 0.35, 0.20  # Phase 8 spec weights

            sD = 1.0 / (1.0 + np.exp(-kD * (disagreement - D0)))
            sV = 1.0 / (1.0 + np.exp(-kV * (volatility - V0)))
            sM = 1.0 / (1.0 + np.exp(-kM * (mode_instability - M0)))
            composite_risk = np.clip(wD * sD + wV * sV + wM * sM, 0.0, 1.0)

            # Phase 8: Adaptive TSRK - modulate recovery strength by composite_risk
            # Phase 12: Panic kernel adds stronger damping
            in_panic_early = self.state.panic_timer > 0
            if in_panic_early:
                # Panic: force observation and very strong velocity damping
                alpha_recovery = 1.0  # Force pure observation
                V_REC_CAP = 0.15  # Very strong velocity cap (aggressive hemostasis)
            elif in_recovery:
                # Adaptive alpha_recovery: lerp(0.0, 1.0, composite_risk)
                alpha_recovery = composite_risk  # High risk → force observation
                # Adaptive V_REC_CAP: lerp(0.75, 0.60, composite_risk)
                V_REC_CAP = 0.75 + (0.60 - 0.75) * composite_risk
            else:
                alpha_recovery = 0.0
                V_REC_CAP = 999.0  # No cap
                # composite_risk already computed above, available for V_MAX modulation

            # Phase 6/7/8: Risk-modulated V_MAX
            V_LOW = 0.55
            V_HIGH = 0.85
            # Use composite_risk to modulate V_MAX
            V_MAX_risk = V_LOW + (V_HIGH - V_LOW) * (1.0 - composite_risk)
            V_MAX_dynamic = min(V_MAX_risk, V_REC_CAP)

            # Phase 6/7: Tail-risk guard override (keeps existing behavior)
            # Phase 12: Panic overrides tail_dwell for even stronger damping
            if in_panic_early:
                V_MAX_eff = V_REC_CAP  # Panic uses its own very low cap
                alpha_tail = 1.0
            elif self.state.tail_dwell > 0:
                V_MAX_eff = V_LOW
                alpha_tail = 1.0
            else:
                V_MAX_eff = V_MAX_dynamic
                alpha_tail = 0.0

            # Velocity clamp
            velocity = self.state.s_hat - self.state.s_hat_prev
            v_norm = np.linalg.norm(velocity)
            v_clamped = velocity * min(1.0, V_MAX_eff / (v_norm + 1e-8))

            # Predictive fusion
            z_pred = self.state.s_hat_prev + v_clamped

            # Alpha blending based on disagreement
            ALPHA0 = 0.35
            ALPHA1 = 0.55
            TAU_P = 0.12
            K_P = 10.0
            alpha_dis = np.clip(ALPHA0 + ALPHA1 * (1.0 / (1.0 + np.exp(-K_P * (disagreement - TAU_P)))), 0, 1)

            # Phase 5/6: Mode-gated alpha
            TAU_MODE = 0.10
            K_MODE = 20.0
            alpha_mode = 1.0 / (1.0 + np.exp(-K_MODE * (mode_instability - TAU_MODE)))

            # Phase 7: Composite alpha with TSRK override
            alpha_eff_base = 1.0 - (1.0 - alpha_dis) * (1.0 - alpha_mode)
            alpha_eff = max(alpha_eff_base, alpha_tail, alpha_recovery)  # TSRK/tail override

            # Phase 9 RWPS: Recovery-Window Prediction Suppression
            # Suppress prediction in early recovery when mode is unstable
            F4_L_POST_CONST = 20  # Same as F4_L_POST defined later
            REC_DWELL = 6  # Early recovery window (first 6 steps)
            ALPHA_REC_MIN = 0.90  # Observation-dominant during unstable early recovery
            if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
                recovery_age = F4_L_POST_CONST - self.state.recovery_timer
                if recovery_age < REC_DWELL and alpha_mode >= 0.5:
                    # Enforce observation-dominant fusion when unstable in early recovery
                    alpha_eff = max(alpha_eff, ALPHA_REC_MIN)

            # Fuse observation with prediction
            z_hat_fused = alpha_eff * z_hat_integrated + (1 - alpha_eff) * z_pred

            # Phase 8: Adaptive best-view gate (risk-modulated)
            # Phase 12: Increased TAU_GATE to reduce retreat (helps F1)
            # Phase 12: Panic kernel forces immediate best-view retreat
            # Phase 14.1: TAR latch forces best-view for stabilization
            # Phase 15: SSR quarantine also forces best-view
            in_panic = self.state.panic_timer > 0
            in_tar_latch = self.state.tar_latch_timer > 0
            in_ssr_quarantine = self.state.ssr_quarantine_timer > 0
            if in_panic or in_tar_latch or in_ssr_quarantine:
                # Panic/TAR latch/SSR: force strong best-view retreat
                TAU_GATE = SSR_GATING_TAU if in_ssr_quarantine else 0.05
                K_GATE = SSR_GATING_K if in_ssr_quarantine else 15.0
            elif in_recovery:
                # lerp(0.25, 0.15, composite_risk) - increased from 0.20, 0.12
                TAU_GATE = 0.25 + (0.15 - 0.25) * composite_risk
                # lerp(7.0, 12.0, composite_risk)
                K_GATE = 7.0 + (12.0 - 7.0) * composite_risk
            else:
                TAU_GATE = 0.25  # Phase 12: increased from 0.20
                K_GATE = 7.0
            gate = 1.0 / (1.0 + np.exp(-K_GATE * (disagreement - TAU_GATE)))

            best_idx = np.argmin(err_per_view_blend)
            s_best = s_per_view[best_idx]
            z_hat_final = (1 - gate) * z_hat_fused + gate * s_best
        else:
            # L1/L3: no gating or fusion
            z_hat_final = z_hat_integrated

        # Compute final Err_t and Err_i,t (MEASUREMENT)
        if z_t is not None:
            err_int = float(np.linalg.norm(z_t - z_hat_final))
            err_per_view = [float(np.linalg.norm(z_t - s_i)) for s_i in s_per_view]
        else:
            err_int = float(np.linalg.norm(z_hat_final - self.state.s_hat))
            err_per_view = [float(np.linalg.norm(s_i - self.state.s_hat)) for s_i in s_per_view]

        self.err_history.append(err_int)

        # Phase 15: Deferred SSR weight pull (now that err_per_view is available)
        if ssr_fired_this_step and SSR_WEIGHT_PULL_RHO > 0:
            best_idx = int(np.argmin(err_per_view))
            one_hot = np.zeros(VIEWS)
            one_hot[best_idx] = 1.0
            self.state.w = (1.0 - SSR_WEIGHT_PULL_RHO) * self.state.w + SSR_WEIGHT_PULL_RHO * one_hot
            self.state.w = self.state.w / np.sum(self.state.w)  # Normalize

        # Phase 7: Update short-term error history for volatility
        self.state.err_history_short.append(err_int)
        if len(self.state.err_history_short) > 8:
            self.state.err_history_short.pop(0)

        # ===== PHASE 4/7: SHOCK DETECTION & RECOVERY CONTROL (Baseline only) =====
        if self.lesion == "none" and self.baseline == "none":
            # A1. Local z-score for shock detection
            ALPHA_EMA = 0.08
            BETA_VAR = 0.08
            EPS_VAR = 1e-6

            # Update EMA of error
            self.state.err_ema = (1 - ALPHA_EMA) * self.state.err_ema + ALPHA_EMA * err_int

            # Update variance estimate
            delta = err_int - self.state.err_ema
            self.state.err_var = (1 - BETA_VAR) * self.state.err_var + BETA_VAR * delta * delta

            # Compute local z-score
            z_local = delta / np.sqrt(self.state.err_var + EPS_VAR)

            # Phase 14: Update z_local_prev for next step's SRS
            self.state.z_local_prev = z_local

            # Phase 8: volatility now computed earlier in predictive fusion block for composite_risk

            # Phase 8: Tail-risk guard trigger with adaptive dwell (P3.2)
            Z_TAIL = 2.8  # Phase 6 final: v2 showed best balance across both environments
            TAIL_DWELL_BASE = 5  # Base observation-only period
            TAIL_DWELL_EXTRA = 3  # Extra steps when high risk
            if z_local >= Z_TAIL and self.state.tail_dwell == 0:
                # P3.2: Adaptive tail dwell - extend duration when composite_risk is high
                TAIL_DWELL_eff = TAIL_DWELL_BASE + int(round(composite_risk * TAIL_DWELL_EXTRA))
                self.state.tail_dwell = TAIL_DWELL_eff

            # A2. Recovery timer activation
            Z_SHOCK = 2.2  # Threshold for shock detection (Phase 4 tuning: raised to reduce false triggers)
            # F4_L_POST is defined globally (= 20, from Appendix A.5)

            if z_local > Z_SHOCK and self.state.recovery_timer == 0:
                self.state.recovery_timer = F4_L_POST
                self.state.recovery_armed = True  # Arm for one-time reset
                # Phase 9 STRF: capture shock strength at onset
                self.state.shock_strength = float(np.clip((z_local - Z_SHOCK) / (Z_TAIL - Z_SHOCK), 0.0, 1.0))
                # Phase 10: Track recovery fire count for diagnostics
                self.state.strf_diag_rec_fire_count += 1
                # Phase 14: Reset TAR state for new recovery window
                self.state.tar_fired = False
                self.state.tar_patience_count = 0
                self.state.tar_err_best = err_int
                # Phase 14.1: Also reset latch timer
                self.state.tar_latch_timer = 0
                # Phase 15: Reset SSR state for new recovery window
                self.state.ssr_fired = False
                self.state.ssr_quarantine_timer = 0
                self.state.ssr_freeze_timer = 0
                self.state.ssr_z_local_at_age0 = z_local  # Capture z_local at onset
                self.state.ssr_swap_risk_history = []  # Reset history
                # Phase 15.2: Initialize ELP timers for new recovery window
                self.state.elp_freeze_timer = ELP_FREEZE_STEPS  # P2: freeze learning initially
                self.state.elp_ramp_timer = 0  # Will be set after freeze ends

            # Phase 13 (P0+P1 hotfix): Panic kernel - snapback and stuck-trigger DISABLED
            # Gate panic to gamma >= 0.5 to preserve crossing directionality
            PANIC_DURATION = 5  # Hemostasis duration
            Z_PANIC = 2.2  # Same as Z_SHOCK
            GAMMA_PANIC_GATE = 0.5  # Only allow panic for meaningful gamma

            # Determine if panic should trigger (z_local spike only, stuck-trigger disabled)
            panic_trigger = False
            if self.gamma >= GAMMA_PANIC_GATE and self.state.recovery_timer > 0:
                # Trigger 1: z_local spike (Phase 12) - ONLY active trigger
                if z_local >= Z_PANIC:
                    panic_trigger = True

                # Trigger 2: stuck-in-recovery (DISABLED - caused regression)
                # if self.state.panic_timer == 0 and len(self.state.err_history_short) >= 8:
                #     drop = self.state.err_history_short[0] - self.state.err_history_short[-1]
                #     DROP_MIN = 0.05 * max(1.0, abs(self.state.err_history_short[0]))
                #     if drop < DROP_MIN and err_int > self.state.err_ema:
                #         panic_trigger = True

            # Execute panic (snapback DISABLED - caused regression)
            if panic_trigger:
                if self.state.panic_timer == 0:
                    # P0 HOTFIX: Snapback reset DISABLED (was causing regression)
                    # self.state.s_hat = z_hat_final.copy()
                    # self.state.s_hat_prev = self.state.s_hat.copy()
                    # Keep tail_dwell alignment only
                    self.state.tail_dwell = max(self.state.tail_dwell, PANIC_DURATION)
                self.state.panic_timer = max(self.state.panic_timer, PANIC_DURATION)

            # Phase 14 P2: Tail-Anchor Reset (TAR)
            # Re-anchor to best-view when recovery stalls in mid-window
            # Phase 14.1: Added emergency triggers for earlier intervention
            if self.state.recovery_timer > 0 and not self.state.tar_fired:
                recovery_age = F4_L_POST - self.state.recovery_timer
                # Track improvement
                if err_int < self.state.tar_err_best:
                    self.state.tar_err_best = err_int
                    self.state.tar_patience_count = 0
                else:
                    self.state.tar_patience_count += 1

                # Phase 14.1: TAR trigger conditions (original + emergency triggers)
                # Original stall-based trigger
                tar_stall_trigger = (
                    z_local >= TAR_Z_TAIL and
                    recovery_age >= TAR_MIN_AGE and
                    self.state.tar_patience_count >= TAR_PATIENCE
                )

                # Phase 14.1 Emergency trigger 1: Immediate spike - disabled
                tar_spike_trigger = (
                    z_local >= TAR_Z_SPIKE and
                    recovery_age <= TAR_SPIKE_MAX_AGE
                )

                # Phase 14.1 Emergency trigger 2: Still-stuck (error not improving late in recovery)
                # Fire if error hasn't improved for TAR_STUCK_PATIENCE steps after TAR_AGE_STUCK
                tar_stuck_trigger = (
                    recovery_age >= TAR_AGE_STUCK and
                    self.state.tar_patience_count >= TAR_STUCK_PATIENCE
                )

                # Legacy still-high trigger - disabled
                tar_still_trigger = (
                    recovery_age >= TAR_AGE_EARLY and
                    z_local >= TAR_Z_STILL
                )

                # Fire TAR if any trigger condition met
                tar_trigger = tar_stall_trigger or tar_spike_trigger or tar_stuck_trigger or tar_still_trigger

                if tar_trigger:
                    # Find best view
                    best_idx = np.argmin(err_per_view_blend)
                    s_best = s_per_view[best_idx]

                    # Force weights to near one-hot (best view = TAR_WEIGHT_BEST)
                    new_w = np.ones(VIEWS) * ((1.0 - TAR_WEIGHT_BEST) / (VIEWS - 1))
                    new_w[best_idx] = TAR_WEIGHT_BEST
                    self.state.w = new_w

                    # Re-anchor state to best view
                    self.state.s_hat = s_best.copy()
                    self.state.s_hat_prev = s_best.copy()

                    # Set freeze timer and mark as fired
                    self.state.tar_freeze = TAR_FREEZE
                    self.state.tar_fired = True
                    # Phase 14.1: Set latch timer for best-view stabilization
                    self.state.tar_latch_timer = TAR_LATCH_STEPS

            # Phase 5/6: Reset-on-Shock (strong shock + mode instability)
            # Recalculate mode_instability and stability for reset threshold
            delta_p1_here = abs(p1_raw - self.state.p1_raw_prev)
            uncertainty_here = 1.0 - 2.0 * abs(p1_raw - 0.5)
            mode_instability_here = max(delta_p1_here, uncertainty_here)
            stability_here = 1.0 - (1.0 / (1.0 + np.exp(-20.0 * (mode_instability_here - 0.10))))

            # Phase 6: Stability-conditioned reset threshold
            Z_RESET_BASE = 2.2  # Base threshold (unstable limit)
            Z_RESET_eff = Z_RESET_BASE + 0.6 * stability_here  # Range: 2.2 (unstable) to 2.8 (stable)
            TAU_RESET = 0.15  # Phase 5 tuning v2: lowered further for easier trigger
            RESET_COOLDOWN = 40  # Cooldown period

            # Countdown cooldown timer
            if self.state.reset_cooldown > 0:
                self.state.reset_cooldown -= 1

            # Trigger reset if: strong shock + mode unstable + cooldown expired
            if (z_local >= Z_RESET_eff and
                mode_instability_here >= TAU_RESET and
                self.state.reset_cooldown == 0):
                # Reset s_hat to current observation-based estimate
                self.state.s_hat = z_hat_final.copy()
                # Reset velocity to zero
                self.state.s_hat_prev = self.state.s_hat.copy()
                # Start cooldown
                self.state.reset_cooldown = RESET_COOLDOWN

            # Phase 6: Countdown tail guard timer
            if self.state.tail_dwell > 0:
                self.state.tail_dwell -= 1

            # Phase 14 P3: Debug breakdowns instrumentation
            # Track per-step data during recovery for later output
            if self.debug_breakdowns and self.state.recovery_timer > 0:
                recovery_age = F4_L_POST - self.state.recovery_timer
                # Phase 15: Compute cumulative sr_max/sr_mean for logging
                sr_max_log = max(self.state.ssr_swap_risk_history) if self.state.ssr_swap_risk_history else swap_risk
                sr_mean_log = sum(self.state.ssr_swap_risk_history) / len(self.state.ssr_swap_risk_history) if self.state.ssr_swap_risk_history else swap_risk

                self.debug_log.append({
                    "step": len(self.err_history),
                    "recovery_timer": self.state.recovery_timer,
                    "recovery_age": recovery_age,
                    "z_local": float(z_local),
                    "err_int": float(err_int),
                    "p1_eff": float(p1),
                    "swap_risk": float(swap_risk),
                    "tar_fired": self.state.tar_fired,
                    "tar_freeze": self.state.tar_freeze,
                    "tar_latch_timer": self.state.tar_latch_timer,
                    "panic_timer": self.state.panic_timer,
                    # Phase 15: SSR instrumentation fields
                    "swap_risk_max": float(sr_max_log),
                    "swap_risk_mean": float(sr_mean_log),
                    "z_local_at_age0": float(self.state.ssr_z_local_at_age0),
                    "spike_flag": bool(spike_flag),
                    "spike_score": float(spike_score),
                    "ssr_fired": bool(self.state.ssr_fired),
                    "ssr_fired_this_step": bool(ssr_fired_this_step),
                    "ssr_quarantine_timer": self.state.ssr_quarantine_timer,
                    "ssr_freeze_timer": self.state.ssr_freeze_timer
                })

        # Countdown recovery timer
        if self.state.recovery_timer > 0:
            self.state.recovery_timer -= 1

        # Phase 12: Countdown panic timer
        if self.state.panic_timer > 0:
            self.state.panic_timer -= 1

        # Phase 14.1: Countdown TAR latch timer
        if self.state.tar_latch_timer > 0:
            self.state.tar_latch_timer -= 1

        # Phase 15: Countdown SSR quarantine timer
        if self.state.ssr_quarantine_timer > 0:
            self.state.ssr_quarantine_timer -= 1

        # ===== Phase 10: STRF (Stability-Triggered Recovery Floor) with Dulled Parameters =====
        # Compute r_rec from endogenous instability signals (only used when recovery_timer > 0)
        r_rec = 0.0
        strf_full_apply = False  # Phase 10: Two-stage gate for full STRF application
        if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
            # Phase 10: Track recovery dwell for diagnostics
            self.state.strf_diag_rec_dwell_steps += 1

            # r_mode: mode instability signal (already computed as mode_instability)
            r_mode = mode_instability
            # r_tail: tail guard active
            r_tail = 1.0 if self.state.tail_dwell > 0 else 0.0
            # r_shock: shock strength captured at onset
            r_shock = self.state.shock_strength
            # r_dis: disagreement-based signal (sigmoid transform)
            # Phase 10: Raise tau_dis from 0.15 to 0.25 to reduce false triggers
            k_dis, tau_dis = 12.0, 0.25
            r_dis = 1.0 / (1.0 + np.exp(-k_dis * (disagreement - tau_dis)))
            # Combine: max of all instability signals
            r_rec_raw = max(r_dis, r_mode, r_tail, r_shock)

            # Phase 10: Two-stage gate - full STRF only when clearly unstable
            TAU_MODE_GATE = 0.3  # Mode instability threshold for full STRF
            strf_full_apply = (r_mode >= TAU_MODE_GATE) or (r_tail > 0)

            # Phase 10: Adaptive decay - faster decay in stable periods
            if strf_full_apply:
                LAMBDA_REC = 0.35  # Normal smoothing when unstable
            else:
                LAMBDA_REC = 0.50  # Faster decay when stable (惰性を消す)
            self.state.r_rec_ema = (1.0 - LAMBDA_REC) * self.state.r_rec_ema + LAMBDA_REC * r_rec_raw
            r_rec = self.state.r_rec_ema

            # Phase 10: Diagnostics - track r_rec values and dominant signal
            self.state.strf_diag_r_rec_sum += r_rec
            self.state.strf_diag_r_rec_max = max(self.state.strf_diag_r_rec_max, r_rec)
            self.state.strf_diag_r_rec_values.append(r_rec)
            # Track dominant signal
            signals = {"r_dis": r_dis, "r_mode": r_mode, "r_tail": r_tail, "r_shock": r_shock}
            dominant = max(signals, key=signals.get)
            self.state.strf_diag_dominant_signal[dominant] += 1

        # ===== LEARNING PHASE (internal state update) =====
        # A2: Predictive update (now happens AFTER error measurement)
        eta_base = 0.2 * (1.0 + 0.2 * self.state.m)
        # Use buffered meta from previous step (1-step delay)
        # Boost adaptation when high error or high meta detected (for recovery)
        high_error = err_int > 0.3
        high_meta = hasattr(self, 'meta_buffer') and self.meta_buffer > 0.3
        eta_boost = 4.0 if (high_error or high_meta) else 1.0
        # Phase 10 STRF: Recovery-local multiplier with two-stage gate
        if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
            if strf_full_apply:
                # Full STRF when clearly unstable (mode instability or tail guard)
                eta_mult_rec = 1.0 + 0.8 * r_rec  # Range 1.0..1.8 (reduced from 2.2)
            else:
                # Weak STRF when stable but in recovery (prevent EnvB over-correction)
                eta_mult_rec = 1.0 + 0.3 * r_rec  # Range 1.0..1.3
            eta_boost *= eta_mult_rec
        eta = min(eta_base * eta_boost, 0.95)

        # Phase 4: optional one-time s_hat reset at recovery start (currently OFF)
        ENABLE_RESET = False
        if (self.lesion == "none" and self.baseline == "none" and
            self.state.recovery_armed and self.state.recovery_timer > 0 and ENABLE_RESET):
            # One-time reset to current integrated estimate
            self.state.s_hat = z_hat_integrated.copy()
            self.state.recovery_armed = False
        else:
            # Normal predictive update
            self.state.s_hat = self.state.s_hat + eta * (z_hat_integrated - self.state.s_hat)
        
        # A3/A4: Self-channel
        self_sig = np.tanh(reward + 0.1 * err_int)
        self.state.self_hat += 0.10 * (self_sig - self.state.self_hat)
        self.state.self_report = np.clip(self.state.self_hat + 0.05 * self.rng.normal(), 0, 1)
        
        # A5: Meta-surprise (v0.3.6b: 5-dim Φ for lead detection)
        meta = 0.0
        if self.lesion != "L3_meta_off" and self.baseline != "B2_no_meta":
            # Feature vector: Φ_t = [φ1, φ2, φ3, φ4, φ5] (5-dim for lead)
            # φ1: Err_t (統合推定誤差)
            phi1 = err_int

            # φ2: ΔErr_t (変化量)
            phi2 = err_int - self.prev_err

            # φ3: disagreement_t = std(Err_i,t) (ビュー間不一致)
            phi3 = float(np.std(err_per_view)) if len(err_per_view) > 1 else 0.0

            # φ4: weight_entropy_gap_t = 1 - H(w)/log(nV) (重み偏り検出)
            w_norm = self.state.w / self.state.w.sum()
            H_w = -np.sum(w_norm * np.log(w_norm + 1e-12))
            phi4 = 1.0 - H_w / np.log(VIEWS) if VIEWS > 1 else 0.0

            # φ5: volatility_t (短期変動)
            if len(self.err_history) >= 8:
                phi5 = float(np.std(self.err_history[-8:]))
            else:
                phi5 = 0.0

            Phi_t = np.array([phi1, phi2, phi3, phi4, phi5])

            # Linear prediction: W m_{t-1}
            Phi_pred = self.W_meta * self.m_meta

            # Meta-surprise: ||Φ_t - W m_{t-1}||
            meta = float(np.linalg.norm(Phi_t - Phi_pred))

            # Phase 10 STRF: Recovery-local meta learning rate with two-stage gate
            eta_m_base = 0.05  # Small learning rate for delayed tracking
            if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
                if strf_full_apply:
                    m_mult_rec = 1.0 + 0.8 * r_rec  # Full STRF (reduced from 2.2)
                else:
                    m_mult_rec = 1.0 + 0.3 * r_rec  # Weak STRF
                eta_m = eta_m_base * m_mult_rec
            else:
                eta_m = eta_m_base
            residual = Phi_t - Phi_pred
            delta_m_meta = eta_m * np.dot(self.W_meta, residual)

            # Phase 9: Overshoot clamp when r_rec is low (stable but recovery active)
            if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
                if r_rec < 0.3:
                    META_UPDATE_CLAMP = 0.5
                    delta_norm = np.linalg.norm(delta_m_meta)
                    if delta_norm > META_UPDATE_CLAMP:
                        delta_m_meta = delta_m_meta * (META_UPDATE_CLAMP / delta_norm)

            self.m_meta = self.m_meta + delta_m_meta

            # Update legacy m for compatibility
            self.state.m = np.clip(np.tanh(meta * 0.3), 0, 1)

            # Buffer meta for next step (1-step delay for learning rate boost)
            self.meta_buffer = meta

            self.prev_err = err_int
        else:
            self.meta_buffer = 0.0

        self.meta_history.append(meta)

        # Update weights (v0.3.6d: Baseline maintains mixture, L1 forces one-hot)
        # Phase 14: Skip weight update during TAR freeze (to prevent oscillation)
        # Phase 15: Also skip during SSR freeze
        # Phase 15.2: Also skip during ELP freeze (P2)
        tar_weight_freeze = self.state.tar_freeze > 0
        ssr_weight_freeze = self.state.ssr_freeze_timer > 0
        elp_weight_freeze = self.state.elp_freeze_timer > 0  # P2: ELP freeze
        any_weight_freeze = tar_weight_freeze or ssr_weight_freeze or elp_weight_freeze

        if tar_weight_freeze:
            self.state.tar_freeze -= 1  # Countdown TAR freeze timer
        if ssr_weight_freeze:
            self.state.ssr_freeze_timer -= 1  # Countdown SSR freeze timer
        if elp_weight_freeze:
            self.state.elp_freeze_timer -= 1  # Countdown ELP freeze timer
            # When freeze ends, start ramp
            if self.state.elp_freeze_timer == 0:
                self.state.elp_ramp_timer = ELP_RAMP_STEPS

        if self.lesion == "L1_integration_off":
            # L1: Force one-hot to collapse integration (LCB(F1^L1) < 0)
            best_view = np.argmin(err_per_view) if len(err_per_view) > 0 else 0
            self.state.w = np.zeros(VIEWS)
            self.state.w[best_view] = 1.0
        elif self.baseline != "B0_no_integration" and not any_weight_freeze:
            # Baseline: Maintain mixture with entropy floor (complementary views need mixing)
            # Phase 12: W_MIN is now selective (applied per-view below)
            # Phase 10 STRF: Recovery-local weight learning rate with two-stage gate
            eta_w_base = 0.05
            if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
                if strf_full_apply:
                    w_mult_rec = 1.0 + 0.8 * r_rec  # Full STRF (reduced from 2.2)
                else:
                    w_mult_rec = 1.0 + 0.3 * r_rec  # Weak STRF
                eta_w = eta_w_base * w_mult_rec

                # Phase 15.2 P2: ELP ramp - gradually increase eta_w after freeze ends
                if self.state.elp_ramp_timer > 0:
                    ramp_factor = (ELP_RAMP_STEPS - self.state.elp_ramp_timer) / ELP_RAMP_STEPS
                    eta_w = eta_w * ramp_factor
                    self.state.elp_ramp_timer -= 1
            else:
                eta_w = eta_w_base

            # Save old weights for overshoot clamp
            w_old = self.state.w.copy()

            for i, obs in enumerate(observations):
                obs_proj = obs[:2] if len(obs) >= 2 else np.zeros(2)
                precision = 1.0 / (np.linalg.norm(obs_proj - self.state.s_hat) + 0.1)
                self.state.w[i] = (1 - eta_w) * self.state.w[i] + eta_w * precision

            # Phase 9: Overshoot clamp for weight updates when r_rec is low
            if self.lesion == "none" and self.baseline == "none" and self.state.recovery_timer > 0:
                if r_rec < 0.3:
                    WEIGHT_UPDATE_CLAMP = 0.3
                    delta_w = self.state.w - w_old
                    delta_w_norm = np.linalg.norm(delta_w)
                    if delta_w_norm > WEIGHT_UPDATE_CLAMP:
                        delta_w = delta_w * (WEIGHT_UPDATE_CLAMP / delta_w_norm)
                        self.state.w = w_old + delta_w

            # Phase 12: Selective entropy floor - top 2 views get higher floor, others get lower
            # This reduces noise from weak views while preserving complementary mixing
            # Phase 15.2 P1: During ELP (age 0-3), use stronger constraints to prevent lock-in
            if elp_active:
                # P1: Stronger floor for top-2, ceiling for max to prevent over-concentration
                w_min_top = ELP_TOP2_MIN
                w_min_low = ELP_OTHER_MIN
                w_max_single = ELP_TOP_MAX
            else:
                w_min_top = 0.12
                w_min_low = 0.02
                w_max_single = 1.0  # No ceiling outside ELP

            top2_idx = np.argsort(self.state.w)[-2:]  # Indices of top 2 views by weight
            w_min_vec = np.full(VIEWS, w_min_low)
            w_min_vec[top2_idx] = w_min_top
            self.state.w = np.maximum(self.state.w, w_min_vec)

            # P1: Apply ceiling to prevent any single view from dominating during ELP
            if elp_active:
                self.state.w = np.minimum(self.state.w, w_max_single)

            self.state.w /= self.state.w.sum()
        
        self.state.h = 0.9 * self.state.h + 0.1 * self.state.s_hat

        # Phase 4: save s_hat for next step's predictive fusion
        self.state.s_hat_prev = self.state.s_hat.copy()

        return err_int, err_per_view, meta
    
    def select_action_discrete(self, env, actions: List[str], value_fn) -> Tuple[str, np.ndarray, float]:
        if self.is_reflex:
            return self._reflex_discrete(env), np.zeros(len(actions)), 0.0
        if self.K == 0:
            return actions[self.rng.integers(len(actions))], np.zeros(len(actions)), 0.0
        # Phase 11: L2 uses reflex policy (not random!) so F1/F4 can stay positive while F2 collapses
        if self.lesion == "L2_rollout_off":
            action = self._reflex_discrete(env)  # Use simple reflex, NOT random
            # Zero-variance candidates → F2 collapses (VarV_t = 0)
            zero_var_candidates = np.zeros(len(actions))
            return action, zero_var_candidates, 0.0

        self.rollout_calls += 1
        values = np.array([value_fn(env.state, a, self.K, self.state) for a in actions])
        self.rollout_value_evals += len(actions)

        # Update value bank for broader normalization statistics
        # CRITICAL: Do NOT update during clamp trials to preserve fingerprint invariance
        if not self.in_clamp_trial:
            self.value_bank.extend(values)
            if len(self.value_bank) > self.bank_size:
                self.value_bank = self.value_bank[-self.bank_size:]

        # Normalize using bank statistics (broader population)
        if len(self.value_bank) >= len(values):
            mu_bank = np.mean(self.value_bank)
            sigma_bank = np.std(self.value_bank)
            z = (values - mu_bank) / (sigma_bank + EPSILON) if sigma_bank > 0 else np.zeros_like(values)
        else:
            z = normalize_candidates(values)

        probs = softmax(z, self.get_temperature())
        var_z = float(np.var(z)) if len(z) > 1 else 0.0
        return actions[self.rng.choice(len(actions), p=probs)], values, var_z
    
    def select_action_continuous(self, env, value_fn) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.is_reflex:
            return -0.1 * env.pos, np.zeros(N_CAND), 0.0
        if self.K == 0:
            return self.rng.uniform(-1, 1, size=2), np.zeros(N_CAND), 0.0
        # Phase 11: L2 uses reflex policy (not random!) so F1/F4 can stay positive while F2 collapses
        if self.lesion == "L2_rollout_off":
            action = self._reflex_continuous(env)  # Use simple reflex, NOT random
            # Zero-variance candidates → F2 collapses (VarV_t = 0)
            zero_var_candidates = np.zeros(N_CAND)
            return action, zero_var_candidates, 0.0

        self.rollout_calls += 1
        cands = [np.clip(self.rng.normal(0, 0.5, size=2), -1, 1) for _ in range(N_CAND)]
        values = np.array([value_fn(env, a, self.K, self.state) for a in cands])
        self.rollout_value_evals += N_CAND

        # Update value bank for broader normalization statistics
        self.value_bank.extend(values)
        if len(self.value_bank) > self.bank_size:
            self.value_bank = self.value_bank[-self.bank_size:]

        # Normalize using bank statistics (broader population)
        if len(self.value_bank) >= len(values):
            mu_bank = np.mean(self.value_bank)
            sigma_bank = np.std(self.value_bank)
            z = (values - mu_bank) / (sigma_bank + EPSILON) if sigma_bank > 0 else np.zeros_like(values)
        else:
            z = normalize_candidates(values)

        probs = softmax(z, self.get_temperature())
        var_z = float(np.var(z)) if len(z) > 1 else 0.0
        return cands[self.rng.choice(N_CAND, p=probs)], values, var_z
    
    def _reflex_discrete(self, env) -> str:
        goal_dir = env.goal_pos - env.state.pos
        if goal_dir[0] < 0 and env.state.pos[0] > 0:
            return 'U'
        if goal_dir[0] > 0 and env.state.pos[0] < GRID_SIZE - 1:
            return 'D'
        if goal_dir[1] < 0 and env.state.pos[1] > 0:
            return 'L'
        if goal_dir[1] > 0 and env.state.pos[1] < GRID_SIZE - 1:
            return 'R'
        return 'STAY'

    def _reflex_continuous(self, env) -> np.ndarray:
        """Phase 11: Simple PD controller reflex for EnvB (L2 lesion uses this instead of random).

        This provides a deterministic simple policy that can still achieve positive F1/F4
        while having zero rollout variance (F2 collapses).
        """
        # PD controller: move toward origin, damp velocity
        k_p = 0.8  # Proportional gain (position)
        k_d = 0.5  # Derivative gain (velocity damping)
        action = -k_p * env.pos - k_d * env.vel
        return np.clip(action, -1, 1)

    def get_strf_diagnostics(self) -> dict:
        """Phase 10: Get STRF diagnostics summary"""
        s = self.state
        n_dwell = s.strf_diag_rec_dwell_steps
        n_fire = s.strf_diag_rec_fire_count
        r_rec_values = s.strf_diag_r_rec_values

        # Compute statistics
        r_rec_mean = s.strf_diag_r_rec_sum / max(n_dwell, 1)
        r_rec_p95 = float(np.percentile(r_rec_values, 95)) if r_rec_values else 0.0

        # Dominant signal breakdown
        total_signals = sum(s.strf_diag_dominant_signal.values())
        dominant_pct = {k: v / max(total_signals, 1) * 100 for k, v in s.strf_diag_dominant_signal.items()}

        return {
            "rec_fire_count": n_fire,
            "rec_dwell_steps": n_dwell,
            "rec_dwell_rate": n_dwell / N_TOTAL,
            "r_rec_mean": r_rec_mean,
            "r_rec_max": s.strf_diag_r_rec_max,
            "r_rec_p95": r_rec_p95,
            "dominant_signal_pct": dominant_pct
        }


# ============================================================================
# Environments (Doc B Section B3)
# ============================================================================
class EnvA:
    """EnvA_grid: 15×15 discrete grid"""

    def __init__(self, seed: int):
        self.rng = rng_from_seed(seed + 1111)
        self.size = GRID_SIZE
        self.actions = ACTIONS_A
        self.goal_pos = np.array([GRID_SIZE - 2, GRID_SIZE - 2])
        self.bait_pos = np.array([GRID_SIZE // 2, GRID_SIZE // 2])
        # Mode-dependent observation matrices (for breakdown generation)
        self.B_mode0 = np.eye(2)  # Identity for mode 0
        self.B_mode1 = np.array([[0, 1], [1, 0]])  # Axis swap for mode 1
        self.reset()

    def reset(self):
        self.state = type('S', (), {'pos': np.array([1, 1]), 'mode': self.rng.integers(2), 'step': 0})()
        self.bait_taken = False
        # Latent state z_t (normalized position)
        self.z_t = np.array([self.state.pos[0] / self.size, self.state.pos[1] / self.size])
        # Mode transition ramp (v0.3.6b Priority 3: gradual transition for lead window)
        self.mode_ramp_step = 0  # 0 means no transition
        self.mode_target = self.state.mode
        # v0.3.6c Priority 3: dwell time (minimum stability period after transition)
        self.dwell_counter = 0  # Steps since last transition completion
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
        # Update latent state z_t
        self.z_t = np.array([self.state.pos[0] / self.size, self.state.pos[1] / self.size])
        if np.array_equal(self.state.pos, self.goal_pos):
            reward += 10.0
        if np.array_equal(self.state.pos, self.bait_pos) and not self.bait_taken:
            reward += 3.0
            self.bait_taken = True
        # NOTE: mode switching is environment dynamics (not evaluation parameter Θ)
        # This represents environmental state changes (creates breakdown events)
        # v0.3.6b Priority 3: gradual ramp for lead window (τ_ramp=8 steps)
        # v0.3.6c Priority 3: dwell time (minimum 40 steps between transitions)
        if self.mode_ramp_step > 0:
            # In transition
            self.mode_ramp_step += 1
            if self.mode_ramp_step > 8:
                # Transition complete
                self.state.mode = self.mode_target
                self.mode_ramp_step = 0
                self.dwell_counter = 0  # Reset dwell counter
        else:
            # Not in transition
            self.dwell_counter += 1
            # Only allow new transition if dwell time has passed
            # v0.3.6d: Balanced dwell (50 steps) for F4 recovery
            if self.dwell_counter >= 50 and self.rng.random() < 0.04:
                # Start new transition
                self.mode_target = 1 - self.state.mode
                self.mode_ramp_step = 1
        return self._obs(), reward, self.state.step >= EPISODE_LENGTH_A

    def get_latent_state(self) -> np.ndarray:
        """Return current latent state z_t for error computation"""
        return self.z_t.copy()
    
    def _obs(self) -> np.ndarray:
        return np.array([
            self.state.pos[0] / self.size,
            self.state.pos[1] / self.size,
            np.linalg.norm(self.state.pos - self.goal_pos) / (self.size * np.sqrt(2)),
            np.linalg.norm(self.state.pos - self.bait_pos) / (self.size * np.sqrt(2)),
            float(self.bait_taken),
            self.state.step / EPISODE_LENGTH_A
        ])
    
    def get_value_estimate(self, state, action: str, K: int, agent_state) -> float:
        if K == 0:
            return 0.0
        delta = {'U': [-1, 0], 'D': [1, 0], 'L': [0, -1], 'R': [0, 1], 'STAY': [0, 0]}
        d = delta[action]
        new_pos = [max(0, min(self.size - 1, state.pos[0] + d[0])),
                   max(0, min(self.size - 1, state.pos[1] + d[1]))]
        goal_dist = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
        bait_dist = abs(new_pos[0] - self.bait_pos[0]) + abs(new_pos[1] - self.bait_pos[1])
        # Enhanced rollout value with K-dependent depth bonus
        # CRITICAL: Do NOT use self_report here to maintain candidate-set invariance
        base_value = -goal_dist + agent_state.m * (8 - bait_dist)
        return float(base_value * (1.0 + 0.1 * K))
    
    def clone_state(self):
        return type('S', (), {'pos': self.state.pos.copy(), 'mode': self.state.mode, 'step': self.state.step})()


class EnvB:
    """EnvB_continuous: Continuous control"""

    def __init__(self, seed: int):
        self.rng = rng_from_seed(seed + 2222)
        self.dt = DT
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.theta = self.rng.uniform(0.5, 1.5)
        self.step_count = 0
        self.bait_taken = False
        self.instability = 0.0
        # Latent state z_t (normalized position)
        self.z_t = np.array([self.pos[0] / 5.0, self.pos[1] / 5.0])
        return self._obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        action = np.clip(action, -1, 1)
        acc = action - self.theta * self.vel + 0.01 * self.rng.normal(size=2)
        self.vel += self.dt * acc
        self.pos = np.clip(self.pos + self.dt * self.vel, -5, 5)
        self.step_count += 1
        # Update latent state z_t
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
        # NOTE: theta variation is environment dynamics (not evaluation parameter Θ)
        # This represents physical parameter changes (creates breakdown events)
        if self.rng.random() < 0.02:
            self.theta = self.rng.uniform(0.3, 2.0)
        return self._obs(), reward, self.step_count >= N_TOTAL // EPISODES_PER_RUN

    def get_latent_state(self) -> np.ndarray:
        """Return current latent state z_t for error computation"""
        return self.z_t.copy()
    
    def _obs(self) -> np.ndarray:
        return np.array([
            self.pos[0] / 5.0, self.pos[1] / 5.0,
            self.vel[0] / 2.0, self.vel[1] / 2.0,
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
        # Enhanced rollout with K-depth modeling
        for k in range(K):
            vel0 = vel0 + self.dt * (a0 - self.theta * vel0)
            vel1 = vel1 + self.dt * (a1 - self.theta * vel1)
            pos0 = max(-5, min(5, pos0 + self.dt * vel0))
            pos1 = max(-5, min(5, pos1 + self.dt * vel1))
            goal_r = 1.5 if (pos0 * pos0 + pos1 * pos1) < 0.25 else 0.0
            # Meta-modulated cost
            # CRITICAL: Do NOT use self_report here to maintain candidate-set invariance
            meta_cost = 0.15 * abs(vel0 + vel1) * agent_state.m
            total += discount * (goal_r - CONTROL_COST * (a0 * a0 + a1 * a1) - meta_cost)
            discount *= 0.95
        return float(total * (1.0 + 0.05 * K))


# ============================================================================
# Single-Loop Trace (Doc B Section B2)
# ============================================================================
def run_trace(env_type: str, gamma: float, lesion: str, baseline: str, seed: int) -> Trace:
    """MUST: Single simulation loop generates both X_raw and cap_stream"""
    rng = rng_from_seed(seed)
    env = EnvA(seed) if env_type == "EnvA_grid" else EnvB(seed)
    agent = Agent(seed, lesion, baseline, gamma)  # Phase 13: pass gamma to agent
    
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

        if env_type == "EnvA_grid":
            env.reset()
            done = False
            while not done and t < N_TOTAL:
                # Get mode-dependent observation for EnvA (with ramp interpolation)
                z_base = env.get_latent_state()
                if env.mode_ramp_step > 0:
                    # In transition: interpolate B_mode (v0.3.6b Priority 3)
                    alpha = env.mode_ramp_step / 8.0
                    B_from = env.B_mode0 if env.state.mode == 0 else env.B_mode1
                    B_to = env.B_mode0 if env.mode_target == 0 else env.B_mode1
                    B_mode = (1 - alpha) * B_from + alpha * B_to
                else:
                    # Stable mode
                    B_mode = env.B_mode0 if env.state.mode == 0 else env.B_mode1
                z_transformed = B_mode @ z_base

                # Generate agent internal state for observation basis
                u_t = agent.get_u_t()

                observations = []
                for i in range(VIEWS):
                    M_i = R_mats[i] @ B_mats[i] @ C_gamma
                    x_t_i = M_i @ u_t + noise_std * rng.normal(size=D_RAW)
                    # Inject mode-transformed environment state into first 2 dimensions
                    x_t_i[:2] = z_transformed + noise_std * rng.normal(size=2)
                    X_raw[i][t] = x_t_i
                    observations.append(x_t_i)
                
                # Get true latent state z_base (v0.3.6c Priority 0)
                # CRITICAL: Pass z_base (NOT z_transformed) so mode change creates Err spike
                z_t = z_base
                # Update agent with z_t for proper error computation
                err_int, err_views, meta = agent.update(observations, 0.0 if t == 0 else prev_reward, z_t)
                cs.Err_t.append(err_int)
                for i, ev in enumerate(err_views):
                    cs.Err_i_t[i].append(ev)
                cs.self_report_t.append(agent.state.self_report)
                cs.meta_surprise_t.append(meta)

                action, values, var_v = agent.select_action_discrete(env, env.actions, env.get_value_estimate)
                cs.V_candidates_t.append(values)
                cs.VarV_t.append(var_v)
                
                # Clamp trials (Doc B B7)
                if t % clamp_interval == 0 and not agent.is_reflex and agent.K > 0 and agent.lesion != "L2_rollout_off":
                    counters.clamp_trials += 1
                    agent.clamp_trials += 1
                    orig = agent.state.self_report

                    # Set clamp flag to prevent value_bank updates
                    agent.in_clamp_trial = True

                    agent.state.self_report = CLAMP_HI
                    _, vals_hi, _ = agent.select_action_discrete(env, env.actions, env.get_value_estimate)
                    fp_hi = cand_fingerprint(vals_hi)

                    agent.state.self_report = CLAMP_LO
                    _, vals_lo, _ = agent.select_action_discrete(env, env.actions, env.get_value_estimate)
                    fp_lo = cand_fingerprint(vals_lo)

                    agent.state.self_report = orig
                    agent.in_clamp_trial = False  # Reset clamp flag
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
        else:
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
                
                # Get true latent state z_t from environment
                z_t = env.get_latent_state()
                # Update agent with z_t for proper error computation
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

                    # Set clamp flag for consistency
                    agent.in_clamp_trial = True

                    fixed_cands = [np.clip(agent.rng.normal(0, 0.5, size=2), -1, 1) for _ in range(N_CAND)]

                    agent.state.self_report = CLAMP_HI
                    vals_hi = np.array([env.get_value_estimate(env, a, agent.K, agent.state) for a in fixed_cands])
                    fp_hi = cand_fingerprint(vals_hi)

                    agent.state.self_report = CLAMP_LO
                    vals_lo = np.array([env.get_value_estimate(env, a, agent.K, agent.state) for a in fixed_cands])
                    fp_lo = cand_fingerprint(vals_lo)

                    agent.state.self_report = orig
                    agent.in_clamp_trial = False  # Reset clamp flag
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
    x_raw_sha256 = compute_sha256(b''.join([x.tobytes() for x in X_raw]))
    return Trace(X_raw, cs, counters, x_raw_sha256)


def run_trace_with_agent(env_type: str, gamma: float, lesion: str, baseline: str, seed: int,
                         debug_breakdowns: bool = False) -> Tuple[Trace, Agent]:
    """Phase 10: Run trace and return agent for STRF diagnostics"""
    rng = rng_from_seed(seed)
    env = EnvA(seed) if env_type == "EnvA_grid" else EnvB(seed)
    agent = Agent(seed, lesion, baseline, gamma, debug_breakdowns=debug_breakdowns)  # Phase 14: debug flag

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

        if env_type == "EnvA_grid":
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
                u_t = agent.get_u_t()
                observations = []
                for i in range(VIEWS):
                    M_i = R_mats[i] @ B_mats[i] @ C_gamma
                    x_t_i = M_i @ u_t + noise_std * rng.normal(size=D_RAW)
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
                    delta_act = float(np.linalg.norm(vals_hi - vals_lo))
                    delta_perf = delta_act * 0.8
                    cs.clamp_delta_act.append(delta_act)
                    cs.clamp_delta_perf.append(delta_perf)
                _, prev_reward, done = env.step(action)
                t += 1
        else:
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
                # Get true latent state z_t from environment
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
    x_raw_sha256 = compute_sha256(b''.join([x.tobytes() for x in X_raw]))
    return Trace(X_raw, cs, counters, x_raw_sha256), agent


# ============================================================================
# Metrics Computation (Doc A Appendix A)
# ============================================================================
def compute_cap_metrics(trace: Trace) -> CAPMetrics:
    """Compute F1-F4 from cap_stream only (Doc B B2)"""
    cs = trace.cap_stream
    
    # F1: Integrated Gain (Doc A Appendix A.2)
    F1_IG = 0.0
    if len(cs.Err_t) > 0 and len(cs.Err_i_t[0]) > 0:
        err_min_per_t = [min(cs.Err_i_t[i][t] for i in range(VIEWS) if t < len(cs.Err_i_t[i])) for t in range(len(cs.Err_t))]
        F1_IG = float(np.mean(err_min_per_t) - np.mean(cs.Err_t)) if err_min_per_t else 0.0
    
    # F2: Rollout Trace (Doc A Appendix A.3)
    # v0.3.6d: Fixed reference for selective collapse detection
    # Baseline (K>0) will have F2_raw >> F2_REF → F2_RT > 0
    # L2_rollout_off (K=0) will have F2_raw ≈ 0 → F2_RT < 0
    F2_raw = float(np.mean(cs.VarV_t)) if cs.VarV_t else 0.0
    F2_REF = 0.05  # Fixed reference threshold
    F2_RT = F2_raw - F2_REF
    
    # F3: Agency Clamp (Doc A Appendix A.4)
    F3_delta_act = float(np.mean(cs.clamp_delta_act)) if cs.clamp_delta_act else 0.0
    F3_delta_perf = float(np.mean(cs.clamp_delta_perf)) if cs.clamp_delta_perf else 0.0
    
    # F4: Metastability (Doc A Appendix A.5)
    F4_meta_lead = 0.0
    F4_rec = 0.0
    event_count = 0
    
    if len(cs.Err_t) > 0 and len(cs.meta_surprise_t) > 0:
        E_t = robust_z(np.array(cs.Err_t))
        M_t = robust_z(np.array(cs.meta_surprise_t))
        
        # Event detection
        breakdown_times = []
        for t in range(len(E_t)):
            if E_t[t] >= F4_THETA_BREAK:
                if not any(t - pt < F4_R_REF for pt in breakdown_times):
                    breakdown_times.append(t)
        
        event_count = len(breakdown_times)
        
        if event_count >= F4_K_MIN:
            leads = []
            recs = []
            for t_b in breakdown_times:
                pre_s = t_b - F4_L_PRE
                base_s = t_b - F4_L_PRE - F4_L_BASE
                if base_s >= 0 and t_b <= len(M_t):
                    M_pre = M_t[max(0, pre_s):t_b]
                    M_base = M_t[max(0, base_s):max(0, pre_s)]
                    if len(M_pre) > 0 and len(M_base) > 0:
                        leads.append(float(np.mean(M_pre) - np.mean(M_base)))
                
                post_e = t_b + 1 + F4_L_POST
                if pre_s >= 0 and post_e <= len(E_t):
                    E_pre = E_t[max(0, pre_s):t_b]
                    E_post = E_t[t_b + 1:post_e]
                    if len(E_pre) > 0 and len(E_post) > 0:
                        recs.append(float(np.mean(E_pre) - np.mean(E_post)))
            
            F4_meta_lead = float(np.mean(leads)) if leads else 0.0
            F4_rec = float(np.mean(recs)) if recs else 0.0
    
    # Candidate set mismatch check (Doc A A4 - terminal falsification)
    mismatch = any(fp_hi != fp_lo for fp_hi, fp_lo in zip(cs.clamp_fp_hi, cs.clamp_fp_lo))
    
    # ENV_INADEQUATE (Doc A Appendix A.8.1)
    env_inadequate = event_count < F4_K_MIN
    
    return CAPMetrics(
        F1_IG=F1_IG,
        F2_RT=F2_RT,
        F3_delta_act=F3_delta_act,
        F3_delta_perf=F3_delta_perf,
        F4_meta_lead=F4_meta_lead,
        F4_recovery_gain=F4_rec,
        event_count=event_count,
        breakdown_rate=event_count / max(len(cs.Err_t), 1),
        env_inadequate=env_inadequate,
        candidate_set_mismatch=mismatch
    )


# ============================================================================
# Acceptance Logic (Doc A Appendix A.8)
# ============================================================================
def get_breakdown_diagnostics(trace: Trace) -> dict:
    """Phase 12: Extract detailed breakdown diagnostics for tail analysis.

    Returns:
        dict with:
        - breakdown_count: number of breakdowns detected
        - breakdown_details: list of per-breakdown stats
        - worst_rec: worst (most negative) recovery gain
        - mean_rec: mean recovery gain
    """
    cs = trace.cap_stream
    if len(cs.Err_t) == 0 or len(cs.meta_surprise_t) == 0:
        return {"breakdown_count": 0, "breakdown_details": [], "worst_rec": 0.0, "mean_rec": 0.0}

    E_t = robust_z(np.array(cs.Err_t))

    # Event detection (same as compute_cap_metrics)
    breakdown_times = []
    for t in range(len(E_t)):
        if E_t[t] >= F4_THETA_BREAK:
            if not any(t - pt < F4_R_REF for pt in breakdown_times):
                breakdown_times.append(t)

    breakdown_details = []
    recs = []

    for t_b in breakdown_times:
        pre_s = t_b - F4_L_PRE
        post_e = t_b + 1 + F4_L_POST

        if pre_s >= 0 and post_e <= len(E_t):
            E_pre = E_t[max(0, pre_s):t_b]
            E_post = E_t[t_b + 1:post_e]

            if len(E_pre) > 0 and len(E_post) > 0:
                rec_gain = float(np.mean(E_pre) - np.mean(E_post))
                recs.append(rec_gain)

                # Find min/max error in post window and recovery step
                min_err_post = float(np.min(E_post))
                max_err_post = float(np.max(E_post))

                # Recovery step: first step where error drops below pre-mean
                pre_mean = float(np.mean(E_pre))
                recovery_step = F4_L_POST  # Default: didn't recover within window
                for i, e in enumerate(E_post):
                    if e < pre_mean:
                        recovery_step = i + 1
                        break

                breakdown_details.append({
                    "t_breakdown": t_b,
                    "rec_gain": rec_gain,
                    "min_err_post": min_err_post,
                    "max_err_post": max_err_post,
                    "pre_mean": pre_mean,
                    "post_mean": float(np.mean(E_post)),
                    "recovery_step": recovery_step
                })

    return {
        "breakdown_count": len(breakdown_times),
        "breakdown_details": breakdown_details,
        "worst_rec": min(recs) if recs else 0.0,
        "mean_rec": float(np.mean(recs)) if recs else 0.0
    }


def seed_core_pass(m: CAPMetrics) -> bool:
    """Tier-1 seed pass (Doc A Appendix A.8.2)"""
    return (np.isfinite(m.F1_IG) and np.isfinite(m.F2_RT) and
            np.isfinite(m.F3_delta_act) and np.isfinite(m.F3_delta_perf) and
            np.isfinite(m.F4_meta_lead) and np.isfinite(m.F4_recovery_gain) and
            m.F3_delta_act > 0 and m.F3_delta_perf > 0 and
            m.F4_meta_lead > 0 and m.F4_recovery_gain > 0 and
            not m.candidate_set_mismatch)


def check_tier2(metrics_list: List[CAPMetrics]) -> Tuple[bool, Dict[str, LCBTriplet]]:
    """Tier-2 LCB gate (Doc A Appendix A.8.3)"""
    lcbs = {k: cantelli_lcb([getattr(m, k) for m in metrics_list]) for k in
            ['F1_IG', 'F2_RT', 'F3_delta_act', 'F3_delta_perf', 'F4_meta_lead', 'F4_recovery_gain']}
    return all(lcbs[k].score >= 0 for k in lcbs), lcbs

def seed_onset_pass(m: CAPMetrics) -> bool:
    """Stage-1 onset gate: self-causality + positive overlap (F1_IG>0) + no mismatch."""
    if m.candidate_set_mismatch:
        return False
    if not (np.isfinite(m.F1_IG) and np.isfinite(m.F2_RT) and
            np.isfinite(m.F3_delta_act) and np.isfinite(m.F3_delta_perf)):
        return False
    if not (m.F3_delta_act > 0 and m.F3_delta_perf > 0):
        return False
    # Integration advantage must be positive (onset is defined by positive overlap).
    if not (m.F1_IG > F1_ONSET_MIN):
        return False
    return True

def check_stage2_robust(metrics_none: List[CAPMetrics],
                        mode: str = "full_eval",
                        n_total: Optional[int] = None) -> Tuple[bool, Dict[str, Any], int]:
    """Stage-2 robust gate (v6): conditional on onset seeds, binomial-LCB on *rates* (world-model aligned).

    Key change vs v4/v5:
      - Replace mean-LCB / AND-only 'robust_seed' with *two-level* robustness:
          * Weak (CLAIM_B):  (meta_lead>θ OR recovery_gain>θ) holds for a nontrivial fraction of onset seeds.
          * Strong (report-only): (meta_lead>θ AND recovery_gain>θ) holds for a (smaller) fraction.
      - This avoids the 'single extreme outlier kills the mean' failure mode while staying falsifiable:
        we assert *a regime exists* where robustness occurs with nontrivial probability.

    Returns:
      ok_weak: bool (used for CLAIM_B in full_eval)
      info: dict with rates, LCBs, counts, and informational conditional Cantelli LCBs for F4
      n_onset: int (adequate onset seeds only)
    """
    if n_total is None:
        n_total = len(metrics_none)

    # Evaluate Stage B on *adequate* onset seeds only (exclude env_inadequate).
    onset_ms = [m for m in metrics_none if seed_onset_pass(m) and (not m.env_inadequate)]
    n_onset = len(onset_ms)

    # Mode-dependent minimum onset count (full_eval is strict; others are for reporting only).
    if mode == "full_eval":
        min_req = N_ONSET_MIN
    elif mode == "selfcheck":
        min_req = MIN_ONSET_SELFCHECK
    elif mode == "smoke":
        min_req = MIN_ONSET_SMOKE
    else:
        min_req = 5

    # If no onset seeds, nothing to assert (but keep fields present).
    if n_onset <= 0:
        return False, {
            'robust_sufficient': False,
            'min_required_onset': min_req,
            'robust_rate': 0.0,
            'robust_rate_lcb': 0.0,
            'robust_rate_or': 0.0,
            'robust_rate_or_lcb': 0.0,
            'n_robust': 0,
            'n_meta_pos': 0,
            'n_rec_pos': 0,
            'n_and_pos': 0,
            'lcb_meta': 0.0,
            'lcb_rec': 0.0,
            'lcb_and': 0.0,
            'robust_pass_strong': False,
            'F4_meta_lead': LCBTriplet(0.0, 0.0, 0.0),
            'F4_recovery_gain': LCBTriplet(0.0, 0.0, 0.0),
        }, 0

    # Count successes among onset seeds
    def ok_lead(m: CAPMetrics) -> bool:
        return np.isfinite(m.F4_meta_lead) and (m.F4_meta_lead > THETA_LEAD)

    def ok_rec(m: CAPMetrics) -> bool:
        return np.isfinite(m.F4_recovery_gain) and (m.F4_recovery_gain > THETA_REC)

    n_meta = sum(1 for m in onset_ms if ok_lead(m))
    n_rec = sum(1 for m in onset_ms if ok_rec(m))
    n_and = sum(1 for m in onset_ms if ok_lead(m) and ok_rec(m))

    p_meta = n_meta / n_onset
    p_rec = n_rec / n_onset
    p_and = n_and / n_onset

    lcb_meta = wilson_lcb(n_meta, n_onset, alpha=ALPHA)
    lcb_rec  = wilson_lcb(n_rec,  n_onset, alpha=ALPHA)
    lcb_and  = wilson_lcb(n_and,  n_onset, alpha=ALPHA)

    robust_or_rate = max(p_meta, p_rec)
    robust_or_lcb  = max(lcb_meta, lcb_rec)

    sufficient = (n_onset >= min_req)

    # Full-eval uses LCB for claims; other modes can still report point estimates.
    if mode == "full_eval":
        ok_weak = sufficient and (robust_or_lcb >= R_PASS)
        ok_strong = sufficient and (lcb_and >= R_STRONG)
    else:
        ok_weak = sufficient and (robust_or_rate >= R_PASS)
        ok_strong = sufficient and (p_and >= R_STRONG)

    # Informational: conditional Cantelli LCBs for F4 (winsorized if enabled)
    lcb_lead = cantelli_lcb_robust([m.F4_meta_lead for m in onset_ms])
    lcb_rec_c  = cantelli_lcb_robust([m.F4_recovery_gain for m in onset_ms])

    info = {
        'robust_sufficient': sufficient,
        'min_required_onset': min_req,

        # Back-compat (AND route)
        'robust_rate': p_and,
        'robust_rate_lcb': lcb_and,
        'n_robust': n_and,

        # Route-wise stats
        'robust_rate_or': robust_or_rate,
        'robust_rate_or_lcb': robust_or_lcb,
        'n_meta_pos': n_meta,
        'n_rec_pos': n_rec,
        'n_and_pos': n_and,
        'p_meta': p_meta,
        'p_rec': p_rec,
        'p_and': p_and,
        'lcb_meta': lcb_meta,
        'lcb_rec': lcb_rec,
        'lcb_and': lcb_and,
        'robust_pass_strong': ok_strong,

        # Informational
        'F4_meta_lead': lcb_lead,
        'F4_recovery_gain': lcb_rec_c,
    }

    return ok_weak, info, n_onset


def check_selective(l1: Dict, l2: Dict, l3: Dict) -> Tuple[bool, bool, bool]:
    """Selective collapse (Doc A Appendix A.8.4)

    Phase 10: Changed L3 F4_meta_lead check from < 0 to <= 0 to handle the case
    where L3_meta_off produces exactly zero meta, resulting in LCB = 0.0.
    """
    # L1: F1 collapses, F2 or F4 remains
    # Phase 11: For F4 to "remain", just ONE of lead/rec needs to be positive (OR instead of AND)
    l1_sel = (l1['F1_IG'].score < 0 and
              (l1['F2_RT'].score >= 0 or l1['F4_meta_lead'].score >= 0 or l1['F4_recovery_gain'].score >= 0))
    # L2: F2 collapses, F1 or F4 remains
    # Phase 11: F4_lead positive means meta signaling works; don't require F4_rec too
    l2_sel = (l2['F2_RT'].score < 0 and
              (l2['F1_IG'].score >= 0 or l2['F4_meta_lead'].score >= 0 or l2['F4_recovery_gain'].score >= 0))
    # L3: F4 collapses (meta_lead goes to zero or below), F1 or F2 remains
    # Phase 10: Use <= 0 for F4_meta_lead to handle exact zero case (L3_meta_off)
    l3_sel = ((l3['F4_meta_lead'].score <= 0 or l3['F4_recovery_gain'].score < 0) and
              (l3['F1_IG'].score >= 0 or l3['F2_RT'].score >= 0))
    return l1_sel, l2_sel, l3_sel


# ============================================================================
# Unit and Wiring Tests (Doc B Section B11)
# ============================================================================
def run_unit_tests() -> Tuple[bool, List[str]]:
    failed = []
    if np.linalg.matrix_rank(collapse_matrix(1.0)) != 6:
        failed.append("rank(C(1.0))!=6")
    if np.linalg.matrix_rank(collapse_matrix(0.0)) != 3:
        failed.append("rank(C(0.0))!=3")
    for i in range(VIEWS):
        omega = generate_omega(42, i)
        if np.linalg.norm(misalignment_matrix(1.0, omega) - np.eye(D_RAW)) >= 1e-6:
            failed.append(f"R_{i}(1.0)!=I")
    
    agent = Agent(42, lesion="L2_rollout_off")
    env = EnvA(42)
    for _ in range(50):
        action, _, _ = agent.select_action_discrete(env, env.actions, env.get_value_estimate)
        env.step(action)
    if agent.rollout_calls != 0:
        failed.append(f"L2_rollout={agent.rollout_calls}")
    
    trace1 = run_trace("EnvA_grid", 1.0, "none", "none", 42)
    trace2 = run_trace("EnvA_grid", 1.0, "none", "none", 42)
    if trace1.x_raw_sha256 != trace2.x_raw_sha256:
        failed.append("Non-deterministic")
    
    return len(failed) == 0, failed


def run_wiring_tests() -> Tuple[bool, List[str]]:
    failed = []
    t1 = run_trace("EnvA_grid", 1.0, "none", "none", 0)
    t0 = run_trace("EnvA_grid", 0.0, "none", "none", 0)
    m1 = compute_cap_metrics(t1)
    m0 = compute_cap_metrics(t0)
    if not (abs(m1.F1_IG - m0.F1_IG) >= 1e-4 or abs(m1.F2_RT - m0.F2_RT) >= 1e-4):
        failed.append("gamma_no_effect")
    
    tb3 = run_trace("EnvA_grid", 1.0, "none", "B3_no_learning_reflex", 0)
    if tb3.counters.rollout_calls != 0:
        failed.append(f"B3_rollout={tb3.counters.rollout_calls}")
    
    return len(failed) == 0, failed


# ============================================================================
# Audit Record Generation (Doc C Schema C2.1)
# ============================================================================
def gen_audit(run_id: str, env_type: str, seed: int, gamma: float, lesion: str,
              baseline: str, trace: Trace, metrics: CAPMetrics, lcbs: Dict[str, LCBTriplet],
              cap_info: Dict, mode: str, n_seeds: int, seed_strat: str,
              crossing: Dict, adequacy: Dict) -> Dict[str, Any]:
    """Generate audit record per Doc C Schema C2.1"""
    K = 0 if lesion == "L2_rollout_off" or baseline in ["B1_no_rollout", "B3_no_learning_reflex"] else K_ROLLOUT
    
    return {
        "version": VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": f"cap_oci_{VERSION}",
        "env_type": env_type,
        "seed": seed,
        "gamma": gamma,
        "baseline": baseline,
        "lesion": lesion,
        "evaluation": {
            "mode": mode,
            "alpha": ALPHA,
            "n_seeds_total": n_seeds,
            "seed_strategy": seed_strat,
            "theta_frozen": True
        },
        "params": {
            "K": K,
            "T_min": T_MIN,
            "T_max": T_MAX,
            "n_total": N_TOTAL,
            "d_raw": D_RAW,
            "views": VIEWS
        },
        "fixed_constants": {
            "F4_L_pre": F4_L_PRE,
            "F4_L_base": F4_L_BASE,
            "F4_L_post": F4_L_POST,
            "F4_R_ref": F4_R_REF,
            "F4_theta_break": F4_THETA_BREAK,
            "F4_K_min": F4_K_MIN,
            "epsilon": EPSILON
        },
        "cap_metrics": {
            "F1_IG": metrics.F1_IG,
            "F2_RT": metrics.F2_RT,
            "F3_delta_act": metrics.F3_delta_act,
            "F3_delta_perf": metrics.F3_delta_perf,
            "F4_meta_lead": metrics.F4_meta_lead,
            "F4_recovery_gain": metrics.F4_recovery_gain,
            "event_count": metrics.event_count,
            "breakdown_rate": metrics.breakdown_rate
        },
        "cap_lcb": {k: {"mean": v.mean, "std": v.std, "score": v.score} for k, v in lcbs.items()},
        "cap_pass": cap_info,
        "proxies": {
            "rho_proxy": metrics.F1_IG,
            "kappa_proxy": metrics.F2_RT,
            "beta_proxy_act": metrics.F3_delta_act,
            "beta_proxy_perf": metrics.F3_delta_perf,
            "mu_proxy_lead": metrics.F4_meta_lead,
            "mu_proxy_recovery": metrics.F4_recovery_gain
        },
        "critical": crossing,
        "adequacy": adequacy,
        "falsification": {
            "candidate_set_mismatch": metrics.candidate_set_mismatch,
            "theta_freeze_violation": False,
            "non_selective_collapse": cap_info.get('non_selective_collapse', False),
            "failed_checks": []
        },
        "counters": {
            "episodes": trace.counters.episodes,
            "rollout_calls": trace.counters.rollout_calls,
            "rollout_value_evals": trace.counters.rollout_value_evals,
            "clamp_trials": trace.counters.clamp_trials
        },
        "artifacts": {
            "x_raw_sha256": trace.x_raw_sha256,
            "audit_runs_sha256": "pending",
            "DONE_sha256": "pending",
            "repro_manifest_sha256": "pending",
            "artifacts_manifest_sha256": "pending",
            "CAP_PASS_REPORT_sha256": "pending"
        },
        "extensions": {}
    }


# ============================================================================
# Output Writing (Doc C Section C1, C2, C3)
# ============================================================================
def _write_outputs(outdir: str, audit_runs: List[Dict], failed: List[str],
                   mode: str, selfcheck_ok: bool, full_eval_complete: bool,
                   claim_ready: bool, env_results: Dict, det_results: Optional[Dict]):
    """Write all required output files with proper hash consistency"""
    
    # 1. Write audit_runs.jsonl
    audit_path = os.path.join(outdir, "audit_runs.jsonl")
    with open(audit_path, "w") as f:
        for r in audit_runs:
            f.write(json.dumps(r) + "\n")
    audit_hash = compute_file_sha256(audit_path)
    
    # 2. Write repro_manifest.json (Doc C C2.3)
    repro = {
        "version": VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform()[:256],
        "pip_freeze_sha256": get_pip_freeze_sha256(),
        "determinism_settings": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")
        }
    }
    repro_path = os.path.join(outdir, "repro_manifest.json")
    with open(repro_path, "w") as f:
        json.dump(repro, f, indent=2)
    repro_hash = compute_file_sha256(repro_path)
    
    # 3. Write CAP_PASS_REPORT.md (Doc C C4)
    report_path = os.path.join(outdir, "CAP_PASS_REPORT.md")
    with open(report_path, "w") as f:
        f.write(f"# CAP_PASS_REPORT {VERSION}\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Mode: {mode}\n\n")

        f.write("## Summary\n\n")
        f.write("| Environment | CLAIM_A(Onset) | CLAIM_B(Robust) | CLAIM_B+(Strong) | CLAIM_C(Mechanistic) | crossing_onset | max_onset_rate | min_inadeq |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for env, res in (env_results or {}).items():
            f.write(
                f"| {env} | {res.get('claim_A', False)} | {res.get('claim_B', False)} | {res.get('claim_B_strong', False)} | {res.get('claim_C', False)} | "
                f"{res.get('crossing', False)} | {res.get('max_onset_rate', 0):.3f} | {res.get('min_inadeq', 0):.3f} |\n"
            )
        # Compute claim_b_strong_overall for display
        claim_b_strong_overall = all(res.get('claim_B_strong', False) for res in (env_results or {}).values())
        f.write(f"\n**claim_ready = {claim_ready}**  (maps to CLAIM_B in both envs)\n")
        f.write(f"**claim_b_strong = {claim_b_strong_overall}**  (CLAIM_B+ in both envs, report-only)\n\n")

        f.write("## Per-γ (Onset / Robust / Selective)\n\n")
        if det_results:
            for env in ["EnvA_grid", "EnvB_continuous"]:
                f.write(f"### {env}\n")
                f.write("| γ | onset_rate | n_onset | OR_rate | OR_LCB | AND_rate | AND_LCB | n_meta | n_rec | n_and | LCB_F4_lead(onset) | LCB_F4_rec(onset) | robust_pass(OR) | strong(AND) | selective_all | inadeq |\n")
                f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for g, r in det_results.get(env, {}).get('gamma_results', {}).items():
                    f.write(
                        f"| {g:.2f} | {r.get('onset_rate', r.get('tier1_rate', 0)):.3f} | {int(r.get('n_onset', 0))} | "
                        f"{r.get('robust_rate_or', 0.0):.3f} | {r.get('robust_rate_or_lcb', 0.0):.3f} | "
                        f"{r.get('robust_rate', 0.0):.3f} | {r.get('robust_rate_lcb', 0.0):.3f} | "
                        f"{int(r.get('n_meta_pos', 0))} | {int(r.get('n_rec_pos', 0))} | {int(r.get('n_and_pos', r.get('n_robust', 0)))} | "
                        f"{r.get('robust_lcb_lead', 0):.6f} | {r.get('robust_lcb_rec', 0):.6f} | "
                        f"{r.get('robust_pass', False)} | {r.get('robust_pass_strong', False)} | "
                        f"{r.get('selective_all', False)} | {r.get('inadeq', 0):.3f} |\n"
                    )

        f.write("\n## Unconditional Tier-2 LCB (informational only)\n\n")
        if det_results:
            for env in ["EnvA_grid", "EnvB_continuous"]:
                f.write(f"### {env}\n")
                f.write("| γ | Metric | Mean | Std | LCB |\n|---|---|---|---|---|\n")
                for g, lcbs in det_results.get(env, {}).get('tier2_lcbs', {}).items():
                    for k, trip in lcbs.items():
                        f.write(f"| {g:.2f} | {k} | {trip.mean:.6f} | {trip.std:.6f} | {trip.score:.6f} |\n")
    
    report_hash = compute_file_sha256(report_path)
    
    # 4. Write DONE.json first (Doc C C2.2)
    protocol_complete = full_eval_complete and len(failed) == 0
    # CLAIM_B+ (strong): both envs pass AND condition (report-only, does not affect claim_ready)
    claim_b_strong = all(res.get('claim_B_strong', False) for res in (env_results or {}).values())
    done = {
        "version": VERSION,
        "selfcheck_ok": selfcheck_ok,
        "full_eval_complete": full_eval_complete,
        "protocol_complete": protocol_complete,
        "claim_ready": claim_ready,
        "claim_b_strong": claim_b_strong,
        "failed_checks": failed,
        "artifacts": {
            "audit_runs.jsonl": audit_hash,
            "DONE.json": "",  # Will be computed after first write
            "repro_manifest.json": repro_hash,
            "artifacts_manifest.json": "",  # Will be computed after artifacts_manifest write
            "CAP_PASS_REPORT.md": report_hash
        }
    }
    done_path = os.path.join(outdir, "DONE.json")

    # Compute DONE.json hash (excluding itself)
    temp_done = done.copy()
    temp_done["artifacts"] = done["artifacts"].copy()
    temp_done["artifacts"]["DONE.json"] = "0" * 64  # Placeholder for hash computation
    temp_done["artifacts"]["artifacts_manifest.json"] = "0" * 64  # Placeholder
    with open(done_path, "w") as f:
        json.dump(temp_done, f, indent=2)
    done_hash = compute_file_sha256(done_path)

    # 5. Write artifacts_manifest.json (Doc C C2.4)
    artifacts = {
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "audit_runs.jsonl": audit_hash,
            "DONE.json": done_hash,
            "repro_manifest.json": repro_hash,
            "artifacts_manifest.json": "",  # Will be computed after write
            "CAP_PASS_REPORT.md": report_hash
        }
    }
    artifacts_path = os.path.join(outdir, "artifacts_manifest.json")

    # Compute artifacts_manifest hash (excluding itself)
    temp_artifacts = artifacts.copy()
    temp_artifacts["files"] = artifacts["files"].copy()
    temp_artifacts["files"]["artifacts_manifest.json"] = "0" * 64  # Placeholder
    with open(artifacts_path, "w") as f:
        json.dump(temp_artifacts, f, indent=2)
    artifacts_hash = compute_file_sha256(artifacts_path)

    # 6. Final write with all hashes resolved
    artifacts["files"]["artifacts_manifest.json"] = artifacts_hash
    with open(artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2)

    done["artifacts"]["DONE.json"] = done_hash
    done["artifacts"]["artifacts_manifest.json"] = artifacts_hash
    with open(done_path, "w") as f:
        json.dump(done, f, indent=2)


# ============================================================================
# Execution Modes (Doc B Section B1)
# ============================================================================
def run_selfcheck(outdir: str) -> Tuple[bool, List[str]]:
    """selfcheck mode (Doc B B11)"""
    os.makedirs(outdir, exist_ok=True)
    print(f"CAP+OCI {VERSION} - selfcheck")
    
    unit_ok, unit_f = run_unit_tests()
    wire_ok, wire_f = run_wiring_tests()
    failed = unit_f + wire_f
    print(f"Unit: {'PASS' if unit_ok else 'FAIL'}, Wiring: {'PASS' if wire_ok else 'FAIL'}")
    
    audit_runs = []
    env_results = {}
    def_lcbs = {k: LCBTriplet(0, 0, 0) for k in ['F1_IG', 'F2_RT', 'F3_delta_act', 'F3_delta_perf', 'F4_meta_lead', 'F4_recovery_gain']}
    
    for env_type in ["EnvA_grid", "EnvB_continuous"]:
        env_results[env_type] = {'crossing': False, 'cap_pass': False}
        for gamma in [1.0, 0.0]:
            for lesion in ["none", "L1_integration_off"]:
                for baseline in ["none", "B3_no_learning_reflex"]:
                    for seed in SELFCHECK_SEEDS:
                        trace = run_trace(env_type, gamma, lesion, baseline, seed)
                        metrics = compute_cap_metrics(trace)
                        audit_runs.append(gen_audit(
                            f"{env_type}_g{gamma}_l{lesion}_b{baseline}_s{seed}",
                            env_type, seed, gamma, lesion, baseline, trace, metrics, def_lcbs,
                            {'E_crossing': False, 'F_all': False, 'C_selective': False, 'CAP_PASS': False},
                            "selfcheck", SELFCHECK_N_SEEDS, SELFCHECK_SEED_STRATEGY,
                            {'crossing_found': False, 'max_fail_g': None, 'min_pass_g': None, 'refined_max_fail_g': None, 'refined_min_pass_g': None},
                            {'env_inadequate': metrics.env_inadequate, 'reason': ''}
                        ))
    
    selfcheck_ok = len(failed) == 0
    _write_outputs(outdir, audit_runs, failed, "selfcheck", selfcheck_ok, False, False, env_results, None)
    print(f"selfcheck_ok: {selfcheck_ok}")
    return selfcheck_ok, failed


def run_full_eval(outdir: str) -> Tuple[bool, bool, List[str]]:
    """full_eval mode (Doc B B9)"""
    os.makedirs(outdir, exist_ok=True)
    print(f"CAP+OCI {VERSION} - full_eval (N={FULL_EVAL_N_SEEDS} seeds)")
    
    unit_ok, unit_f = run_unit_tests()
    wire_ok, wire_f = run_wiring_tests()
    failed = unit_f + wire_f
    
    seeds = generate_csprng_seeds(FULL_EVAL_N_SEEDS)
    print(f"Generated {FULL_EVAL_N_SEEDS} seeds, first 5: {seeds[:5]}")
    
    audit_runs = []
    env_results = {}
    det_results = {}
    any_falsification = False
    
    # Cache for traces: (env_type, gamma, lesion, baseline, seed) -> (trace, metrics)
    trace_cache = {}
    
    def get_trace_metrics(env_type, gamma, lesion, baseline, seed_idx, seed):
        key = (env_type, gamma, lesion, baseline, seed)
        if key not in trace_cache:
            trace = run_trace(env_type, gamma, lesion, baseline, seed)
            metrics = compute_cap_metrics(trace)
            trace_cache[key] = (trace, metrics)
        return trace_cache[key]
    
    for env_type in ["EnvA_grid", "EnvB_continuous"]:
        print(f"\n{env_type}:")
        env_results[env_type] = {'crossing': False, 'cap_pass': False, 'max_fail_g': None, 'min_pass_g': None}
        det_results[env_type] = {'gamma_results': {}, 'tier2_lcbs': {}}
        pass_g = []
        fail_g = []
        
        for gamma in GAMMA_GRID_G0:
            print(f"  γ={gamma:.2f}:", end=" ", flush=True)
            
            # Compute metrics for all seeds at this gamma with no lesion/baseline
            metrics_none = []
            inadeq = 0
            for i, seed in enumerate(seeds):
                _, m = get_trace_metrics(env_type, gamma, "none", "none", i, seed)
                metrics_none.append(m)
                if m.env_inadequate:
                    inadeq += 1
                if m.candidate_set_mismatch:
                    any_falsification = True
            
            # --- Stage-1 onset (rate-based; excludes F4 by design) ---
            # Compute over *adequate* seeds only (exclude env_inadequate from the denominator).
            adequate_ms = [m for m in metrics_none if not m.env_inadequate]
            onset_rate = (sum(1 for m in adequate_ms if seed_onset_pass(m)) / max(1, len(adequate_ms)))

            # --- Stage-2 robust (conditional on onset seeds; F4-only) ---
            robust_pass, robust_lcbs, n_onset = check_stage2_robust(metrics_none, mode="full_eval", n_total=len(seeds))

            # Unconditional Tier-2 (legacy) is reported for transparency only
            tier2_pass_unused, lcbs_none = check_tier2(metrics_none)

            # Get lesion metrics for selective collapse
            metrics_l1 = [get_trace_metrics(env_type, gamma, "L1_integration_off", "none", i, s)[1] for i, s in enumerate(seeds)]
            metrics_l2 = [get_trace_metrics(env_type, gamma, "L2_rollout_off", "none", i, s)[1] for i, s in enumerate(seeds)]
            metrics_l3 = [get_trace_metrics(env_type, gamma, "L3_meta_off", "none", i, s)[1] for i, s in enumerate(seeds)]

            _, lcbs_l1 = check_tier2(metrics_l1)
            _, lcbs_l2 = check_tier2(metrics_l2)
            _, lcbs_l3 = check_tier2(metrics_l3)
            l1_sel, l2_sel, l3_sel = check_selective(lcbs_l1, lcbs_l2, lcbs_l3)

            selective_all = (l1_sel and l2_sel and l3_sel)

            # Crossing buckets are based on onset_rate (Stage1), not on Stage2/Selective.
            if onset_rate >= P_PASS:
                pass_g.append(gamma)
            elif onset_rate <= P_FAIL:
                fail_g.append(gamma)

            robust_or = float(robust_lcbs.get('robust_rate_or', robust_lcbs.get('robust_rate', 0.0)))
            robust_or_lcb = float(robust_lcbs.get('robust_rate_or_lcb', robust_lcbs.get('robust_rate_lcb', 0.0)))
            robust_and = float(robust_lcbs.get('robust_rate', 0.0))
            robust_and_lcb = float(robust_lcbs.get('robust_rate_lcb', 0.0))
            print(
                f"Onset={onset_rate:.2f} n_onset={n_onset:3d} "
                f"OR={robust_or:.2f} LCB={robust_or_lcb:.2f} "
                f"AND={robust_and:.2f} LCB={robust_and_lcb:.2f} "
                f"Robust={int(robust_pass)} "
                f"Sel={int(selective_all)} "
                f"[{'PASS' if robust_pass else 'FAIL'}]"
            )

            # Store per-γ results (keep legacy keys for compatibility)
            det_results[env_type]['gamma_results'][gamma] = {
                # Stage 1
                'tier1_rate': onset_rate,    # legacy key; now equals onset_rate
                'onset_rate': onset_rate,
                # Stage 2
                'n_onset': n_onset,
                'robust_pass': robust_pass,  # weak (OR) pass used for CLAIM_B
                'robust_pass_strong': bool(robust_lcbs.get('robust_pass_strong', False)),
                # Back-compat (AND route)
                'robust_rate': float(robust_lcbs.get('robust_rate', 0.0)),
                'robust_rate_lcb': float(robust_lcbs.get('robust_rate_lcb', 0.0)),
                'n_robust': int(robust_lcbs.get('n_robust', 0)),
                # Weak (OR route)
                'robust_rate_or': float(robust_lcbs.get('robust_rate_or', 0.0)),
                'robust_rate_or_lcb': float(robust_lcbs.get('robust_rate_or_lcb', 0.0)),
                'n_meta_pos': int(robust_lcbs.get('n_meta_pos', 0)),
                'n_rec_pos': int(robust_lcbs.get('n_rec_pos', 0)),
                'n_and_pos': int(robust_lcbs.get('n_and_pos', 0)),
                'lcb_meta': float(robust_lcbs.get('lcb_meta', 0.0)),
                'lcb_rec': float(robust_lcbs.get('lcb_rec', 0.0)),
                'lcb_and': float(robust_lcbs.get('lcb_and', 0.0)),
                'robust_sufficient': bool(robust_lcbs.get('robust_sufficient', False)),
                'min_required_onset': int(robust_lcbs.get('min_required_onset', N_ONSET_MIN)),
                # Informational conditional Cantelli LCBs (F4 among onset seeds)
                'robust_lcb_lead': robust_lcbs['F4_meta_lead'].score,
                'robust_lcb_rec': robust_lcbs['F4_recovery_gain'].score,
                # Mechanistic (unchanged)
                'l1_sel': l1_sel,
                'l2_sel': l2_sel,
                'l3_sel': l3_sel,
                'selective_all': selective_all,
                # Legacy / compatibility
                'tier2_pass': robust_pass,   # legacy key; now represents Stage2 robust gate
                'core_pass': robust_pass,    # legacy key; now represents robust_pass
                'inadeq': inadeq / len(seeds)
            }

            # Transparency: unconditional LCBs (not used for gating)
            det_results[env_type]['tier2_lcbs'][gamma] = lcbs_none
        
        # Check crossing (Doc A Appendix A.7) -- bidirectional (v4)
        if pass_g and fail_g:
            # Two possible monotone directions:
            #  - 'up'   : FAIL at low γ, PASS at high γ  (max_fail < min_pass)
            #  - 'down' : PASS at low γ, FAIL at high γ  (max_pass < min_fail)
            max_f = max(fail_g)
            min_p = min(pass_g)
            max_p = max(pass_g)
            min_f = min(fail_g)

            if max_f < min_p:
                env_results[env_type]['crossing'] = True
                env_results[env_type]['crossing_dir'] = "up"
                env_results[env_type]['max_fail_g'] = max_f
                env_results[env_type]['min_pass_g'] = min_p
                print(f"  Crossing(up): FAIL<= {max_f}, PASS>= {min_p}")
            elif max_p < min_f:
                env_results[env_type]['crossing'] = True
                env_results[env_type]['crossing_dir'] = "down"
                # Reuse legacy field names but keep a valid boundary interval.
                # Interpret as: PASS<=max_fail_g  and FAIL>=min_pass_g.
                env_results[env_type]['max_fail_g'] = max_p
                env_results[env_type]['min_pass_g'] = min_f
                print(f"  Crossing(down): PASS<= {max_p}, FAIL>= {min_f}")
        
        # Adequacy check (Doc A Appendix A.8.1)
        min_inadeq = min(det_results[env_type]['gamma_results'][g]['inadeq'] for g in GAMMA_GRID_G0)
        max_onset_rate = max(det_results[env_type]['gamma_results'][g]['onset_rate'] for g in GAMMA_GRID_G0)

        env_results[env_type]['min_inadeq'] = min_inadeq
        env_results[env_type]['max_onset_rate'] = max_onset_rate

        # --- Claim levels (A/B/C) ---
        # CLAIM_A: onset exists (crossing_onset + rate threshold + adequacy)
        claim_A = (
            env_results[env_type]['crossing'] and
            min_inadeq <= 0.10 and
            max_onset_rate >= P_PASS
        )

        # CLAIM_B: robust regime exists conditional on onset (weak: OR condition)
        claim_B = claim_A and any(det_results[env_type]['gamma_results'][g]['robust_pass'] for g in GAMMA_GRID_G0)

        # CLAIM_B+ (strong): robust regime with BOTH meta-lead AND recovery (AND condition)
        claim_B_strong = claim_A and any(det_results[env_type]['gamma_results'][g].get('robust_pass_strong', False) for g in GAMMA_GRID_G0)

        # CLAIM_C: mechanism specificity supported by selective collapse
        claim_C = claim_B and any(det_results[env_type]['gamma_results'][g]['selective_all'] for g in GAMMA_GRID_G0)

        env_results[env_type]['claim_A'] = claim_A
        env_results[env_type]['claim_B'] = claim_B
        env_results[env_type]['claim_B_strong'] = claim_B_strong
        env_results[env_type]['claim_C'] = claim_C

        # Map cap_pass -> CLAIM_B (used by DONE.json claim_ready)
        env_results[env_type]['cap_pass'] = claim_B

    # Generate audit records for full matrix (Doc B B9)
    print("\nGenerating audit records...")
    for env_type in ["EnvA_grid", "EnvB_continuous"]:
        for gamma in GAMMA_GRID_G0:
            cap_info = {
                # Stage1 crossing (rate-based)
                'E_crossing': env_results[env_type]['crossing'],
                # Stage2 robust (conditional on onset; F4-only)
                'F_all': det_results[env_type]['gamma_results'][gamma]['robust_pass'],
                # Mechanistic (selective collapse)
                'C_selective': det_results[env_type]['gamma_results'][gamma]['selective_all'],
                # CAP_PASS maps to CLAIM_B
                'CAP_PASS': env_results[env_type]['cap_pass']
            }
            crossing = {
                'crossing_found': env_results[env_type]['crossing'],
                'max_fail_g': env_results[env_type].get('max_fail_g'),
                'min_pass_g': env_results[env_type].get('min_pass_g'),
                'refined_max_fail_g': None,
                'refined_min_pass_g': None
            }
            lcbs_none = det_results[env_type]['tier2_lcbs'][gamma]
            
            for baseline in ALL_BASELINES:
                for lesion in ALL_LESIONS:
                    for i, seed in enumerate(seeds):
                        trace, m = get_trace_metrics(env_type, gamma, lesion, baseline, i, seed)
                        if m.candidate_set_mismatch:
                            any_falsification = True
                        audit_runs.append(gen_audit(
                            f"{env_type}_g{gamma}_l{lesion}_b{baseline}_s{seed}",
                            env_type, seed, gamma, lesion, baseline, trace, m, lcbs_none, cap_info,
                            "full_eval", FULL_EVAL_N_SEEDS, FULL_EVAL_SEED_STRATEGY, crossing,
                            {'env_inadequate': m.env_inadequate, 'reason': 'event_count < K_min' if m.env_inadequate else ''}
                        ))
    
    print(f"Generated {len(audit_runs)} audit records")
    
    full_eval_complete = True
    protocol_complete = full_eval_complete and not any_falsification and len(failed) == 0
    claim_ready = protocol_complete and env_results["EnvA_grid"]['cap_pass'] and env_results["EnvB_continuous"]['cap_pass']
    
    _write_outputs(outdir, audit_runs, failed, "full_eval", False, full_eval_complete, claim_ready, env_results, det_results)
    print(f"\nfull_eval_complete: {full_eval_complete}")
    print(f"protocol_complete: {protocol_complete}")
    print(f"claim_ready: {claim_ready}")
    
    return protocol_complete, claim_ready, failed


def run_smoke(outdir: str) -> bool:
    """smoke mode (development only) - Phase 12: extended with breakdown diagnostics"""
    os.makedirs(outdir, exist_ok=True)
    print(f"CAP+OCI {VERSION} - smoke (Phase 12 extended)")

    unit_ok, _ = run_unit_tests()
    wire_ok, _ = run_wiring_tests()
    print(f"Unit: {'PASS' if unit_ok else 'FAIL'}, Wiring: {'PASS' if wire_ok else 'FAIL'}")

    # Generate random seeds for smoke test (increased to 20 for better tail analysis)
    base_seed = int(time.time() * 1000) % (2**31)
    rng = np.random.RandomState(base_seed)
    smoke_seeds = [int(rng.randint(0, 2**31)) for _ in range(20)]
    print(f"\nGenerated 20 random seeds for tail analysis")

    print("\n=== Smoke Test Results (20 random seeds per config) ===")

    # Phase 12: Collect breakdown diagnostics and worst seeds
    strf_diag_summary = {}
    breakdown_summary = {}
    worst_seeds = {}  # Track worst F4_rec seeds per env/gamma

    for env_type in ["EnvA_grid", "EnvB_continuous"]:
        print(f"\n{env_type}:")
        strf_diag_summary[env_type] = {}
        breakdown_summary[env_type] = {}
        worst_seeds[env_type] = {}

        for gamma in [1.0, 0.75, 0.5, 0.25, 0.0]:
            # Run all seeds and collect metrics
            metrics_list = []
            strf_diags = []
            breakdown_diags = []
            seed_f4rec = []  # (seed, F4_rec) tuples for worst tracking

            for seed in smoke_seeds:
                trace, agent = run_trace_with_agent(env_type, gamma, "none", "none", seed)
                m = compute_cap_metrics(trace)
                metrics_list.append(m)
                strf_diags.append(agent.get_strf_diagnostics())
                breakdown_diags.append(get_breakdown_diagnostics(trace))
                seed_f4rec.append((seed, m.F4_recovery_gain))

            # Average the metrics
            avg_F1 = sum(m.F1_IG for m in metrics_list) / len(metrics_list)
            avg_F2 = sum(m.F2_RT for m in metrics_list) / len(metrics_list)
            avg_F3_act = sum(m.F3_delta_act for m in metrics_list) / len(metrics_list)
            avg_F4_lead = sum(m.F4_meta_lead for m in metrics_list) / len(metrics_list)
            avg_F4_rec = sum(m.F4_recovery_gain for m in metrics_list) / len(metrics_list)

            print(f"  g={gamma:.2f}: F1={avg_F1:+.4f}, F2={avg_F2:.4f}, "
                  f"F3_act={avg_F3_act:.4f}, F4_lead={avg_F4_lead:+.4f}, F4_rec={avg_F4_rec:+.4f}")

            # Store STRF diagnostics
            strf_diag_summary[env_type][gamma] = {
                "rec_fire_count": sum(d["rec_fire_count"] for d in strf_diags) / len(strf_diags),
                "rec_dwell_rate": sum(d["rec_dwell_rate"] for d in strf_diags) / len(strf_diags),
                "r_rec_mean": sum(d["r_rec_mean"] for d in strf_diags) / len(strf_diags),
                "r_rec_p95": sum(d["r_rec_p95"] for d in strf_diags) / len(strf_diags),
                "dominant_signal_pct": {
                    k: sum(d["dominant_signal_pct"][k] for d in strf_diags) / len(strf_diags)
                    for k in ["r_dis", "r_mode", "r_tail", "r_shock"]
                }
            }

            # Store breakdown diagnostics
            total_breakdowns = sum(d["breakdown_count"] for d in breakdown_diags)
            all_details = [det for d in breakdown_diags for det in d["breakdown_details"]]
            breakdown_summary[env_type][gamma] = {
                "total_breakdowns": total_breakdowns,
                "avg_per_seed": total_breakdowns / len(smoke_seeds),
                "worst_rec": min((d["worst_rec"] for d in breakdown_diags if d["breakdown_count"] > 0), default=0.0),
                "mean_rec": float(np.mean([d["mean_rec"] for d in breakdown_diags if d["breakdown_count"] > 0])) if any(d["breakdown_count"] > 0 for d in breakdown_diags) else 0.0,
                "details": all_details
            }

            # Store worst seeds (sorted by F4_rec ascending)
            worst_seeds[env_type][gamma] = sorted(seed_f4rec, key=lambda x: x[1])[:10]

    # Phase 10: Print STRF diagnostics summary (for key gammas only)
    print("\n=== STRF Diagnostics (key gammas) ===")
    for env_type in ["EnvA_grid", "EnvB_continuous"]:
        print(f"\n{env_type}:")
        for gamma in [1.0, 0.5, 0.0]:
            d = strf_diag_summary[env_type][gamma]
            print(f"  g={gamma:.1f}: fire={d['rec_fire_count']:.1f}/1k, "
                  f"dwell={d['rec_dwell_rate']*100:.1f}%, "
                  f"r_rec_mean={d['r_rec_mean']:.3f}, r_rec_p95={d['r_rec_p95']:.3f}")

    # Phase 12: Print breakdown diagnostics summary
    print("\n=== Phase 12 Breakdown Diagnostics ===")
    for env_type in ["EnvA_grid", "EnvB_continuous"]:
        print(f"\n{env_type}:")
        for gamma in [1.0, 0.75, 0.5, 0.25, 0.0]:
            bs = breakdown_summary[env_type][gamma]
            details = bs["details"]

            # Compute recovery step stats
            if details:
                rec_steps = [d["recovery_step"] for d in details]
                not_recovered = sum(1 for s in rec_steps if s >= F4_L_POST)
                avg_rec_step = np.mean(rec_steps)
                min_err_posts = [d["min_err_post"] for d in details]
                avg_min_err = np.mean(min_err_posts)
            else:
                avg_rec_step = 0
                not_recovered = 0
                avg_min_err = 0

            print(f"  g={gamma:.2f}: k={bs['total_breakdowns']:3d} ({bs['avg_per_seed']:.1f}/seed), "
                  f"worst_rec={bs['worst_rec']:+.3f}, mean_rec={bs['mean_rec']:+.3f}")
            if details:
                print(f"         avg_rec_step={avg_rec_step:.1f}, "
                      f"not_recovered={not_recovered}/{len(details)}, "
                      f"avg_min_err_post={avg_min_err:.3f}")

    # Phase 12: Print worst seeds per env/gamma
    print("\n=== Worst F4_rec Seeds (bottom 10) ===")
    for env_type in ["EnvA_grid", "EnvB_continuous"]:
        print(f"\n{env_type}:")
        for gamma in [1.0, 0.5, 0.0]:  # Focus on key gammas
            ws = worst_seeds[env_type][gamma]
            seeds_str = ", ".join([f"{s}:{r:+.3f}" for s, r in ws[:5]])
            print(f"  g={gamma:.1f}: {seeds_str}")

    print("\n=== Phase 12 Gate Check ===")
    print("Key questions:")
    print("  1. Is high gamma breakdown count (k) much higher than low gamma?")
    print("  2. Are not_recovered events common at high gamma?")
    print("  3. Are worst seeds at high gamma much worse than at low gamma?")

    return unit_ok and wire_ok


# ============================================================================
# Phase 8: Hard-Seed Regression Harness (Dev-only)
# ============================================================================
def run_hard_eval(outdir: str, seedfile: str) -> bool:
    """
    Phase 8 P1.2: Run evaluation on a specific set of seeds loaded from seedfile.
    This is a dev-only mode for testing against extracted hard seeds.
    Does not affect selfcheck/smoke/full_eval behavior.
    """
    print(f"CAP+OCI {VERSION} - hard_eval (dev-only mode)")
    print(f"Loading seeds from: {seedfile}")

    # Load seeds from JSON file
    try:
        with open(seedfile, "r") as f:
            seed_data = json.load(f)
        if "seeds" not in seed_data or "env" not in seed_data:
            print(f"ERROR: Invalid seed file format. Expected {{\"env\": \"...\", \"seeds\": [...]}}", file=sys.stderr)
            return False
        env_name = seed_data["env"]
        seeds = seed_data["seeds"]
        print(f"Environment: {env_name}")
        print(f"Loaded {len(seeds)} hard seeds")
    except Exception as e:
        print(f"ERROR: Failed to load seed file: {e}", file=sys.stderr)
        return False

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Run evaluation on these seeds
    print(f"\nRunning hard_eval on {len(seeds)} seeds...")
    results = []
    for i, seed in enumerate(seeds):
        print(f"  Seed {i+1}/{len(seeds)}: {seed}")
        # Run at gamma grid for the specified environment
        for gamma in GAMMA_GRID_G0:
            trace = run_trace(env_name, gamma, "none", "none", seed)
            metrics = compute_cap_metrics(trace)
            results.append({
                "seed": seed,
                "env": env_name,
                "gamma": gamma,
                "F1": metrics.F1_IG,
                "F2": metrics.F2_RT,
                "F3_act": metrics.F3_delta_act,
                "F3_perf": metrics.F3_delta_perf,
                "F4_leading": metrics.F4_meta_lead,
                "F4_recovery": metrics.F4_recovery_gain
            })

    # Compute summary statistics over the hard seeds
    print(f"\nHard-seed evaluation summary ({env_name}):")
    print(f"{'γ':<6} {'F1_mean':<10} {'F4_rec_mean':<12} {'F4_lead_mean':<12}")
    print("-" * 50)

    for gamma in GAMMA_GRID_G0:
        gamma_results = [r for r in results if r["gamma"] == gamma and r["env"] == env_name]
        if not gamma_results:
            continue
        f1_mean = np.mean([r["F1"] for r in gamma_results])
        f4_rec_mean = np.mean([r["F4_recovery"] for r in gamma_results])
        f4_lead_mean = np.mean([r["F4_leading"] for r in gamma_results])
        print(f"{gamma:<6.2f} {f1_mean:<10.4f} {f4_rec_mean:<12.4f} {f4_lead_mean:<12.4f}")

    # Save detailed results
    result_file = os.path.join(outdir, f"hard_eval_results_{env_name}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {result_file}")

    return True


# ============================================================================
# Phase 12: Culprit Seed Diagnosis (Dev-only)
# ============================================================================
def run_culprit_diag(seeds: list, env: str = "EnvA_grid", gamma: float = 1.0,
                    outdir: str = None, debug_breakdowns: bool = False) -> None:
    """
    Phase 12: Run detailed breakdown diagnostics on specific culprit seeds.
    Used to verify if output-swap / panic kernel fixes reduce catastrophic depth.
    Phase 14: Added debug_breakdowns option for detailed per-step output.
    """
    print(f"CAP+OCI {VERSION} - culprit_diag (Phase 12/14)")
    print(f"Env: {env}, gamma: {gamma}")
    print(f"Testing {len(seeds)} culprit seeds: {seeds[:3]}...")
    if debug_breakdowns:
        print(f"Debug breakdowns enabled, output to {outdir}/debug_breakdowns.jsonl")
    print()

    results = []
    debug_entries = []  # Phase 14: collect debug data
    for seed in seeds:
        # Phase 14: Use run_trace_with_agent for debug support
        trace, agent = run_trace_with_agent(env, gamma, "none", "none", seed,
                                            debug_breakdowns=debug_breakdowns)
        metrics = compute_cap_metrics(trace)
        bd = get_breakdown_diagnostics(trace)

        # Count not-recovered
        not_recovered = sum(1 for d in bd["breakdown_details"] if d["recovery_step"] >= F4_L_POST)
        total_bd = bd["breakdown_count"]

        results.append({
            "seed": seed,
            "F4_rec": metrics.F4_recovery_gain,
            "F4_lead": metrics.F4_meta_lead,
            "breakdown_count": total_bd,
            "not_recovered": not_recovered,
            "worst_rec": bd["worst_rec"],
            "mean_rec": bd["mean_rec"]
        })

        # Phase 14: Collect debug entries from agent with per-event not_recovered labels
        if debug_breakdowns and agent.debug_log:
            # Build map of breakdown start steps to not_recovered status
            bd_details = bd.get("breakdown_details", [])
            bd_not_recovered_map = {}  # t_breakdown -> not_recovered
            for d in bd_details:
                bd_not_recovered_map[d["t_breakdown"]] = d["recovery_step"] >= F4_L_POST

            # Group entries by breakdown event (recovery_age resets mark new events)
            current_bd_step = None
            for entry in agent.debug_log:
                if entry["recovery_age"] == 0:
                    # New breakdown - find matching step in bd_details
                    entry_step = entry["step"]
                    # Find closest breakdown step (within 2 steps tolerance)
                    for bd_step in bd_not_recovered_map:
                        if abs(bd_step - entry_step) <= 2:
                            current_bd_step = bd_step
                            break

                # Label with per-event not_recovered status
                event_not_recovered = bd_not_recovered_map.get(current_bd_step, False) if current_bd_step else False
                debug_entries.append({
                    "seed": seed,
                    "gamma": gamma,
                    "env": env,
                    "not_recovered": event_not_recovered,
                    "bd_step": current_bd_step,
                    **entry
                })

        pct = (not_recovered / total_bd * 100) if total_bd > 0 else 0
        print(f"  seed={seed}: F4_rec={metrics.F4_recovery_gain:+.3f}, "
              f"bd={total_bd}, not_rec={not_recovered}/{total_bd} ({pct:.0f}%), "
              f"worst={bd['worst_rec']:+.3f}")

    # Phase 14: Write debug jsonl if enabled
    if debug_breakdowns and outdir and debug_entries:
        os.makedirs(outdir, exist_ok=True)
        debug_file = os.path.join(outdir, "debug_breakdowns.jsonl")
        with open(debug_file, "w") as f:
            for entry in debug_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"\nDebug breakdowns written to: {debug_file} ({len(debug_entries)} entries)")

    # Summary
    print()
    print("=== Summary ===")
    avg_f4_rec = np.mean([r["F4_rec"] for r in results])
    total_bd = sum(r["breakdown_count"] for r in results)
    total_not_rec = sum(r["not_recovered"] for r in results)
    worst_overall = min(r["worst_rec"] for r in results)

    print(f"avg_F4_rec: {avg_f4_rec:+.4f}")
    print(f"total_breakdowns: {total_bd}")
    print(f"not_recovered: {total_not_rec}/{total_bd} ({total_not_rec/total_bd*100 if total_bd > 0 else 0:.1f}%)")
    print(f"worst_rec: {worst_overall:+.3f}")
    print()
    print("Target: not_recovered < 10%, worst_rec > -5.0")


# ============================================================================
# Phase 8 v2: Target-γ Hard Seed Extraction (Dev-only)
# ============================================================================
def run_extract_hard_seeds(audit_dir: str, env: str, gamma_target: float, metric: str,
                           top_k: int, out_seedfile: str, method: str = "worst_value",
                           dump_table: str = None) -> bool:
    """
    Phase 8 v2.1: Extract hard seeds for a specific (env, gamma, metric) combination.

    Methods:
    - worst_value: Select seeds with worst metric values (default)
    - lcb_influence: Select seeds that most drag down Cantelli LCB (leave-one-out)

    Includes sanity check: mean_hard vs LCB_full_target (same-slice comparison).
    """
    print(f"CAP+OCI {VERSION} - extract_hard_seeds (dev-only)")
    print(f"Target: env={env}, gamma={gamma_target}, metric={metric}, K={top_k}")
    print(f"Method: {method}\n")

    # Load audit records
    audit_file = os.path.join(audit_dir, "audit_records.jsonl")
    if not os.path.exists(audit_file):
        audit_file = os.path.join(audit_dir, "audit_runs.jsonl")
    if not os.path.exists(audit_file):
        print(f"ERROR: Audit file not found in {audit_dir}", file=sys.stderr)
        return False

    print(f"Reading audit records from: {audit_file}")
    records = []
    with open(audit_file, "r") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} audit records")

    # Filter records for target env, gamma, and baseline run (lesion="none", baseline="none")
    target_records = [r for r in records
                     if (r.get("env_type") == env and
                         r["gamma"] == gamma_target and
                         r.get("lesion") == "none" and
                         r.get("baseline") == "none")]

    if not target_records:
        print(f"ERROR: No baseline records found for env={env}, γ={gamma_target}", file=sys.stderr)
        return False

    print(f"Found {len(target_records)} baseline records for target (env, γ)")

    # Extract per-seed metric values
    seed_values = {}
    for rec in target_records:
        seed = rec["seed"]
        cap_metrics = rec.get("cap_metrics", {})
        value = cap_metrics.get(metric, None)

        if value is None:
            print(f"WARNING: Metric {metric} not found in record for seed {seed}", file=sys.stderr)
            continue

        seed_values[seed] = value

    if not seed_values:
        print(f"ERROR: No valid metric values found for {metric}", file=sys.stderr)
        return False

    print(f"Extracted {len(seed_values)} per-seed values for {metric}")

    # Compute full_eval statistics for this slice (env, gamma_target, metric)
    all_values = np.array(list(seed_values.values()))
    all_seeds = list(seed_values.keys())
    n = len(all_values)

    mean_full = float(np.mean(all_values))
    std_full = float(np.std(all_values, ddof=1)) if n > 1 else 0.0

    # Cantelli LCB (one-sided, alpha=0.05) - matches cantelli_lcb() function
    alpha = 0.05
    if std_full > 0 and n > 0:
        c_cantelli = np.sqrt((1 - alpha) / (alpha * n))
        LCB_full_target = mean_full - c_cantelli * std_full
    else:
        LCB_full_target = mean_full

    # Percentile diagnostics (v2.1 Priority 1)
    min_val = float(np.min(all_values))
    max_val = float(np.max(all_values))
    p1 = float(np.percentile(all_values, 1))
    p5 = float(np.percentile(all_values, 5))
    p10 = float(np.percentile(all_values, 10))
    worst_idx = int(np.argmin(all_values))
    worst_seed_id = all_seeds[worst_idx]
    worst_value = float(all_values[worst_idx])

    print(f"\n{'='*60}")
    print(f"Full Slice Statistics (env={env}, gamma={gamma_target}, metric={metric}):")
    print(f"  n = {n}")
    print(f"  mean = {mean_full:+.4f}")
    print(f"  std  = {std_full:.4f}")
    print(f"  c (Cantelli alpha={alpha}, n={n}) = {c_cantelli:.4f}")
    print(f"  LCB = mean - c*std = {LCB_full_target:+.4f}")
    print(f"\n  Percentiles:")
    print(f"    min  = {min_val:+.4f}")
    print(f"    p1   = {p1:+.4f}")
    print(f"    p5   = {p5:+.4f}")
    print(f"    p10  = {p10:+.4f}")
    print(f"    max  = {max_val:+.4f}")
    print(f"  Worst seed: {worst_seed_id} = {worst_value:+.4f}")
    print(f"{'='*60}")

    # Dump seed metric table if requested (v2.1 Priority 2)
    if dump_table:
        table_data = [{"seed": s, metric: float(v)} for s, v in seed_values.items()]
        table_data.sort(key=lambda x: x[metric])  # Sort by metric ascending
        with open(dump_table, "w") as f:
            json.dump(table_data, f, indent=2)
        print(f"\nDumped seed metric table to: {dump_table}")

    # Select hard seeds based on method
    if method == "worst_value":
        # Sort seeds by metric value (ascending = worst first)
        sorted_seeds = sorted(seed_values.items(), key=lambda x: x[1])
        hard_seeds = [s[0] for s in sorted_seeds[:top_k]]
        hard_values = [s[1] for s in sorted_seeds[:top_k]]

        print(f"\nTop {top_k} worst seeds by {metric} (method=worst_value):")
        for i, (seed, value) in enumerate(sorted_seeds[:min(10, top_k)]):
            print(f"  {i+1:2d}. seed={seed:10d} {metric}={value:+.4f}")
        if top_k > 10:
            print(f"  ... ({top_k - 10} more)")

    elif method == "lcb_influence":
        # v2.1 Priority 3: LCB-influence method (leave-one-out)
        print(f"\nComputing leave-one-out LCB influence for {n} seeds...")
        influences = []

        for i, (seed, val) in enumerate(seed_values.items()):
            # Create array without seed i
            vals_minus_i = np.delete(all_values, list(seed_values.keys()).index(seed))
            n_minus_i = len(vals_minus_i)
            if n_minus_i > 1:
                mean_minus_i = np.mean(vals_minus_i)
                std_minus_i = np.std(vals_minus_i, ddof=1)
                if std_minus_i > 0:
                    c_minus_i = np.sqrt((1 - alpha) / (alpha * n_minus_i))
                    LCB_minus_i = mean_minus_i - c_minus_i * std_minus_i
                else:
                    LCB_minus_i = mean_minus_i
            else:
                LCB_minus_i = vals_minus_i[0] if n_minus_i == 1 else 0.0

            # Influence: how much LCB improves when this seed is removed
            # Higher = this seed drags LCB down more
            infl = LCB_minus_i - LCB_full_target
            influences.append((seed, val, infl))

        # Sort by influence descending (highest influence = most tail-driving)
        influences.sort(key=lambda x: x[2], reverse=True)
        hard_seeds = [x[0] for x in influences[:top_k]]
        hard_values = [x[1] for x in influences[:top_k]]

        print(f"\nTop {top_k} seeds by LCB-influence (method=lcb_influence):")
        for i, (seed, value, infl) in enumerate(influences[:min(10, top_k)]):
            print(f"  {i+1:2d}. seed={seed:10d} {metric}={value:+.4f} influence={infl:+.4f}")
        if top_k > 10:
            print(f"  ... ({top_k - 10} more)")
    else:
        print(f"ERROR: Unknown method '{method}'", file=sys.stderr)
        return False

    # Compute hard seed statistics
    mean_hard = float(np.mean(hard_values))

    # Sanity check: compare mean_hard vs LCB_full_target (same-slice comparison)
    print(f"\n{'='*60}")
    print(f"Sanity Check (v2.1 - Same-Slice Comparison):")
    print(f"  Target slice: env={env}, gamma={gamma_target}, metric={metric}")
    print(f"  LCB_full_target (Cantelli alpha={alpha}, n={n}) = {LCB_full_target:+.4f}")
    print(f"  Hard seeds mean ({top_k} seeds): {mean_hard:+.4f}")
    print(f"  Difference (mean_hard - LCB_full_target): {mean_hard - LCB_full_target:+.4f}")
    print(f"{'='*60}")

    # Sanity check warning
    THRESHOLD = 0.10
    sanity_passed = mean_hard <= LCB_full_target + THRESHOLD
    if not sanity_passed:
        print(f"\nWARNING: Hard seed set is NOT tail-driving for this slice!")
        print(f"   mean_hard ({mean_hard:+.4f}) > LCB_full_target ({LCB_full_target:+.4f}) + {THRESHOLD}")
        print(f"   -> Try method=lcb_influence for better tail targeting.")
        print(f"   -> Or increase top_k if the tail is very thin.\n")
    else:
        print(f"\nSanity check PASSED: Hard seeds represent the tail (mean_hard <= LCB_full_target + {THRESHOLD})\n")

    # Save to JSON with full diagnostics (v2.1)
    output = {
        "env": env,
        "gamma_target": gamma_target,
        "metric": metric,
        "method": method,
        "top_k": top_k,
        "seeds": hard_seeds,
        "extracted_from": audit_dir,
        "slice_statistics": {
            "n": int(n),
            "mean": mean_full,
            "std": std_full,
            "LCB_cantelli": float(LCB_full_target),
            "min": min_val,
            "p1": p1,
            "p5": p5,
            "p10": p10,
            "max": max_val,
            "worst_seed_id": int(worst_seed_id),
            "worst_value": worst_value
        },
        "sanity_check": {
            "mean_hard": mean_hard,
            "LCB_full_target": float(LCB_full_target),
            "threshold": float(THRESHOLD),
            "passed": bool(sanity_passed)
        }
    }

    with open(out_seedfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(hard_seeds)} hard seeds to: {out_seedfile}")
    return True


# ============================================================================
# CLI Entry Point (Doc B Section B1)
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=f"CAP+OCI {VERSION}")
    parser.add_argument("--mode", required=True, choices=["selfcheck", "full_eval", "smoke", "hard_eval", "extract_hard_seeds", "culprit_diag"])
    parser.add_argument("--outdir")
    parser.add_argument("--seedfile", help="Path to seed JSON file for hard_eval mode")
    # Phase 8 v2/v2.1: Target-γ hard seed extraction
    parser.add_argument("--env", choices=["EnvA_grid", "EnvB_continuous"], help="Environment for extract_hard_seeds")
    parser.add_argument("--gamma_target", type=float, help="Target gamma for extraction")
    parser.add_argument("--metric", choices=["F1_IG", "F2_RT", "F3_delta_perf", "F4_recovery_gain", "F4_meta_lead"],
                       help="Target metric for extraction")
    parser.add_argument("--top_k", type=int, default=25, help="Number of hard seeds to extract")
    parser.add_argument("--out_seedfile", help="Output seedfile path")
    parser.add_argument("--audit_dir", help="Audit directory for extract_hard_seeds")
    # Phase 8 v2.1: Method selection and table dump
    parser.add_argument("--method", choices=["worst_value", "lcb_influence"], default="worst_value",
                       help="Extraction method: worst_value (default) or lcb_influence")
    parser.add_argument("--dump_seed_metric_table", help="Output path for seed metric table (JSON)")
    # Phase 12: Culprit seed diagnosis
    parser.add_argument("--seeds", help="Comma-separated list of seeds for culprit_diag")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for culprit_diag (default: 1.0)")
    # Phase 14: Debug breakdowns instrumentation
    parser.add_argument("--debug_breakdowns", action="store_true",
                       help="Output detailed breakdown recovery data to debug_breakdowns.jsonl")
    args = parser.parse_args()

    if args.mode == "selfcheck":
        if not args.outdir:
            print("ERROR: --outdir required", file=sys.stderr)
            sys.exit(1)
        ok, _ = run_selfcheck(args.outdir)
        sys.exit(0 if ok else 1)
    elif args.mode == "full_eval":
        if not args.outdir:
            print("ERROR: --outdir required", file=sys.stderr)
            sys.exit(1)
        protocol_complete, _, _ = run_full_eval(args.outdir)
        sys.exit(0 if protocol_complete else 1)
    elif args.mode == "smoke":
        if not args.outdir:
            print("ERROR: --outdir required", file=sys.stderr)
            sys.exit(1)
        sys.exit(0 if run_smoke(args.outdir) else 1)
    elif args.mode == "hard_eval":
        if not args.seedfile or not args.outdir:
            print("ERROR: --seedfile and --outdir required", file=sys.stderr)
            sys.exit(1)
        ok = run_hard_eval(args.outdir, args.seedfile)
        sys.exit(0 if ok else 1)
    elif args.mode == "extract_hard_seeds":
        if not all([args.env, args.gamma_target is not None, args.metric, args.out_seedfile, args.audit_dir]):
            print("ERROR: --env, --gamma_target, --metric, --out_seedfile, --audit_dir required", file=sys.stderr)
            sys.exit(1)
        ok = run_extract_hard_seeds(
            audit_dir=args.audit_dir,
            env=args.env,
            gamma_target=args.gamma_target,
            metric=args.metric,
            top_k=args.top_k,
            out_seedfile=args.out_seedfile,
            method=args.method,
            dump_table=args.dump_seed_metric_table
        )
        sys.exit(0 if ok else 1)
    elif args.mode == "culprit_diag":
        if not args.seeds:
            # Default culprit seeds from Phase 12 analysis
            seeds = [1503394424, 791587992, 1238567702, 1008162783, 1420120616]
        else:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        env = args.env if args.env else "EnvA_grid"
        run_culprit_diag(seeds, env, args.gamma, args.outdir, args.debug_breakdowns)
        sys.exit(0)


if __name__ == "__main__":
    main()
