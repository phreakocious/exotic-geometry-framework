#!/usr/bin/env python3
"""
Investigation: Deep Prime Gap Sequential Structure

primes_deep.py found 14 metrics that survive distribution-matching — they detect
sequential correlation in prime gaps that no marginal model can explain. The top
survivor is Lorentzian:causal_order_preserved (d=10.5), a metric measuring
temporal asymmetry. This investigation probes what those 14 are, whether gaps
are time-reversible, and whether the structure is a small-prime artifact.

Five directions:
  D1: Anatomy of the 14 dist-matched survivors — identify and classify them.
  D2: Time-reversibility — do prime gaps have an arrow of time?
  D3: Surrogate hierarchy — how much structure is linear vs nonlinear?
  D4: Residue-conditioned gap geometry — Lemke Oliver-Soundararajan bias.
  D5: Scale stability of the 14 survivors — artifact or fundamental?

Budget: ~450 analyzer calls, estimated ~3 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000


# =============================================================================
# PRIME GENERATION
# =============================================================================

def sieve_primes(limit):
    """Sieve of Eratosthenes up to limit. Returns sorted array of primes."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


PRIME_LIMIT = 20_000_000
ALL_PRIMES = sieve_primes(PRIME_LIMIT)
print(f"Sieved {len(ALL_PRIMES)} primes up to {PRIME_LIMIT:,}")


def get_primes_from(start_idx, count):
    """Get `count` consecutive primes starting at index `start_idx`."""
    end = start_idx + count
    if end > len(ALL_PRIMES):
        raise ValueError(f"Need {end} primes but only have {len(ALL_PRIMES)}")
    return ALL_PRIMES[start_idx:end]


def generate_prime_gaps(trial_seed, size=DATA_SIZE, start_idx=None):
    """Generate real prime gap sequence as uint8."""
    if start_idx is None:
        start_idx = 1000 + trial_seed * 137
    primes = get_primes_from(start_idx, size + 10)
    gaps = np.diff(primes)[:size]
    return np.clip(gaps, 0, 255).astype(np.uint8)


def generate_cramer_model(trial_seed, size=DATA_SIZE, start_value=10000):
    """Cramér random model: each n is 'prime' with probability 1/ln(n)."""
    rng = np.random.RandomState(trial_seed)
    gaps = []
    n = start_value
    last_prime = n
    while len(gaps) < size:
        n += 1
        if n > 10 * PRIME_LIMIT:
            break
        prob = 1.0 / np.log(n)
        if rng.random() < prob:
            gap = n - last_prime
            gaps.append(gap)
            last_prime = n
    gaps = np.array(gaps[:size])
    return np.clip(gaps, 0, 255).astype(np.uint8)


def generate_dist_matched(trial_seed, real_gaps, size=DATA_SIZE):
    """Distribution-matched: sample from empirical distribution, destroys ordering."""
    rng = np.random.RandomState(trial_seed)
    sampled = rng.choice(real_gaps, size=size, replace=True)
    return np.clip(sampled, 0, 255).astype(np.uint8)


def generate_reversed_gaps(trial_seed, size=DATA_SIZE, start_idx=None):
    """Generate real prime gaps, then reverse the sequence."""
    return generate_prime_gaps(trial_seed, size, start_idx)[::-1].copy()


def generate_residue_gaps(trial_seed, residue, mod=6, size=DATA_SIZE):
    """Gaps conditioned on starting prime's residue class mod `mod`."""
    start_idx = 1000 + trial_seed * 137
    # Need ~3x primes since only ~50% pass mod 6 filter
    primes = get_primes_from(start_idx, size * 3 + 100)
    gaps = np.diff(primes)
    p_residues = primes[:-1] % mod
    conditioned = gaps[p_residues == residue][:size]
    return np.clip(conditioned, 0, 255).astype(np.uint8)


# =============================================================================
# IAAFT SURROGATE (from surrogate.py)
# =============================================================================

def iaaft_surrogate(data, rng, n_iter=100):
    """Iterative Amplitude Adjusted Fourier Transform surrogate.
    Preserves BOTH the power spectrum AND the marginal distribution.
    Gold standard null hypothesis for nonlinear structure.
    """
    data_f = data.astype(np.float64)
    n = len(data_f)
    target_fft = np.fft.rfft(data_f)
    target_amplitudes = np.abs(target_fft)
    sorted_data = np.sort(data_f)
    surrogate = data_f.copy()
    rng.shuffle(surrogate)
    for _ in range(n_iter):
        surr_fft = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_fft)
        matched_fft = target_amplitudes * np.exp(1j * surr_phases)
        surrogate = np.fft.irfft(matched_fft, n=n)
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_data[rank_order]
    return np.clip(surrogate, 0, 255).astype(np.uint8)


def block_shuffle(data, block_size, rng):
    """Shuffle data in blocks of `block_size`, preserving within-block order."""
    n = len(data)
    n_blocks = n // block_size
    blocks = [data[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    rng.shuffle(blocks)
    remainder = data[n_blocks*block_size:]
    return np.concatenate(blocks + ([remainder] if len(remainder) > 0 else []))


# =============================================================================
# STATISTICS
# =============================================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    if not np.isfinite(d):
        return 0.0
    return d


def count_significant(metrics_a, metrics_b, metric_names, n_total_tests):
    """Count metrics significantly different (|d|>0.8, Bonferroni)."""
    alpha = 0.05 / max(n_total_tests, 1)
    sig = 0
    for km in metric_names:
        va = metrics_a.get(km, [])
        vb = metrics_b.get(km, [])
        if len(va) < 2 or len(vb) < 2:
            continue
        d = cohens_d(va, vb)
        _, p = stats.ttest_ind(va, vb, equal_var=False)
        if not np.isfinite(p):
            continue
        if abs(d) > 0.8 and p < alpha:
            sig += 1
    return sig


def per_metric_significance(metrics_a, metrics_b, metric_names, n_total_tests):
    """Per-metric Cohen's d, p-value, significance flag."""
    alpha = 0.05 / max(n_total_tests, 1)
    result = {}
    for km in metric_names:
        va = metrics_a.get(km, [])
        vb = metrics_b.get(km, [])
        if len(va) < 2 or len(vb) < 2:
            result[km] = (0.0, 1.0, False)
            continue
        d = cohens_d(va, vb)
        _, p = stats.ttest_ind(va, vb, equal_var=False)
        if not np.isfinite(d) or not np.isfinite(p):
            result[km] = (0.0, 1.0, False)
            continue
        is_sig = abs(d) > 0.8 and p < alpha
        result[km] = (d, p, is_sig)
    return result


def collect_metrics(analyzer, data, target_dict):
    """Run analyzer on data and append results to target_dict."""
    results = analyzer.analyze(data)
    for r in results.results:
        for mn, mv in r.metrics.items():
            target_dict[f"{r.geometry_name}:{mn}"].append(mv)


# =============================================================================
# GEOMETRY FAMILY MAPPING
# =============================================================================

GEOMETRY_FAMILIES = {
    'Lattice/Discrete': ['E8 Lattice', 'Cantor (base 3)', '2-adic'],
    'Torus': ['Torus T^2', 'Clifford Torus'],
    'Curved': ['Hyperbolic (Poincaré)', 'Spherical S²'],
    'Heisenberg': ['Heisenberg (Nil)', 'Heisenberg (Nil) (centered)'],
    'Thurston': ['Sol (Thurston)', 'S² × ℝ (Thurston)',
                 'H² × ℝ (Thurston)', 'SL(2,ℝ) (Thurston)'],
    'Algebraic': ['Tropical', 'Projective ℙ²'],
    'Statistical': ['Wasserstein', 'Fisher Information', 'Persistent Homology'],
    'Physical': ['Lorentzian', 'Symplectic', 'Spiral (logarithmic)'],
    'Aperiodic': ['Penrose (Quasicrystal)', 'Ammann-Beenker (Octagonal)',
                  'Einstein (Hat Monotile)'],
    'Higher-Order': ['Higher-Order Statistics'],
}

GEOM_TO_FAMILY = {}
for fam, names in GEOMETRY_FAMILIES.items():
    for name in names:
        GEOM_TO_FAMILY[name] = fam

FAMILY_COLORS = {
    'Lattice/Discrete': '#E91E63',
    'Torus': '#FF5722',
    'Curved': '#FF9800',
    'Heisenberg': '#FFC107',
    'Thurston': '#8BC34A',
    'Algebraic': '#4CAF50',
    'Statistical': '#00BCD4',
    'Physical': '#2196F3',
    'Aperiodic': '#9C27B0',
    'Higher-Order': '#F44336',
}


def metric_to_family(metric_name):
    """Extract geometry family from 'GeometryName:metric_name' string."""
    geom_name = metric_name.split(':')[0]
    return GEOM_TO_FAMILY.get(geom_name, 'Unknown')


# =============================================================================
# DIRECTION 1: ANATOMY OF THE 14 SURVIVORS (~100 calls)
# =============================================================================

def direction1_survivors(analyzer, metric_names):
    """Identify and classify the dist-matched survivor metrics."""
    print("\n" + "=" * 78)
    print("DIRECTION 1: Anatomy of the Dist-Matched Survivors")
    print("=" * 78)
    print(f"4 conditions × {N_TRIALS} trials = 100 calls")
    print(f"Question: what ARE the metrics that survive distribution-matching?\n")

    n_total = len(metric_names)

    # Collect all baselines
    real_metrics = defaultdict(list)
    cramer_metrics = defaultdict(list)
    random_metrics = defaultdict(list)

    # Build reference gap pool for dist-matched
    ref_gaps = []

    print("  Real prime gaps...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_prime_gaps(trial)
        collect_metrics(analyzer, data, real_metrics)
        ref_gaps.extend(data.tolist())
    ref_gaps = np.array(ref_gaps)
    print("done")

    print("  Cramér model...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_cramer_model(trial)
        collect_metrics(analyzer, data, cramer_metrics)
    print("done")

    print("  Random baseline...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial + 5000)
        data = rng.randint(0, 256, DATA_SIZE, dtype=np.uint8)
        collect_metrics(analyzer, data, random_metrics)
    print("done")

    # Distribution-matched
    dist_metrics = defaultdict(list)
    print("  Dist-matched...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_dist_matched(trial, ref_gaps)
        collect_metrics(analyzer, data, dist_metrics)
    print("done")

    # Per-metric significance: real vs dist-matched
    pms_dist = per_metric_significance(real_metrics, dist_metrics, metric_names, n_total)
    survivors = [(km, d, p) for km, (d, p, is_sig) in pms_dist.items() if is_sig]
    survivors.sort(key=lambda x: -abs(x[1]))

    # Also compute real vs random and real vs Cramér for context
    n_vs_random = count_significant(real_metrics, random_metrics, metric_names, n_total)
    n_vs_cramer = count_significant(real_metrics, cramer_metrics, metric_names, n_total)
    n_vs_dist = len(survivors)

    print(f"\n  Context:")
    print(f"    Real vs Random:       {n_vs_random} sig")
    print(f"    Real vs Cramér:       {n_vs_cramer} sig")
    print(f"    Real vs Dist-matched: {n_vs_dist} sig (pure sequential correlation)")

    print(f"\n  Dist-matched survivors ({n_vs_dist} total):")
    family_counts = defaultdict(int)
    for km, d, p in survivors:
        fam = metric_to_family(km)
        family_counts[fam] += 1
        print(f"    {km:<55} d={d:+8.2f}  [{fam}]")

    print(f"\n  By geometry family:")
    for fam, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"    {fam:<20} {count}")

    return {
        'real_metrics': real_metrics,
        'cramer_metrics': cramer_metrics,
        'random_metrics': random_metrics,
        'dist_metrics': dist_metrics,
        'ref_gaps': ref_gaps,
        'survivors': survivors,
        'survivor_names': [km for km, d, p in survivors],
        'pms_dist': pms_dist,
        'family_counts': dict(family_counts),
        'n_vs_random': n_vs_random,
        'n_vs_cramer': n_vs_cramer,
        'n_vs_dist': n_vs_dist,
    }


# =============================================================================
# DIRECTION 2: TIME-REVERSIBILITY (~50 calls)
# =============================================================================

def direction2_reversibility(analyzer, metric_names, d1):
    """Do prime gaps have an arrow of time?"""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Time-Reversibility")
    print("=" * 78)
    print(f"2 × {N_TRIALS} reversed trials = 50 calls")
    print(f"Question: do prime gaps have an arrow of time?\n")

    n_total = len(metric_names)

    # Reversed real gaps
    rev_real_metrics = defaultdict(list)
    print("  Reversed real gaps...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_reversed_gaps(trial)
        collect_metrics(analyzer, data, rev_real_metrics)
    print("done")

    # Reversed Cramér gaps
    rev_cramer_metrics = defaultdict(list)
    print("  Reversed Cramér gaps...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_cramer_model(trial)[::-1].copy()
        collect_metrics(analyzer, data, rev_cramer_metrics)
    print("done")

    # Forward data reused from D1
    real_metrics = d1['real_metrics']
    cramer_metrics = d1['cramer_metrics']

    # Compare forward vs reversed
    n_real_fwd_rev = count_significant(real_metrics, rev_real_metrics,
                                       metric_names, n_total)
    n_cramer_fwd_rev = count_significant(cramer_metrics, rev_cramer_metrics,
                                          metric_names, n_total)

    print(f"\n  Forward vs Reversed:")
    print(f"    Real gaps:   {n_real_fwd_rev} sig — ", end="")
    if n_real_fwd_rev > 0:
        print("IRREVERSIBLE — prime gaps have an arrow of time!")
    else:
        print("reversible — no temporal asymmetry detected")

    print(f"    Cramér gaps: {n_cramer_fwd_rev} sig — ", end="")
    if n_cramer_fwd_rev > 0:
        print("some asymmetry (expected from density drift)")
    else:
        print("reversible (expected for i.i.d.-like model)")

    # Per-metric breakdown for real fwd vs rev
    pms_rev = per_metric_significance(real_metrics, rev_real_metrics,
                                       metric_names, n_total)
    rev_sig = [(km, d, p) for km, (d, p, is_sig) in pms_rev.items() if is_sig]
    rev_sig.sort(key=lambda x: -abs(x[1]))

    if rev_sig:
        print(f"\n  Top time-asymmetric metrics (real gaps):")
        for km, d, p in rev_sig[:10]:
            fam = metric_to_family(km)
            print(f"    {km:<55} d={d:+8.2f}  [{fam}]")

    return {
        'n_real_fwd_rev': n_real_fwd_rev,
        'n_cramer_fwd_rev': n_cramer_fwd_rev,
        'rev_sig': rev_sig,
    }


# =============================================================================
# DIRECTION 3: SURROGATE HIERARCHY (~50 calls)
# =============================================================================

def direction3_surrogates(analyzer, metric_names, d1):
    """How much sequential structure is linear vs nonlinear?"""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Surrogate Hierarchy")
    print("=" * 78)
    print(f"2 surrogate types × {N_TRIALS} trials = 50 calls")
    print(f"Question: how much structure is linear vs nonlinear?\n")

    n_total = len(metric_names)
    real_metrics = d1['real_metrics']

    # Shuffle comparison: real vs random already done in D1
    # But we need real vs shuffled (same distribution, random order)
    shuf_metrics = defaultdict(list)
    print("  Block-shuffled (block=5)...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_prime_gaps(trial)
        rng = np.random.RandomState(trial + 7000)
        shuffled = block_shuffle(data, block_size=5, rng=rng)
        collect_metrics(analyzer, shuffled, shuf_metrics)
    print("done")

    # IAAFT
    iaaft_metrics = defaultdict(list)
    print("  IAAFT surrogates...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_prime_gaps(trial)
        rng = np.random.RandomState(trial + 8000)
        surrogate = iaaft_surrogate(data, rng)
        collect_metrics(analyzer, surrogate, iaaft_metrics)
    print("done")

    # Full shuffle for comparison
    full_shuf_metrics = d1['dist_metrics']  # dist-matched ≈ full shuffle

    # Count significant differences
    n_vs_full_shuf = d1['n_vs_dist']
    n_vs_block_shuf = count_significant(real_metrics, shuf_metrics,
                                         metric_names, n_total)
    n_vs_iaaft = count_significant(real_metrics, iaaft_metrics,
                                    metric_names, n_total)

    print(f"\n  Real vs Surrogate Hierarchy:")
    print(f"    Full shuffle (dist-matched): {n_vs_full_shuf:>3} sig — all ordering destroyed")
    print(f"    Block-shuffle (block=5):     {n_vs_block_shuf:>3} sig — short-range preserved")
    print(f"    IAAFT:                       {n_vs_iaaft:>3} sig — spectrum+dist preserved")

    if n_vs_full_shuf > 0:
        linear_frac = 1 - (n_vs_iaaft / max(n_vs_full_shuf, 1))
        print(f"\n  Decomposition:")
        print(f"    Total sequential:    {n_vs_full_shuf} sig")
        print(f"    Linear component:    ~{n_vs_full_shuf - n_vs_iaaft} sig ({linear_frac*100:.0f}%)")
        print(f"    Nonlinear component: ~{n_vs_iaaft} sig ({(1-linear_frac)*100:.0f}%)")

    return {
        'n_vs_full_shuf': n_vs_full_shuf,
        'n_vs_block_shuf': n_vs_block_shuf,
        'n_vs_iaaft': n_vs_iaaft,
    }


# =============================================================================
# DIRECTION 4: RESIDUE-CONDITIONED GAP GEOMETRY (~100 calls)
# =============================================================================

def direction4_residue(analyzer, metric_names):
    """Does the Lemke Oliver-Soundararajan bias show up geometrically?"""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Residue-Conditioned Gap Geometry")
    print("=" * 78)
    print(f"4 conditions × {N_TRIALS} trials = 100 calls")
    print(f"Question: does mod-6 residue class affect gap geometry?\n")

    n_total = len(metric_names)

    # Gaps conditioned on p ≡ 1 (mod 6)
    mod1_metrics = defaultdict(list)
    mod1_ref_gaps = []
    print("  Gaps from p ≡ 1 (mod 6)...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_residue_gaps(trial, residue=1, mod=6)
        collect_metrics(analyzer, data, mod1_metrics)
        mod1_ref_gaps.extend(data.tolist())
    mod1_ref_gaps = np.array(mod1_ref_gaps)
    print("done")

    # Gaps conditioned on p ≡ 5 (mod 6)
    mod5_metrics = defaultdict(list)
    mod5_ref_gaps = []
    print("  Gaps from p ≡ 5 (mod 6)...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_residue_gaps(trial, residue=5, mod=6)
        collect_metrics(analyzer, data, mod5_metrics)
        mod5_ref_gaps.extend(data.tolist())
    mod5_ref_gaps = np.array(mod5_ref_gaps)
    print("done")

    # Dist-matched for each residue class
    dist1_metrics = defaultdict(list)
    print("  Dist-matched for mod6=1...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_dist_matched(trial + 3000, mod1_ref_gaps)
        collect_metrics(analyzer, data, dist1_metrics)
    print("done")

    dist5_metrics = defaultdict(list)
    print("  Dist-matched for mod6=5...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_dist_matched(trial + 4000, mod5_ref_gaps)
        collect_metrics(analyzer, data, dist5_metrics)
    print("done")

    # Comparisons
    n_mod1_vs_mod5 = count_significant(mod1_metrics, mod5_metrics,
                                        metric_names, n_total)
    n_mod1_vs_dist1 = count_significant(mod1_metrics, dist1_metrics,
                                         metric_names, n_total)
    n_mod5_vs_dist5 = count_significant(mod5_metrics, dist5_metrics,
                                         metric_names, n_total)

    print(f"\n  Residue comparisons:")
    print(f"    mod6=1 vs mod6=5:          {n_mod1_vs_mod5:>3} sig — ", end="")
    if n_mod1_vs_mod5 > 0:
        print("residue classes are geometrically distinct!")
    else:
        print("no geometric difference detected")
    print(f"    mod6=1 vs its dist-matched: {n_mod1_vs_dist1:>3} sig (sequential structure)")
    print(f"    mod6=5 vs its dist-matched: {n_mod5_vs_dist5:>3} sig (sequential structure)")

    # Per-metric for mod1 vs mod5
    pms_residue = per_metric_significance(mod1_metrics, mod5_metrics,
                                           metric_names, n_total)
    residue_sig = [(km, d, p) for km, (d, p, is_sig) in pms_residue.items() if is_sig]
    residue_sig.sort(key=lambda x: -abs(x[1]))

    if residue_sig:
        print(f"\n  Top residue-distinguishing metrics (mod6=1 vs mod6=5):")
        for km, d, p in residue_sig[:8]:
            fam = metric_to_family(km)
            print(f"    {km:<55} d={d:+8.2f}  [{fam}]")

    # Gap distribution comparison
    print(f"\n  Gap distribution:")
    print(f"    mod6=1 mean gap: {mod1_ref_gaps.mean():.2f}")
    print(f"    mod6=5 mean gap: {mod5_ref_gaps.mean():.2f}")

    return {
        'n_mod1_vs_mod5': n_mod1_vs_mod5,
        'n_mod1_vs_dist1': n_mod1_vs_dist1,
        'n_mod5_vs_dist5': n_mod5_vs_dist5,
        'residue_sig': residue_sig,
    }


# =============================================================================
# DIRECTION 5: SCALE STABILITY OF SURVIVORS (~150 calls)
# =============================================================================

def direction5_scale(analyzer, metric_names, d1):
    """Do the dist-matched survivors persist at all prime scales?"""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Scale Stability of Survivors")
    print("=" * 78)
    print(f"4 scales × {N_TRIALS} × 2 = 200 calls (10K reused from D1)")
    print(f"Question: do survivors persist across scales or are they small-prime artifacts?\n")

    n_total = len(metric_names)
    survivor_names = d1['survivor_names']

    # Scales: prime index → approximate prime value
    scales = {
        'near 1K':   {'idx': 168,    'start_value': 1000},
        'near 10K':  {'idx': 1229,   'start_value': 10000},
        'near 100K': {'idx': 9592,   'start_value': 100000},
        'near 1M':   {'idx': 78498,  'start_value': 1000000},
    }

    scale_results = {}  # scale → n_sig
    scale_survivor_status = {}  # scale → {metric: (d, is_sig)}

    for scale_name, params in scales.items():
        # Check if we can reuse D1 data (10K scale)
        if scale_name == 'near 10K':
            real_m = d1['real_metrics']
            # Need dist-matched for this scale
            ref_gaps = d1['ref_gaps']
            dist_m = d1['dist_metrics']
            print(f"  {scale_name}: reusing D1 data...", end=" ", flush=True)
        else:
            print(f"  {scale_name}...", end=" ", flush=True)
            real_m = defaultdict(list)
            ref_gaps_list = []
            for trial in range(N_TRIALS):
                offset = trial * 50
                data = generate_prime_gaps(trial, start_idx=params['idx'] + offset)
                collect_metrics(analyzer, data, real_m)
                ref_gaps_list.extend(data.tolist())
            ref_gaps = np.array(ref_gaps_list)

            dist_m = defaultdict(list)
            for trial in range(N_TRIALS):
                data = generate_dist_matched(trial + 6000, ref_gaps)
                collect_metrics(analyzer, data, dist_m)

        # Count total sig and per-survivor status
        n_sig = count_significant(real_m, dist_m, metric_names, n_total)
        scale_results[scale_name] = n_sig

        pms = per_metric_significance(real_m, dist_m, metric_names, n_total)
        status = {}
        for km in survivor_names:
            d_val, p_val, is_sig = pms.get(km, (0.0, 1.0, False))
            status[km] = (d_val, is_sig)
        scale_survivor_status[scale_name] = status

        # Mean gap for context
        primes = get_primes_from(params['idx'], min(DATA_SIZE + 1, 2100))
        mean_gap = np.mean(np.diff(primes))
        n_survivors_here = sum(1 for s in status.values() if s[1])
        print(f"{n_sig} total sig, {n_survivors_here}/{len(survivor_names)} survivors persist (mean gap {mean_gap:.1f})")

    # Summary: always-sig survivors
    scale_names = list(scales.keys())
    always_sig = []
    for km in survivor_names:
        if all(scale_survivor_status[s][km][1] for s in scale_names):
            always_sig.append(km)

    print(f"\n  Scale summary:")
    for s in scale_names:
        n_persist = sum(1 for km in survivor_names if scale_survivor_status[s][km][1])
        print(f"    {s:<12} {scale_results[s]:>3} total sig, {n_persist}/{len(survivor_names)} survivors")
    print(f"\n  Always-significant survivors (all 4 scales): {len(always_sig)}/{len(survivor_names)}")
    for km in always_sig:
        fam = metric_to_family(km)
        print(f"    {km:<55} [{fam}]")

    return {
        'scale_results': scale_results,
        'scale_survivor_status': scale_survivor_status,
        'scale_names': scale_names,
        'always_sig': always_sig,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 78)
    print("INVESTIGATION: Deep Prime Gap Sequential Structure")
    print("=" * 78)

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get metric names from a test run
    test_data = np.random.RandomState(0).randint(0, 256, DATA_SIZE, dtype=np.uint8)
    test_result = analyzer.analyze(test_data)
    metric_names = []
    for r in test_result.results:
        for mn in r.metrics:
            metric_names.append(f"{r.geometry_name}:{mn}")
    print(f"Tracking {len(metric_names)} metrics across {len(test_result.results)} geometries\n")

    d1 = direction1_survivors(analyzer, metric_names)
    d2 = direction2_reversibility(analyzer, metric_names, d1)
    d3 = direction3_surrogates(analyzer, metric_names, d1)
    d4 = direction4_residue(analyzer, metric_names)
    d5 = direction5_scale(analyzer, metric_names, d1)

    return metric_names, d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(metric_names, d1, d2, d3, d4, d5):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ── Panel 1 (top-left): D1 — Survivor bar chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)

    survivors = d1['survivors']
    if survivors:
        # Show up to 20 survivors, sorted by |d|
        show = survivors[:20]
        labels = [km.replace(':', ':\n', 1) if len(km) > 35 else km for km, d, p in show]
        d_vals = [d for km, d, p in show]
        colors = [FAMILY_COLORS.get(metric_to_family(km), '#888') for km, d_val, p in show]

        y_pos = np.arange(len(show))
        ax1.barh(y_pos, [abs(d) for d in d_vals], color=colors, alpha=0.85, edgecolor='#333')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=5.5, color=FG)
        ax1.invert_yaxis()
        ax1.set_xlabel('|Cohen\'s d|', color=FG, fontsize=9)

        # Legend for families
        seen = set()
        for km, d_val, p in show:
            fam = metric_to_family(km)
            if fam not in seen:
                ax1.barh([], [], color=FAMILY_COLORS.get(fam, '#888'), label=fam)
                seen.add(fam)
        ax1.legend(fontsize=5.5, facecolor='#222', edgecolor='#444', labelcolor=FG,
                   loc='lower right')
    else:
        ax1.text(0.5, 0.5, 'No survivors', transform=ax1.transAxes,
                 ha='center', va='center', color=FG, fontsize=14)

    ax1.set_title(f'D1: Dist-Matched Survivors ({len(survivors)})',
                  fontsize=11, fontweight='bold', color=FG)

    # ── Panel 2 (top-center): D2 — Time arrow ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)

    bar_labels = ['Real gaps', 'Cramér gaps']
    bar_vals = [d2['n_real_fwd_rev'], d2['n_cramer_fwd_rev']]
    bar_colors = ['#E91E63', '#2196F3']

    bars = ax2.bar(range(len(bar_labels)), bar_vals, color=bar_colors,
                   alpha=0.85, edgecolor='#333', width=0.5)
    ax2.set_xticks(range(len(bar_labels)))
    ax2.set_xticklabels(bar_labels, fontsize=9, color=FG)
    ax2.set_ylabel('Sig metrics (forward vs reversed)', color=FG, fontsize=9)
    for i, v in enumerate(bar_vals):
        ax2.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=11,
                 fontweight='bold')

    # Inset: top discriminating metrics
    if d2['rev_sig']:
        inset_text = "Top time-asymmetric:\n"
        for km, d_val, p in d2['rev_sig'][:5]:
            short = km.split(':')[1] if ':' in km else km
            inset_text += f"  {short}: d={d_val:+.1f}\n"
        ax2.text(0.98, 0.98, inset_text.strip(), transform=ax2.transAxes,
                 fontsize=6, va='top', ha='right', color='#aaa',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#444'))

    ax2.set_title('D2: Arrow of Time', fontsize=11, fontweight='bold', color=FG)

    # ── Panel 3 (top-right): D3 — Surrogate hierarchy ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    surr_labels = ['Full shuffle\n(dist-matched)', 'Block-shuffle\n(block=5)', 'IAAFT\n(nonlinear only)']
    surr_vals = [d3['n_vs_full_shuf'], d3['n_vs_block_shuf'], d3['n_vs_iaaft']]
    surr_colors = ['#E91E63', '#FF9800', '#4CAF50']

    bars3 = ax3.bar(range(len(surr_labels)), surr_vals, color=surr_colors,
                    alpha=0.85, edgecolor='#333', width=0.55)
    ax3.set_xticks(range(len(surr_labels)))
    ax3.set_xticklabels(surr_labels, fontsize=8, color=FG)
    ax3.set_ylabel('Sig metrics (real vs surrogate)', color=FG, fontsize=9)
    for i, v in enumerate(surr_vals):
        ax3.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=11,
                 fontweight='bold')

    # Annotate linear vs nonlinear decomposition
    if d3['n_vs_full_shuf'] > 0:
        n_linear = d3['n_vs_full_shuf'] - d3['n_vs_iaaft']
        ax3.annotate('', xy=(0, d3['n_vs_full_shuf'] * 0.5),
                     xytext=(2, d3['n_vs_iaaft'] + 1),
                     arrowprops=dict(arrowstyle='->', color='#aaa', lw=1))
        ax3.text(1.5, max(surr_vals) * 0.85,
                 f"Linear: ~{n_linear}\nNonlinear: ~{d3['n_vs_iaaft']}",
                 fontsize=8, color='#aaa', ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#444'))

    ax3.set_title('D3: Surrogate Hierarchy', fontsize=11, fontweight='bold', color=FG)

    # ── Panel 4 (bottom-left): D4 — Residue classes ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    res_labels = ['mod6=1\nvs mod6=5', 'mod6=1\nvs dist-match', 'mod6=5\nvs dist-match']
    res_vals = [d4['n_mod1_vs_mod5'], d4['n_mod1_vs_dist1'], d4['n_mod5_vs_dist5']]
    res_colors = ['#9C27B0', '#E91E63', '#2196F3']

    ax4.bar(range(len(res_labels)), res_vals, color=res_colors,
            alpha=0.85, edgecolor='#333', width=0.55)
    ax4.set_xticks(range(len(res_labels)))
    ax4.set_xticklabels(res_labels, fontsize=8, color=FG)
    ax4.set_ylabel('Significant metrics', color=FG, fontsize=9)
    for i, v in enumerate(res_vals):
        ax4.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=11,
                 fontweight='bold')

    ax4.set_title('D4: Residue-Conditioned Geometry', fontsize=11,
                  fontweight='bold', color=FG)

    # ── Panel 5 (bottom-center): D5 — Scale stability ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    scale_names = d5['scale_names']
    scale_labels = [s.replace('near ', '') for s in scale_names]
    total_sig = [d5['scale_results'][s] for s in scale_names]
    survivor_names = d1['survivor_names']

    # Total sig line
    ax5.plot(range(len(scale_names)), total_sig, 's-', color='#FFB300',
             linewidth=2.5, markersize=10, alpha=0.85, label='Total sig vs dist-match',
             zorder=5)
    for i, v in enumerate(total_sig):
        ax5.text(i, v + 0.8, str(v), ha='center', color='#FFB300', fontsize=9,
                 fontweight='bold')

    # Survivor persistence count
    surv_persist = []
    for s in scale_names:
        n_persist = sum(1 for km in survivor_names
                        if d5['scale_survivor_status'][s][km][1])
        surv_persist.append(n_persist)
    ax5.plot(range(len(scale_names)), surv_persist, 'o--', color='#E91E63',
             linewidth=2, markersize=8, alpha=0.85, label='D1 survivors persisting',
             zorder=5)
    for i, v in enumerate(surv_persist):
        ax5.text(i, v - 1.2, str(v), ha='center', color='#E91E63', fontsize=9,
                 fontweight='bold')

    ax5.set_xticks(range(len(scale_names)))
    ax5.set_xticklabels(scale_labels, fontsize=9, color=FG)
    ax5.set_xlabel('Prime scale', color=FG, fontsize=9)
    ax5.set_ylabel('Significant metrics', color=FG, fontsize=9)
    ax5.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    n_always = len(d5['always_sig'])
    ax5.text(0.98, 0.02, f"Always-sig: {n_always}/{len(survivor_names)}",
             transform=ax5.transAxes, fontsize=8, va='bottom', ha='right',
             color='#aaa', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#444'))

    ax5.set_title('D5: Scale Stability', fontsize=11, fontweight='bold', color=FG)

    # ── Panel 6 (bottom-right): Summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    n_total = len(metric_names)
    lines = [
        "Deep Prime Gap Sequential Structure",
        "",
        "D1: Dist-Matched Survivors",
        f"  Real vs Random:       {d1['n_vs_random']:>3} / {n_total} sig",
        f"  Real vs Cramér:       {d1['n_vs_cramer']:>3} / {n_total} sig",
        f"  Real vs Dist-matched: {d1['n_vs_dist']:>3} / {n_total} sig",
        "",
        "D2: Arrow of Time",
        f"  Real fwd vs rev:   {d2['n_real_fwd_rev']:>3} sig",
        f"  Cramér fwd vs rev: {d2['n_cramer_fwd_rev']:>3} sig",
        "",
        "D3: Surrogate Hierarchy",
        f"  vs Full shuffle:   {d3['n_vs_full_shuf']:>3} sig (all ordering)",
        f"  vs Block-shuffle:  {d3['n_vs_block_shuf']:>3} sig (long-range)",
        f"  vs IAAFT:          {d3['n_vs_iaaft']:>3} sig (nonlinear)",
        "",
        "D4: Residue Classes (mod 6)",
        f"  mod6=1 vs mod6=5:    {d4['n_mod1_vs_mod5']:>3} sig",
        f"  mod6=1 vs dist-match:{d4['n_mod1_vs_dist1']:>3} sig",
        f"  mod6=5 vs dist-match:{d4['n_mod5_vs_dist5']:>3} sig",
        "",
        "D5: Scale Stability",
    ]
    for s in d5['scale_names']:
        n_persist = sum(1 for km in d1['survivor_names']
                        if d5['scale_survivor_status'][s][km][1])
        lines.append(f"  {s:<12} {d5['scale_results'][s]:>3} sig, "
                     f"{n_persist}/{len(d1['survivor_names'])} survive")
    lines.append(f"  Always-sig: {len(d5['always_sig'])}/{len(d1['survivor_names'])}")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor='#444'))

    fig.suptitle('Deep Prime Gap Sequential Structure: Beyond Distribution-Matching',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'primes_deep2.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved primes_deep2.png")
    plt.close(fig)


if __name__ == '__main__':
    metric_names, d1, d2, d3, d4, d5 = main()
    make_figure(metric_names, d1, d2, d3, d4, d5)
