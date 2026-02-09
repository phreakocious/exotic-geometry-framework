#!/usr/bin/env python3
"""
Investigation: Geometric Signatures of Classical Arithmetic Functions.

Can exotic geometries detect structure in the fundamental functions of
multiplicative number theory? The key hook: μ(n) being "random" is equivalent
to the Riemann Hypothesis — if geometries see structure in the Mertens function
beyond a random walk, that's remarkable.

8 encodings tested:
  1. divisor_count   — d(n) = number of divisors, clipped to uint8
  2. totient_mod256  — φ(n) mod 256
  3. totient_ratio   — floor(255 × φ(n)/n), encodes "primality ratio"
  4. omega           — Ω(n) (total prime factors w/ multiplicity) × 12
  5. moebius         — μ(n) ∈ {-1,0,1} → {0, 128, 255} (3 values, degenerate)
  6. liouville       — λ(n) = (-1)^Ω(n) → {0, 255} (2 values, degenerate)
  7. mertens_mod256  — M(n) = Σ_{k≤n} μ(k) mod 256 (RH-relevant)
  8. zeta_spacing    — normalized spacings of Riemann zeta zeros (requires mpmath)

Five directions:
  D1: Each encoding vs random (is there detectable structure?)
  D2: All 28 pairwise encoding comparisons
  D3: Classical limit theorem baselines (Erdős–Kac, random walk, GUE)
  D4: Shuffle validation (sequential vs distributional structure)
  D5: Scale sensitivity (small n vs large n)

Budget: ~700 analyzer calls, estimated 5 minutes + sieve precompute.
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
# SIEVE FUNCTIONS
# =============================================================================

def sieve_primes(limit):
    """Sieve of Eratosthenes up to limit. Returns sorted array of primes."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


def sieve_omega(limit):
    """Ω(n): number of prime factors with multiplicity for n=0..limit."""
    omega = np.zeros(limit + 1, dtype=np.int32)
    for p in range(2, limit + 1):
        if omega[p] == 0:  # p is prime
            for multiple in range(p, limit + 1, p):
                n = multiple
                while n % p == 0:
                    omega[multiple] += 1
                    n //= p
    return omega


def sieve_moebius(limit):
    """μ(n): Möbius function for n=0..limit. Returns int8 array in {-1, 0, 1}.

    μ(1)=1, μ(n)=0 if n has squared prime factor, μ(n)=(-1)^k if n is product
    of k distinct primes.
    """
    mu = np.ones(limit + 1, dtype=np.int8)
    mu[0] = 0
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for p in range(2, limit + 1):
        if is_prime[p]:
            # Standard sieve: mark composites (only for p ≤ √limit)
            if p * p <= limit:
                is_prime[p*p::p] = False
            # Flip sign for all multiples of p (one more distinct prime factor)
            mu[p::p] *= -1
            # Zero out multiples of p² (not squarefree)
            if p * p <= limit:
                mu[p*p::p*p] = 0
    return mu


def sieve_divisor_count(limit):
    """d(n): number of divisors for n=0..limit. O(N log N)."""
    d = np.zeros(limit + 1, dtype=np.int32)
    for i in range(1, limit + 1):
        d[i::i] += 1
    return d


def sieve_totient(limit):
    """φ(n): Euler's totient for n=0..limit. O(N log log N)."""
    phi = np.arange(limit + 1, dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for p in range(2, limit + 1):
        if is_prime[p]:
            if p * p <= limit:
                is_prime[p*p::p] = False
            # φ(kp) = φ(kp) * (1 - 1/p) = φ(kp) - φ(kp)//p
            phi[p::p] -= phi[p::p] // p
    return phi


# =============================================================================
# PRECOMPUTATION
# =============================================================================

PRIME_LIMIT = 2_000_000

print("Sieving number-theoretic functions up to {:,}...".format(PRIME_LIMIT))
print("  Primes...", end=" ", flush=True)
ALL_PRIMES = sieve_primes(PRIME_LIMIT)
print(f"{len(ALL_PRIMES)} found")

print("  Ω(n)...", end=" ", flush=True)
OMEGA = sieve_omega(PRIME_LIMIT)
print("done")

print("  μ(n)...", end=" ", flush=True)
MOEBIUS = sieve_moebius(PRIME_LIMIT)
print("done")

print("  d(n)...", end=" ", flush=True)
DIVISOR_COUNTS = sieve_divisor_count(PRIME_LIMIT)
print("done")

print("  φ(n)...", end=" ", flush=True)
TOTIENTS = sieve_totient(PRIME_LIMIT)
print("done")

# Spot-check sieves
assert DIVISOR_COUNTS[12] == 6, f"d(12) should be 6, got {DIVISOR_COUNTS[12]}"
assert TOTIENTS[12] == 4, f"φ(12) should be 4, got {TOTIENTS[12]}"
assert MOEBIUS[12] == 0, f"μ(12) should be 0, got {MOEBIUS[12]}"
assert OMEGA[12] == 3, f"Ω(12) should be 3, got {OMEGA[12]}"
# Mertens spot-check: M(1)=1, M(2)=0, M(3)=-1, M(4)=-1, M(5)=-2
_M = np.cumsum(MOEBIUS[1:6].astype(np.int64))
assert list(_M) == [1, 0, -1, -1, -2], f"Mertens M(1..5) wrong: {list(_M)}"
print("  Spot-checks passed")

# Zeta zeros (optional)
HAS_ZETA = False
ZETA_ZEROS = []
try:
    from mpmath import zetazero
    ZETA_LIMIT = 2200  # ~3-4 min to compute; enough for 25 trials
    print(f"  Zeta zeros (n=1..{ZETA_LIMIT})...", end=" ", flush=True)
    ZETA_ZEROS = [float(zetazero(n).imag) for n in range(1, ZETA_LIMIT + 1)]
    HAS_ZETA = True
    print(f"done ({len(ZETA_ZEROS)} zeros)")
except ImportError:
    print("  mpmath not available — skipping zeta_spacing encoding")
except Exception as e:
    print(f"  Zeta computation failed: {e}")


# =============================================================================
# ENCODING DEFINITIONS
# =============================================================================

SIEVE_ENCODINGS = ['divisor_count', 'totient_mod256', 'totient_ratio', 'omega',
                   'moebius', 'liouville', 'mertens_mod256']

if HAS_ZETA:
    ENCODINGS = SIEVE_ENCODINGS + ['zeta_spacing']
else:
    ENCODINGS = list(SIEVE_ENCODINGS)

ENC_COLORS = {
    'divisor_count': '#E91E63',
    'totient_mod256': '#FF5722',
    'totient_ratio': '#FF9800',
    'omega': '#4CAF50',
    'moebius': '#2196F3',
    'liouville': '#9C27B0',
    'mertens_mod256': '#00BCD4',
    'zeta_spacing': '#FFD54F',
}


# =============================================================================
# ENCODING GENERATORS
# =============================================================================

def generate_number_theory(encoding, trial_seed, size=DATA_SIZE, start_idx=None):
    """Generate a uint8 array of length `size` from a number-theoretic function."""
    if start_idx is None:
        start_idx = 1000 + trial_seed * 137
    end_idx = start_idx + size

    if end_idx > PRIME_LIMIT:
        raise ValueError(f"Need indices up to {end_idx} but PRIME_LIMIT={PRIME_LIMIT}")

    if encoding == 'divisor_count':
        return np.clip(DIVISOR_COUNTS[start_idx:end_idx], 0, 255).astype(np.uint8)

    elif encoding == 'totient_mod256':
        return (TOTIENTS[start_idx:end_idx] % 256).astype(np.uint8)

    elif encoding == 'totient_ratio':
        n_values = np.arange(start_idx, end_idx, dtype=np.float64)
        ratio = TOTIENTS[start_idx:end_idx].astype(np.float64) / n_values
        return np.clip(ratio * 255, 0, 255).astype(np.uint8)

    elif encoding == 'omega':
        return np.clip(OMEGA[start_idx:end_idx] * 12, 0, 255).astype(np.uint8)

    elif encoding == 'moebius':
        mu = MOEBIUS[start_idx:end_idx]
        result = np.full(size, 128, dtype=np.uint8)
        result[mu == -1] = 0
        result[mu == 1] = 255
        return result

    elif encoding == 'liouville':
        return np.where(OMEGA[start_idx:end_idx] % 2 == 0, 255, 0).astype(np.uint8)

    elif encoding == 'mertens_mod256':
        # M(n) = Σ_{k=1}^{n} μ(k). Precompute cumulative sum from μ[1].
        mu_cumsum = np.cumsum(MOEBIUS[1:end_idx + 1].astype(np.int64))
        # mu_cumsum[i] = M(i+1). M(n) = mu_cumsum[n-1].
        mertens = mu_cumsum[start_idx - 1:end_idx - 1]
        return (mertens % 256).astype(np.uint8)

    elif encoding == 'zeta_spacing':
        if not HAS_ZETA:
            raise ValueError("mpmath not available for zeta_spacing")
        # Small offsets: 2200 zeros supports 25 trials × offset 8
        offset = trial_seed * 8
        if offset + size + 1 > len(ZETA_ZEROS):
            offset = max(0, len(ZETA_ZEROS) - size - 1)
        zeros = ZETA_ZEROS[offset:offset + size + 1]
        spacings = np.diff(zeros)
        mean_spacing = np.mean(spacings) if len(spacings) > 0 else 1.0
        normalized = spacings / max(mean_spacing, 1e-10)
        return np.clip(normalized * 128, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown encoding: {encoding}")


# =============================================================================
# CLASSICAL MODEL GENERATORS (D3)
# =============================================================================

def generate_erdos_kac_model(trial_seed, size=DATA_SIZE, start_idx=None):
    """Erdős–Kac: Ω(n) ~ Normal(ln ln n, √(ln ln n))."""
    rng = np.random.RandomState(trial_seed)
    if start_idx is None:
        start_idx = 1000 + trial_seed * 137
    n_values = np.arange(start_idx, start_idx + size, dtype=np.float64)
    mean_omega = np.log(np.log(n_values))
    std_omega = np.sqrt(np.clip(np.log(np.log(n_values)), 0.1, None))
    omega_model = rng.normal(mean_omega, std_omega)
    return np.clip(np.round(omega_model) * 12, 0, 255).astype(np.uint8)


def generate_mertens_walk(trial_seed, size=DATA_SIZE):
    """Random walk matching μ(n) step distribution.
    P(step=+1) = P(step=-1) = 3/π², P(step=0) = 1 - 6/π².
    """
    rng = np.random.RandomState(trial_seed)
    p_squarefree = 6.0 / (np.pi ** 2)  # ~0.6079
    steps = np.zeros(size, dtype=np.int64)
    is_squarefree = rng.random(size) < p_squarefree
    signs = rng.choice([-1, 1], size=size)
    steps[is_squarefree] = signs[is_squarefree]
    walk = np.cumsum(steps)
    return (walk % 256).astype(np.uint8)


def generate_divisor_matched(trial_seed, ref_divisors, size=DATA_SIZE):
    """I.I.D. sample from empirical d(n) distribution."""
    rng = np.random.RandomState(trial_seed)
    return rng.choice(ref_divisors, size=size, replace=True)


def generate_gue_spacings(trial_seed, size=DATA_SIZE):
    """Wigner surmise: P(s) = (π/2) s exp(-πs²/4).
    Inverse CDF: s = √(-4/π × ln(1-U)).
    """
    rng = np.random.RandomState(trial_seed)
    u = rng.uniform(0, 1, size=size)
    spacings = np.sqrt(-4.0 / np.pi * np.log(1.0 - u))
    return np.clip(spacings * 128, 0, 255).astype(np.uint8)


# =============================================================================
# STATISTICS
# =============================================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


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
# DIRECTION 1: ENCODINGS vs RANDOM
# =============================================================================

def direction1_vs_random(analyzer, metric_names):
    """Can exotic geometries detect structure in arithmetic functions?"""
    print("\n" + "=" * 78)
    print("DIRECTION 1: Arithmetic Functions vs Random")
    print("=" * 78)
    print(f"{len(ENCODINGS)} encodings, {N_TRIALS} trials each, vs os.urandom baseline")
    print(f"Question: which functions carry detectable geometric structure?\n")

    all_metrics = {}

    # Random baseline
    print("  Random baseline...", end=" ", flush=True)
    all_metrics['random'] = defaultdict(list)
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial + 5000)
        data = rng.randint(0, 256, DATA_SIZE, dtype=np.uint8)
        collect_metrics(analyzer, data, all_metrics['random'])
    print("done")

    # Each encoding
    for enc in ENCODINGS:
        print(f"  {enc}...", end=" ", flush=True)
        all_metrics[enc] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_number_theory(enc, trial)
            collect_metrics(analyzer, data, all_metrics[enc])
        print("done")

    # Count significant metrics for each encoding vs random
    n_total = len(metric_names)
    results = {}
    print(f"\n  {'Encoding':<20}  {'Sig metrics':>11}")
    print(f"  {'-'*20}  {'-'*11}")
    for enc in ENCODINGS:
        n_sig = count_significant(all_metrics[enc], all_metrics['random'],
                                  metric_names, n_total)
        results[enc] = n_sig
        print(f"  {enc:<20}  {n_sig:>11}")

    # Top 5 metrics per encoding (skip degenerate ones for brevity)
    for enc in ENCODINGS:
        if enc in ('moebius', 'liouville'):
            continue
        pms = per_metric_significance(all_metrics[enc], all_metrics['random'],
                                      metric_names, n_total)
        sig_list = [(km, d, p) for km, (d, p, is_sig) in pms.items() if is_sig]
        sig_list.sort(key=lambda x: -abs(x[1]))
        if sig_list:
            print(f"\n  {enc} top 5:")
            for km, d, p in sig_list[:5]:
                print(f"    {km:<45} d={d:+8.2f}  p={p:.2e}")

    return {'all_metrics': all_metrics, 'results': results}


# =============================================================================
# DIRECTION 2: PAIRWISE ENCODING COMPARISONS
# =============================================================================

def direction2_pairwise(metric_names, d1):
    """Can different arithmetic function encodings be distinguished?"""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Pairwise Encoding Comparisons")
    print("=" * 78)
    n_pairs = len(ENCODINGS) * (len(ENCODINGS) - 1) // 2
    print(f"{n_pairs} pairs from {len(ENCODINGS)} encodings (reusing D1 data, 0 extra calls)")
    print(f"Question: do different functions capture different structure?\n")

    all_metrics = d1['all_metrics']
    n_total = len(metric_names)
    pair_results = []

    for i, enc1 in enumerate(ENCODINGS):
        for enc2 in ENCODINGS[i+1:]:
            n_sig = count_significant(all_metrics[enc1], all_metrics[enc2],
                                      metric_names, n_total)
            pair_results.append((enc1, enc2, n_sig))

    # Print matrix
    print(f"  {'':>18}", end="")
    for enc in ENCODINGS:
        print(f"  {enc[:7]:>7}", end="")
    print()

    pair_lookup = {}
    for enc1, enc2, n_sig in pair_results:
        pair_lookup[(enc1, enc2)] = n_sig
        pair_lookup[(enc2, enc1)] = n_sig

    for enc1 in ENCODINGS:
        print(f"  {enc1:<18}", end="")
        for enc2 in ENCODINGS:
            if enc1 == enc2:
                print(f"  {'—':>7}", end="")
            else:
                n = pair_lookup.get((enc1, enc2), 0)
                print(f"  {n:>7}", end="")
        print()

    return {'pair_results': pair_results}


# =============================================================================
# DIRECTION 3: CLASSICAL LIMIT THEOREM BASELINES
# =============================================================================

def direction3_baselines(analyzer, metric_names, d1):
    """Can classical probabilistic models explain the geometric signatures?"""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Classical Limit Theorem Baselines")
    print("=" * 78)
    print(f"Compare each function to its classical probabilistic model")
    print(f"Question: is there structure beyond what the theorems predict?\n")

    all_metrics = d1['all_metrics']
    n_total = len(metric_names)

    # Build reference divisor array for distribution-matched model
    ref_divs = np.clip(DIVISOR_COUNTS[1000:1000 + DATA_SIZE * N_TRIALS],
                       0, 255).astype(np.uint8)

    # Define models: {name: (real_encoding, generator)}
    models = [
        ('Erdős–Kac (Ω)', 'omega', lambda t: generate_erdos_kac_model(t)),
        ('Rand Walk (M)', 'mertens_mod256', lambda t: generate_mertens_walk(t)),
        ('Dist-match (d)', 'divisor_count', lambda t: generate_divisor_matched(t, ref_divs)),
    ]
    if HAS_ZETA:
        models.append(('Wigner (ζ)', 'zeta_spacing', lambda t: generate_gue_spacings(t)))

    # Collect model metrics
    model_metrics = {}
    for model_name, real_enc, gen_fn in models:
        print(f"  {model_name}...", end=" ", flush=True)
        model_metrics[model_name] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = gen_fn(trial)
            collect_metrics(analyzer, data, model_metrics[model_name])
        print("done")

    # Compare each model to its real encoding AND to random
    print(f"\n  {'Model':<20} {'vs Real':>8} {'vs Random':>10}  Interpretation")
    print(f"  {'-'*20} {'-'*8} {'-'*10}  {'-'*30}")
    model_results = {}
    for model_name, real_enc, _ in models:
        n_vs_real = count_significant(model_metrics[model_name], all_metrics[real_enc],
                                      metric_names, n_total)
        n_vs_rand = count_significant(model_metrics[model_name], all_metrics['random'],
                                      metric_names, n_total)
        model_results[model_name] = {
            'vs_real': n_vs_real,
            'vs_random': n_vs_rand,
            'real_encoding': real_enc,
        }
        if n_vs_real == 0:
            interp = "theorem fully explains"
        elif n_vs_real < 10:
            interp = "mostly explained"
        else:
            interp = f"structure BEYOND theorem"
        print(f"  {model_name:<20} {n_vs_real:>8} {n_vs_rand:>10}  {interp}")

    # Top discriminating metrics for models with vs_real > 0
    for model_name, real_enc, _ in models:
        if model_results[model_name]['vs_real'] > 0:
            pms = per_metric_significance(model_metrics[model_name], all_metrics[real_enc],
                                          metric_names, n_total)
            sig_list = [(km, d, p) for km, (d, p, is_sig) in pms.items() if is_sig]
            sig_list.sort(key=lambda x: -abs(x[1]))
            if sig_list:
                print(f"\n  {model_name} vs Real — top metrics beyond theorem:")
                for km, d, p in sig_list[:5]:
                    print(f"    {km:<45} d={d:+8.2f}")

    return {
        'model_metrics': model_metrics,
        'model_results': model_results,
        'models': [(n, e) for n, e, _ in models],
    }


# =============================================================================
# DIRECTION 4: SHUFFLE VALIDATION
# =============================================================================

def direction4_shuffle(analyzer, metric_names, d1):
    """Which arithmetic function structures depend on ordering vs distribution?"""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Shuffle Validation")
    print("=" * 78)
    print(f"Shuffle each encoding and re-analyze. Ordering effects: destroyed by shuffle.")
    print(f"Multiplicative functions on consecutive n share factors → ordering matters?\n")

    all_metrics = d1['all_metrics']
    n_total = len(metric_names)

    shuffle_metrics = {}
    for enc in ENCODINGS:
        print(f"  Shuffling {enc}...", end=" ", flush=True)
        shuffle_metrics[enc] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_number_theory(enc, trial)
            rng = np.random.RandomState(trial + 9000)
            rng.shuffle(data)
            collect_metrics(analyzer, data, shuffle_metrics[enc])
        print("done")

    print(f"\n  {'Encoding':<20}  {'orig vs rand':>12}  {'shuf vs rand':>12}  {'orig vs shuf':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}")

    results = {}
    for enc in ENCODINGS:
        n_orig_rand = d1['results'][enc]
        n_shuf_rand = count_significant(shuffle_metrics[enc], all_metrics['random'],
                                        metric_names, n_total)
        n_orig_shuf = count_significant(all_metrics[enc], shuffle_metrics[enc],
                                        metric_names, n_total)
        results[enc] = {
            'orig_vs_rand': n_orig_rand,
            'shuf_vs_rand': n_shuf_rand,
            'orig_vs_shuf': n_orig_shuf,
        }
        label = ""
        if n_orig_shuf > 5:
            label = " ← ORDERING"
        elif n_orig_rand > 5 and n_orig_shuf <= 5:
            label = " ← distributional"
        print(f"  {enc:<20}  {n_orig_rand:>12}  {n_shuf_rand:>12}  {n_orig_shuf:>12}{label}")

    return {'shuffle_metrics': shuffle_metrics, 'results': results}


# =============================================================================
# DIRECTION 5: SCALE SENSITIVITY
# =============================================================================

def direction5_scale(analyzer, metric_names, d1):
    """Do geometric signatures evolve for larger n?"""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Scale Sensitivity")
    print("=" * 78)
    print(f"d(n) and Ω(n) at 4 ranges (near 1K, 10K, 100K, 1M)")
    print(f"Question: do signatures change as n grows?\n")

    ranges = {
        'near 1K': 1000,
        'near 10K': 10000,
        'near 100K': 100000,
        'near 1M': 1000000,
    }
    focus_encodings = ['divisor_count', 'omega']

    range_metrics = {}
    for rn, start in ranges.items():
        for enc in focus_encodings:
            print(f"  {enc} {rn}...", end=" ", flush=True)
            key = (rn, enc)
            range_metrics[key] = defaultdict(list)
            for trial in range(N_TRIALS):
                offset = trial * 137
                data = generate_number_theory(enc, trial, start_idx=start + offset)
                collect_metrics(analyzer, data, range_metrics[key])
            print("done")

    n_total = len(metric_names)

    # Each range vs random
    print(f"\n  {'Range':<12} {'Encoding':<18} {'vs random':>10}")
    print(f"  {'-'*12} {'-'*18} {'-'*10}")
    scale_results = {}
    for rn in ranges:
        for enc in focus_encodings:
            key = (rn, enc)
            n_sig = count_significant(range_metrics[key], d1['all_metrics']['random'],
                                      metric_names, n_total)
            scale_results[key] = n_sig
            print(f"  {rn:<12} {enc:<18} {n_sig:>10}")

    # Pairwise range comparisons for each encoding
    range_names = list(ranges.keys())
    pair_results = {}
    for enc in focus_encodings:
        print(f"\n  Pairwise range comparisons ({enc}):")
        pairs = []
        for i, rn1 in enumerate(range_names):
            for rn2 in range_names[i+1:]:
                n_sig = count_significant(range_metrics[(rn1, enc)],
                                          range_metrics[(rn2, enc)],
                                          metric_names, n_total)
                pairs.append((rn1, rn2, n_sig))
                print(f"    {rn1:<12} vs {rn2:<12}  {n_sig:>3} sig")
        pair_results[enc] = pairs

    return {
        'range_metrics': range_metrics,
        'scale_results': scale_results,
        'pair_results': pair_results,
        'focus_encodings': focus_encodings,
        'range_names': range_names,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 78)
    print("INVESTIGATION: Geometric Signatures of Classical Arithmetic Functions")
    print("=" * 78)

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get metric names from a test run
    test_data = np.random.RandomState(0).randint(0, 256, DATA_SIZE, dtype=np.uint8)
    test_result = analyzer.analyze(test_data)
    metric_names = []
    for r in test_result.results:
        for mn in r.metrics:
            metric_names.append(f"{r.geometry_name}:{mn}")
    print(f"Tracking {len(metric_names)} metrics across {len(test_result.results)} geometries")
    print(f"Encodings: {len(ENCODINGS)} ({', '.join(ENCODINGS)})\n")

    d1 = direction1_vs_random(analyzer, metric_names)
    d2 = direction2_pairwise(metric_names, d1)
    d3 = direction3_baselines(analyzer, metric_names, d1)
    d4 = direction4_shuffle(analyzer, metric_names, d1)
    d5 = direction5_scale(analyzer, metric_names, d1)

    return metric_names, d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(metric_names, d1, d2, d3, d4, d5):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ── Panel 1 (top-left): D1 — Detection strength per encoding ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)

    enc_names = ENCODINGS
    det_counts = [d1['results'][enc] for enc in enc_names]
    colors = [ENC_COLORS[enc] for enc in enc_names]
    ax1.barh(range(len(enc_names)), det_counts, color=colors,
             alpha=0.85, edgecolor='#333')
    ax1.set_yticks(range(len(enc_names)))
    ax1.set_yticklabels([e.replace('_', ' ') for e in enc_names], fontsize=8, color=FG)
    ax1.set_xlabel('Significant metrics vs random', color=FG, fontsize=9)
    ax1.set_title('D1: Detection Strength', fontsize=11, fontweight='bold', color=FG)
    ax1.invert_yaxis()
    for i, v in enumerate(det_counts):
        ax1.text(v + 0.5, i, str(v), va='center', color=FG, fontsize=8, fontweight='bold')

    # ── Panel 2 (top-center): D2 — Pairwise heatmap ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)

    n_enc = len(ENCODINGS)
    mat = np.zeros((n_enc, n_enc))
    pair_lookup = {}
    for enc1, enc2, n_sig in d2['pair_results']:
        pair_lookup[(enc1, enc2)] = n_sig
        pair_lookup[(enc2, enc1)] = n_sig
    for i, enc1 in enumerate(ENCODINGS):
        for j, enc2 in enumerate(ENCODINGS):
            if i != j:
                mat[i, j] = pair_lookup.get((enc1, enc2), 0)

    im = ax2.imshow(mat, cmap='YlOrRd', interpolation='nearest', vmin=0)
    ax2.set_xticks(range(n_enc))
    ax2.set_yticks(range(n_enc))
    short = [e[:7] for e in ENCODINGS]
    ax2.set_xticklabels(short, fontsize=6, color=FG, rotation=45, ha='right')
    ax2.set_yticklabels(short, fontsize=6, color=FG)
    for i in range(n_enc):
        for j in range(n_enc):
            if i != j:
                ax2.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                         fontsize=6, fontweight='bold',
                         color='white' if mat[i, j] > mat.max() * 0.5 else FG)
    ax2.set_title('D2: Pairwise Matrix', fontsize=11, fontweight='bold', color=FG)
    cb = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cb.set_label('Sig metrics', color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG)

    # ── Panel 3 (top-right): D3 — Classical baselines ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    model_names = [n for n, _ in d3['models']]
    x_pos = np.arange(len(model_names))
    bar_w = 0.35
    vs_real = [d3['model_results'][m]['vs_real'] for m in model_names]
    vs_rand = [d3['model_results'][m]['vs_random'] for m in model_names]

    ax3.bar(x_pos - bar_w/2, vs_real, bar_w, label='vs Real',
            color='#E91E63', alpha=0.85, edgecolor='#333')
    ax3.bar(x_pos + bar_w/2, vs_rand, bar_w, label='vs Random',
            color='#2196F3', alpha=0.85, edgecolor='#333')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([m.replace(' ', '\n') for m in model_names],
                        fontsize=7, color=FG)
    ax3.set_ylabel('Significant metrics', color=FG, fontsize=9)
    ax3.set_title('D3: Classical Baselines', fontsize=11, fontweight='bold', color=FG)
    ax3.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)
    for i, v in enumerate(vs_real):
        ax3.text(i - bar_w/2, v + 0.5, str(v), ha='center', color='#E91E63',
                 fontsize=8, fontweight='bold')
    for i, v in enumerate(vs_rand):
        ax3.text(i + bar_w/2, v + 0.5, str(v), ha='center', color='#2196F3',
                 fontsize=8, fontweight='bold')

    # ── Panel 4 (bottom-left): D4 — Shuffle validation ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    x = np.arange(len(ENCODINGS))
    bw = 0.25
    orig_vals = [d4['results'][enc]['orig_vs_rand'] for enc in ENCODINGS]
    shuf_vals = [d4['results'][enc]['shuf_vs_rand'] for enc in ENCODINGS]
    diff_vals = [d4['results'][enc]['orig_vs_shuf'] for enc in ENCODINGS]

    ax4.bar(x - bw, orig_vals, bw, label='orig vs rand',
            color='#E91E63', alpha=0.85, edgecolor='#333')
    ax4.bar(x, shuf_vals, bw, label='shuf vs rand',
            color='#FF9800', alpha=0.85, edgecolor='#333')
    ax4.bar(x + bw, diff_vals, bw, label='orig vs shuf',
            color='#2196F3', alpha=0.85, edgecolor='#333')
    ax4.set_xticks(x)
    ax4.set_xticklabels([e[:7] for e in ENCODINGS], fontsize=7, color=FG,
                        rotation=30, ha='right')
    ax4.set_ylabel('Sig metrics', color=FG, fontsize=9)
    ax4.set_title('D4: Shuffle Validation', fontsize=11, fontweight='bold', color=FG)
    ax4.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    # ── Panel 5 (bottom-center): D5 — Scale sensitivity ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    range_names = d5['range_names']
    focus_enc = d5['focus_encodings']
    line_colors = ['#E91E63', '#4CAF50']

    for idx, enc in enumerate(focus_enc):
        vals = [d5['scale_results'][(rn, enc)] for rn in range_names]
        ax5.plot(range(len(range_names)), vals, 'o-', color=line_colors[idx],
                 linewidth=2, markersize=8, alpha=0.85, label=enc.replace('_', ' '))
        for i, v in enumerate(vals):
            ax5.text(i, v + 0.5, str(v), ha='center', color=line_colors[idx],
                     fontsize=8, fontweight='bold')

    ax5.set_xticks(range(len(range_names)))
    ax5.set_xticklabels(range_names, fontsize=8, color=FG)
    ax5.set_xlabel('Range', color=FG, fontsize=9)
    ax5.set_ylabel('Sig metrics vs random', color=FG, fontsize=9)
    ax5.set_title('D5: Scale Sensitivity', fontsize=11, fontweight='bold', color=FG)
    ax5.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    # ── Panel 6 (bottom-right): Summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    lines = [
        "Arithmetic Functions — Key Findings",
        "",
        "D1: Encodings vs Random",
    ]
    for enc in ENCODINGS:
        v = d1['results'][enc]
        lines.append(f"  {enc:<18} {v:>3} sig")

    lines.append("")
    lines.append("D3: Classical Baselines (vs Real)")
    for model_name, _ in d3['models']:
        v = d3['model_results'][model_name]['vs_real']
        lines.append(f"  {model_name:<18} {v:>3} sig")

    lines.append("")
    lines.append("D4: Ordering vs Distribution")
    n_ordering = sum(1 for enc in ENCODINGS if d4['results'][enc]['orig_vs_shuf'] > 5)
    n_distrib = sum(1 for enc in ENCODINGS
                    if d4['results'][enc]['orig_vs_rand'] > 5
                    and d4['results'][enc]['orig_vs_shuf'] <= 5)
    lines.append(f"  {n_ordering} ordering-dependent")
    lines.append(f"  {n_distrib} purely distributional")

    lines.append("")
    lines.append("D5: Scale Evolution")
    for enc in focus_enc:
        line = f"  {enc[:12]}:"
        for rn in range_names:
            v = d5['scale_results'][(rn, enc)]
            line += f" {v}"
        lines.append(line)

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor='#444'))

    fig.suptitle('Classical Arithmetic Functions: Exotic Geometry Analysis',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'number_theory.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved number_theory.png")
    plt.close(fig)


if __name__ == '__main__':
    metric_names, d1, d2, d3, d4, d5 = main()
    make_figure(metric_names, d1, d2, d3, d4, d5)
