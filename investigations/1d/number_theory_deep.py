#!/usr/bin/env python3
"""
Investigation: Beyond Multiplicative Number Theory — Additive & Structural Sequences.

The first number_theory.py covered multiplicative functions (d(n), φ(n), μ(n), ζ zeros).
This investigation explores additive and structural number theory:

8 new encodings:
  1. partition_fn    — p(n), number of integer partitions (Ramanujan territory)
  2. cf_sqrt2        — continued fraction coefficients of √2: [1; 2, 2, 2, ...] (periodic)
  3. cf_e            — continued fraction of e: [2; 1, 2, 1, 1, 4, 1, 1, 6, ...] (patterned)
  4. cf_pi           — continued fraction of π: [3; 7, 15, 1, 292, ...] (appears random)
  5. stern_brocot    — Stern's diatomic sequence s(n), enumerates rationals
  6. sum_of_digits   — s₁₀(n) = digit sum in base 10
  7. fibonacci_mod   — Fibonacci numbers mod 256 (Pisano period π(256) = 1536)
  8. sigma_ratio     — σ(n)/n in [0,255], encodes abundant/perfect/deficient

Five directions:
  D1: Each encoding vs random (is there detectable structure?)
  D2: Algebraic vs transcendental continued fractions (√2, e vs π, random)
  D3: Partition function vs Hardy-Ramanujan model
  D4: Shuffle validation (sequential vs distributional structure)
  D5: Cross-comparison with multiplicative functions (d(n), Ω(n), Mertens)

Methodology: N_TRIALS=25, DATA_SIZE=2000, Cohen's d > 0.8, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05

# --- Discover metric names ---
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_ALPHA = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn

print(f"1D metrics: {N_METRICS}, Bonferroni α={BONF_ALPHA:.2e}")


# =========================================================================
# PRECOMPUTATION
# =========================================================================

def compute_partitions_mod256(limit):
    """Compute p(n) mod 256 for n=0..limit using DP.
    Inner loop has data dependencies (uses updated values), so must be scalar."""
    p = [0] * (limit + 1)
    p[0] = 1
    for k in range(1, limit + 1):
        for n in range(k, limit + 1):
            p[n] = (p[n] + p[n - k]) & 0xFF
    return np.array(p, dtype=np.uint8)


def compute_cf_coefficients(value_str, n_coeffs):
    """Compute continued fraction coefficients using mpmath for high precision."""
    try:
        from mpmath import mp, mpf, floor as mpfloor
        mp.dps = max(n_coeffs * 2, 1000)  # 2 digits per coefficient
        if value_str == 'sqrt2':
            x = mp.sqrt(2)
        elif value_str == 'e':
            x = mp.e
        elif value_str == 'pi':
            x = mp.pi
        elif value_str == 'golden':
            x = (1 + mp.sqrt(5)) / 2
        else:
            raise ValueError(f"Unknown constant: {value_str}")

        coeffs = []
        for _ in range(n_coeffs):
            a = int(mpfloor(x))
            coeffs.append(a)
            frac = x - a
            if frac < mpf('1e-50'):
                break
            x = 1 / frac
        return coeffs
    except ImportError:
        # Fallback: compute from known patterns
        if value_str == 'sqrt2':
            return [1] + [2] * (n_coeffs - 1)
        elif value_str == 'e':
            # e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...]
            coeffs = [2]
            k = 1
            while len(coeffs) < n_coeffs:
                coeffs.extend([1, 2*k, 1])
                k += 1
            return coeffs[:n_coeffs]
        elif value_str == 'pi':
            # Hardcode first ~50 known coefficients of pi
            pi_cf = [3,7,15,1,292,1,1,1,2,1,3,1,14,2,1,1,2,2,2,2,1,84,2,1,1,
                     15,3,13,1,4,2,6,6,99,1,2,2,6,3,5,1,1,6,8,1,7,1,2,3,7]
            if n_coeffs > len(pi_cf):
                print(f"  Warning: Only {len(pi_cf)} pi CF coeffs without mpmath")
            return pi_cf[:n_coeffs]
        elif value_str == 'golden':
            return [1] * n_coeffs
        raise


def stern_brocot_sequence(n):
    """Stern's diatomic sequence: s(0)=0, s(1)=1, s(2k)=s(k), s(2k+1)=s(k)+s(k+1)."""
    s = np.zeros(n, dtype=np.int64)
    if n > 1:
        s[1] = 1
    for i in range(2, n):
        if i % 2 == 0:
            s[i] = s[i // 2]
        else:
            s[i] = s[i // 2] + s[i // 2 + 1]
    return s


def sieve_sigma(limit):
    """σ(n): sum of divisors for n=1..limit."""
    sigma = np.zeros(limit + 1, dtype=np.int64)
    for d in range(1, limit + 1):
        sigma[d::d] += d
    return sigma


print("Precomputing sequences...")

# Partition function (mod 256 to avoid overflow, O(n²) so keep limit modest)
PART_LIMIT = 10000
print("  Partition function (mod 256)...", end=" ", flush=True)
PARTITIONS = compute_partitions_mod256(PART_LIMIT)
print(f"done ({PART_LIMIT} terms)")

# Continued fractions
CF_N = 3000  # enough for 25 trials with offset stride 50
print("  Continued fractions...", end=" ", flush=True)
CF_SQRT2 = compute_cf_coefficients('sqrt2', CF_N)
CF_E = compute_cf_coefficients('e', CF_N)
CF_PI = compute_cf_coefficients('pi', CF_N)
print(f"√2: {len(CF_SQRT2)}, e: {len(CF_E)}, π: {len(CF_PI)} coefficients")

# Stern-Brocot
print("  Stern-Brocot...", end=" ", flush=True)
SB_LIMIT = 2_000_000
STERN = stern_brocot_sequence(SB_LIMIT)
print(f"done ({SB_LIMIT} terms)")

# Sum of divisors
print("  σ(n)...", end=" ", flush=True)
SIGMA = sieve_sigma(2_000_000)
print("done")

print("  Spot-checks:")
assert PARTITIONS[5] == 7, f"p(5) should be 7, got {PARTITIONS[5]}"
assert PARTITIONS[100] == 76, f"p(100) mod 256 should be 76, got {PARTITIONS[100]}"
assert STERN[7] == 3, f"s(7) should be 3, got {STERN[7]}"
assert SIGMA[12] == 28, f"σ(12) should be 28, got {SIGMA[12]}"  # 1+2+3+4+6+12
print("  Passed")


# =========================================================================
# ENCODING GENERATORS
# =========================================================================

def gen_partition_fn(trial, size):
    """p(n) mod 256 starting from random offset."""
    rng = np.random.default_rng(42 + trial)
    start = rng.integers(100, PART_LIMIT - size)
    return PARTITIONS[start:start + size].copy()


def gen_cf_sqrt2(trial, size):
    """Continued fraction coefficients of √2, clipped to uint8."""
    start = trial * 50  # shift window per trial
    coeffs = CF_SQRT2[start:start + size]
    return np.clip(coeffs, 0, 255).astype(np.uint8)[:size]


def gen_cf_e(trial, size):
    """Continued fraction coefficients of e, clipped to uint8."""
    start = trial * 50
    coeffs = CF_E[start:start + size]
    return np.clip(coeffs, 0, 255).astype(np.uint8)[:size]


def gen_cf_pi(trial, size):
    """Continued fraction coefficients of π, clipped to uint8."""
    # We may not have enough for all trials at full size
    n_avail = len(CF_PI)
    if n_avail < size:
        # Repeat with different offsets
        rng = np.random.default_rng(42 + trial)
        idx = rng.permutation(n_avail)[:size]
        coeffs = [CF_PI[i] for i in idx]
    else:
        start = min(trial * 50, max(0, n_avail - size))
        coeffs = CF_PI[start:start + size]
    return np.clip(coeffs, 0, 255).astype(np.uint8)


def gen_stern_brocot(trial, size):
    """Stern's diatomic sequence mod 256."""
    rng = np.random.default_rng(42 + trial)
    start = rng.integers(1000, SB_LIMIT - size)
    return (STERN[start:start + size] % 256).astype(np.uint8)


def gen_digit_sum(trial, size):
    """Sum of digits (base 10), clipped to uint8."""
    rng = np.random.default_rng(42 + trial)
    start = rng.integers(1000, 1_900_000)
    vals = np.array([sum(int(c) for c in str(n))
                     for n in range(start, start + size)])
    return np.clip(vals, 0, 255).astype(np.uint8)


def gen_fibonacci_mod(trial, size):
    """Fibonacci numbers mod 256. Pisano period π(256) = 1536."""
    # Start from a shifted position
    a, b = 0, 1
    skip = trial * 60
    for _ in range(skip):
        a, b = b, (a + b) % 256
    vals = []
    for _ in range(size):
        vals.append(a)
        a, b = b, (a + b) % 256
    return np.array(vals, dtype=np.uint8)


def gen_sigma_ratio(trial, size):
    """σ(n)/n mapped to [0, 255]. σ(n)/n = 1 for primes, 2 for perfect numbers."""
    rng = np.random.default_rng(42 + trial)
    start = rng.integers(1000, 1_900_000)
    ns = np.arange(start, start + size, dtype=np.float64)
    ratios = SIGMA[start:start + size].astype(np.float64) / ns
    # Map [1, 4] → [0, 255] (most values are in [1, 3])
    mapped = np.clip((ratios - 1.0) * 85, 0, 255)
    return mapped.astype(np.uint8)


def gen_random(trial, size):
    return np.random.default_rng(9000 + trial).integers(0, 256, size, dtype=np.uint8)


ENCODINGS = {
    'partition_fn':  gen_partition_fn,
    'cf_sqrt2':      gen_cf_sqrt2,
    'cf_e':          gen_cf_e,
    'cf_pi':         gen_cf_pi,
    'stern_brocot':  gen_stern_brocot,
    'digit_sum':     gen_digit_sum,
    'fibonacci_mod': gen_fibonacci_mod,
    'sigma_ratio':   gen_sigma_ratio,
}


# =========================================================================
# ANALYSIS UTILITIES
# =========================================================================

def collect_metrics(analyzer, data_arrays):
    out = {m: [] for m in METRIC_NAMES}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def compare(data_a, data_b):
    sig = 0
    findings = []
    for m in METRIC_NAMES:
        a = np.array(data_a[m])
        b = np.array(data_b[m])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < BONF_ALPHA and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# D1: Each encoding vs random
# =========================================================================
def direction_1(analyzer):
    print("\n" + "=" * 78)
    print("D1: EACH ENCODING VS RANDOM")
    print("=" * 78)

    random_arrays = [gen_random(t, DATA_SIZE) for t in range(N_TRIALS)]
    random_data = collect_metrics(analyzer, random_arrays)

    all_data = {}
    d1_results = {}

    for name, gen_fn in ENCODINGS.items():
        print(f"  {name:18s}...", end=" ", flush=True)
        arrays = [gen_fn(t, DATA_SIZE) for t in range(N_TRIALS)]

        # Quick stats on the encoding
        sample = arrays[0]
        ent_vals = np.bincount(sample, minlength=256)
        ent_p = ent_vals[ent_vals > 0] / len(sample)
        entropy = float(-np.sum(ent_p * np.log2(ent_p)))
        n_distinct = len(np.unique(sample))

        data = collect_metrics(analyzer, arrays)
        all_data[name] = data

        n_sig, findings = compare(data, random_data)
        d1_results[name] = n_sig
        print(f"{n_sig:3d} sig  (H={entropy:.1f} bits, {n_distinct} distinct)")
        for m, d, p in findings[:3]:
            print(f"    {m:45s}  d={d:+8.2f}")

    return all_data, random_data, d1_results


# =========================================================================
# D2: Continued fraction coefficients — algebraic vs transcendental vs random
# =========================================================================
def direction_2(all_data, random_data):
    print("\n" + "=" * 78)
    print("D2: CONTINUED FRACTIONS — ALGEBRAIC VS TRANSCENDENTAL VS RANDOM")
    print("=" * 78)
    print("  √2 is periodic (algebraic), e has a pattern (transcendental),")
    print("  π appears random (transcendental). Khinchin's theorem predicts")
    print("  geometric mean of CF coefficients → K ≈ 2.6854... for almost all reals.")

    cf_names = ['cf_sqrt2', 'cf_e', 'cf_pi']

    # Each CF vs random
    print(f"\n  vs Random:")
    for name in cf_names:
        n_sig, findings = compare(all_data[name], random_data)
        print(f"    {name:12s}: {n_sig:2d} sig")

    # Pairwise among CFs
    print(f"\n  Pairwise:")
    for i, n1 in enumerate(cf_names):
        for n2 in cf_names[i+1:]:
            n_sig, findings = compare(all_data[n1], all_data[n2])
            top = findings[0] if findings else ("—", 0, 1)
            print(f"    {n1:12s} vs {n2:12s}: {n_sig:2d} sig", end="")
            if findings:
                print(f"  best: {findings[0][0]:35s} d={findings[0][1]:+.2f}", end="")
            print()

    # CF statistics
    print(f"\n  CF coefficient statistics:")
    for name, coeffs in [('√2', CF_SQRT2[:2000]), ('e', CF_E[:2000]), ('π', CF_PI[:min(len(CF_PI), 2000)])]:
        arr = np.array(coeffs, dtype=np.float64)
        geo_mean = np.exp(np.mean(np.log(np.maximum(arr, 1))))
        print(f"    {name:3s}: mean={np.mean(arr):.1f}, median={np.median(arr):.0f}, "
              f"geo_mean={geo_mean:.3f} (Khinchin K=2.685)")


# =========================================================================
# D3: Partition function vs Hardy-Ramanujan model
# =========================================================================
def direction_3(analyzer, all_data):
    print("\n" + "=" * 78)
    print("D3: PARTITION FUNCTION VS HARDY-RAMANUJAN MODEL")
    print("=" * 78)
    print("  p(n) ~ exp(π√(2n/3)) / (4n√3). Does geometry see structure")
    print("  beyond this leading asymptotic?")

    # Hardy-Ramanujan model: generate p(n) from the formula, then mod 256
    def gen_hardy_ramanujan(trial, size):
        rng = np.random.default_rng(42 + trial)
        start = rng.integers(100, PART_LIMIT - size)
        ns = np.arange(start, start + size, dtype=np.float64)
        # HR approximation (will overflow for large n, but mod 256 is OK)
        log_p = np.pi * np.sqrt(2.0 * ns / 3.0) - np.log(4 * ns * np.sqrt(3))
        # Add Gaussian noise scaled to approximate the error term
        noise = rng.normal(0, 0.5, size)
        vals = np.exp(log_p + noise)
        return (vals % 256).astype(np.uint8)

    # Distribution-matched random
    def gen_dist_matched(trial, size):
        """Random bytes matching the marginal distribution of p(n) mod 256."""
        rng = np.random.default_rng(42 + trial)
        start = rng.integers(100, PART_LIMIT - size)
        real = PARTITIONS[start:start + size].copy()
        # Shuffle to destroy ordering
        perm = rng.permutation(size)
        return real[perm]

    hr_arrays = [gen_hardy_ramanujan(t, DATA_SIZE) for t in range(N_TRIALS)]
    hr_data = collect_metrics(analyzer, hr_arrays)

    dm_arrays = [gen_dist_matched(t, DATA_SIZE) for t in range(N_TRIALS)]
    dm_data = collect_metrics(analyzer, dm_arrays)

    # Real p(n) vs Hardy-Ramanujan
    n_sig, findings = compare(all_data['partition_fn'], hr_data)
    print(f"\n  p(n) vs Hardy-Ramanujan model: {n_sig} sig")
    for m, d, p in findings[:5]:
        print(f"    {m:45s}  d={d:+8.2f}")

    # Real p(n) vs distribution-matched shuffle
    n_sig2, findings2 = compare(all_data['partition_fn'], dm_data)
    print(f"\n  p(n) vs distribution-matched (shuffled): {n_sig2} sig")
    print(f"  → {n_sig2} metrics detect sequential ordering beyond distribution")
    for m, d, p in findings2[:5]:
        print(f"    {m:45s}  d={d:+8.2f}")

    return hr_data, dm_data


# =========================================================================
# D4: Shuffle validation — ordering vs distribution
# =========================================================================
def direction_4(analyzer, all_data):
    print("\n" + "=" * 78)
    print("D4: SHUFFLE VALIDATION — ORDERING VS DISTRIBUTION")
    print("=" * 78)

    results = {}
    for name, gen_fn in ENCODINGS.items():
        arrays = [gen_fn(t, DATA_SIZE) for t in range(N_TRIALS)]
        shuf_arrays = []
        for arr in arrays:
            s = arr.copy()
            np.random.default_rng(1000).shuffle(s)
            shuf_arrays.append(s)
        shuf_data = collect_metrics(analyzer, shuf_arrays)

        n_sig, findings = compare(all_data[name], shuf_data)
        results[name] = n_sig
        label = "ordering-dep" if n_sig > 5 else "distributional"
        print(f"  {name:18s}: {n_sig:2d} sig (orig vs shuf) → {label}")

    return results


# =========================================================================
# D5: Cross-comparison with multiplicative functions
# =========================================================================
def direction_5(analyzer, all_data, random_data):
    print("\n" + "=" * 78)
    print("D5: CROSS-COMPARISON WITH EXISTING SEQUENCES")
    print("=" * 78)
    print("  Comparing new additive/structural sequences against")
    print("  multiplicative functions from number_theory.py")

    # Import existing encodings
    from investigations.number_theory_sieve import (sieve_divisor_count, sieve_omega,
                                                     sieve_moebius)

    # ... actually, let's just regenerate the key ones inline
    def sieve_divisor_count_local(limit):
        d = np.zeros(limit + 1, dtype=np.int32)
        for i in range(1, limit + 1):
            d[i::i] += 1
        return d

    def sieve_omega_local(limit):
        omega = np.zeros(limit + 1, dtype=np.int32)
        for p in range(2, limit + 1):
            if omega[p] == 0:
                for multiple in range(p, limit + 1, p):
                    n = multiple
                    while n % p == 0:
                        omega[multiple] += 1
                        n //= p
        return omega

    MULT_LIMIT = 200_000
    print("  Computing multiplicative functions...", end=" ", flush=True)
    div_counts = sieve_divisor_count_local(MULT_LIMIT)
    omega = sieve_omega_local(MULT_LIMIT)
    print("done")

    def gen_divisor_count(trial, size):
        rng = np.random.default_rng(42 + trial)
        start = rng.integers(1000, MULT_LIMIT - size)
        return np.clip(div_counts[start:start + size], 0, 255).astype(np.uint8)

    def gen_omega(trial, size):
        rng = np.random.default_rng(42 + trial)
        start = rng.integers(1000, MULT_LIMIT - size)
        return np.clip(omega[start:start + size] * 12, 0, 255).astype(np.uint8)

    mult_names = {'divisor_count': gen_divisor_count, 'omega': gen_omega}
    mult_data = {}
    for name, gen_fn in mult_names.items():
        arrays = [gen_fn(t, DATA_SIZE) for t in range(N_TRIALS)]
        mult_data[name] = collect_metrics(analyzer, arrays)

    # Cross-comparison matrix
    all_names = list(ENCODINGS.keys()) + list(mult_names.keys())
    all_combined = {**all_data, **mult_data}

    print(f"\n  Pairwise cross-comparison (new additive vs old multiplicative):")
    for new_name in ENCODINGS.keys():
        for old_name in mult_names.keys():
            n_sig, findings = compare(all_combined[new_name], all_combined[old_name])
            top_str = ""
            if findings:
                top_str = f"  best: {findings[0][0]:30s} d={findings[0][1]:+.2f}"
            print(f"    {new_name:18s} vs {old_name:15s}: {n_sig:3d} sig{top_str}")


# =========================================================================
# FIGURE
# =========================================================================
def make_figure(all_data, random_data, d1_results, d4_results):
    print("\nGenerating figure...", flush=True)

    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 14), facecolor='#181818')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                           height_ratios=[1.0, 1.2])

    names = list(ENCODINGS.keys())
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3',
              '#9C27B0', '#00BCD4', '#FFC107', '#8BC34A']

    # D1: Detection bar chart
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    sigs = [d1_results[n] for n in names]
    ax.bar(range(len(names)), sigs, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Sig metrics vs random', fontsize=9)
    ax.set_title('D1: Detection — each encoding vs random', fontsize=11, fontweight='bold')

    # D4: Shuffle validation
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    shuf_sigs = [d4_results.get(n, 0) for n in names]
    ax.bar(range(len(names)), shuf_sigs, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Sig metrics (orig vs shuf)', fontsize=9)
    ax.set_title('D2: Ordering dependence', fontsize=11, fontweight='bold')

    # Pairwise matrix for all 8 encodings
    ax = _dark_ax(fig.add_subplot(gs[1, :]))
    n = len(names)
    mat = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j > i:
                n_sig, _ = compare(all_data[n1], all_data[n2])
                mat[i, j] = n_sig
                mat[j, i] = n_sig

    im = ax.imshow(mat, cmap='magma', interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                       fontsize=7, fontweight='bold',
                       color='white' if mat[i,j] > mat.max()/2 else '#aaa')
    ax.set_title('D3: Pairwise distance matrix (sig metrics)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle('Additive & Structural Number Theory: Geometric Signatures',
                 fontsize=14, fontweight='bold', color='white', y=0.995)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'number_theory_deep.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved number_theory_deep.png")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    all_data, random_data, d1_results = direction_1(analyzer)
    direction_2(all_data, random_data)
    direction_3(analyzer, all_data)
    d4_results = direction_4(analyzer, all_data)
    # D5 is slow (recomputes sieves) — only run if explicitly needed
    # direction_5(analyzer, all_data, random_data)

    make_figure(all_data, random_data, d1_results, d4_results)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    ranked = sorted(d1_results.items(), key=lambda x: -x[1])
    for name, n_sig in ranked:
        shuf = d4_results.get(name, 0)
        label = "ordering-dep" if shuf > 5 else "distributional"
        print(f"  {name:18s}: {n_sig:3d} sig vs random, {shuf:2d} ordering  [{label}]")
