#!/usr/bin/env python3
"""
Investigation: Geometric Signatures of Prime Number Sequences.

Can exotic geometries detect structure in prime number sequences? The primes
are the most studied objects in number theory — their distribution is
approximately random (prime number theorem) but with deep, subtle correlations
(twin primes, Lemke Oliver–Soundararajan bias, Cramér's conjecture).

7 encodings tested:
  1. prime_gaps      — g_n = p_{n+1} - p_n (natural uint8)
  2. primes_mod256   — p_n mod 256
  3. last_digit      — last decimal digit, spread to uint8 range
  4. binary_primes   — binary representations concatenated and packed
  5. gap_pairs       — two consecutive gaps packed into one byte
  6. prime_mod30     — p_n mod 30# (8 residue classes)
  7. gap_diff        — second differences of gaps (g_{n+1} - g_n + 128)

Five directions:
  D1: Each encoding vs random (is there detectable structure?)
  D2: All 21 pairwise encoding comparisons
  D3: Primes vs near-primes (Cramér random model, semiprimes, composites)
  D4: Shuffle validation (sequential vs distributional structure)
  D5: Starting range sensitivity (small primes vs large primes)

Budget: ~500 analyzer calls, estimated 3-4 minutes.
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


def sieve_omega(limit):
    """Count of prime factors with multiplicity for integers up to limit.
    omega[n] = number of prime factors of n (with multiplicity).
    """
    omega = np.zeros(limit + 1, dtype=np.int32)
    for p in range(2, limit + 1):
        if omega[p] == 0:  # p is prime
            for multiple in range(p, limit + 1, p):
                n = multiple
                while n % p == 0:
                    omega[multiple] += 1
                    n //= p
    return omega


# Precompute primes — sieve up to 2M covers all our needs
PRIME_LIMIT = 2_000_000
ALL_PRIMES = sieve_primes(PRIME_LIMIT)
print(f"Sieved {len(ALL_PRIMES)} primes up to {PRIME_LIMIT:,}")


# =============================================================================
# ENCODING GENERATORS
# =============================================================================

def get_primes_from(start_idx, count):
    """Get `count` consecutive primes starting at index `start_idx`."""
    end = start_idx + count
    if end > len(ALL_PRIMES):
        raise ValueError(f"Need {end} primes but only have {len(ALL_PRIMES)}")
    return ALL_PRIMES[start_idx:end]


def generate_prime_data(encoding, trial_seed, size=DATA_SIZE, start_idx=None):
    """Generate a uint8 array of length `size` from prime numbers.

    trial_seed determines which primes are used (offset into the sieve).
    start_idx overrides the automatic offset (used by D5 for range control).
    """
    rng = np.random.RandomState(trial_seed)

    if start_idx is None:
        # Each trial uses a different starting offset to get independent samples.
        # Primes near index 1000-50000 (roughly primes 7919-611953).
        start_idx = 1000 + trial_seed * 137  # spread across sieve

    # Most encodings need size+2 primes (for gaps, differences, etc.)
    primes = get_primes_from(start_idx, size + 10)

    if encoding == 'prime_gaps':
        # g_n = p_{n+1} - p_n. Gaps are typically 2-50, well within uint8.
        gaps = np.diff(primes)[:size]
        return np.clip(gaps, 0, 255).astype(np.uint8)

    elif encoding == 'primes_mod256':
        return (primes[:size] % 256).astype(np.uint8)

    elif encoding == 'last_digit':
        # Last decimal digit of primes > 5 is always 1, 3, 7, or 9.
        # Map to spread across uint8: {1:0, 3:64, 7:128, 9:192}
        digits = primes[:size] % 10
        mapping = {0: 128, 1: 0, 2: 128, 3: 64, 4: 128, 5: 128,
                   6: 128, 7: 128, 8: 128, 9: 192}
        result = np.array([mapping[d] for d in digits], dtype=np.uint8)
        return result

    elif encoding == 'binary_primes':
        # Concatenate binary representations, pack into bytes
        bits = []
        for p in primes:
            bits.extend([int(b) for b in bin(p)[2:]])
            if len(bits) >= size * 8:
                break
        bits = bits[:size * 8]
        # Pad if needed
        while len(bits) < size * 8:
            bits.append(0)
        # Pack 8 bits per byte
        result = np.zeros(size, dtype=np.uint8)
        for i in range(size):
            byte_val = 0
            for bit in range(8):
                byte_val = (byte_val << 1) | bits[i * 8 + bit]
            result[i] = byte_val
        return result

    elif encoding == 'gap_pairs':
        # Pack two consecutive gaps into one byte: high nibble + low nibble
        gaps = np.diff(primes)
        result = np.zeros(size, dtype=np.uint8)
        for i in range(size):
            g1 = int(gaps[2 * i]) if 2 * i < len(gaps) else 0
            g2 = int(gaps[2 * i + 1]) if 2 * i + 1 < len(gaps) else 0
            result[i] = ((g1 % 16) << 4) | (g2 % 16)
        return result

    elif encoding == 'prime_mod30':
        # Primes > 5 fall into 8 residue classes mod 30: {1,7,11,13,17,19,23,29}
        # Map to evenly spaced uint8 values
        mod30_map = {1: 0, 7: 36, 11: 73, 13: 109, 17: 146, 19: 182, 23: 219, 29: 255}
        residues = primes[:size] % 30
        result = np.array([mod30_map.get(int(r), 128) for r in residues],
                          dtype=np.uint8)
        return result

    elif encoding == 'gap_diff':
        # Second differences: g_{n+1} - g_n + 128 (centered at 128)
        gaps = np.diff(primes)
        diffs = np.diff(gaps)[:size]
        result = np.clip(diffs + 128, 0, 255).astype(np.uint8)
        # Pad if needed
        if len(result) < size:
            result = np.concatenate([result, np.full(size - len(result), 128, dtype=np.uint8)])
        return result

    else:
        raise ValueError(f"Unknown encoding: {encoding}")


# =============================================================================
# COMPARISON GENERATORS
# =============================================================================

def generate_cramer_model(trial_seed, size=DATA_SIZE, start_value=10000):
    """Cramér random model: each n is 'prime' with probability 1/ln(n).
    Returns gap sequence as uint8 (like prime_gaps encoding).
    """
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


def generate_semiprime_gaps(trial_seed, size=DATA_SIZE):
    """Gaps between consecutive semiprimes (numbers with Ω(n) = 2).
    Uses precomputed omega counts.
    """
    rng = np.random.RandomState(trial_seed)
    # Find semiprimes in a range — offset by trial_seed for variety
    start = 10000 + trial_seed * 500
    limit = start + size * 20  # semiprimes are denser than primes
    if limit > PRIME_LIMIT:
        limit = PRIME_LIMIT
    omega = sieve_omega(limit)
    semiprimes = np.where(omega[start:limit] == 2)[0] + start
    if len(semiprimes) < size + 1:
        # Fall back to smaller offset
        semiprimes = np.where(omega[4:limit] == 2)[0] + 4
    gaps = np.diff(semiprimes)[:size]
    if len(gaps) < size:
        gaps = np.concatenate([gaps, np.ones(size - len(gaps), dtype=np.int64)])
    return np.clip(gaps, 0, 255).astype(np.uint8)


def generate_composite_mod256(trial_seed, size=DATA_SIZE):
    """Composite numbers mod 256 — comparison for primes_mod256."""
    rng = np.random.RandomState(trial_seed)
    start = 10000 + trial_seed * 200
    composites = []
    n = start
    prime_set = set(ALL_PRIMES[ALL_PRIMES < PRIME_LIMIT])
    while len(composites) < size:
        if n not in prime_set:
            composites.append(n)
        n += 1
    return (np.array(composites[:size]) % 256).astype(np.uint8)


# =============================================================================
# STATISTICS
# =============================================================================

ENCODINGS = ['prime_gaps', 'primes_mod256', 'last_digit', 'binary_primes',
             'gap_pairs', 'prime_mod30', 'gap_diff']

ENC_COLORS = {
    'prime_gaps': '#E91E63',
    'primes_mod256': '#FF5722',
    'last_digit': '#FF9800',
    'binary_primes': '#4CAF50',
    'gap_pairs': '#2196F3',
    'prime_mod30': '#9C27B0',
    'gap_diff': '#00BCD4',
}

COMPARISONS = ['random', 'cramer_model', 'semiprime_gaps', 'composite_mod256']

COMP_COLORS = {
    'random': '#888888',
    'cramer_model': '#FFD54F',
    'semiprime_gaps': '#AED581',
    'composite_mod256': '#CE93D8',
}


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
        _, p = stats.ttest_ind(va, vb)
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
        _, p = stats.ttest_ind(va, vb)
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
    """Can exotic geometries detect structure in prime number sequences?"""
    print("\n" + "=" * 78)
    print("DIRECTION 1: Prime Encodings vs Random")
    print("=" * 78)
    print(f"7 encodings, {N_TRIALS} trials each, vs os.urandom baseline")
    print(f"Question: which encodings carry detectable geometric structure?\n")

    # Collect metrics for all encodings and random baseline
    all_metrics = {}

    # Random baseline
    print("  Random baseline...", end=" ", flush=True)
    all_metrics['random'] = defaultdict(list)
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial + 5000)
        data = rng.randint(0, 256, DATA_SIZE, dtype=np.uint8)
        collect_metrics(analyzer, data, all_metrics['random'])
    print("done")

    # Prime encodings
    for enc in ENCODINGS:
        print(f"  {enc}...", end=" ", flush=True)
        all_metrics[enc] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_prime_data(enc, trial)
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

    # Top 5 metrics per encoding
    for enc in ENCODINGS:
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
    """Can different prime encodings be distinguished from each other?"""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Pairwise Encoding Comparisons")
    print("=" * 78)
    print(f"21 pairs from 7 encodings (reusing D1 data, 0 extra calls)")
    print(f"Question: do different encodings capture different structure?\n")

    all_metrics = d1['all_metrics']
    n_total = len(metric_names)
    pair_results = []

    for i, enc1 in enumerate(ENCODINGS):
        for enc2 in ENCODINGS[i+1:]:
            pms = per_metric_significance(all_metrics[enc1], all_metrics[enc2],
                                          metric_names, n_total)
            sig_list = [(km, d, p) for km, (d, p, is_sig) in pms.items() if is_sig]
            sig_list.sort(key=lambda x: -abs(x[1]))
            n_sig = len(sig_list)
            best_km = sig_list[0][0] if sig_list else ''
            best_d = sig_list[0][1] if sig_list else 0.0
            best_p = sig_list[0][2] if sig_list else 1.0
            pair_results.append((enc1, enc2, n_sig, best_km, best_d, best_p))

    # Print matrix
    print(f"  {'':>20}", end="")
    for enc in ENCODINGS:
        print(f"  {enc[:8]:>8}", end="")
    print()

    pair_lookup = {}
    for enc1, enc2, n_sig, km, d, p in pair_results:
        pair_lookup[(enc1, enc2)] = n_sig
        pair_lookup[(enc2, enc1)] = n_sig

    for enc1 in ENCODINGS:
        print(f"  {enc1:<20}", end="")
        for enc2 in ENCODINGS:
            if enc1 == enc2:
                print(f"  {'—':>8}", end="")
            else:
                n = pair_lookup.get((enc1, enc2), 0)
                print(f"  {n:>8}", end="")
        print()

    return {'pair_results': pair_results}


# =============================================================================
# DIRECTION 3: PRIMES vs NEAR-PRIMES
# =============================================================================

def direction3_near_primes(analyzer, metric_names, d1):
    """Can geometries distinguish primes from other number-theoretic sequences?"""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Primes vs Near-Primes")
    print("=" * 78)
    print(f"Compare prime_gaps to: Cramér random model, semiprime gaps, composite mod 256")
    print(f"Also compare primes_mod256 to composite_mod256")
    print(f"Question: is there structure specific to primality?\n")

    all_metrics = d1['all_metrics']
    n_total = len(metric_names)

    # Generate comparison sequences
    comp_metrics = {}
    for comp_name in ['cramer_model', 'semiprime_gaps', 'composite_mod256']:
        print(f"  Generating {comp_name}...", end=" ", flush=True)
        comp_metrics[comp_name] = defaultdict(list)
        for trial in range(N_TRIALS):
            if comp_name == 'cramer_model':
                data = generate_cramer_model(trial)
            elif comp_name == 'semiprime_gaps':
                data = generate_semiprime_gaps(trial)
            elif comp_name == 'composite_mod256':
                data = generate_composite_mod256(trial)
            collect_metrics(analyzer, data, comp_metrics[comp_name])
        print("done")

    # Compare prime_gaps to gap-type comparisons
    print(f"\n  Prime gaps vs alternatives:")
    gap_results = {}
    for comp_name in ['cramer_model', 'semiprime_gaps']:
        n_sig = count_significant(all_metrics['prime_gaps'], comp_metrics[comp_name],
                                  metric_names, n_total)
        gap_results[comp_name] = n_sig
        print(f"    prime_gaps vs {comp_name:<20}  {n_sig:>3} sig")

        # Top 3 discriminators
        pms = per_metric_significance(all_metrics['prime_gaps'], comp_metrics[comp_name],
                                      metric_names, n_total)
        sig_list = [(km, d, p) for km, (d, p, is_sig) in pms.items() if is_sig]
        sig_list.sort(key=lambda x: -abs(x[1]))
        for km, d, p in sig_list[:3]:
            print(f"      {km:<40} d={d:+8.2f}")

    # Compare primes_mod256 to composites_mod256
    print(f"\n  Primes mod 256 vs composites mod 256:")
    n_sig = count_significant(all_metrics['primes_mod256'], comp_metrics['composite_mod256'],
                              metric_names, n_total)
    gap_results['composite_mod256'] = n_sig
    print(f"    primes_mod256 vs composite_mod256  {n_sig:>3} sig")

    # Also compare each comparison to random
    print(f"\n  Comparisons vs random:")
    for comp_name in ['cramer_model', 'semiprime_gaps', 'composite_mod256']:
        n_sig = count_significant(comp_metrics[comp_name], all_metrics['random'],
                                  metric_names, n_total)
        print(f"    {comp_name:<20} vs random  {n_sig:>3} sig")

    return {'comp_metrics': comp_metrics, 'gap_results': gap_results}


# =============================================================================
# DIRECTION 4: SHUFFLE VALIDATION
# =============================================================================

def direction4_shuffle(analyzer, metric_names, d1):
    """Which encoding structures depend on ordering vs distribution?"""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Shuffle Validation")
    print("=" * 78)
    print(f"Shuffle each encoding and re-analyze. Compare original vs shuffled vs random.")
    print(f"Ordering effects: destroyed by shuffling. Distributional: preserved.\n")

    all_metrics = d1['all_metrics']
    n_total = len(metric_names)

    # Generate shuffled versions
    shuffle_metrics = {}
    for enc in ENCODINGS:
        print(f"  Shuffling {enc}...", end=" ", flush=True)
        shuffle_metrics[enc] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_prime_data(enc, trial)
            rng = np.random.RandomState(trial + 9000)
            rng.shuffle(data)
            collect_metrics(analyzer, data, shuffle_metrics[enc])
        print("done")

    # Compare original vs shuffled, and shuffled vs random
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
# DIRECTION 5: STARTING RANGE SENSITIVITY
# =============================================================================

def direction5_range(analyzer, metric_names, d1):
    """Does the geometric signature evolve for larger primes?"""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Starting Range Sensitivity")
    print("=" * 78)
    print(f"prime_gaps encoding at 4 different starting points in the sieve")
    print(f"Question: do large primes have different gap geometry than small primes?\n")

    # Ranges: approximate prime indices for primes near 10^3, 10^4, 10^5, 10^6
    # π(1000)≈168, π(10000)≈1229, π(100000)≈9592, π(1000000)≈78498
    ranges = {
        'near 1K': 168,        # primes starting around 1000
        'near 10K': 1229,      # primes starting around 10000
        'near 100K': 9592,     # primes starting around 100000
        'near 1M': 78498,      # primes starting around 1000000
    }

    range_metrics = {}
    for range_name, start_idx in ranges.items():
        print(f"  Collecting prime_gaps {range_name} (idx={start_idx})...", end=" ", flush=True)
        range_metrics[range_name] = defaultdict(list)
        for trial in range(N_TRIALS):
            offset = trial * 50  # small offset per trial
            data = generate_prime_data('prime_gaps', trial, start_idx=start_idx + offset)
            collect_metrics(analyzer, data, range_metrics[range_name])
        print("done")

    n_total = len(metric_names)
    range_names = list(ranges.keys())

    # Each range vs random
    print(f"\n  {'Range':<15}  {'vs random':>10}  {'mean gap':>10}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}")
    for rn in range_names:
        n_sig = count_significant(range_metrics[rn], d1['all_metrics']['random'],
                                  metric_names, n_total)
        # Compute mean gap for context
        start_idx = ranges[rn]
        primes = get_primes_from(start_idx, DATA_SIZE + 1)
        mean_gap = np.mean(np.diff(primes))
        print(f"  {rn:<15}  {n_sig:>10}  {mean_gap:>10.1f}")

    # Pairwise range comparisons
    print(f"\n  Pairwise range comparisons:")
    pair_results = []
    for i, rn1 in enumerate(range_names):
        for rn2 in range_names[i+1:]:
            n_sig = count_significant(range_metrics[rn1], range_metrics[rn2],
                                      metric_names, n_total)
            pair_results.append((rn1, rn2, n_sig))
            print(f"    {rn1:<12} vs {rn2:<12}  {n_sig:>3} sig")

    return {'range_metrics': range_metrics, 'pair_results': pair_results}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 78)
    print("INVESTIGATION: Geometric Signatures of Prime Number Sequences")
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

    d1 = direction1_vs_random(analyzer, metric_names)
    d2 = direction2_pairwise(metric_names, d1)
    d3 = direction3_near_primes(analyzer, metric_names, d1)
    d4 = direction4_shuffle(analyzer, metric_names, d1)
    d5 = direction5_range(analyzer, metric_names, d1)

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
    bars = ax1.barh(range(len(enc_names)), det_counts, color=colors,
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
    for enc1, enc2, n_sig, km, d, p in d2['pair_results']:
        i1, i2 = ENCODINGS.index(enc1), ENCODINGS.index(enc2)
        mat[i1, i2] = n_sig
        mat[i2, i1] = n_sig
    im = ax2.imshow(mat, cmap='YlOrRd', interpolation='nearest', vmin=0)
    ax2.set_xticks(range(n_enc))
    ax2.set_yticks(range(n_enc))
    short = [e.replace('primes_', 'p_').replace('prime_', 'p_').replace('binary_primes', 'binary')
             for e in ENCODINGS]
    ax2.set_xticklabels(short, fontsize=6, color=FG, rotation=45, ha='right')
    ax2.set_yticklabels(short, fontsize=6, color=FG)
    for i in range(n_enc):
        for j in range(n_enc):
            if i != j:
                ax2.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                         fontsize=7, fontweight='bold',
                         color='white' if mat[i,j] > mat.max() * 0.5 else FG)
    ax2.set_title('D2: Pairwise Encoding Matrix', fontsize=11, fontweight='bold', color=FG)
    cb = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cb.set_label('Sig metrics', color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG)

    # ── Panel 3 (top-right): D3 — Primes vs near-primes ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    comp_names = ['cramer_model', 'semiprime_gaps', 'composite_mod256']
    comp_labels = ['Cramér model\n(fake primes)', 'Semiprime\ngaps', 'Composites\nmod 256']
    comp_counts = [d3['gap_results'].get(c, 0) for c in comp_names]
    comp_colors = [COMP_COLORS[c] for c in comp_names]

    bars3 = ax3.bar(range(len(comp_names)), comp_counts, color=comp_colors,
                    alpha=0.85, edgecolor='#333')
    ax3.set_xticks(range(len(comp_names)))
    ax3.set_xticklabels(comp_labels, fontsize=8, color=FG)
    ax3.set_ylabel('Sig metrics', color=FG, fontsize=9)
    ax3.set_title('D3: Primes vs Near-Primes', fontsize=11, fontweight='bold', color=FG)
    for i, v in enumerate(comp_counts):
        ax3.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=9, fontweight='bold')

    # ── Panel 4 (bottom-left): D4 — Shuffle validation ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    x = np.arange(len(ENCODINGS))
    bar_width = 0.25
    orig_vals = [d4['results'][enc]['orig_vs_rand'] for enc in ENCODINGS]
    shuf_vals = [d4['results'][enc]['shuf_vs_rand'] for enc in ENCODINGS]
    diff_vals = [d4['results'][enc]['orig_vs_shuf'] for enc in ENCODINGS]

    ax4.bar(x - bar_width, orig_vals, bar_width, label='orig vs rand',
            color='#E91E63', alpha=0.85, edgecolor='#333')
    ax4.bar(x, shuf_vals, bar_width, label='shuf vs rand',
            color='#FF9800', alpha=0.85, edgecolor='#333')
    ax4.bar(x + bar_width, diff_vals, bar_width, label='orig vs shuf',
            color='#2196F3', alpha=0.85, edgecolor='#333')
    ax4.set_xticks(x)
    ax4.set_xticklabels([e[:8] for e in ENCODINGS], fontsize=7, color=FG, rotation=30, ha='right')
    ax4.set_ylabel('Sig metrics', color=FG, fontsize=9)
    ax4.set_title('D4: Shuffle Validation', fontsize=11, fontweight='bold', color=FG)
    ax4.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    # ── Panel 5 (bottom-center): D5 — Range sensitivity ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    range_names = list(d5['range_metrics'].keys())
    n_total = len(metric_names)
    range_vs_rand = []
    for rn in range_names:
        n_sig = count_significant(d5['range_metrics'][rn], d1['all_metrics']['random'],
                                  metric_names, n_total)
        range_vs_rand.append(n_sig)

    ax5.plot(range(len(range_names)), range_vs_rand, 'o-', color='#E91E63',
             linewidth=2, markersize=8, alpha=0.85)
    ax5.set_xticks(range(len(range_names)))
    ax5.set_xticklabels(range_names, fontsize=8, color=FG)
    ax5.set_ylabel('Sig metrics vs random', color=FG, fontsize=9)
    ax5.set_title('D5: Range Sensitivity (prime_gaps)', fontsize=11,
                  fontweight='bold', color=FG)
    for i, v in enumerate(range_vs_rand):
        ax5.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=9, fontweight='bold')

    # Add pairwise info
    if d5['pair_results']:
        pair_text = "Pairwise:\n"
        for rn1, rn2, n_sig in d5['pair_results']:
            pair_text += f"  {rn1} vs {rn2}: {n_sig}\n"
        ax5.text(0.98, 0.98, pair_text.strip(), transform=ax5.transAxes, fontsize=6,
                 verticalalignment='top', horizontalalignment='right',
                 fontfamily='monospace', color='#aaa',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#444'))

    # ── Panel 6 (bottom-right): Summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    lines = [
        "Prime Sequences — Key Findings",
        "",
        "D1: Encodings vs Random",
    ]
    for enc in ENCODINGS:
        v = d1['results'][enc]
        lines.append(f"  {enc:<18} {v:>3} sig")

    lines.append("")
    lines.append("D3: Primes vs Near-Primes")
    for c in ['cramer_model', 'semiprime_gaps', 'composite_mod256']:
        v = d3['gap_results'].get(c, 0)
        lines.append(f"  vs {c:<18} {v:>3} sig")

    lines.append("")
    lines.append("D4: Ordering vs Distribution")
    n_ordering = sum(1 for enc in ENCODINGS if d4['results'][enc]['orig_vs_shuf'] > 5)
    n_distrib = sum(1 for enc in ENCODINGS
                    if d4['results'][enc]['orig_vs_rand'] > 5 and d4['results'][enc]['orig_vs_shuf'] <= 5)
    lines.append(f"  {n_ordering} ordering-dependent")
    lines.append(f"  {n_distrib} purely distributional")

    lines.append("")
    lines.append("D5: Range Evolution")
    for i, rn in enumerate(range_names):
        lines.append(f"  {rn:<12} {range_vs_rand[i]:>3} sig vs rand")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor='#444'))

    fig.suptitle('Prime Number Sequences: Exotic Geometry Analysis',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'primes.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved primes.png")
    plt.close(fig)


if __name__ == '__main__':
    metric_names, d1, d2, d3, d4, d5 = main()
    make_figure(metric_names, d1, d2, d3, d4, d5)
