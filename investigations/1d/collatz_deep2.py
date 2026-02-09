#!/usr/bin/env python3
"""
Investigation: Deep Collatz Geometric Exploration II.

Follows up on collatz_deep.py, which revealed:
  - Sharp k=1→2 phase transition (86→35 sig metrics)
  - Ordering peaks at mod 4 (63 metrics)
  - Optimal delay τ=2 (78 sig)
  - U-shaped bitplane signal (LSB=77, MSB=68)
  - Tropical slopes match theory exactly (μ=1.5855 vs 1.5850)

Five new directions probing WHY 3n+1 converges:

1. 2-adic Valuation Sequence — the v₂(3n+1) halving engine
2. Phase Transition Anatomy — WHICH metrics die at k=1→2?
3. Cross-Scale Composition — bitplane + delay embedding together
4. Drift Rate Detection — metrics that track drift across the family
5. Inverse Collatz Tree — encoding the backward BFS tree from 1

Total budget: ~725 analyzer calls, estimated 4-5 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict, deque
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer, TropicalGeometry
from exotic_geometry_framework import delay_embed, bitplane_extract
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000


# =============================================================================
# TRAJECTORY GENERATORS (from collatz_deep.py)
# =============================================================================

def variant_step(n, k):
    """Single step for the (2k+1)n+1 family."""
    if n % 2 == 0:
        return n // 2
    return (2 * k + 1) * n + 1


def variant_trajectory(start, k, max_steps=100000, cap=10**15):
    """Trajectory for (2k+1)n+1 variant. Stops at 1, max_steps, or cap."""
    n = start
    traj = [n]
    while n != 1 and len(traj) < max_steps:
        n = variant_step(n, k)
        traj.append(n)
        if n > cap:
            break
    return traj


def generate_variant_data(k, trial_seed, size=DATA_SIZE):
    """Generate byte data from (2k+1)n+1 trajectory, chaining starts."""
    rng = np.random.RandomState(trial_seed)
    traj = []
    while len(traj) < size:
        start = rng.randint(10**6, 10**8)
        seg = variant_trajectory(start, k)
        traj.extend(seg)
    arr = np.array(traj[:size], dtype=np.uint64)
    return (arr % 256).astype(np.uint8)


def generate_hailstone_large(trial_seed, size=DATA_SIZE):
    """Hailstone large (3n+1) from collatz.py — canonical encoding."""
    rng = np.random.RandomState(trial_seed)
    traj = []
    while len(traj) < size:
        start = rng.randint(10**9, 10**10)
        seg = variant_trajectory(start, k=1)
        traj.extend(seg)
    arr = np.array(traj[:size], dtype=np.uint64)
    return (arr % 256).astype(np.uint8)


def generate_random(trial_seed, size=DATA_SIZE):
    """Uniform random baseline."""
    return np.random.RandomState(trial_seed).randint(0, 256, size, dtype=np.uint8)


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
    """Count metrics significantly different between two groups (|d|>0.8, Bonferroni)."""
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
    """Per-metric Cohen's d, p-value, and significance flag.

    Returns: {metric: (d, p, is_sig)}
    """
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


# =============================================================================
# DIRECTION 1: 2-ADIC VALUATION SEQUENCE
# =============================================================================

def extract_v2_sequence(start, k, min_odd_steps=500):
    """Walk (2k+1)n+1 trajectory, extract v₂ and Syracuse values at each odd step.

    v₂ = number of trailing zeros after odd step (how many halvings follow)
    syracuse = the odd part after all halvings

    Returns (v2_vals, syracuse_vals) as lists.
    """
    n = start
    v2_vals = []
    syracuse_vals = []
    steps = 0
    while len(v2_vals) < min_odd_steps and steps < 10**6:
        if n <= 0:
            break
        if n % 2 == 1:
            # Odd step: apply (2k+1)n+1
            m = (2 * k + 1) * n + 1
            # Count trailing zeros (v₂)
            if m == 0:
                break
            v2 = 0
            tmp = m
            while tmp > 0 and tmp % 2 == 0:
                v2 += 1
                tmp //= 2
            v2_vals.append(v2)
            syracuse_vals.append(tmp)  # odd part
            n = tmp
        else:
            n = n // 2
        steps += 1
        if n > 10**15:
            break
    return v2_vals, syracuse_vals


def generate_v2_random(trial_seed, size=DATA_SIZE):
    """IID Geometric(1/2) samples — matches v₂ distribution for random odd numbers."""
    rng = np.random.RandomState(trial_seed)
    # Geometric(1/2): P(X=k) = (1/2)^k for k=1,2,...
    vals = rng.geometric(0.5, size=size)
    return (np.array(vals, dtype=np.uint64) % 256).astype(np.uint8)


def direction1_v2_valuation(analyzer, metric_names, shared_random):
    """The v₂(3n+1) distribution — the engine of convergence."""
    print("\n" + "=" * 78)
    print("DIRECTION 1: 2-adic Valuation Sequence")
    print("=" * 78)
    print("v₂ = trailing zeros after odd step = number of halvings")
    print("Theory: for random odds, v₂ ~ Geometric(1/2), mean=2")
    print(f"4 encodings, {N_TRIALS} trials each\n")

    # Collect v₂ sequences for histogram and empirical distribution
    print("  Extracting v₂ sequences...", flush=True)
    all_v2_3n1 = []
    all_v2_5n1 = []
    for seed in range(50):
        rng = np.random.RandomState(seed + 10000)
        start = rng.randint(10**6, 10**8)
        if start % 2 == 0:
            start += 1
        v2_3, _ = extract_v2_sequence(start, k=1, min_odd_steps=1000)
        v2_5, _ = extract_v2_sequence(start, k=2, min_odd_steps=1000)
        all_v2_3n1.extend(v2_3)
        all_v2_5n1.extend(v2_5)

    print(f"    3n+1: {len(all_v2_3n1)} odd steps, mean v₂ = {np.mean(all_v2_3n1):.3f}")
    print(f"    5n+1: {len(all_v2_5n1)} odd steps, mean v₂ = {np.mean(all_v2_5n1):.3f}")

    # 4 encodings of the v₂ sequence for geometric analysis
    encoding_results = {}  # encoding -> n_sig

    for enc_name in ['v2_raw', 'v2_diff', 'v2_syracuse', 'v2_runlength']:
        print(f"  Encoding: {enc_name}...", end=" ", flush=True)
        enc_metrics = defaultdict(list)
        rand_metrics = defaultdict(list)

        for trial in range(N_TRIALS):
            rng = np.random.RandomState(trial + 11000)
            start = rng.randint(10**6, 10**8)
            if start % 2 == 0:
                start += 1
            v2_vals, syr_vals = extract_v2_sequence(start, k=1, min_odd_steps=DATA_SIZE + 100)

            if enc_name == 'v2_raw':
                # v₂ values mod 256
                arr = np.array(v2_vals[:DATA_SIZE], dtype=np.uint64)
                data = (arr % 256).astype(np.uint8)
                # Random: geometric(1/2) mod 256
                rand_data = generate_v2_random(trial + 12000, size=DATA_SIZE)

            elif enc_name == 'v2_diff':
                # Consecutive differences, shifted to uint8
                v2a = np.array(v2_vals[:DATA_SIZE + 1], dtype=np.int64)
                diffs = np.diff(v2a)[:DATA_SIZE]
                # Shift: add 128 to center around 128, clip
                shifted = np.clip(diffs + 128, 0, 255).astype(np.uint8)
                data = shifted
                # Random: diffs of geometric samples
                rv = np.random.RandomState(trial + 12000).geometric(0.5, size=DATA_SIZE + 1).astype(np.int64)
                rd = np.diff(rv)[:DATA_SIZE]
                rand_data = np.clip(rd + 128, 0, 255).astype(np.uint8)

            elif enc_name == 'v2_syracuse':
                # Syracuse (odd part) values mod 256
                arr = np.array(syr_vals[:DATA_SIZE], dtype=np.uint64)
                data = (arr % 256).astype(np.uint8)
                # Random: uniform odd numbers mod 256
                rng2 = np.random.RandomState(trial + 12000)
                rand_vals = rng2.randint(0, 128, size=DATA_SIZE) * 2 + 1
                rand_data = (rand_vals % 256).astype(np.uint8)

            elif enc_name == 'v2_runlength':
                # Binary: above/below median v₂, bit-packed
                v2a = np.array(v2_vals[:DATA_SIZE * 8], dtype=np.float64)
                median_v2 = np.median(v2a)
                bits = (v2a > median_v2).astype(np.uint8)
                # Pack 8 bits into bytes
                n_bytes = len(bits) // 8
                packed = np.zeros(n_bytes, dtype=np.uint8)
                for b in range(8):
                    packed |= (bits[b::8][:n_bytes] << b)
                data = packed[:DATA_SIZE]
                if len(data) < DATA_SIZE:
                    data = np.pad(data, (0, DATA_SIZE - len(data)),
                                  constant_values=128).astype(np.uint8)
                # Random: uniform bytes (random bits)
                rand_data = generate_random(trial + 12000, size=DATA_SIZE)

            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    enc_metrics[f"{r.geometry_name}:{mn}"].append(mv)

            r_rand = analyzer.analyze(rand_data)
            for r in r_rand.results:
                for mn, mv in r.metrics.items():
                    rand_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        n_total = 4 * len(metric_names)  # 4 encodings
        n_sig = count_significant(enc_metrics, rand_metrics, metric_names, n_total)
        encoding_results[enc_name] = n_sig
        print(f"{n_sig} sig metrics vs random")

    # Also test 5n+1 Syracuse for comparison
    print(f"  Encoding: v2_syracuse (5n+1)...", end=" ", flush=True)
    enc_5n1 = defaultdict(list)
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial + 13000)
        start = rng.randint(10**6, 10**8)
        if start % 2 == 0:
            start += 1
        _, syr_vals = extract_v2_sequence(start, k=2, min_odd_steps=DATA_SIZE + 100)
        arr = np.array(syr_vals[:DATA_SIZE], dtype=np.uint64)
        data = (arr % 256).astype(np.uint8)
        results = analyzer.analyze(data)
        for r in results.results:
            for mn, mv in r.metrics.items():
                enc_5n1[f"{r.geometry_name}:{mn}"].append(mv)
    # Compare 5n+1 syracuse vs same random baseline as 3n+1 syracuse
    n_sig_5n1 = count_significant(enc_5n1, shared_random, metric_names,
                                  5 * len(metric_names))
    print(f"{n_sig_5n1} sig metrics vs random")
    encoding_results['v2_syracuse_5n1'] = n_sig_5n1

    return {
        'encoding_results': encoding_results,
        'v2_3n1': all_v2_3n1,
        'v2_5n1': all_v2_5n1,
    }


# =============================================================================
# DIRECTION 2: PHASE TRANSITION ANATOMY
# =============================================================================

def direction2_phase_anatomy(analyzer, metric_names, shared_random):
    """WHICH of the ~120 metrics die at k=1→2?"""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Phase Transition Anatomy")
    print("=" * 78)
    print("Classifying each metric: convergence-specific, universal, divergence-specific")
    print(f"{N_TRIALS} trials for k=1 and k=2\n")

    # Collect metrics for k=1 (3n+1) and k=2 (5n+1)
    k_data = {}
    for k in [1, 2]:
        label = f"({2*k+1})n+1"
        print(f"  k={k} {label}...", end=" ", flush=True)
        k_metrics = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_variant_data(k, trial + 20000 + 1000 * k)
            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    k_metrics[f"{r.geometry_name}:{mn}"].append(mv)
        k_data[k] = dict(k_metrics)
        print("done")

    # Per-metric significance for each k vs random
    n_total = 2 * len(metric_names)
    sig_k1 = per_metric_significance(k_data[1], shared_random, metric_names, n_total)
    sig_k2 = per_metric_significance(k_data[2], shared_random, metric_names, n_total)

    # Classify each metric
    categories = {
        'convergence_specific': [],  # sig for k=1 only
        'universal': [],             # sig for both
        'divergence_specific': [],   # sig for k=2 only
        'neither': [],               # sig for neither
    }

    for km in metric_names:
        d1, p1, is_sig1 = sig_k1[km]
        d2, p2, is_sig2 = sig_k2[km]
        if is_sig1 and is_sig2:
            categories['universal'].append(km)
        elif is_sig1 and not is_sig2:
            categories['convergence_specific'].append(km)
        elif not is_sig1 and is_sig2:
            categories['divergence_specific'].append(km)
        else:
            categories['neither'].append(km)

    print(f"\n  Classification:")
    print(f"    Convergence-specific (k=1 only): {len(categories['convergence_specific'])}")
    print(f"    Universal (both):                {len(categories['universal'])}")
    print(f"    Divergence-specific (k=2 only):  {len(categories['divergence_specific'])}")
    print(f"    Neither:                         {len(categories['neither'])}")

    # Group by geometry type
    geom_counts = defaultdict(lambda: {'conv': 0, 'univ': 0, 'div': 0})
    for km in categories['convergence_specific']:
        geom = km.split(':')[0]
        geom_counts[geom]['conv'] += 1
    for km in categories['universal']:
        geom = km.split(':')[0]
        geom_counts[geom]['univ'] += 1
    for km in categories['divergence_specific']:
        geom = km.split(':')[0]
        geom_counts[geom]['div'] += 1

    print(f"\n  Geometry breakdown (convergence-specific / universal / divergence-specific):")
    for geom in sorted(geom_counts.keys()):
        c = geom_counts[geom]
        total = c['conv'] + c['univ'] + c['div']
        if total > 0:
            print(f"    {geom:<30} {c['conv']:>2} / {c['univ']:>2} / {c['div']:>2}")

    # Build heatmap data: significant metrics sorted by category, Cohen's d values
    heatmap_metrics = []
    heatmap_cats = []
    heatmap_d_k1 = []
    heatmap_d_k2 = []

    for cat, label in [('convergence_specific', 'conv'),
                       ('universal', 'univ'),
                       ('divergence_specific', 'div')]:
        for km in sorted(categories[cat], key=lambda m: -abs(sig_k1[m][0])):
            heatmap_metrics.append(km)
            heatmap_cats.append(label)
            heatmap_d_k1.append(sig_k1[km][0])
            heatmap_d_k2.append(sig_k2[km][0])

    return {
        'categories': categories,
        'sig_k1': sig_k1,
        'sig_k2': sig_k2,
        'geom_counts': dict(geom_counts),
        'heatmap_metrics': heatmap_metrics,
        'heatmap_cats': heatmap_cats,
        'heatmap_d_k1': heatmap_d_k1,
        'heatmap_d_k2': heatmap_d_k2,
    }


# =============================================================================
# DIRECTION 3: CROSS-SCALE COMPOSITION
# =============================================================================

def direction3_cross_scale(analyzer, metric_names, shared_random):
    """Bitplane and delay embedding composed — does the combination beat individuals?"""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Cross-Scale Composition")
    print("=" * 78)
    print("Composing bitplane + delay embed transforms")
    print("Baselines from collatz_deep: BP0=77, DE2=78, BP7=68")
    print(f"{N_TRIALS} trials each, 20000-byte base data\n")

    BASE_SIZE = 20000  # large enough to survive both reductions

    compositions = {
        'BP0→DE2': ('bitplane_then_delay', 0, 2),   # bitplane(LSB) then delay(τ=2)
        'DE2→BP0': ('delay_then_bitplane', 2, 0),   # delay(τ=2) then bitplane(LSB)
        'BP7→DE2': ('bitplane_then_delay', 7, 2),   # bitplane(MSB) then delay(τ=2)
        'DE1→BP0': ('delay_then_bitplane', 1, 0),   # delay(τ=1) then bitplane(LSB)
    }

    comp_results = {}

    for comp_name, (mode, param1, param2) in compositions.items():
        print(f"  {comp_name}...", end=" ", flush=True)
        comp_metrics = defaultdict(list)
        rand_metrics = defaultdict(list)

        for trial in range(N_TRIALS):
            # Collatz data
            data = generate_hailstone_large(trial + 30000, size=BASE_SIZE)
            rand_data = generate_random(trial + 31000, size=BASE_SIZE)

            if mode == 'bitplane_then_delay':
                plane, tau = param1, param2
                # bitplane first (needs 8x input), then delay embed
                bp = bitplane_extract(data, plane)  # BASE_SIZE/8 = 2500 bytes
                composed = delay_embed(bp, tau)       # ~2*(2500-tau) bytes
                composed = composed[:DATA_SIZE]

                bp_r = bitplane_extract(rand_data, plane)
                rand_composed = delay_embed(bp_r, tau)
                rand_composed = rand_composed[:DATA_SIZE]

            elif mode == 'delay_then_bitplane':
                tau, plane = param1, param2
                # delay embed first, then bitplane
                de = delay_embed(data, tau)           # ~2*(BASE_SIZE-tau) bytes
                composed = bitplane_extract(de, plane)  # de_len/8 bytes
                composed = composed[:DATA_SIZE]

                de_r = delay_embed(rand_data, tau)
                rand_composed = bitplane_extract(de_r, plane)
                rand_composed = rand_composed[:DATA_SIZE]

            # Pad if needed
            if len(composed) < DATA_SIZE:
                composed = np.pad(composed, (0, DATA_SIZE - len(composed)),
                                  constant_values=128).astype(np.uint8)
            if len(rand_composed) < DATA_SIZE:
                rand_composed = np.pad(rand_composed, (0, DATA_SIZE - len(rand_composed)),
                                       constant_values=128).astype(np.uint8)

            results = analyzer.analyze(composed)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    comp_metrics[f"{r.geometry_name}:{mn}"].append(mv)

            r_rand = analyzer.analyze(rand_composed)
            for r in r_rand.results:
                for mn, mv in r.metrics.items():
                    rand_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        n_total = len(compositions) * len(metric_names)
        n_sig = count_significant(comp_metrics, rand_metrics, metric_names, n_total)
        comp_results[comp_name] = n_sig
        print(f"{n_sig} sig metrics")

    # Reference baselines from collatz_deep
    baselines = {'BP0': 77, 'DE2': 78, 'BP7': 68}
    print(f"\n  Individual baselines: BP0={baselines['BP0']}, DE2={baselines['DE2']}, BP7={baselines['BP7']}")
    for name, n_sig in comp_results.items():
        print(f"    {name}: {n_sig}  ({'↑' if n_sig > max(baselines.values()) else '↓' if n_sig < min(baselines.values()) else '~'})")

    return {
        'comp_results': comp_results,
        'baselines': baselines,
    }


# =============================================================================
# DIRECTION 4: DRIFT RATE DETECTION
# =============================================================================

def compute_empirical_drift(k, n_trajs=200, max_steps=5000):
    """Average log₂(trajectory[i+1]/trajectory[i]) across many trajectories."""
    rng = np.random.RandomState(k + 40000)
    log_ratios = []
    for _ in range(n_trajs):
        start = rng.randint(10**6, 10**8)
        n = start
        for _ in range(max_steps):
            prev = n
            n = variant_step(n, k)
            if n <= 0 or prev <= 0:
                break
            if n > 10**15:
                break
            log_ratios.append(np.log2(n) - np.log2(prev))
            if n == 1:
                break
    return np.mean(log_ratios) if log_ratios else 0.0


def direction4_drift_detection(analyzer, metric_names, shared_random):
    """Which metrics correlate with theoretical drift rate across (2k+1)n+1?"""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Drift Rate Detection")
    print("=" * 78)
    print("Finding metrics whose VALUES track the drift rate across k=1..8")
    print(f"{N_TRIALS} trials per k\n")

    k_values = list(range(1, 9))

    # Compute empirical drift rates
    print("  Computing drift rates...", flush=True)
    drift_rates = {}
    for k in k_values:
        dr = compute_empirical_drift(k)
        drift_rates[k] = dr
        conv = "converges" if dr < 0 else "DIVERGES"
        print(f"    k={k} ({2*k+1}n+1): drift = {dr:+.4f} log₂/step  ({conv})")

    # Collect per-k metric means
    print(f"\n  Collecting metrics for k=1..8...", flush=True)
    k_metric_means = {}  # k -> {metric: mean_value}
    for k in k_values:
        print(f"    k={k}...", end=" ", flush=True)
        trial_metrics = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_variant_data(k, trial + 41000 + 1000 * k)
            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    trial_metrics[f"{r.geometry_name}:{mn}"].append(mv)
        k_metric_means[k] = {km: np.mean(vals) for km, vals in trial_metrics.items()}
        print("done")

    # For each metric: Pearson correlation of its mean value across k vs drift rate
    print(f"\n  Computing correlations...", flush=True)
    drift_arr = np.array([drift_rates[k] for k in k_values])
    correlations = {}  # metric -> (r, p)
    for km in metric_names:
        metric_arr = np.array([k_metric_means[k].get(km, 0.0) for k in k_values])
        if np.std(metric_arr) < 1e-10:
            correlations[km] = (0.0, 1.0)
            continue
        r_val, p_val = stats.pearsonr(metric_arr, drift_arr)
        correlations[km] = (r_val, p_val)

    # "Convergence detectors": |r| > 0.8 and p < 0.05
    detectors = [(km, r, p) for km, (r, p) in correlations.items()
                 if abs(r) > 0.8 and p < 0.05]
    detectors.sort(key=lambda x: -abs(x[1]))

    print(f"\n  Convergence detectors (|r| > 0.8, p < 0.05): {len(detectors)}")
    for km, r, p in detectors[:10]:
        print(f"    {km:<45} r={r:+.3f}  p={p:.1e}")

    # Prepare scatter data for top 5 detectors
    scatter_data = {}
    for km, r, p in detectors[:5]:
        metric_vals = [k_metric_means[k].get(km, 0.0) for k in k_values]
        scatter_data[km] = {
            'k_values': k_values,
            'metric_vals': metric_vals,
            'drift_vals': [drift_rates[k] for k in k_values],
            'r': r,
            'p': p,
        }

    return {
        'drift_rates': drift_rates,
        'correlations': correlations,
        'detectors': detectors,
        'scatter_data': scatter_data,
        'k_values': k_values,
    }


# =============================================================================
# DIRECTION 5: INVERSE COLLATZ TREE
# =============================================================================

def build_inverse_tree(root=1, max_nodes=20000):
    """BFS the Collatz predecessor tree backward from root.

    Predecessors of n:
      - 2n (always, since n/2 → n means 2n → n)
      - (n-1)/3 if n ≡ 1 mod 3 and (n-1)/3 is odd (odd step 3m+1=n means m=(n-1)/3)

    Returns (bfs_order, branching) where:
      bfs_order: list of node values in BFS visit order
      branching: list of child counts (1 or 2) per node
    """
    visited = {root}
    queue = deque([root])
    bfs_order = []
    branching = []

    while queue and len(bfs_order) < max_nodes:
        n = queue.popleft()
        bfs_order.append(n)

        children = []
        # Predecessor 1: 2n
        p1 = 2 * n
        if p1 not in visited:
            visited.add(p1)
            children.append(p1)
            queue.append(p1)

        # Predecessor 2: (n-1)/3 if valid
        if n > 1 and n % 3 == 1:
            p2 = (n - 1) // 3
            if p2 > 0 and p2 % 2 == 1 and p2 not in visited:
                visited.add(p2)
                children.append(p2)
                queue.append(p2)

        branching.append(len(children))

    return bfs_order, branching


def direction5_inverse_tree(analyzer, metric_names, shared_random):
    """Encode and analyze the backward BFS tree from 1."""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Inverse Collatz Tree")
    print("=" * 78)
    print("BFS backward from 1: predecessors 2n and (n-1)/3 when valid")
    print(f"2 encodings, {N_TRIALS} trials via offset slicing\n")

    # Build the deterministic tree (large)
    print("  Building inverse tree...", end=" ", flush=True)
    bfs_order, branching = build_inverse_tree(root=1, max_nodes=50000)
    print(f"{len(bfs_order)} nodes")

    # Statistics on branching
    branch_arr = np.array(branching)
    n_single = np.sum(branch_arr == 1)
    n_double = np.sum(branch_arr == 2)
    n_zero = np.sum(branch_arr == 0)
    print(f"    Branching: {n_zero} leaf, {n_single} single ({n_single/len(branching)*100:.1f}%), "
          f"{n_double} double ({n_double/len(branching)*100:.1f}%)")

    encoding_results = {}

    for enc_name in ['bfs_order', 'branching']:
        print(f"  Encoding: {enc_name}...", end=" ", flush=True)
        enc_metrics = defaultdict(list)

        for trial in range(N_TRIALS):
            # Offset slicing into the deterministic BFS order for trial variation
            offset = trial * 500
            if enc_name == 'bfs_order':
                vals = bfs_order[offset:offset + DATA_SIZE]
                if len(vals) < DATA_SIZE:
                    vals = bfs_order[:DATA_SIZE]
                data = (np.array(vals, dtype=np.uint64) % 256).astype(np.uint8)
            elif enc_name == 'branching':
                vals = branching[offset:offset + DATA_SIZE * 8]
                if len(vals) < DATA_SIZE * 8:
                    vals = branching[:DATA_SIZE * 8]
                # Bit-pack: 1-child=0, 2-children=1
                bits = np.array(vals[:DATA_SIZE * 8], dtype=np.uint8)
                bits = np.clip(bits - 1, 0, 1)  # 0→0, 1→0, 2→1
                n_bytes = len(bits) // 8
                packed = np.zeros(n_bytes, dtype=np.uint8)
                for b in range(8):
                    packed |= (bits[b::8][:n_bytes] << b)
                data = packed[:DATA_SIZE]
                if len(data) < DATA_SIZE:
                    data = np.pad(data, (0, DATA_SIZE - len(data)),
                                  constant_values=128).astype(np.uint8)

            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    enc_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        n_total = 2 * len(metric_names)
        n_sig = count_significant(enc_metrics, shared_random, metric_names, n_total)
        encoding_results[enc_name] = n_sig
        print(f"{n_sig} sig metrics vs random")

    return {
        'encoding_results': encoding_results,
        'bfs_order': bfs_order[:5000],  # keep truncated for figure
        'branching': branching[:5000],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("DEEP COLLATZ GEOMETRIC EXPLORATION II")
    print("=" * 78)
    print(f"\nParameters: {N_TRIALS} trials, {DATA_SIZE} bytes per sample")
    print("Five directions: v₂ valuation, phase anatomy, cross-scale,")
    print("  drift detection, inverse tree\n")

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get metric names from a quick dummy run
    dummy = generate_random(0)
    dummy_results = analyzer.analyze(dummy)
    metric_names = []
    for r in dummy_results.results:
        for mn in r.metrics:
            metric_names.append(f"{r.geometry_name}:{mn}")
    metric_names = sorted(set(metric_names))
    print(f"  {len(metric_names)} metrics across {len(dummy_results.results)} geometries\n")

    # Shared random baseline (collected once, reused by all directions)
    print("  Collecting shared random baseline...", flush=True)
    shared_random = defaultdict(list)
    for trial in range(N_TRIALS):
        data = generate_random(trial + 10500)
        results = analyzer.analyze(data)
        for r in results.results:
            for mn, mv in r.metrics.items():
                shared_random[f"{r.geometry_name}:{mn}"].append(mv)
    shared_random = dict(shared_random)
    print("  done\n")

    d1 = direction1_v2_valuation(analyzer, metric_names, shared_random)
    d2 = direction2_phase_anatomy(analyzer, metric_names, shared_random)
    d3 = direction3_cross_scale(analyzer, metric_names, shared_random)
    d4 = direction4_drift_detection(analyzer, metric_names, shared_random)
    d5 = direction5_inverse_tree(analyzer, metric_names, shared_random)

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"\n  D1 — 2-adic Valuation Sequence:")
    for enc, n_sig in d1['encoding_results'].items():
        print(f"    {enc:<25} {n_sig:>3} sig metrics vs random")

    print(f"\n  D2 — Phase Transition Anatomy:")
    for cat in ['convergence_specific', 'universal', 'divergence_specific', 'neither']:
        n = len(d2['categories'][cat])
        print(f"    {cat:<25} {n:>3} metrics")

    print(f"\n  D3 — Cross-Scale Composition:")
    for name, n_sig in d3['comp_results'].items():
        print(f"    {name:<25} {n_sig:>3} sig metrics")
    print(f"    Baselines: BP0={d3['baselines']['BP0']}, DE2={d3['baselines']['DE2']}, BP7={d3['baselines']['BP7']}")

    print(f"\n  D4 — Drift Rate Detection:")
    print(f"    {len(d4['detectors'])} convergence detectors (|r|>0.8, p<0.05)")
    for km, r, p in d4['detectors'][:5]:
        short = km.split(':')[1]
        print(f"    {short:<35} r={r:+.3f}")

    print(f"\n  D5 — Inverse Collatz Tree:")
    for enc, n_sig in d5['encoding_results'].items():
        print(f"    {enc:<25} {n_sig:>3} sig metrics vs random")

    print(f"\n[Deep investigation II complete]")

    return d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION — 6-panel layout (22x14, dark theme)
# =============================================================================

def make_figure(d1, d2, d3, d4, d5):
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

    # ── Top-left: D1 — v₂ histogram + inset bar chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)

    # v₂ histograms
    v2_3n1 = np.array(d1['v2_3n1'])
    v2_5n1 = np.array(d1['v2_5n1'])
    max_v2 = min(15, max(v2_3n1.max(), v2_5n1.max()) + 1)
    bins = np.arange(0.5, max_v2 + 1.5, 1)

    ax1.hist(v2_3n1, bins=bins, density=True, alpha=0.6, color='#E91E63',
             label='3n+1', edgecolor='#333')
    ax1.hist(v2_5n1, bins=bins, density=True, alpha=0.6, color='#2196F3',
             label='5n+1', edgecolor='#333')

    # Geometric(1/2) theory overlay
    x_theory = np.arange(1, max_v2 + 1)
    y_theory = 0.5 ** x_theory  # P(X=k) = (1/2)^k
    ax1.plot(x_theory, y_theory, 'w--', linewidth=2, alpha=0.8, label='Geom(1/2) theory')

    ax1.set_xlabel('v₂ (number of halvings)', color=FG, fontsize=9)
    ax1.set_ylabel('Density', color=FG, fontsize=9)
    ax1.set_title('D1: 2-adic Valuation Distribution', fontsize=11,
                  fontweight='bold', color=FG)
    ax1.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    # Inset: sig metrics per encoding
    ax1_ins = ax1.inset_axes([0.55, 0.45, 0.42, 0.48])
    ax1_ins.set_facecolor('#222')
    enc_names = [k for k in d1['encoding_results'] if k != 'v2_syracuse_5n1']
    enc_vals = [d1['encoding_results'][k] for k in enc_names]
    short_names = [n.replace('v2_', '') for n in enc_names]
    colors_ins = ['#E91E63', '#FF9800', '#4CAF50', '#9C27B0']
    bars_ins = ax1_ins.bar(range(len(enc_names)), enc_vals,
                           color=colors_ins[:len(enc_names)], alpha=0.85,
                           edgecolor='#333')
    ax1_ins.set_xticks(range(len(enc_names)))
    ax1_ins.set_xticklabels(short_names, fontsize=6, color=FG, rotation=30)
    ax1_ins.set_ylabel('sig', fontsize=7, color=FG)
    ax1_ins.tick_params(colors=FG, labelsize=6)
    for spine in ax1_ins.spines.values():
        spine.set_edgecolor('#444')
    for i, v in enumerate(enc_vals):
        ax1_ins.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=7,
                     fontweight='bold')

    # ── Top-center: D2 — Heatmap of Cohen's d ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)

    if d2['heatmap_metrics']:
        n_metrics = len(d2['heatmap_metrics'])
        # Limit display to top 40 metrics for readability
        display_n = min(40, n_metrics)
        d_k1 = np.array(d2['heatmap_d_k1'][:display_n])
        d_k2 = np.array(d2['heatmap_d_k2'][:display_n])
        cats = d2['heatmap_cats'][:display_n]

        heatmap_data = np.column_stack([d_k1, d_k2])
        im = ax2.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                        vmin=-5, vmax=5, interpolation='nearest')

        # Color-code row labels by category
        cat_colors = {'conv': '#E91E63', 'univ': '#4CAF50', 'div': '#2196F3'}
        # Add category color bar on the left
        for i, cat in enumerate(cats):
            ax2.plot(-0.8, i, 's', color=cat_colors[cat], markersize=4,
                     clip_on=False)

        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['k=1\n(3n+1)', 'k=2\n(5n+1)'], fontsize=8, color=FG)
        ax2.set_yticks([])  # too many to label individually
        ax2.set_ylabel(f'Metrics (n={display_n})', color=FG, fontsize=9)

        # Legend for categories
        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0], [0], marker='s', color='none', markerfacecolor='#E91E63',
                   label=f"Conv-specific ({len(d2['categories']['convergence_specific'])})", markersize=6),
            Line2D([0], [0], marker='s', color='none', markerfacecolor='#4CAF50',
                   label=f"Universal ({len(d2['categories']['universal'])})", markersize=6),
            Line2D([0], [0], marker='s', color='none', markerfacecolor='#2196F3',
                   label=f"Div-specific ({len(d2['categories']['divergence_specific'])})", markersize=6),
        ]
        ax2.legend(handles=legend_els, fontsize=6, facecolor='#222',
                   edgecolor='#444', labelcolor=FG, loc='lower right')

        cb = fig.colorbar(im, ax=ax2, shrink=0.6, pad=0.02)
        cb.set_label("Cohen's d vs random", color=FG, fontsize=8)
        cb.ax.tick_params(colors=FG, labelsize=7)

    ax2.set_title('D2: Phase Transition Anatomy', fontsize=11,
                  fontweight='bold', color=FG)

    # ── Top-right: D3 — Composite sig counts with reference lines ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    comp_names = list(d3['comp_results'].keys())
    comp_vals = [d3['comp_results'][k] for k in comp_names]
    colors3 = ['#E91E63', '#FF9800', '#4CAF50', '#9C27B0']

    bars3 = ax3.bar(range(len(comp_names)), comp_vals,
                    color=colors3[:len(comp_names)], alpha=0.85, edgecolor='#333')

    # Reference lines for individual baselines
    for label, val, color, ls in [('BP0=77', 77, '#E91E63', '--'),
                                   ('DE2=78', 78, '#4CAF50', '--'),
                                   ('BP7=68', 68, '#2196F3', ':')]:
        ax3.axhline(y=val, color=color, linestyle=ls, linewidth=1.5, alpha=0.5,
                    label=label)

    ax3.set_xticks(range(len(comp_names)))
    ax3.set_xticklabels(comp_names, fontsize=8, color=FG, rotation=15)
    ax3.set_ylabel('Significant metrics vs random', color=FG, fontsize=9)
    ax3.set_title('D3: Cross-Scale Composition', fontsize=11,
                  fontweight='bold', color=FG)
    ax3.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)
    for i, v in enumerate(comp_vals):
        ax3.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=9,
                 fontweight='bold')

    # ── Bottom-left: D4 — Scatter of drift vs metric for top detectors ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    if d4['scatter_data']:
        colors4 = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
        for idx, (km, sdata) in enumerate(d4['scatter_data'].items()):
            drift_vals = sdata['drift_vals']
            metric_vals = sdata['metric_vals']
            # Normalize metric values to [0, 1] for display
            mv_arr = np.array(metric_vals)
            mv_range = mv_arr.max() - mv_arr.min()
            if mv_range > 1e-10:
                mv_norm = (mv_arr - mv_arr.min()) / mv_range
            else:
                mv_norm = np.zeros_like(mv_arr)
            short = km.split(':')[1].replace('_', ' ')
            c = colors4[idx % len(colors4)]
            ax4.scatter(drift_vals, mv_norm, color=c, s=40, alpha=0.8,
                       label=f"{short} (r={sdata['r']:+.2f})", zorder=3)
            # Connect points with lines
            ax4.plot(drift_vals, mv_norm, color=c, alpha=0.3, linewidth=1)

        # Mark convergent/divergent boundary at drift=0
        ax4.axvline(x=0, color='#FF5722', linestyle='--', linewidth=2, alpha=0.7)
        ax4.text(0.02, 0.98, '← converges | diverges →', color='#FF5722',
                fontsize=7, transform=ax4.transAxes, va='top')

    ax4.set_xlabel('Empirical drift rate (log₂/step)', color=FG, fontsize=9)
    ax4.set_ylabel('Normalized metric value', color=FG, fontsize=9)
    ax4.set_title('D4: Drift Rate Detectors', fontsize=11,
                  fontweight='bold', color=FG)
    ax4.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor=FG,
               loc='lower right')

    # ── Bottom-center: D5 — Tree encoding results ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    tree_names = list(d5['encoding_results'].keys())
    tree_vals = [d5['encoding_results'][k] for k in tree_names]
    colors5 = ['#E91E63', '#4CAF50']

    bars5 = ax5.bar(range(len(tree_names)), tree_vals,
                    color=colors5[:len(tree_names)], alpha=0.85, edgecolor='#333')
    ax5.set_xticks(range(len(tree_names)))
    ax5.set_xticklabels(tree_names, fontsize=9, color=FG)
    ax5.set_ylabel('Significant metrics vs random', color=FG, fontsize=9)
    ax5.set_title('D5: Inverse Collatz Tree', fontsize=11,
                  fontweight='bold', color=FG)
    for i, v in enumerate(tree_vals):
        ax5.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=10,
                 fontweight='bold')

    # Inset: branching pattern heatmap (first 1000 nodes in 25x40 grid)
    branch_data = np.array(d5['branching'][:1000])
    if len(branch_data) >= 1000:
        grid = branch_data.reshape(25, 40)
        ax5_ins = ax5.inset_axes([0.55, 0.45, 0.42, 0.48])
        ax5_ins.set_facecolor('#222')
        ax5_ins.imshow(grid, cmap='RdYlGn', aspect='auto', interpolation='nearest',
                       vmin=0, vmax=2)
        ax5_ins.set_title('Branching pattern', fontsize=7, color=FG)
        ax5_ins.set_xticks([])
        ax5_ins.set_yticks([])
        for spine in ax5_ins.spines.values():
            spine.set_edgecolor('#444')

    # ── Bottom-right: Summary text panel ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    lines = [
        "Deep Collatz II — Key Findings",
        "",
        "D1: 2-adic Valuation",
    ]
    for enc, n_sig in d1['encoding_results'].items():
        if enc == 'v2_syracuse_5n1':
            lines.append(f"  syracuse(5n+1): {n_sig} sig")
        else:
            lines.append(f"  {enc}: {n_sig} sig")
    lines.append(f"  Mean v₂: 3n+1={np.mean(d1['v2_3n1']):.2f}, "
                 f"5n+1={np.mean(d1['v2_5n1']):.2f}")

    lines.append("")
    lines.append("D2: Phase Anatomy")
    lines.append(f"  Conv-specific: {len(d2['categories']['convergence_specific'])}")
    lines.append(f"  Universal:     {len(d2['categories']['universal'])}")
    lines.append(f"  Div-specific:  {len(d2['categories']['divergence_specific'])}")

    lines.append("")
    lines.append("D3: Cross-Scale Composition")
    for name, n_sig in d3['comp_results'].items():
        lines.append(f"  {name}: {n_sig} sig")

    lines.append("")
    lines.append("D4: Drift Detectors")
    lines.append(f"  {len(d4['detectors'])} metrics track drift")
    if d4['detectors']:
        top = d4['detectors'][0]
        short = top[0].split(':')[1]
        lines.append(f"  Best: {short} (r={top[1]:+.3f})")

    lines.append("")
    lines.append("D5: Inverse Tree")
    for enc, n_sig in d5['encoding_results'].items():
        lines.append(f"  {enc}: {n_sig} sig")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=8.5,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor='#444'))

    fig.suptitle('Deep Collatz Geometric Exploration II',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'collatz_deep2.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    d1, d2, d3, d4, d5 = main()
    make_figure(d1, d2, d3, d4, d5)
