#!/usr/bin/env python3
"""
Investigation: Do different hash functions have distinguishable geometric signatures?

Hypothesis: Hash functions designed to be pseudorandom should all look similar to
random data. But weaker/older hashes (MD5, SHA-1) might have subtle structural
differences from stronger ones (SHA-256, SHA-3, BLAKE2).

Null hypothesis: All hash outputs are geometrically indistinguishable from each other
and from random data.

Methodology:
1. Generate hash outputs by hashing sequential integers (0, 1, 2, ...)
2. Concatenate outputs and analyze as byte streams
3. Compare all hash functions against each other and against os.urandom
4. Use shuffle baseline, Cohen's d, p-values, Bonferroni correction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import hashlib
import struct
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer

PYTHON = sys.executable
N_TRIALS = 20
DATA_SIZE = 2000  # bytes per trial
N_HASHES_PER_TRIAL = DATA_SIZE // 32 + 1  # enough hashes to fill DATA_SIZE

def generate_hash_data(hash_func_name, trial_seed, size=DATA_SIZE):
    """Generate data by hashing sequential integers starting from trial_seed * 10000."""
    offset = trial_seed * 10000
    buf = bytearray()
    i = 0
    while len(buf) < size:
        msg = struct.pack('>Q', offset + i)
        if hash_func_name == 'md5':
            h = hashlib.md5(msg).digest()
        elif hash_func_name == 'sha1':
            h = hashlib.sha1(msg).digest()
        elif hash_func_name == 'sha256':
            h = hashlib.sha256(msg).digest()
        elif hash_func_name == 'sha3_256':
            h = hashlib.sha3_256(msg).digest()
        elif hash_func_name == 'blake2b':
            h = hashlib.blake2b(msg, digest_size=32).digest()
        elif hash_func_name == 'sha512':
            h = hashlib.sha512(msg).digest()
        elif hash_func_name == 'random':
            h = os.urandom(32)
        else:
            raise ValueError(f"Unknown hash: {hash_func_name}")
        buf.extend(h)
        i += 1
    return np.frombuffer(bytes(buf[:size]), dtype=np.uint8)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def main():
    print("=" * 70)
    print("INVESTIGATION: Hash Function Geometric Signatures")
    print("=" * 70)

    hash_funcs = ['md5', 'sha1', 'sha256', 'sha3_256', 'blake2b', 'sha512', 'random']

    # Key metrics to track (one per geometry, the most discriminative)
    key_metrics = {
        'E8 Lattice': 'unique_roots',
        'Torus T²': 'coverage',
        'Hyperbolic H²': 'mean_radius',
        'Heisenberg': 'twist_rate',
        'Heisenberg (centered)': 'twist_rate',
        'Spherical S²': 'concentration',
        'Sol': 'anisotropy',
        'Tropical': 'linearity',
        'Projective P²': 'collinearity',
        'Persistent Homology': 'n_significant',
        'Penrose': 'fivefold_balance',
        'Cantor Set': 'cantor_dimension',
        '2-adic': 'ultrametric_dim',
        'Lorentzian': 'causal_density',
        'Symplectic': 'symplectic_area',
        'Spiral': 'growth_rate',
        'Fisher': 'fisher_info',
        'Wasserstein': 'w1_distance',
    }

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Collect metrics for all hash functions across trials
    all_metrics = {hf: defaultdict(list) for hf in hash_funcs}

    print(f"\nRunning {N_TRIALS} trials per hash function, {DATA_SIZE} bytes each...")
    print(f"Hash functions: {hash_funcs}")
    print()

    for trial in range(N_TRIALS):
        if trial % 5 == 0:
            print(f"  Trial {trial}/{N_TRIALS}...")
        for hf in hash_funcs:
            data = generate_hash_data(hf, trial)
            results = analyzer.analyze(data)
            for r in results.results:
                for metric_name, metric_val in r.metrics.items():
                    full_key = f"{r.geometry_name}:{metric_name}"
                    all_metrics[hf][full_key].append(metric_val)

    print("\n" + "=" * 70)
    print("RESULTS: Hash Function Geometric Signatures")
    print("=" * 70)

    # 1. Summary table: mean values for key metrics
    print("\n--- Mean Values (Key Metrics) ---\n")
    header = f"{'Metric':<35}" + "".join(f"{hf:>10}" for hf in hash_funcs)
    print(header)
    print("-" * len(header))

    for geom_name, metric_name in key_metrics.items():
        full_key = f"{geom_name}:{metric_name}"
        if full_key in all_metrics[hash_funcs[0]]:
            row = f"{geom_name[:25]+':'+metric_name[:8]:<35}"
            for hf in hash_funcs:
                vals = all_metrics[hf][full_key]
                row += f"{np.mean(vals):>10.3f}"
            print(row)

    # 2. Statistical comparison: each hash vs random baseline
    print("\n\n--- Each Hash vs Random (Cohen's d, p-value) ---\n")
    print(f"{'Metric':<40} {'Hash':<10} {'d':>8} {'p':>12} {'Sig?':>6}")
    print("-" * 80)

    n_comparisons = len(key_metrics) * (len(hash_funcs) - 1)
    alpha_bonferroni = 0.05 / n_comparisons

    significant_findings = []

    for geom_name, metric_name in key_metrics.items():
        full_key = f"{geom_name}:{metric_name}"
        if full_key not in all_metrics['random']:
            continue
        random_vals = all_metrics['random'][full_key]

        for hf in hash_funcs:
            if hf == 'random':
                continue
            hash_vals = all_metrics[hf][full_key]
            d = cohens_d(hash_vals, random_vals)
            _, p = stats.ttest_ind(hash_vals, random_vals)
            sig = "***" if p < alpha_bonferroni else ""

            if abs(d) > 0.5:  # medium+ effect size
                print(f"{full_key:<40} {hf:<10} {d:>8.3f} {p:>12.2e} {sig:>6}")
                if p < alpha_bonferroni:
                    significant_findings.append((full_key, hf, d, p))

    # 3. Pairwise hash comparison - can we tell hashes apart from each other?
    print("\n\n--- Pairwise Hash Comparisons (significant only) ---\n")
    print(f"{'Metric':<35} {'Hash1':<10} {'Hash2':<10} {'d':>8} {'p':>12}")
    print("-" * 80)

    pairwise_sigs = []
    hash_only = [h for h in hash_funcs if h != 'random']
    n_pairs = len(list(__import__('itertools').combinations(hash_only, 2))) * len(key_metrics)
    alpha_pair = 0.05 / max(n_pairs, 1)

    for geom_name, metric_name in key_metrics.items():
        full_key = f"{geom_name}:{metric_name}"
        if full_key not in all_metrics[hash_only[0]]:
            continue

        for h1, h2 in __import__('itertools').combinations(hash_only, 2):
            vals1 = all_metrics[h1][full_key]
            vals2 = all_metrics[h2][full_key]
            d = cohens_d(vals1, vals2)
            _, p = stats.ttest_ind(vals1, vals2)

            if abs(d) > 0.8 and p < alpha_pair:
                print(f"{full_key:<35} {h1:<10} {h2:<10} {d:>8.3f} {p:>12.2e}")
                pairwise_sigs.append((full_key, h1, h2, d, p))

    # 4. Shuffle test: does ordering matter for hash outputs?
    print("\n\n--- Shuffle Test (do hash outputs have sequential structure?) ---\n")
    print(f"{'Hash':<12} {'Metric':<35} {'Real':>10} {'Shuffled':>10} {'Ratio':>8}")
    print("-" * 80)

    shuffle_findings = []
    for hf in hash_funcs:
        if hf == 'random':
            continue
        # Use one larger sample for shuffle test
        data = generate_hash_data(hf, 999, size=5000)
        results_real = analyzer.analyze(data)

        shuffled = data.copy()
        np.random.shuffle(shuffled)
        results_shuf = analyzer.analyze(shuffled)

        for r_real, r_shuf in zip(results_real.results, results_shuf.results):
            for metric in r_real.metrics:
                real_val = r_real.metrics[metric]
                shuf_val = r_shuf.metrics[metric]
                ratio = real_val / (shuf_val + 1e-10)
                if ratio > 1.5 or ratio < 0.67:
                    full_key = f"{r_real.geometry_name}:{metric}"
                    print(f"{hf:<12} {full_key:<35} {real_val:>10.3f} {shuf_val:>10.3f} {ratio:>8.2f}")
                    shuffle_findings.append((hf, full_key, real_val, shuf_val, ratio))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBonferroni-corrected alpha (vs random): {alpha_bonferroni:.6f}")
    print(f"Number of comparisons (vs random): {n_comparisons}")

    if significant_findings:
        print(f"\n{len(significant_findings)} significant differences found (hash vs random):")
        for full_key, hf, d, p in significant_findings:
            print(f"  {hf} on {full_key}: d={d:.3f}, p={p:.2e}")
    else:
        print("\nNo significant differences between any hash and random.")
        print("CONCLUSION: All tested hash functions are geometrically indistinguishable from random.")
        print("This is expected for well-designed hash functions.")

    if pairwise_sigs:
        print(f"\n{len(pairwise_sigs)} significant pairwise differences between hashes:")
        for full_key, h1, h2, d, p in pairwise_sigs:
            print(f"  {h1} vs {h2} on {full_key}: d={d:.3f}, p={p:.2e}")
    else:
        print("\nNo significant pairwise differences between hash functions.")

    if shuffle_findings:
        print(f"\n{len(shuffle_findings)} metrics show ordering effects in hash output.")
    else:
        print("\nNo ordering effects detected (hash outputs are geometrically independent of input order).")

    print("\n[Investigation complete]")

if __name__ == '__main__':
    main()
