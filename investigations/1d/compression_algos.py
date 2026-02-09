#!/usr/bin/env python3
"""
Investigation: Geometric signatures of compressed data.

Questions:
1. Does compressed data look random to exotic geometries?
2. Can we distinguish different compression algorithms?
3. Can we tell what TYPE of data was compressed (text vs image vs random)?
4. Does compression level affect geometric signature?

Compression algorithms: zlib, bz2, lzma (all in stdlib)
Source data types: random, English text, structured, repeating, prime gaps
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import zlib
import bz2
import lzma
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer

N_TRIALS = 20
SOURCE_SIZE = 10000  # larger source for meaningful compression

def generate_source(source_type, trial_seed, size=SOURCE_SIZE):
    """Generate different types of source data for compression."""
    rng = np.random.RandomState(trial_seed)

    if source_type == 'random':
        return bytes(rng.randint(0, 256, size, dtype=np.uint8))

    elif source_type == 'english':
        # Simulated English with word-like patterns
        words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
                 'and', 'then', 'she', 'said', 'hello', 'world', 'this', 'is',
                 'a', 'test', 'of', 'compression', 'with', 'repeated', 'words',
                 'that', 'should', 'compress', 'well', 'because', 'natural',
                 'language', 'has', 'patterns', 'like', 'common', 'letters']
        text = []
        while len(' '.join(text)) < size:
            text.append(rng.choice(words))
        return ' '.join(text)[:size].encode('ascii')

    elif source_type == 'structured':
        # CSV-like structured data
        lines = []
        while len('\n'.join(lines)) < size:
            vals = [str(rng.randint(0, 1000)) for _ in range(5)]
            lines.append(','.join(vals))
        return '\n'.join(lines)[:size].encode('ascii')

    elif source_type == 'repeating':
        # Highly repetitive
        pattern = bytes(rng.randint(0, 256, 32, dtype=np.uint8))
        return (pattern * (size // 32 + 1))[:size]

    elif source_type == 'binary_image':
        # Simulated grayscale image (smooth gradients with noise)
        width = 100
        height = size // width
        img = np.zeros(width * height, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                val = int(128 + 50 * np.sin(x / 10) * np.cos(y / 15))
                img[y * width + x] = np.clip(val + rng.randint(-5, 6), 0, 255)
        return bytes(img)

    elif source_type == 'low_entropy':
        # Only a few distinct values
        symbols = rng.randint(0, 4, size)
        return bytes(symbols.astype(np.uint8))

    else:
        raise ValueError(f"Unknown source: {source_type}")


def compress_data(data, algorithm, level=None):
    """Compress data with given algorithm."""
    if algorithm == 'zlib':
        return zlib.compress(data, level if level is not None else 6)
    elif algorithm == 'zlib_fast':
        return zlib.compress(data, 1)
    elif algorithm == 'zlib_best':
        return zlib.compress(data, 9)
    elif algorithm == 'bz2':
        return bz2.compress(data, compresslevel=level if level is not None else 9)
    elif algorithm == 'lzma':
        return lzma.compress(data)
    elif algorithm == 'none':
        return data
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("=" * 70)
    print("INVESTIGATION: Geometric Signatures of Compressed Data")
    print("=" * 70)

    source_types = ['random', 'english', 'structured', 'repeating', 'binary_image', 'low_entropy']
    algorithms = ['none', 'zlib', 'zlib_fast', 'zlib_best', 'bz2', 'lzma']

    key_metrics = [
        'E8 Lattice:unique_roots',
        'E8 Lattice:alignment_mean',
        'Torus T^2:coverage',
        'Torus T^2:chi2_uniformity',
        'Heisenberg (Nil):twist_rate',
        'Sol:anisotropy',
        'Tropical:linearity',
        'Fisher Information:trace_fisher',
        'Cantor (base 3):cantor_dimension',
        'Persistent Homology:n_significant_features',
        'Penrose (Quasicrystal):fivefold_balance',
    ]

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Part 1: How does compression change geometric signatures?
    print(f"\n{'='*70}")
    print("PART 1: Compression vs No Compression")
    print(f"{'='*70}")

    # Structure: source -> algorithm -> metric -> list
    all_data = {}
    compression_ratios = {}

    for src in source_types:
        print(f"\n  Source: {src}")
        all_data[src] = {}
        compression_ratios[src] = {}

        for alg in algorithms:
            all_data[src][alg] = defaultdict(list)
            ratios = []

            for trial in range(N_TRIALS):
                source = generate_source(src, trial)
                compressed = compress_data(source, alg)
                ratios.append(len(compressed) / len(source))

                # Take first 2000 bytes of compressed output
                analysis_data = np.frombuffer(compressed[:2000], dtype=np.uint8)
                if len(analysis_data) < 100:
                    continue

                try:
                    results = analyzer.analyze(analysis_data)
                    for r in results.results:
                        for mn, mv in r.metrics.items():
                            all_data[src][alg][f"{r.geometry_name}:{mn}"].append(mv)
                except Exception:
                    continue

            compression_ratios[src][alg] = np.mean(ratios)
            print(f"    {alg:>12}: ratio={np.mean(ratios):.3f} ({len(all_data[src][alg].get(key_metrics[0], []))} trials)")

    # Also generate random baseline
    random_metrics = defaultdict(list)
    for trial in range(N_TRIALS):
        data = np.frombuffer(os.urandom(2000), dtype=np.uint8)
        results = analyzer.analyze(data)
        for r in results.results:
            for mn, mv in r.metrics.items():
                random_metrics[f"{r.geometry_name}:{mn}"].append(mv)

    # Summary table: key metrics for each source+algorithm
    print(f"\n{'='*70}")
    print("PART 2: Geometric Fingerprints")
    print(f"{'='*70}")

    for src in source_types:
        print(f"\n--- {src} (compression ratios: " +
              ", ".join(f"{a}={compression_ratios[src][a]:.2f}" for a in algorithms) + ") ---\n")

        header = f"{'Metric':<35}" + "".join(f"{a[:8]:>10}" for a in algorithms) + f"{'random':>10}"
        print(header)
        print("-" * len(header))

        for km in key_metrics[:6]:
            row = f"{km.split(':')[1][:34]:<35}"
            for alg in algorithms:
                vals = all_data[src][alg].get(km, [0])
                row += f"{np.mean(vals):>10.3f}"
            ref = random_metrics.get(km, [0])
            row += f"{np.mean(ref):>10.3f}"
            print(row)

    # Part 3: Can we distinguish compressed from random?
    print(f"\n{'='*70}")
    print("PART 3: Compressed Data vs True Random")
    print(f"{'='*70}")

    n_tests = len(key_metrics)
    alpha = 0.05 / max(n_tests, 1)

    significant = []
    for src in source_types:
        for alg in algorithms:
            if alg == 'none':
                continue
            for km in key_metrics:
                if km not in all_data[src][alg] or km not in random_metrics:
                    continue
                if len(all_data[src][alg][km]) < 2 or len(random_metrics[km]) < 2:
                    continue
                d = cohens_d(all_data[src][alg][km], random_metrics[km])
                _, p = stats.ttest_ind(all_data[src][alg][km], random_metrics[km], equal_var=False)
                if abs(d) > 0.8 and p < alpha:
                    significant.append((src, alg, km, d, p))

    if significant:
        print(f"\n{len(significant)} significant differences from random:\n")
        print(f"{'Source':<15} {'Algorithm':<12} {'Metric':<35} {'d':>8} {'p':>12}")
        print("-" * 85)
        for src, alg, km, d, p in sorted(significant, key=lambda x: -abs(x[3]))[:30]:
            print(f"{src:<15} {alg:<12} {km.split(':')[1][:34]:<35} {d:>+8.2f} {p:>12.2e}")
    else:
        print("\nNo significant differences! Compressed data looks random.")

    # Part 4: Can we distinguish compression algorithms from each other?
    print(f"\n{'='*70}")
    print("PART 4: Distinguishing Compression Algorithms")
    print(f"{'='*70}")

    # For each source type, can we tell zlib from bz2 from lzma?
    comp_algs = ['zlib', 'bz2', 'lzma']
    n_pair_tests = len(key_metrics)  # per comparison family
    alpha_pair = 0.05 / max(n_pair_tests, 1)

    alg_diffs = []
    for src in source_types:
        for a1, a2 in [('zlib', 'bz2'), ('zlib', 'lzma'), ('bz2', 'lzma')]:
            best_d = 0
            best_km = None
            best_p = 1
            for km in key_metrics:
                if km not in all_data[src][a1] or km not in all_data[src][a2]:
                    continue
                if len(all_data[src][a1][km]) < 2 or len(all_data[src][a2][km]) < 2:
                    continue
                d = cohens_d(all_data[src][a1][km], all_data[src][a2][km])
                _, p = stats.ttest_ind(all_data[src][a1][km], all_data[src][a2][km], equal_var=False)
                if abs(d) > abs(best_d):
                    best_d = d
                    best_km = km
                    best_p = p
            if best_km:
                alg_diffs.append((src, a1, a2, best_km, best_d, best_p))

    print(f"\n{'Source':<15} {'Alg1':<8} {'Alg2':<8} {'Best Metric':<30} {'d':>8} {'Sig?':>6}")
    print("-" * 80)
    for src, a1, a2, km, d, p in sorted(alg_diffs, key=lambda x: -abs(x[4])):
        sig = "YES" if abs(d) > 0.8 and p < alpha_pair else ""
        print(f"{src:<15} {a1:<8} {a2:<8} {km.split(':')[1][:29]:<30} {d:>+8.2f} {sig:>6}")

    # Part 5: Can we tell what was compressed?
    print(f"\n{'='*70}")
    print("PART 5: Identifying Source Data Type from Compressed Output")
    print(f"{'='*70}")

    # For zlib compression, can we distinguish source types?
    alg = 'zlib'
    n_src_tests = len(key_metrics)
    alpha_src = 0.05 / max(n_src_tests, 1)

    print(f"\nUsing {alg} compression:\n")
    print(f"{'Source1':<15} {'Source2':<15} {'Best Metric':<30} {'d':>8} {'Sig?':>6}")
    print("-" * 80)

    for i, s1 in enumerate(source_types):
        for s2 in source_types[i+1:]:
            best_d = 0
            best_km = None
            best_p = 1
            for km in key_metrics:
                if km not in all_data[s1][alg] or km not in all_data[s2][alg]:
                    continue
                if len(all_data[s1][alg][km]) < 2 or len(all_data[s2][alg][km]) < 2:
                    continue
                d = cohens_d(all_data[s1][alg][km], all_data[s2][alg][km])
                _, p = stats.ttest_ind(all_data[s1][alg][km], all_data[s2][alg][km], equal_var=False)
                if abs(d) > abs(best_d):
                    best_d = d
                    best_km = km
                    best_p = p
            if best_km:
                sig = "YES" if abs(best_d) > 0.8 and best_p < alpha_src else ""
                print(f"{s1:<15} {s2:<15} {best_km.split(':')[1][:29]:<30} {best_d:>+8.2f} {sig:>6}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    n_comp_vs_random = len(significant)
    n_alg_diffs = sum(1 for _, _, _, _, d, p in alg_diffs if abs(d) > 0.8 and p < alpha_pair)

    print(f"\nCompressed vs random: {n_comp_vs_random} significant differences")
    print(f"Algorithm distinction: {n_alg_diffs} distinguishable pairs")
    print(f"Compression ratios vary from {min(min(compression_ratios[s][a] for a in algorithms if a != 'none') for s in source_types):.3f} to {max(max(compression_ratios[s][a] for a in algorithms if a != 'none') for s in source_types):.3f}")

    if n_comp_vs_random == 0:
        print("\nConclusion: Compressed data is geometrically indistinguishable from random.")
        print("This makes sense - good compression maximizes entropy of the output.")
    else:
        print(f"\nConclusion: {n_comp_vs_random} geometric differences detected between compressed and random data!")

    print("\n[Investigation complete]")


if __name__ == '__main__':
    main()
