#!/usr/bin/env python3
"""
Investigation: Can exotic geometries detect PRNG weaknesses?

We know from previous work that E8 catches RANDU *if* you use the right
representation (plane residuals). This investigation systematically tests:

1. Multiple PRNGs: MT19937, SystemRandom, RANDU, LCG (glibc), XorShift, PCG
2. Multiple representations: raw bytes, pairs, triples, modular residuals
3. All 21 geometries
4. Rigorous statistics

Hypothesis: Weaker PRNGs will show geometric signatures that differ from
cryptographic random, but the representation matters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import struct
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer

N_TRIALS = 20
DATA_SIZE = 2000

class RANDU:
    """Infamous IBM PRNG. Falls on 15 planes in 3D."""
    def __init__(self, seed=1):
        self.state = seed & 0x7FFFFFFF
    def next(self):
        self.state = (65539 * self.state) % (2**31)
        return self.state

class LCG_glibc:
    """glibc LCG: state = (1103515245 * state + 12345) mod 2^31"""
    def __init__(self, seed=1):
        self.state = seed
    def next(self):
        self.state = (1103515245 * self.state + 12345) % (2**31)
        return self.state

class XorShift32:
    """Simple XorShift32"""
    def __init__(self, seed=1):
        self.state = seed if seed != 0 else 1
    def next(self):
        x = self.state & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x

class MINSTD:
    """Minimal standard PRNG: Park-Miller"""
    def __init__(self, seed=1):
        self.state = seed if seed > 0 else 1
    def next(self):
        self.state = (16807 * self.state) % (2**31 - 1)
        return self.state


def generate_prng_data(prng_name, trial_seed, size=DATA_SIZE, representation='raw_bytes'):
    """Generate data from various PRNGs in various representations."""
    raw_values = []
    n_values = size * 4  # generate extra for representations that consume more

    if prng_name == 'mt19937':
        rng = np.random.RandomState(trial_seed)
        raw_values = [int(rng.randint(0, 2**31)) for _ in range(n_values)]

    elif prng_name == 'system_random':
        import secrets
        raw_values = [secrets.randbelow(2**31) for _ in range(n_values)]

    elif prng_name == 'randu':
        r = RANDU(seed=trial_seed + 1)
        raw_values = [r.next() for _ in range(n_values)]

    elif prng_name == 'lcg_glibc':
        r = LCG_glibc(seed=trial_seed + 1)
        raw_values = [r.next() for _ in range(n_values)]

    elif prng_name == 'xorshift32':
        r = XorShift32(seed=trial_seed + 1)
        raw_values = [r.next() for _ in range(n_values)]

    elif prng_name == 'minstd':
        r = MINSTD(seed=trial_seed + 1)
        raw_values = [r.next() for _ in range(n_values)]

    else:
        raise ValueError(f"Unknown PRNG: {prng_name}")

    raw_values = np.array(raw_values, dtype=np.int64)

    if representation == 'raw_bytes':
        # Take low byte of each value
        return (raw_values[:size] & 0xFF).astype(np.uint8)

    elif representation == 'high_bytes':
        # Take high byte (bits 23-30)
        return ((raw_values[:size] >> 23) & 0xFF).astype(np.uint8)

    elif representation == 'pairs_mod':
        # Pairs: (v[i] + v[i+1]) mod 256
        vals = raw_values[:size + 1]
        return ((vals[:-1] + vals[1:])[:size] % 256).astype(np.uint8)

    elif representation == 'triples_residual':
        # RANDU-breaking: compute 9*x - 6*y + z for consecutive triples
        n = min(len(raw_values) // 3, size)
        result = []
        for i in range(n):
            x = raw_values[3*i]
            y = raw_values[3*i + 1]
            z = raw_values[3*i + 2]
            residual = (9 * x - 6 * y + z) % 256
            result.append(int(residual))
        return np.array(result[:size], dtype=np.uint8)

    elif representation == 'xor_consecutive':
        # XOR consecutive values
        vals = raw_values[:size + 1]
        return ((vals[:-1] ^ vals[1:])[:size] % 256).astype(np.uint8)

    elif representation == 'bit_pattern':
        # Bit extraction: pack individual bits from high bits
        bits = []
        for v in raw_values:
            for bit in range(8):
                bits.append((int(v) >> (bit + 16)) & 1)
                if len(bits) >= size:
                    break
            if len(bits) >= size:
                break
        return np.array(bits[:size], dtype=np.uint8) * 255  # 0 or 255

    else:
        raise ValueError(f"Unknown representation: {representation}")


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("=" * 70)
    print("INVESTIGATION: PRNG Weakness Detection via Exotic Geometries")
    print("=" * 70)

    prngs = ['system_random', 'mt19937', 'randu', 'lcg_glibc', 'xorshift32', 'minstd']
    representations = ['raw_bytes', 'high_bytes', 'pairs_mod', 'triples_residual', 'xor_consecutive']

    key_metrics = [
        'E8 Lattice:unique_roots',
        'Torus T²:coverage',
        'Sol:anisotropy',
        'Heisenberg (centered):twist_rate',
        'Tropical:linearity',
        'Persistent Homology:n_significant',
        'Penrose:fivefold_balance',
        'Cantor Set:cantor_dimension',
        'Lorentzian:causal_density',
        'Symplectic:symplectic_area',
    ]

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Structure: prng -> representation -> metric -> list of values
    all_data = {p: {r: defaultdict(list) for r in representations} for p in prngs}

    total = len(prngs) * len(representations) * N_TRIALS
    count = 0

    for rep in representations:
        print(f"\n--- Representation: {rep} ---")
        for prng in prngs:
            print(f"  {prng}...", end=" ", flush=True)
            for trial in range(N_TRIALS):
                count += 1
                try:
                    data = generate_prng_data(prng, trial, DATA_SIZE, rep)
                    results = analyzer.analyze(data)
                    for r in results.results:
                        for mn, mv in r.metrics.items():
                            full_key = f"{r.geometry_name}:{mn}"
                            all_data[prng][rep][full_key].append(mv)
                except Exception as e:
                    pass  # skip failures silently
            print("done")

    # Analysis
    reference = 'system_random'

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # For each representation, compare each PRNG to system_random
    significant_findings = []
    n_tests = len(prngs) * len(representations) * len(key_metrics)
    alpha = 0.05 / max(n_tests, 1)

    for rep in representations:
        print(f"\n{'='*70}")
        print(f"Representation: {rep}")
        print(f"{'='*70}")
        print(f"\n{'Metric':<40} {'PRNG':<14} {'Mean':>8} {'Ref':>8} {'d':>8} {'p':>12} {'Sig':>5}")
        print("-" * 95)

        for km in key_metrics:
            if km not in all_data[reference][rep] or len(all_data[reference][rep][km]) < 2:
                continue
            ref_vals = all_data[reference][rep][km]

            for prng in prngs:
                if prng == reference:
                    continue
                if km not in all_data[prng][rep] or len(all_data[prng][rep][km]) < 2:
                    continue
                prng_vals = all_data[prng][rep][km]

                d = cohens_d(prng_vals, ref_vals)
                _, p = stats.ttest_ind(prng_vals, ref_vals)
                sig = "***" if p < alpha and abs(d) > 0.8 else ""

                if abs(d) > 0.8:
                    print(f"{km:<40} {prng:<14} {np.mean(prng_vals):>8.2f} {np.mean(ref_vals):>8.2f} {d:>8.2f} {p:>12.2e} {sig:>5}")
                    if sig:
                        significant_findings.append((rep, km, prng, d, p))

    # Summary: which PRNGs are caught by which geometry+representation combo?
    print("\n" + "=" * 70)
    print("DETECTION MATRIX: Which geometry+representation catches which PRNG?")
    print("=" * 70)

    # Build detection matrix
    detection = {}  # (prng, rep, geom) -> (d, p)
    for rep in representations:
        for km in key_metrics:
            if km not in all_data[reference][rep]:
                continue
            ref_vals = all_data[reference][rep][km]
            for prng in prngs:
                if prng == reference:
                    continue
                if km not in all_data[prng][rep]:
                    continue
                prng_vals = all_data[prng][rep][km]
                d = cohens_d(prng_vals, ref_vals)
                _, p = stats.ttest_ind(prng_vals, ref_vals)
                geom = km.split(':')[0]
                key = (prng, rep, geom)
                if key not in detection or abs(d) > abs(detection[key][0]):
                    detection[key] = (d, p)

    # Print matrix: PRNG x Representation, showing which geometries detect it
    for prng in prngs:
        if prng == reference:
            continue
        print(f"\n{prng}:")
        print(f"  {'Representation':<25} {'Detected by geometries (d > 0.8, p < {alpha:.1e})'}")
        print(f"  {'-'*70}")
        for rep in representations:
            detections = []
            for km in key_metrics:
                geom = km.split(':')[0]
                key = (prng, rep, geom)
                if key in detection:
                    d, p = detection[key]
                    if abs(d) > 0.8 and p < alpha:
                        detections.append(f"{geom}(d={d:.1f})")
            if detections:
                print(f"  {rep:<25} {', '.join(detections)}")
            else:
                print(f"  {rep:<25} (not detected)")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if significant_findings:
        # Group by PRNG
        by_prng = defaultdict(list)
        for rep, km, prng, d, p in significant_findings:
            by_prng[prng].append((rep, km, d, p))

        for prng, findings in sorted(by_prng.items()):
            print(f"\n{prng}: {len(findings)} significant detection(s)")
            for rep, km, d, p in findings:
                print(f"  [{rep}] {km}: d={d:.2f}, p={p:.2e}")

        # Which representation is most powerful?
        by_rep = defaultdict(int)
        for rep, km, prng, d, p in significant_findings:
            by_rep[rep] += 1
        print(f"\nMost powerful representations:")
        for rep, count in sorted(by_rep.items(), key=lambda x: -x[1]):
            print(f"  {rep}: {count} detections")

        # Which geometry is most powerful?
        by_geom = defaultdict(int)
        for rep, km, prng, d, p in significant_findings:
            geom = km.split(':')[0]
            by_geom[geom] += 1
        print(f"\nMost powerful geometries:")
        for geom, count in sorted(by_geom.items(), key=lambda x: -x[1]):
            print(f"  {geom}: {count} detections")

    else:
        print("\nNo significant findings! All PRNGs look random in all representations.")
        print("This could mean:")
        print("  1. The PRNGs are better than expected")
        print("  2. We need different representations")
        print("  3. We need larger sample sizes")

    # Compare to shuffle baseline for detected PRNGs
    print("\n" + "=" * 70)
    print("SHUFFLE VALIDATION (for significant findings)")
    print("=" * 70)

    validated = []
    for rep, km, prng, d, p in significant_findings[:10]:  # top 10
        data = generate_prng_data(prng, 999, 5000, rep)
        results = analyzer.analyze(data)
        shuffled = data.copy()
        np.random.shuffle(shuffled)
        results_shuf = analyzer.analyze(shuffled)

        for r, rs in zip(results.results, results_shuf.results):
            for mn in r.metrics:
                full_key = f"{r.geometry_name}:{mn}"
                if full_key == km:
                    ratio = r.metrics[mn] / (rs.metrics[mn] + 1e-10)
                    survived = abs(ratio - 1.0) > 0.2
                    status = "VALIDATED" if survived else "ARTIFACT (shuffle explains it)"
                    print(f"  [{rep}] {prng} {km}: real={r.metrics[mn]:.3f} shuf={rs.metrics[mn]:.3f} ratio={ratio:.3f} -> {status}")
                    if survived:
                        validated.append((rep, km, prng, d, p, ratio))

    print(f"\n{len(validated)} findings survived shuffle validation out of {min(len(significant_findings), 10)} tested")

    print("\n[Investigation complete]")


if __name__ == '__main__':
    main()
