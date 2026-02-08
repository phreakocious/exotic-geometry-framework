#!/usr/bin/env python3
"""
Investigation: Bit-level steganography detection via exotic geometries.

Follow-up to the byte-level failure. Instead of analyzing raw bytes,
extract the LSBs as a separate bitstream, pack them into bytes, and
analyze THOSE geometrically.

Hypothesis: LSB replacement makes the LSB stream random (message bits).
Clean carrier LSBs have structure (correlated with higher bits).
Geometric analysis of the LSB stream should detect this difference.

Also test: higher bit planes (bit 1, bit 2, etc.) for comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer

N_TRIALS = 30  # more trials for subtle effects
DATA_SIZE = 4000  # larger data for bit extraction

def generate_carrier(carrier_type, trial_seed, size=DATA_SIZE):
    """Generate carrier data with natural LSB correlations."""
    rng = np.random.RandomState(trial_seed)

    if carrier_type == 'gradient':
        x = np.linspace(0, 255, size)
        noise = rng.normal(0, 3, size)
        return np.clip(x + noise, 0, 255).astype(np.uint8)

    elif carrier_type == 'photo_blocks':
        data = np.zeros(size, dtype=np.float64)
        pos = 0
        while pos < size:
            block_size = rng.randint(30, 150)
            block_val = rng.randint(0, 256)
            end = min(pos + block_size, size)
            data[pos:end] = block_val + rng.normal(0, 2, end - pos)
            pos = end
        return np.clip(data, 0, 255).astype(np.uint8)

    elif carrier_type == 'natural_texture':
        t = np.linspace(0, 30 * np.pi, size)
        pattern = 128 + 60 * np.sin(t) + 30 * np.sin(3.7 * t) + 10 * np.cos(11.3 * t)
        noise = rng.normal(0, 4, size)
        return np.clip(pattern + noise, 0, 255).astype(np.uint8)

    elif carrier_type == 'dark_image':
        # Mostly dark with some bright spots
        data = rng.exponential(15, size)
        bright_spots = rng.random(size) < 0.05
        data[bright_spots] = rng.randint(100, 256, bright_spots.sum())
        return np.clip(data, 0, 255).astype(np.uint8)

    elif carrier_type == 'high_contrast':
        # Bimodal distribution (like B&W photos)
        data = np.where(rng.random(size) < 0.4,
                        rng.normal(40, 10, size),
                        rng.normal(200, 15, size))
        return np.clip(data, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown carrier: {carrier_type}")


def embed_lsb(carrier, rate, seed=42):
    """LSB replacement at given rate."""
    rng = np.random.RandomState(seed)
    stego = carrier.copy()
    n_embed = int(len(carrier) * rate)
    if n_embed == 0:
        return stego

    positions = np.sort(rng.choice(len(carrier), n_embed, replace=False))
    message_bits = rng.randint(0, 2, n_embed)

    for i, pos in enumerate(positions):
        stego[pos] = (stego[pos] & 0xFE) | message_bits[i]

    return stego


def extract_bit_plane(data, bit=0):
    """Extract a specific bit plane and pack into bytes."""
    bits = (data >> bit) & 1
    # Pack 8 bits into each byte
    n_bytes = len(bits) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(8):
        if i * n_bytes + n_bytes <= len(bits):
            packed |= (bits[i::8][:n_bytes] << i).astype(np.uint8)
    return packed


def extract_lsb_pairs(data):
    """Extract LSB pairs: pack consecutive LSB pairs into bytes for more structure."""
    lsbs = data & 1
    n = len(lsbs) // 8
    result = np.zeros(n, dtype=np.uint8)
    for i in range(8):
        result |= (lsbs[i::8][:n] << i).astype(np.uint8)
    return result


def extract_lsb_transitions(data):
    """Count LSB transitions (0->1, 1->0) in sliding windows."""
    lsbs = data & 1
    transitions = np.abs(np.diff(lsbs.astype(np.int8)))
    # Window of 8: count transitions
    n = len(transitions) // 8
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        result[i] = np.sum(transitions[i*8:(i+1)*8])
    return result


def extract_lsb_correlation(data):
    """Measure correlation between LSB and bit 1 in sliding windows."""
    lsb = data & 1
    bit1 = (data >> 1) & 1
    # XOR gives correlation (same=0, different=1)
    xor = lsb ^ bit1
    # Pack into bytes
    n = len(xor) // 8
    result = np.zeros(n, dtype=np.uint8)
    for i in range(8):
        result |= (xor[i::8][:n] << i).astype(np.uint8)
    return result


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("=" * 70)
    print("INVESTIGATION: Bit-Level Steganography Detection")
    print("=" * 70)

    carrier_types = ['gradient', 'photo_blocks', 'natural_texture', 'dark_image', 'high_contrast']
    embed_rates = [0.05, 0.10, 0.25, 0.50, 1.00]
    extraction_methods = {
        'lsb_plane': lambda d: extract_bit_plane(d, 0),
        'bit1_plane': lambda d: extract_bit_plane(d, 1),
        'lsb_transitions': extract_lsb_transitions,
        'lsb_correlation': extract_lsb_correlation,
    }

    analyzer = GeometryAnalyzer().add_all_geometries()

    key_metrics = [
        'E8 Lattice:unique_roots',
        'Torus T^2:coverage',
        'Torus T^2:chi2_uniformity',
        'Heisenberg (Nil):twist_rate',
        'Heisenberg (Nil) (centered):twist_rate',
        'Tropical:linearity',
        'Persistent Homology:n_significant_features',
        'Fisher Information:trace_fisher',
        'Cantor (base 3):cantor_dimension',
        'Penrose (Quasicrystal):fivefold_balance',
    ]

    all_findings = []

    for method_name, extract_fn in extraction_methods.items():
        print(f"\n{'='*70}")
        print(f"EXTRACTION METHOD: {method_name}")
        print(f"{'='*70}")

        for ct in carrier_types:
            print(f"\n  Carrier: {ct}")

            # Collect clean vs stego metrics
            clean_metrics = defaultdict(list)
            stego_metrics = {rate: defaultdict(list) for rate in embed_rates}

            for trial in range(N_TRIALS):
                carrier = generate_carrier(ct, trial)

                # Clean extraction
                try:
                    extracted = extract_fn(carrier)
                    if len(extracted) < 50:
                        continue
                    results = analyzer.analyze(extracted)
                    for r in results.results:
                        for mn, mv in r.metrics.items():
                            clean_metrics[f"{r.geometry_name}:{mn}"].append(mv)
                except Exception:
                    continue

                # Stego extraction at each rate
                for rate in embed_rates:
                    stego = embed_lsb(carrier, rate, seed=trial * 100)
                    try:
                        extracted = extract_fn(stego)
                        if len(extracted) < 50:
                            continue
                        results = analyzer.analyze(extracted)
                        for r in results.results:
                            for mn, mv in r.metrics.items():
                                stego_metrics[rate][f"{r.geometry_name}:{mn}"].append(mv)
                    except Exception:
                        continue

            # Compare clean vs stego
            n_tests = len(key_metrics) * len(embed_rates)
            alpha = 0.05 / max(n_tests, 1)

            found = False
            for km in key_metrics:
                if km not in clean_metrics or len(clean_metrics[km]) < 5:
                    continue

                for rate in embed_rates:
                    if km not in stego_metrics[rate] or len(stego_metrics[rate][km]) < 5:
                        continue

                    d = cohens_d(stego_metrics[rate][km], clean_metrics[km])
                    _, p = stats.ttest_ind(stego_metrics[rate][km], clean_metrics[km])

                    if abs(d) > 0.8 and p < alpha:
                        if not found:
                            print(f"\n    {'Metric':<40} {'Rate':>6} {'Clean':>8} {'Stego':>8} {'d':>8} {'p':>12}")
                            print(f"    {'-'*84}")
                            found = True
                        print(f"    {km[:39]:<40} {rate:>5.0%} {np.mean(clean_metrics[km]):>8.3f} {np.mean(stego_metrics[rate][km]):>8.3f} {d:>+8.2f} {p:>12.2e} ***")
                        all_findings.append((method_name, ct, km, rate, d, p))

            if not found:
                print(f"    No significant detections")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if all_findings:
        print(f"\n{len(all_findings)} total significant detections!\n")

        # By method
        by_method = defaultdict(int)
        for m, ct, km, rate, d, p in all_findings:
            by_method[m] += 1
        print("By extraction method:")
        for m, count in sorted(by_method.items(), key=lambda x: -x[1]):
            print(f"  {m}: {count} detections")

        # By carrier
        by_carrier = defaultdict(int)
        for m, ct, km, rate, d, p in all_findings:
            by_carrier[ct] += 1
        print("\nBy carrier type:")
        for ct, count in sorted(by_carrier.items(), key=lambda x: -x[1]):
            print(f"  {ct}: {count} detections")

        # By geometry
        by_geom = defaultdict(int)
        for m, ct, km, rate, d, p in all_findings:
            geom = km.split(':')[0]
            by_geom[geom] += 1
        print("\nBy geometry:")
        for g, count in sorted(by_geom.items(), key=lambda x: -x[1]):
            print(f"  {g}: {count} detections")

        # Min detectable rate
        min_rates = {}
        for m, ct, km, rate, d, p in all_findings:
            key = (m, ct)
            if key not in min_rates or rate < min_rates[key]:
                min_rates[key] = rate
        print("\nMinimum detectable embedding rate:")
        for (m, ct), rate in sorted(min_rates.items(), key=lambda x: x[1]):
            print(f"  {m} + {ct}: {rate:.0%}")

        # Shuffle validation on best findings
        print(f"\n--- Shuffle Validation (top findings) ---\n")
        for m, ct, km, rate, d, p in sorted(all_findings, key=lambda x: abs(x[4]), reverse=True)[:5]:
            carrier = generate_carrier(ct, 999)
            stego = embed_lsb(carrier, rate, seed=99999)
            extract_fn = extraction_methods[m]
            extracted = extract_fn(stego)
            results_real = analyzer.analyze(extracted)
            shuffled = extracted.copy()
            np.random.shuffle(shuffled)
            results_shuf = analyzer.analyze(shuffled)

            for r, rs in zip(results_real.results, results_shuf.results):
                for mn in r.metrics:
                    fk = f"{r.geometry_name}:{mn}"
                    if fk == km:
                        ratio = r.metrics[mn] / (rs.metrics[mn] + 1e-10)
                        survived = abs(ratio - 1.0) > 0.15
                        status = "VALIDATED" if survived else "ARTIFACT"
                        print(f"  [{m}] {ct} rate={rate:.0%} {km}: ratio={ratio:.2f} -> {status}")
    else:
        print("\nNo significant detections even at bit level!")
        print("LSB steganography is truly invisible to geometric analysis.")

    print("\n[Investigation complete]")


if __name__ == '__main__':
    main()
