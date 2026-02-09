#!/usr/bin/env python3
"""
Investigation: Can exotic geometries detect cipher weakness?

Compare strong ciphers (AES-256-CTR, ChaCha20) against weak/broken ones
(RC4, DES-ECB, Blowfish-ECB) and also test ECB vs CTR mode for AES.

Hypothesis: ECB mode leaks plaintext structure. RC4 has known biases.
Geometric analysis should detect these weaknesses.

Control: os.urandom (true random baseline)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from Crypto.Cipher import AES, ARC4, DES3, ChaCha20, Blowfish
from Crypto.Random import get_random_bytes
from exotic_geometry_framework import GeometryAnalyzer

N_TRIALS = 20
DATA_SIZE = 2000

def generate_cipher_data(cipher_name, trial_seed, plaintext_type='structured'):
    """Encrypt data with different ciphers and return ciphertext bytes."""
    rng = np.random.RandomState(trial_seed)

    # Generate plaintext
    if plaintext_type == 'structured':
        # Repeating blocks - ECB should leak this structure
        block = bytes(rng.randint(0, 256, 16, dtype=np.uint8))
        pt = block * (DATA_SIZE // 16 + 1)
        pt = pt[:DATA_SIZE]
    elif plaintext_type == 'random':
        pt = bytes(rng.randint(0, 256, DATA_SIZE, dtype=np.uint8))
    elif plaintext_type == 'english':
        # Simulated English text (biased toward ASCII letters)
        chars = []
        for _ in range(DATA_SIZE):
            r = rng.random()
            if r < 0.15:
                chars.append(32)  # space
            elif r < 0.75:
                chars.append(rng.randint(97, 123))  # lowercase
            elif r < 0.85:
                chars.append(rng.randint(65, 91))   # uppercase
            else:
                chars.append(rng.randint(48, 58))    # digits
        pt = bytes(chars)
    else:
        pt = bytes(rng.randint(0, 256, DATA_SIZE, dtype=np.uint8))

    # Pad to block size if needed
    def pad(data, block_size):
        padding = block_size - (len(data) % block_size)
        return data + bytes([padding] * padding)

    if cipher_name == 'aes_ctr':
        key = get_random_bytes(32)
        nonce = get_random_bytes(8)
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
        ct = cipher.encrypt(pt)

    elif cipher_name == 'aes_ecb':
        key = get_random_bytes(32)
        cipher = AES.new(key, AES.MODE_ECB)
        ct = cipher.encrypt(pad(pt, 16))

    elif cipher_name == 'chacha20':
        key = get_random_bytes(32)
        nonce = get_random_bytes(8)
        cipher = ChaCha20.new(key=key, nonce=nonce)
        ct = cipher.encrypt(pt)

    elif cipher_name == 'rc4':
        key = get_random_bytes(16)
        cipher = ARC4.new(key)
        ct = cipher.encrypt(pt)

    elif cipher_name == 'des3_ecb':
        key = DES3.adjust_key_parity(get_random_bytes(24))
        cipher = DES3.new(key, DES3.MODE_ECB)
        ct = cipher.encrypt(pad(pt, 8))

    elif cipher_name == 'blowfish_ecb':
        key = get_random_bytes(16)
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        ct = cipher.encrypt(pad(pt, 8))

    elif cipher_name == 'rc4_biased':
        # RC4 with very short key (amplifies bias)
        key = get_random_bytes(3)  # 24-bit key
        cipher = ARC4.new(key)
        ct = cipher.encrypt(pt)

    elif cipher_name == 'xor_fixed':
        # Terrible "cipher": XOR with fixed key byte
        key_byte = rng.randint(1, 256)
        ct = bytes(b ^ key_byte for b in pt)

    elif cipher_name == 'random':
        ct = os.urandom(DATA_SIZE)

    else:
        raise ValueError(f"Unknown cipher: {cipher_name}")

    return np.frombuffer(ct[:DATA_SIZE], dtype=np.uint8)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("=" * 70)
    print("INVESTIGATION: Cipher Weakness Detection via Exotic Geometries")
    print("=" * 70)

    ciphers = ['aes_ctr', 'aes_ecb', 'chacha20', 'rc4', 'rc4_biased', 'des3_ecb',
               'blowfish_ecb', 'xor_fixed', 'random']

    plaintext_types = ['structured', 'english', 'random']

    key_metrics = [
        'E8 Lattice:unique_roots',
        'E8 Lattice:alignment_mean',
        'Torus T^2:coverage',
        'Torus T^2:chi2_uniformity',
        'Heisenberg (Nil):twist_rate',
        'Heisenberg (Nil) (centered):twist_rate',
        'Sol:anisotropy',
        'Tropical:linearity',
        'Persistent Homology:n_significant_features',
        'Penrose (Quasicrystal):fivefold_balance',
        'Fisher Information:trace_fisher',
        'Cantor (base 3):cantor_dimension',
    ]

    # Use a subset for faster analysis but comprehensive coverage
    analyzer = GeometryAnalyzer().add_all_geometries()

    # Part 1: Structured plaintext (worst case for ECB)
    for pt_type in plaintext_types:
        print(f"\n{'='*70}")
        print(f"PLAINTEXT TYPE: {pt_type}")
        print(f"{'='*70}")

        all_metrics = {c: defaultdict(list) for c in ciphers}

        for trial in range(N_TRIALS):
            if trial % 5 == 0:
                print(f"  Trial {trial}/{N_TRIALS}...")
            for c in ciphers:
                try:
                    data = generate_cipher_data(c, trial, pt_type)
                    results = analyzer.analyze(data)
                    for r in results.results:
                        for mn, mv in r.metrics.items():
                            full_key = f"{r.geometry_name}:{mn}"
                            all_metrics[c][full_key].append(mv)
                except Exception as e:
                    pass

        # Summary table
        print(f"\n--- Key Metrics (mean over {N_TRIALS} trials) ---\n")
        header = f"{'Metric':<40}" + "".join(f"{c[:8]:>10}" for c in ciphers)
        print(header)
        print("-" * len(header))

        for km in key_metrics:
            if km in all_metrics['random']:
                row = f"{km[:39]:<40}"
                for c in ciphers:
                    vals = all_metrics[c].get(km, [0])
                    row += f"{np.mean(vals):>10.3f}"
                print(row)

        # Statistical comparison vs random
        print(f"\n--- Significant differences vs random (Bonferroni) ---\n")
        n_comp = (len(ciphers) - 1) * len(key_metrics)
        alpha = 0.05 / max(n_comp, 1)

        sigs = []
        for km in key_metrics:
            if km not in all_metrics['random'] or len(all_metrics['random'][km]) < 2:
                continue
            ref = all_metrics['random'][km]
            for c in ciphers:
                if c == 'random':
                    continue
                if km not in all_metrics[c] or len(all_metrics[c][km]) < 2:
                    continue
                vals = all_metrics[c][km]
                d = cohens_d(vals, ref)
                _, p = stats.ttest_ind(vals, ref, equal_var=False)
                if abs(d) > 0.8 and p < alpha:
                    print(f"  {c:<15} {km:<40} d={d:+.2f} p={p:.2e} ***")
                    sigs.append((c, km, d, p, pt_type))

        if not sigs:
            print("  No significant differences found.")

    # Part 2: ECB vs CTR deep dive
    print(f"\n{'='*70}")
    print(f"DEEP DIVE: AES-ECB vs AES-CTR with structured plaintext")
    print(f"{'='*70}")

    # More trials for this specific comparison
    ecb_metrics = defaultdict(list)
    ctr_metrics = defaultdict(list)

    for trial in range(40):
        for cipher_name, store in [('aes_ecb', ecb_metrics), ('aes_ctr', ctr_metrics)]:
            data = generate_cipher_data(cipher_name, trial, 'structured')
            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    store[f"{r.geometry_name}:{mn}"].append(mv)

    print(f"\n{'Metric':<45} {'ECB':>10} {'CTR':>10} {'d':>8} {'p':>12}")
    print("-" * 90)

    ecb_detections = []
    all_km = sorted(set(ecb_metrics.keys()) & set(ctr_metrics.keys()))
    for km in all_km:
        ecb_vals = ecb_metrics[km]
        ctr_vals = ctr_metrics[km]
        d = cohens_d(ecb_vals, ctr_vals)
        _, p = stats.ttest_ind(ecb_vals, ctr_vals, equal_var=False)
        if abs(d) > 0.5:
            print(f"  {km[:44]:<45} {np.mean(ecb_vals):>10.3f} {np.mean(ctr_vals):>10.3f} {d:>+8.2f} {p:>12.2e}")
            if abs(d) > 0.8:
                ecb_detections.append((km, d, p))

    # Shuffle validation for ECB
    if ecb_detections:
        print(f"\n--- Shuffle validation for ECB detections ---\n")
        data_ecb = generate_cipher_data('aes_ecb', 999, 'structured')
        results_real = analyzer.analyze(data_ecb)
        shuffled = data_ecb.copy()
        np.random.shuffle(shuffled)
        results_shuf = analyzer.analyze(shuffled)

        for r, rs in zip(results_real.results, results_shuf.results):
            for mn in r.metrics:
                fk = f"{r.geometry_name}:{mn}"
                for km, d, p in ecb_detections:
                    if fk == km:
                        ratio = r.metrics[mn] / (rs.metrics[mn] + 1e-10)
                        survived = abs(ratio - 1.0) > 0.15
                        status = "VALIDATED" if survived else "ARTIFACT"
                        print(f"  {km[:44]}: real={r.metrics[mn]:.3f} shuf={rs.metrics[mn]:.3f} ratio={ratio:.2f} -> {status}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print("\nCipher detection results:")
    print(f"  AES-CTR:     Should look random (control)")
    print(f"  AES-ECB:     {len(ecb_detections)} geometric differences from CTR detected")
    print(f"  ChaCha20:    Should look random")
    print(f"  RC4:         Check above for detections")
    print(f"  XOR-fixed:   Deliberately terrible, should be detected")

    print("\n[Investigation complete]")


if __name__ == '__main__':
    main()
