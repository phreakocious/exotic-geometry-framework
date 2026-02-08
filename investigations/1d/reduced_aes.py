#!/usr/bin/env python3
"""
Investigate Reduced-Round AES with Exotic Geometries
=====================================================

Question: At what round count does AES become geometrically indistinguishable
from os.urandom()?

Full AES-128 uses 10 rounds. Fewer rounds = less diffusion = more detectable
structure (in theory). We test rounds 1, 2, 3, 4, 5, 7, and full 10.

Methodology:
- Implement reduced-round AES manually (SubBytes, ShiftRows, MixColumns, AddRoundKey)
- Use CTR-like mode: encrypt sequential 16-byte counter blocks
- 25 trials per condition, 2000 bytes each
- Compare to os.urandom() baseline using all 21 exotic geometries
- Report Cohen's d effect sizes and Bonferroni-corrected p-values
- Shuffle validation for any positive findings

Author: Claude + TB
Date: 2026-02-05
"""

import numpy as np
import os
import sys
import time
from scipy import stats
from collections import defaultdict

# Add parent directory to path for framework import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from exotic_geometry_framework import GeometryAnalyzer


# =============================================================================
# AES S-BOX (standard Rijndael)
# =============================================================================
AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]

# Round constants for key expansion
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]


# =============================================================================
# GF(2^8) ARITHMETIC for MixColumns
# =============================================================================
def gf_mul(a, b):
    """Multiply two bytes in GF(2^8) with AES irreducible polynomial."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1b  # x^8 + x^4 + x^3 + x + 1
        b >>= 1
    return p


# Precompute GF multiplication tables for the constants used in MixColumns
GF_MUL_2 = [gf_mul(i, 2) for i in range(256)]
GF_MUL_3 = [gf_mul(i, 3) for i in range(256)]


# =============================================================================
# AES ROUND OPERATIONS
# =============================================================================
def sub_bytes(state):
    """Apply S-box substitution to each byte."""
    return [AES_SBOX[b] for b in state]


def shift_rows(state):
    """ShiftRows: cyclic left shifts on rows of the 4x4 state matrix.

    State is stored column-major: state[row + 4*col]
    Row 0: no shift
    Row 1: shift left 1
    Row 2: shift left 2
    Row 3: shift left 3
    """
    # Convert to matrix form [row][col]
    m = [[state[r + 4*c] for c in range(4)] for r in range(4)]
    # Shift each row
    for r in range(1, 4):
        m[r] = m[r][r:] + m[r][:r]
    # Convert back to column-major
    return [m[r][c] for c in range(4) for r in range(4)]


def mix_columns(state):
    """MixColumns: matrix multiply each column by the MixColumns matrix.

    [2 3 1 1]   [s0]
    [1 2 3 1] x [s1]
    [1 1 2 3]   [s2]
    [3 1 1 2]   [s3]
    """
    out = [0] * 16
    for c in range(4):
        i = c * 4
        s0, s1, s2, s3 = state[i], state[i+1], state[i+2], state[i+3]
        out[i]   = GF_MUL_2[s0] ^ GF_MUL_3[s1] ^ s2 ^ s3
        out[i+1] = s0 ^ GF_MUL_2[s1] ^ GF_MUL_3[s2] ^ s3
        out[i+2] = s0 ^ s1 ^ GF_MUL_2[s2] ^ GF_MUL_3[s3]
        out[i+3] = GF_MUL_3[s0] ^ s1 ^ s2 ^ GF_MUL_2[s3]
    return out


def add_round_key(state, round_key):
    """XOR state with round key."""
    return [s ^ k for s, k in zip(state, round_key)]


# =============================================================================
# KEY EXPANSION
# =============================================================================
def key_expansion(key_bytes):
    """Expand 16-byte key into 11 round keys (for up to 10 rounds).

    Returns list of 11 round keys, each a list of 16 bytes.
    """
    # Key is stored column-major
    w = list(key_bytes)  # Start with the original 16-byte key as 4 words

    for i in range(4, 44):  # Generate words 4..43
        temp = w[(i-1)*4 : i*4]  # Previous word
        if i % 4 == 0:
            # RotWord + SubWord + Rcon
            temp = [AES_SBOX[temp[1]], AES_SBOX[temp[2]],
                    AES_SBOX[temp[3]], AES_SBOX[temp[0]]]
            temp[0] ^= RCON[i // 4 - 1]
        prev_4 = w[(i-4)*4 : (i-3)*4]
        new_word = [a ^ b for a, b in zip(prev_4, temp)]
        w.extend(new_word)

    # Split into 11 round keys of 16 bytes each
    round_keys = []
    for r in range(11):
        rk = w[r*16 : (r+1)*16]
        round_keys.append(rk)

    return round_keys


# =============================================================================
# REDUCED-ROUND AES ENCRYPTION
# =============================================================================
def aes_encrypt_block(plaintext_16bytes, round_keys, num_rounds):
    """
    Encrypt a single 16-byte block with a specified number of AES rounds.

    Standard AES-128 uses num_rounds=10:
      - AddRoundKey (initial)
      - Rounds 1..9: SubBytes, ShiftRows, MixColumns, AddRoundKey
      - Round 10: SubBytes, ShiftRows, AddRoundKey (no MixColumns)

    For reduced rounds (e.g., num_rounds=3):
      - AddRoundKey (initial)
      - Rounds 1..2: SubBytes, ShiftRows, MixColumns, AddRoundKey
      - Round 3: SubBytes, ShiftRows, AddRoundKey (no MixColumns)
    """
    state = list(plaintext_16bytes)

    # Initial round key addition
    state = add_round_key(state, round_keys[0])

    # Main rounds (with MixColumns)
    for r in range(1, num_rounds):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[r])

    # Final round (no MixColumns)
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[num_rounds])

    return bytes(state)


def generate_reduced_aes_ctr(num_rounds, num_bytes, key=None, nonce=None):
    """
    Generate pseudo-random bytes using reduced-round AES in CTR mode.

    Uses sequential counter blocks, like real CTR mode.
    """
    if key is None:
        key = os.urandom(16)
    if nonce is None:
        nonce = os.urandom(8)

    round_keys = key_expansion(key)

    output = bytearray()
    counter = 0
    blocks_needed = (num_bytes + 15) // 16

    for ctr in range(blocks_needed):
        # Build counter block: 8-byte nonce + 8-byte counter
        plaintext = bytearray(nonce) + ctr.to_bytes(8, 'big')
        encrypted = aes_encrypt_block(plaintext, round_keys, num_rounds)
        output.extend(encrypted)

    return bytes(output[:num_bytes])


# =============================================================================
# VERIFICATION: Compare our 10-round implementation to PyCryptodome
# =============================================================================
def verify_implementation():
    """Verify our AES implementation matches PyCryptodome for 10 rounds."""
    try:
        from Crypto.Cipher import AES as PyCryptoAES

        key = os.urandom(16)
        plaintext = os.urandom(16)

        # Our implementation
        round_keys = key_expansion(key)
        our_result = aes_encrypt_block(plaintext, round_keys, 10)

        # PyCryptodome ECB (single block)
        cipher = PyCryptoAES.new(key, PyCryptoAES.MODE_ECB)
        ref_result = cipher.encrypt(plaintext)

        if our_result == ref_result:
            print("[VERIFY] Our AES-128 matches PyCryptodome - implementation CORRECT")
            return True
        else:
            print("[VERIFY] MISMATCH!")
            print(f"  Our:  {our_result.hex()}")
            print(f"  Ref:  {ref_result.hex()}")
            return False
    except ImportError:
        print("[VERIFY] PyCryptodome not available, skipping verification")
        return True


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================
def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-15:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    elif ad < 1.2:
        return "large"
    else:
        return "HUGE"


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_experiment():
    print("=" * 80)
    print("REDUCED-ROUND AES: EXOTIC GEOMETRY DETECTION")
    print("=" * 80)
    print()
    print("Question: At what round count does AES become geometrically")
    print("          indistinguishable from os.urandom()?")
    print()

    # Verify our implementation first
    if not verify_implementation():
        print("ABORTING: Implementation does not match reference!")
        return
    print()

    # Parameters
    ROUND_COUNTS = [1, 2, 3, 4, 5, 7, 10]
    N_TRIALS = 25
    N_BYTES = 2000

    # Key metrics we especially care about (from MEMORY.md)
    KEY_METRICS = [
        ("E8 Lattice", "unique_roots"),
        ("E8 Lattice", "alignment_mean"),
        ("Fisher Information", "trace_fisher"),
        ("Tropical", "linearity"),
        ("Cantor (base 3)", "cantor_dimension"),
        ("Heisenberg (Nil)", "twist_rate"),
        ("Penrose (Quasicrystal)", "fivefold_balance"),
        ("Torus T^2", "coverage"),
    ]

    print(f"Parameters: {N_TRIALS} trials x {N_BYTES} bytes per trial")
    print(f"Round counts: {ROUND_COUNTS}")
    print(f"Key metrics tracked: {len(KEY_METRICS)}")
    print()

    # Initialize analyzer
    analyzer = GeometryAnalyzer().add_all_geometries()
    geometry_names = [g.name for g in analyzer.geometries]
    print(f"Geometries loaded: {len(geometry_names)}")
    print(f"  {geometry_names}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Generate baseline (os.urandom)
    # ------------------------------------------------------------------
    print("-" * 80)
    print("STEP 1: Generating os.urandom() baseline...")
    print("-" * 80)

    baseline_metrics = defaultdict(lambda: defaultdict(list))  # {geom: {metric: [values]}}

    t0 = time.time()
    for trial in range(N_TRIALS):
        data = np.frombuffer(os.urandom(N_BYTES), dtype=np.uint8)
        result = analyzer.analyze(data, f"urandom_trial_{trial}")
        for gr in result.results:
            for metric_name, metric_val in gr.metrics.items():
                baseline_metrics[gr.geometry_name][metric_name].append(metric_val)
        if (trial + 1) % 5 == 0:
            print(f"  Baseline trial {trial + 1}/{N_TRIALS} done")

    baseline_time = time.time() - t0
    print(f"  Baseline complete in {baseline_time:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 2: Generate reduced-round AES data
    # ------------------------------------------------------------------
    print("-" * 80)
    print("STEP 2: Generating reduced-round AES data...")
    print("-" * 80)

    # Store all metrics per round count
    # {round_count: {geom: {metric: [values]}}}
    round_metrics = {}

    for n_rounds in ROUND_COUNTS:
        print(f"\n  Round count = {n_rounds}:")
        round_metrics[n_rounds] = defaultdict(lambda: defaultdict(list))

        t0 = time.time()
        for trial in range(N_TRIALS):
            # Each trial uses a fresh random key and nonce
            key = os.urandom(16)
            nonce = os.urandom(8)
            raw = generate_reduced_aes_ctr(n_rounds, N_BYTES, key=key, nonce=nonce)
            data = np.frombuffer(raw, dtype=np.uint8)

            result = analyzer.analyze(data, f"aes_r{n_rounds}_trial_{trial}")
            for gr in result.results:
                for metric_name, metric_val in gr.metrics.items():
                    round_metrics[n_rounds][gr.geometry_name][metric_name].append(metric_val)

            if (trial + 1) % 5 == 0:
                print(f"    Trial {trial + 1}/{N_TRIALS} done")

        elapsed = time.time() - t0
        print(f"    {n_rounds}-round AES complete in {elapsed:.1f}s")

    print()

    # ------------------------------------------------------------------
    # Step 3: Statistical comparison
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: STATISTICAL ANALYSIS")
    print("=" * 80)
    print()

    # Collect ALL geometry:metric pairs
    all_pairs = []
    for geom_name in baseline_metrics:
        for metric_name in baseline_metrics[geom_name]:
            all_pairs.append((geom_name, metric_name))

    n_total_tests = len(all_pairs) * len(ROUND_COUNTS)
    bonferroni = n_total_tests  # Number of simultaneous tests
    print(f"Total tests: {n_total_tests} ({len(all_pairs)} metrics x {len(ROUND_COUNTS)} round counts)")
    print(f"Bonferroni correction factor: {bonferroni}")
    print()

    # Store results for summary
    # {(geom, metric): {round: (d, p_corrected, significant)}}
    all_results = defaultdict(dict)

    for geom_name, metric_name in all_pairs:
        baseline_vals = np.array(baseline_metrics[geom_name][metric_name])

        for n_rounds in ROUND_COUNTS:
            if geom_name not in round_metrics[n_rounds]:
                continue
            if metric_name not in round_metrics[n_rounds][geom_name]:
                continue

            round_vals = np.array(round_metrics[n_rounds][geom_name][metric_name])

            # Filter out NaN/Inf
            valid_b = baseline_vals[np.isfinite(baseline_vals)]
            valid_r = round_vals[np.isfinite(round_vals)]

            if len(valid_b) < 5 or len(valid_r) < 5:
                continue

            # Cohen's d
            d = cohens_d(valid_r, valid_b)

            # Welch's t-test (does not assume equal variance)
            t_stat, p_val = stats.ttest_ind(valid_r, valid_b, equal_var=False)
            p_corrected = min(p_val * bonferroni, 1.0)

            significant = p_corrected < 0.05
            all_results[(geom_name, metric_name)][n_rounds] = (d, p_corrected, significant)

    # ------------------------------------------------------------------
    # Step 3a: KEY METRICS TABLE
    # ------------------------------------------------------------------
    print("-" * 80)
    print("KEY METRICS: Cohen's d by round count (vs os.urandom baseline)")
    print("-" * 80)
    print()

    # Header
    header = f"{'Geometry:Metric':<40s}"
    for n_rounds in ROUND_COUNTS:
        header += f" {'R='+str(n_rounds):>8s}"
    print(header)
    print("-" * (40 + 9 * len(ROUND_COUNTS)))

    for geom_name, metric_name in KEY_METRICS:
        key = (geom_name, metric_name)
        if key not in all_results:
            print(f"{geom_name}:{metric_name:<40s}  [not found]")
            continue

        row = f"{geom_name}:{metric_name}"
        if len(row) > 39:
            row = row[:39]
        row = f"{row:<40s}"

        for n_rounds in ROUND_COUNTS:
            if n_rounds in all_results[key]:
                d, p, sig = all_results[key][n_rounds]
                marker = "*" if sig else " "
                row += f" {d:>7.2f}{marker}"
            else:
                row += f" {'N/A':>8s}"

        print(row)

    print()
    print("  * = significant after Bonferroni correction (p < 0.05)")
    print()

    # ------------------------------------------------------------------
    # Step 3b: ALL significant findings
    # ------------------------------------------------------------------
    print("-" * 80)
    print("ALL SIGNIFICANT FINDINGS (Bonferroni-corrected p < 0.05)")
    print("-" * 80)
    print()

    sig_findings = []
    for (geom_name, metric_name), round_data in all_results.items():
        for n_rounds, (d, p, sig) in round_data.items():
            if sig:
                sig_findings.append((n_rounds, abs(d), d, p, geom_name, metric_name))

    sig_findings.sort(key=lambda x: (x[0], -x[1]))  # Sort by round count, then |d|

    if not sig_findings:
        print("  No significant findings! AES appears random at all round counts.")
    else:
        print(f"  Found {len(sig_findings)} significant results:\n")
        print(f"  {'Rounds':<8s} {'|d|':>6s} {'d':>8s} {'p_corr':>10s} {'Effect':>10s}  {'Geometry:Metric'}")
        print(f"  {'-'*6:<8s} {'-'*6:>6s} {'-'*8:>8s} {'-'*10:>10s} {'-'*10:>10s}  {'-'*30}")

        for n_rounds, abs_d, d, p, geom, metric in sig_findings:
            eff = interpret_d(d)
            print(f"  R={n_rounds:<5d} {abs_d:>6.2f} {d:>8.3f} {p:>10.2e} {eff:>10s}  {geom}:{metric}")

    print()

    # ------------------------------------------------------------------
    # Step 3c: Detection threshold summary
    # ------------------------------------------------------------------
    print("-" * 80)
    print("DETECTION THRESHOLD: At what round count does each geometry lose signal?")
    print("-" * 80)
    print()

    # For each metric, find the highest round count that's still significant
    metric_thresholds = {}
    for (geom_name, metric_name), round_data in all_results.items():
        max_detectable = 0
        for n_rounds, (d, p, sig) in round_data.items():
            if sig and abs(d) >= 0.5:  # At least medium effect
                max_detectable = max(max_detectable, n_rounds)
        if max_detectable > 0:
            metric_thresholds[(geom_name, metric_name)] = max_detectable

    if metric_thresholds:
        # Sort by threshold (highest first)
        sorted_thresholds = sorted(metric_thresholds.items(), key=lambda x: -x[1])
        print(f"  {'Geometry:Metric':<45s} {'Max Detectable Round':>20s}")
        print(f"  {'-'*45} {'-'*20}")
        for (geom, metric), threshold in sorted_thresholds:
            label = f"{geom}:{metric}"
            if len(label) > 44:
                label = label[:44]
            print(f"  {label:<45s} {threshold:>20d}")
    else:
        print("  No geometry detected a medium+ effect at any round count.")

    print()

    # ------------------------------------------------------------------
    # Step 4: Shuffle validation for positive findings
    # ------------------------------------------------------------------
    if sig_findings:
        print("-" * 80)
        print("STEP 4: SHUFFLE VALIDATION of strongest findings")
        print("-" * 80)
        print()

        # Pick top 3 strongest findings (by |d|, at lowest round count)
        # Focus on round 1 findings since those should be strongest
        round1_findings = [(abs_d, d, p, geom, metric) for (nr, abs_d, d, p, geom, metric) in sig_findings if nr == 1]
        round1_findings.sort(key=lambda x: -x[0])
        top_findings = round1_findings[:3]

        if top_findings:
            print("  Validating top findings from R=1 by shuffling the data:")
            print("  (If finding is real, shuffled data should match baseline, not R=1)")
            print()

            for abs_d_orig, d_orig, p_orig, geom, metric in top_findings:
                print(f"  {geom}:{metric} (d={d_orig:.3f}):")

                # Generate R=1 data, shuffle it, measure
                shuffled_vals = []
                for trial in range(N_TRIALS):
                    key = os.urandom(16)
                    nonce = os.urandom(8)
                    raw = generate_reduced_aes_ctr(1, N_BYTES, key=key, nonce=nonce)
                    data = np.frombuffer(raw, dtype=np.uint8).copy()
                    np.random.shuffle(data)  # Destroy structure, keep marginal distribution

                    result = analyzer.analyze(data, f"shuffled_trial_{trial}")
                    for gr in result.results:
                        if gr.geometry_name == geom and metric in gr.metrics:
                            shuffled_vals.append(gr.metrics[metric])

                shuffled_vals = np.array(shuffled_vals)
                baseline_vals = np.array(baseline_metrics[geom][metric])
                r1_vals = np.array(round_metrics[1][geom][metric])

                valid_s = shuffled_vals[np.isfinite(shuffled_vals)]
                valid_b = baseline_vals[np.isfinite(baseline_vals)]
                valid_r = r1_vals[np.isfinite(r1_vals)]

                if len(valid_s) >= 5 and len(valid_b) >= 5:
                    d_shuf_vs_base = cohens_d(valid_s, valid_b)
                    d_shuf_vs_r1 = cohens_d(valid_s, valid_r)
                    d_r1_vs_base = cohens_d(valid_r, valid_b)

                    print(f"    R=1 vs baseline:     d = {d_r1_vs_base:>7.3f}")
                    print(f"    Shuffled vs baseline: d = {d_shuf_vs_base:>7.3f}")
                    print(f"    Shuffled vs R=1:     d = {d_shuf_vs_r1:>7.3f}")

                    if abs(d_shuf_vs_base) < 0.3 and abs(d_r1_vs_base) > 0.5:
                        print(f"    --> VALIDATED: Shuffling destroys signal. Structure is real.")
                    elif abs(d_shuf_vs_base) > 0.5:
                        print(f"    --> ARTIFACT: Shuffled data still differs from baseline.")
                        print(f"        (Marginal distribution effect, not structural)")
                    else:
                        print(f"    --> INCONCLUSIVE")
                    print()
        print()

    # ------------------------------------------------------------------
    # Step 5: Detailed per-round breakdown for key metrics
    # ------------------------------------------------------------------
    print("=" * 80)
    print("DETAILED BREAKDOWN: Mean values per condition")
    print("=" * 80)
    print()

    for geom_name, metric_name in KEY_METRICS:
        baseline_vals = baseline_metrics.get(geom_name, {}).get(metric_name, [])
        if not baseline_vals:
            continue

        valid_b = [v for v in baseline_vals if np.isfinite(v)]
        if not valid_b:
            continue

        print(f"  {geom_name}:{metric_name}")
        print(f"    {'Condition':<15s} {'Mean':>10s} {'Std':>10s} {'d vs random':>12s}")
        print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
        print(f"    {'os.urandom':<15s} {np.mean(valid_b):>10.4f} {np.std(valid_b):>10.4f} {'---':>12s}")

        for n_rounds in ROUND_COUNTS:
            vals = round_metrics.get(n_rounds, {}).get(geom_name, {}).get(metric_name, [])
            valid_r = [v for v in vals if np.isfinite(v)]
            if valid_r:
                d = cohens_d(np.array(valid_r), np.array(valid_b))
                print(f"    {'R=' + str(n_rounds):<15s} {np.mean(valid_r):>10.4f} {np.std(valid_r):>10.4f} {d:>12.3f}")
        print()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    # Count significant findings per round
    sig_per_round = defaultdict(int)
    for (geom_name, metric_name), round_data in all_results.items():
        for n_rounds, (d, p, sig) in round_data.items():
            if sig:
                sig_per_round[n_rounds] += 1

    print(f"  {'Rounds':<10s} {'# Significant Metrics':>25s} {'/ Total':>10s}")
    print(f"  {'-'*10} {'-'*25} {'-'*10}")
    for n_rounds in ROUND_COUNTS:
        total = len(all_pairs)
        n_sig = sig_per_round.get(n_rounds, 0)
        print(f"  R={n_rounds:<7d} {n_sig:>25d} {'/ ' + str(total):>10s}")

    print()

    # Determine threshold
    for n_rounds in ROUND_COUNTS:
        if sig_per_round.get(n_rounds, 0) == 0:
            print(f"  CONCLUSION: AES becomes geometrically indistinguishable from random")
            print(f"              at R={n_rounds} rounds (with {N_TRIALS} trials of {N_BYTES} bytes).")
            break
    else:
        print(f"  CONCLUSION: Some structure detected even at R=10 (full AES)!")
        print(f"              This may indicate a framework artifact or insufficient randomness.")

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility for shuffle validation
    run_experiment()
