#!/usr/bin/env python3
"""
Investigation: Advanced Steganography Detection via Exotic Geometries.

Follow-up to stego.py, which only tested LSB replacement — trivially detectable.
This probes 6 embedding techniques specifically designed to defeat statistical
steganalysis, asking: can exotic geometries detect stego that evades classical tests?

Techniques:
  1. LSB Replace     — set LSB to message bit (baseline)
  2. LSB Match (±1)  — if LSB wrong, randomly +1 or -1 (defeats chi-squared)
  3. LSBMR           — pixel pairs, 2 bits/pair, ≤1 change/pair (defeats RS analysis)
  4. PVD             — embed in adjacent-value differences (defeats LSB-focused detectors)
  5. Spread Spectrum — low-amplitude PRN pattern across all positions
  6. Matrix Embed    — Hamming(7,3) syndrome coding: 3 bits per ≤1 flip

Five directions:
  D1: Raw byte detection (can geometry detect stego without preprocessing?)
  D2: Bitplane detection (bitplane extraction cracked LSB — does it crack these?)
  D3: Embedding rate sensitivity (minimum detectable rate per technique)
  D4: Preprocessing composition (delay_embed + bitplane together)
  D5: Per-geometry sensitivity (which geometries detect which techniques?)

Total budget: ~1510 analyzer calls, estimated 10-12 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer, delay_embed, bitplane_extract
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
N_TRIALS_RATE = 20
DATA_SIZE = 4000


# =============================================================================
# CARRIER GENERATORS (from stego.py)
# =============================================================================

def generate_carrier(carrier_type, trial_seed, size=DATA_SIZE):
    """Generate carrier data with natural LSB correlations."""
    rng = np.random.RandomState(trial_seed)

    if carrier_type == 'photo_blocks':
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

    else:
        raise ValueError(f"Unknown carrier: {carrier_type}")


# =============================================================================
# EMBEDDING TECHNIQUES
# =============================================================================

def embed_lsb_replace(carrier, rate, seed=42):
    """LSB replacement: set LSB to message bit. Trivially detectable."""
    rng = np.random.RandomState(seed)
    stego = carrier.copy().astype(np.int16)
    n_embed = int(len(carrier) * rate)
    if n_embed == 0:
        return carrier.copy()
    positions = np.sort(rng.choice(len(carrier), n_embed, replace=False))
    message_bits = rng.randint(0, 2, n_embed)
    stego[positions] = (stego[positions] & 0xFE) | message_bits
    return np.clip(stego, 0, 255).astype(np.uint8)


def embed_lsb_match(carrier, rate, seed=42):
    """LSB matching (±1): if LSB wrong, randomly add +1 or -1.
    Defeats chi-squared and histogram attacks because it doesn't create
    pairs-of-values artifacts."""
    rng = np.random.RandomState(seed)
    stego = carrier.copy().astype(np.int16)
    n_embed = int(len(carrier) * rate)
    if n_embed == 0:
        return carrier.copy()
    positions = np.sort(rng.choice(len(carrier), n_embed, replace=False))
    message_bits = rng.randint(0, 2, n_embed)
    for i, pos in enumerate(positions):
        if (stego[pos] & 1) != message_bits[i]:
            # Randomly +1 or -1 (instead of forcing LSB)
            delta = rng.choice([-1, 1])
            stego[pos] += delta
    return np.clip(stego, 0, 255).astype(np.uint8)


def embed_lsbmr(carrier, rate, seed=42):
    """LSB Match Revisited (LSBMR): pixel pairs encode 2 bits with ≤1 change per pair.
    Uses relationship between consecutive pixels to embed a second bit in
    the floor(avg) of the pair, defeating RS analysis and chi-squared."""
    rng = np.random.RandomState(seed)
    stego = carrier.copy().astype(np.int16)
    n_pairs = len(carrier) // 2
    n_embed_pairs = int(n_pairs * rate)
    if n_embed_pairs == 0:
        return carrier.copy()
    pair_indices = np.sort(rng.choice(n_pairs, n_embed_pairs, replace=False))
    message_bits = rng.randint(0, 2, n_embed_pairs * 2)

    for idx, pi in enumerate(pair_indices):
        p0, p1 = int(stego[2 * pi]), int(stego[2 * pi + 1])
        m0, m1 = message_bits[2 * idx], message_bits[2 * idx + 1]

        # First bit: LSB of first pixel (via ±1 matching)
        if (p0 & 1) != m0:
            p0 += rng.choice([-1, 1])

        # Second bit: LSB of floor((p0+p1)/2)
        avg_lsb = ((p0 + p1) // 2) & 1
        if avg_lsb != m1:
            # Adjust p1 by ±1 to flip the floor(avg) LSB
            if rng.random() < 0.5:
                p1 += 1
            else:
                p1 -= 1

        stego[2 * pi] = p0
        stego[2 * pi + 1] = p1

    return np.clip(stego, 0, 255).astype(np.uint8)


def embed_pvd(carrier, rate, seed=42):
    """Pixel Value Differencing (PVD): embed bits in adjacent-value differences.
    Capacity proportional to edge strength — more bits in high-difference areas,
    fewer in smooth areas. Defeats all LSB-focused detectors."""
    rng = np.random.RandomState(seed)
    stego = carrier.copy().astype(np.int16)
    n_pairs = len(carrier) // 2
    n_embed_pairs = int(n_pairs * rate)
    if n_embed_pairs == 0:
        return carrier.copy()
    pair_indices = np.sort(rng.choice(n_pairs, n_embed_pairs, replace=False))
    msg_stream = rng.randint(0, 2, n_embed_pairs * 4)  # enough bits
    bit_pos = 0

    # PVD range table (simplified): ranges determine how many bits to embed
    ranges = [(0, 7, 3), (8, 15, 3), (16, 31, 4), (32, 63, 5), (64, 127, 6), (128, 255, 7)]

    for pi in pair_indices:
        p0, p1 = int(stego[2 * pi]), int(stego[2 * pi + 1])
        diff = abs(p0 - p1)

        # Find range
        n_bits = 3  # default
        lower = 0
        for lo, hi, nb in ranges:
            if lo <= diff <= hi:
                n_bits = nb
                lower = lo
                break

        # Limit embedding to available bits
        n_bits = min(n_bits, 3)  # cap to keep distortion low
        if bit_pos + n_bits > len(msg_stream):
            break

        # Extract message value
        msg_val = 0
        for b in range(n_bits):
            msg_val |= (msg_stream[bit_pos + b] << b)
        bit_pos += n_bits

        # New difference
        new_diff = lower + msg_val
        # Adjust pixel pair to achieve new_diff
        mid = (p0 + p1) / 2.0
        if p0 >= p1:
            new_p0 = int(mid + new_diff / 2.0 + 0.5)
            new_p1 = int(mid - new_diff / 2.0 + 0.5)
        else:
            new_p0 = int(mid - new_diff / 2.0 + 0.5)
            new_p1 = int(mid + new_diff / 2.0 + 0.5)

        stego[2 * pi] = new_p0
        stego[2 * pi + 1] = new_p1

    return np.clip(stego, 0, 255).astype(np.uint8)


def embed_spread_spectrum(carrier, rate, seed=42):
    """Spread spectrum: add low-amplitude PRN pattern across positions.
    The message modulates a pseudorandom noise sequence (alpha=2).
    Defeats direct statistical tests because changes are spread uniformly."""
    rng = np.random.RandomState(seed)
    stego = carrier.copy().astype(np.int16)
    n_embed = int(len(carrier) * rate)
    if n_embed == 0:
        return carrier.copy()
    positions = np.sort(rng.choice(len(carrier), n_embed, replace=False))
    message_bits = rng.randint(0, 2, n_embed)
    alpha = 2  # embedding strength
    # PRN sequence (±1)
    prn = rng.choice([-1, 1], n_embed)
    # Embed: stego = carrier + alpha * prn * (2*message - 1)
    modulated = alpha * prn * (2 * message_bits - 1)
    stego[positions] += modulated.astype(np.int16)
    return np.clip(stego, 0, 255).astype(np.uint8)


def embed_matrix(carrier, rate, seed=42):
    """Matrix embedding via Hamming(7,3) syndrome coding.
    Encodes 3 message bits per 7-pixel block with at most 1 pixel change.
    Minimizes total changes — defeats statistical tests that count modifications."""
    rng = np.random.RandomState(seed)
    stego = carrier.copy().astype(np.int16)

    # Hamming(7,3) parity check matrix
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)

    block_size = 7
    n_blocks = len(carrier) // block_size
    n_embed_blocks = int(n_blocks * rate)
    if n_embed_blocks == 0:
        return carrier.copy()

    block_indices = np.sort(rng.choice(n_blocks, n_embed_blocks, replace=False))
    message_bits = rng.randint(0, 2, n_embed_blocks * 3)

    for idx, bi in enumerate(block_indices):
        start = bi * block_size
        block_lsbs = stego[start:start + block_size] & 1

        # Current syndrome
        syndrome = (H @ block_lsbs.astype(np.uint8)) % 2  # 3-bit vector

        # Desired syndrome (message bits)
        desired = message_bits[3 * idx:3 * idx + 3]

        # Error = syndrome XOR desired
        error = syndrome ^ desired

        if np.any(error):
            # Find which column of H matches the error pattern
            error_val = error[0] * 1 + error[1] * 2 + error[2] * 4
            if 1 <= error_val <= 7:
                flip_pos = error_val - 1  # columns are 1-indexed in Hamming
                stego[start + flip_pos] ^= 1  # flip LSB

    return np.clip(stego, 0, 255).astype(np.uint8)


# =============================================================================
# TECHNIQUE / COLOR REGISTRIES
# =============================================================================

TECHNIQUES = {
    'LSB Replace':     embed_lsb_replace,
    'LSB Match':       embed_lsb_match,
    'LSBMR':           embed_lsbmr,
    'PVD':             embed_pvd,
    'Spread Spectrum': embed_spread_spectrum,
    'Matrix Embed':    embed_matrix,
}

TECH_COLORS = {
    'LSB Replace':     '#E91E63',
    'LSB Match':       '#FF9800',
    'LSBMR':           '#FFEB3B',
    'PVD':             '#4CAF50',
    'Spread Spectrum': '#2196F3',
    'Matrix Embed':    '#9C27B0',
}

CARRIERS = ['photo_blocks', 'natural_texture']


# =============================================================================
# STATISTICS (from collatz_deep2.py)
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
        _, p = stats.ttest_ind(va, vb)
        if abs(d) > 0.8 and p < alpha:
            sig += 1
    return sig


def per_metric_significance(metrics_a, metrics_b, metric_names, n_total_tests):
    """Per-metric Cohen's d, p-value, significance flag.
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
        _, p = stats.ttest_ind(va, vb)
        is_sig = abs(d) > 0.8 and p < alpha
        result[km] = (d, p, is_sig)
    return result


# =============================================================================
# HELPER: Collect metrics from analyzer
# =============================================================================

def collect_metrics(analyzer, data, target_dict):
    """Run analyzer on data and append results to target_dict."""
    results = analyzer.analyze(data)
    for r in results.results:
        for mn, mv in r.metrics.items():
            target_dict[f"{r.geometry_name}:{mn}"].append(mv)


# =============================================================================
# DIRECTION 1: RAW BYTE DETECTION
# =============================================================================

def direction1_raw(analyzer, metric_names):
    """Can geometry detect stego without preprocessing?"""
    print("\n" + "=" * 78)
    print("DIRECTION 1: Raw Byte Detection")
    print("=" * 78)
    print(f"All 6 techniques at 100% rate, {N_TRIALS} trials per (technique, carrier)")
    print(f"Question: are advanced techniques invisible at byte level?\n")

    # Collect clean baselines per carrier
    clean = {}  # carrier -> {metric: [values]}
    for carrier_name in CARRIERS:
        print(f"  Clean baseline: {carrier_name}...", end=" ", flush=True)
        clean[carrier_name] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_carrier(carrier_name, trial + 1000)
            collect_metrics(analyzer, data, clean[carrier_name])
        print("done")

    # Collect stego metrics
    n_total = len(metric_names)  # Bonferroni per comparison
    results = {}  # (technique, carrier) -> n_sig
    stego_raw_metrics = {}  # (technique, carrier) -> {metric: [values]} — saved for D5

    for tech_name, embed_fn in TECHNIQUES.items():
        for carrier_name in CARRIERS:
            print(f"  {tech_name} on {carrier_name}...", end=" ", flush=True)
            stego_metrics = defaultdict(list)
            for trial in range(N_TRIALS):
                carrier = generate_carrier(carrier_name, trial + 1000)
                stego = embed_fn(carrier, rate=1.0, seed=trial + 2000)
                collect_metrics(analyzer, stego, stego_metrics)
            stego_raw_metrics[(tech_name, carrier_name)] = stego_metrics
            n_sig = count_significant(stego_metrics, clean[carrier_name],
                                      metric_names, n_total)
            results[(tech_name, carrier_name)] = n_sig
            print(f"{n_sig} sig")

    # Print summary table
    print(f"\n  {'Technique':<20}", end="")
    for cn in CARRIERS:
        print(f"  {cn:>16}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for cn in CARRIERS:
        print(f"  {'-'*16}", end="")
    print()
    for tech in TECHNIQUES:
        print(f"  {tech:<20}", end="")
        for cn in CARRIERS:
            n = results[(tech, cn)]
            print(f"  {n:>16}", end="")
        print()

    return {'results': results, 'clean': clean, 'stego_raw_metrics': stego_raw_metrics}


# =============================================================================
# DIRECTION 2: BITPLANE DETECTION
# =============================================================================

def direction2_bitplane(analyzer, metric_names, d1_clean):
    """Does bitplane extraction crack the advanced techniques too?"""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Bitplane Detection")
    print("=" * 78)
    print(f"bitplane_extract(data, 0) before analysis — cracked LSB replace in stego.py")
    print(f"Also test plane 1 for PVD (modifies higher bits)\n")

    # Clean baselines with bitplane extraction
    clean_bp = {}  # (carrier, plane) -> {metric: [values]}
    for carrier_name in CARRIERS:
        for plane in [0, 1]:
            key = (carrier_name, plane)
            print(f"  Clean BP{plane} baseline: {carrier_name}...", end=" ", flush=True)
            clean_bp[key] = defaultdict(list)
            for trial in range(N_TRIALS):
                data = generate_carrier(carrier_name, trial + 1000)
                bp = bitplane_extract(data, plane)
                collect_metrics(analyzer, bp, clean_bp[key])
            print("done")

    # Stego with bitplane extraction — plane 0 for all, plus plane 1 for PVD
    n_total = len(metric_names)  # Bonferroni per comparison
    results_bp0 = {}  # (technique, carrier) -> n_sig
    results_bp1_pvd = {}  # carrier -> n_sig
    stego_bp0_metrics = {}  # (technique, carrier) -> {metric: [values]} — saved for D5

    for tech_name, embed_fn in TECHNIQUES.items():
        for carrier_name in CARRIERS:
            print(f"  {tech_name} BP0 on {carrier_name}...", end=" ", flush=True)
            stego_metrics = defaultdict(list)
            for trial in range(N_TRIALS):
                carrier = generate_carrier(carrier_name, trial + 1000)
                stego = embed_fn(carrier, rate=1.0, seed=trial + 2000)
                bp = bitplane_extract(stego, 0)
                collect_metrics(analyzer, bp, stego_metrics)
            stego_bp0_metrics[(tech_name, carrier_name)] = stego_metrics
            n_sig = count_significant(stego_metrics, clean_bp[(carrier_name, 0)],
                                      metric_names, n_total)
            results_bp0[(tech_name, carrier_name)] = n_sig
            print(f"{n_sig} sig")

    # PVD on plane 1
    for carrier_name in CARRIERS:
        print(f"  PVD BP1 on {carrier_name}...", end=" ", flush=True)
        stego_metrics = defaultdict(list)
        for trial in range(N_TRIALS):
            carrier = generate_carrier(carrier_name, trial + 1000)
            stego = embed_pvd(carrier, rate=1.0, seed=trial + 2000)
            bp = bitplane_extract(stego, 1)
            collect_metrics(analyzer, bp, stego_metrics)
        n_sig = count_significant(stego_metrics, clean_bp[(carrier_name, 1)],
                                  metric_names, n_total)
        results_bp1_pvd[carrier_name] = n_sig
        print(f"{n_sig} sig")

    # Summary table
    print(f"\n  Bitplane 0:")
    print(f"  {'Technique':<20}", end="")
    for cn in CARRIERS:
        print(f"  {cn:>16}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for cn in CARRIERS:
        print(f"  {'-'*16}", end="")
    print()
    for tech in TECHNIQUES:
        print(f"  {tech:<20}", end="")
        for cn in CARRIERS:
            n = results_bp0[(tech, cn)]
            print(f"  {n:>16}", end="")
        print()

    print(f"\n  PVD on bitplane 1:")
    for cn in CARRIERS:
        print(f"    {cn}: {results_bp1_pvd[cn]} sig")

    return {'results_bp0': results_bp0, 'results_bp1_pvd': results_bp1_pvd,
            'clean_bp': clean_bp, 'stego_bp0_metrics': stego_bp0_metrics}


# =============================================================================
# DIRECTION 3: EMBEDDING RATE SENSITIVITY
# =============================================================================

def direction3_rate_sensitivity(analyzer, metric_names, d1_clean):
    """Minimum detectable rate per technique (raw bytes — strongest detection mode)."""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Embedding Rate Sensitivity")
    print("=" * 78)
    print(f"Carriers: both, raw byte analysis (strongest from D1)")
    print(f"Rates: 5%, 10%, 25%, 50% (100% reused from D1)")
    print(f"{N_TRIALS_RATE} trials per (technique, rate, carrier)\n")

    rates = [0.05, 0.10, 0.25, 0.50]

    n_total = len(metric_names)  # Bonferroni per comparison
    results = {}  # (technique, rate, carrier) -> n_sig

    for tech_name, embed_fn in TECHNIQUES.items():
        for carrier_name in CARRIERS:
            clean_metrics = d1_clean[carrier_name]
            for rate in rates:
                print(f"  {tech_name} rate={rate:.0%} on {carrier_name}...",
                      end=" ", flush=True)
                stego_metrics = defaultdict(list)
                for trial in range(N_TRIALS_RATE):
                    carrier = generate_carrier(carrier_name, trial + 1000)
                    stego = embed_fn(carrier, rate=rate, seed=trial + 3000)
                    collect_metrics(analyzer, stego, stego_metrics)
                n_sig = count_significant(stego_metrics, clean_metrics,
                                          metric_names, n_total)
                results[(tech_name, rate, carrier_name)] = n_sig
                print(f"{n_sig} sig")

    # Summary: rate × technique table (photo_blocks)
    all_rates = rates + [1.0]
    print(f"\n  photo_blocks:")
    print(f"  {'Technique':<20}", end="")
    for r in all_rates:
        print(f"  {r:>6.0%}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in all_rates:
        print(f"  {'-'*6}", end="")
    print()

    return {'results': results, 'rates': rates}


# =============================================================================
# DIRECTION 4: PREPROCESSING COMPOSITION
# =============================================================================

def direction4_composition(analyzer, metric_names, d1_clean):
    """Does delay embedding amplify the raw byte signal?"""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Preprocessing Composition")
    print("=" * 78)
    print(f"Carrier: photo_blocks, rate=50%")
    print(f"Transforms: DE2 (delay τ=2), BP0 (bitplane 0), BP0→DE2, DE2→BP0")
    print(f"{N_TRIALS_RATE} trials per (technique, transform)\n")

    carrier_name = 'photo_blocks'
    rate = 0.50

    transforms = {
        'DE2':     lambda d: delay_embed(d, 2),
        'BP0':     lambda d: bitplane_extract(d, 0),
        'BP0→DE2': lambda d: delay_embed(bitplane_extract(d, 0), 2),
        'DE2→BP0': lambda d: bitplane_extract(delay_embed(d, 2), 0),
    }

    # Clean baselines per transform
    clean_tr = {}
    for tr_name, tr_fn in transforms.items():
        print(f"  Clean {tr_name}...", end=" ", flush=True)
        clean_tr[tr_name] = defaultdict(list)
        for trial in range(N_TRIALS_RATE):
            data = generate_carrier(carrier_name, trial + 1000)
            transformed = tr_fn(data)
            if len(transformed) < 50:
                continue
            collect_metrics(analyzer, transformed, clean_tr[tr_name])
        print("done")

    # Stego per transform
    n_total = len(metric_names)  # Bonferroni per comparison
    results = {}  # (technique, transform) -> n_sig

    for tech_name, embed_fn in TECHNIQUES.items():
        for tr_name, tr_fn in transforms.items():
            print(f"  {tech_name} {tr_name}...", end=" ", flush=True)
            stego_metrics = defaultdict(list)
            for trial in range(N_TRIALS_RATE):
                carrier = generate_carrier(carrier_name, trial + 1000)
                stego = embed_fn(carrier, rate=rate, seed=trial + 4000)
                transformed = tr_fn(stego)
                if len(transformed) < 50:
                    continue
                collect_metrics(analyzer, transformed, stego_metrics)
            n_sig = count_significant(stego_metrics, clean_tr[tr_name],
                                      metric_names, n_total)
            results[(tech_name, tr_name)] = n_sig
            print(f"{n_sig} sig")

    return {'results': results}


# =============================================================================
# DIRECTION 5: PER-GEOMETRY SENSITIVITY
# =============================================================================

def direction5_per_geometry(metric_names, d1_stego_raw, d1_clean):
    """Which geometries detect which techniques? (reuses D1 data — 0 extra calls)"""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Per-Geometry Sensitivity")
    print("=" * 78)
    print(f"Reusing D1 raw byte data — 0 extra analyzer calls")
    print(f"Aggregating per-metric significance by geometry family\n")

    carrier_name = 'natural_texture'  # strongest detection in D1
    n_total = len(metric_names)  # Bonferroni per comparison

    # Reuse stego metrics collected during D1
    stego_by_tech = {tech: d1_stego_raw[(tech, carrier_name)]
                     for tech in TECHNIQUES}

    clean_metrics = d1_clean[carrier_name]

    # Per-metric significance for each technique
    per_metric_by_tech = {}
    for tech_name in TECHNIQUES:
        pms = per_metric_significance(stego_by_tech[tech_name], clean_metrics,
                                      metric_names, n_total)
        per_metric_by_tech[tech_name] = pms

    # Aggregate by geometry family
    geom_families = defaultdict(list)  # geometry_name -> [metric_keys]
    for km in metric_names:
        geom = km.split(':')[0]
        geom_families[geom].append(km)

    # Build heatmap: rows=geometries, cols=techniques, cells=count of sig metrics
    geom_names_sorted = sorted(geom_families.keys())
    tech_names = list(TECHNIQUES.keys())

    heatmap = np.zeros((len(geom_names_sorted), len(tech_names)), dtype=int)
    for gi, geom in enumerate(geom_names_sorted):
        for ti, tech in enumerate(tech_names):
            pms = per_metric_by_tech[tech]
            sig_count = sum(1 for km in geom_families[geom]
                           if pms.get(km, (0, 1, False))[2])
            heatmap[gi, ti] = sig_count

    # Sort geometries by total detection power
    total_per_geom = heatmap.sum(axis=1)
    sort_idx = np.argsort(-total_per_geom)
    heatmap = heatmap[sort_idx]
    geom_names_sorted = [geom_names_sorted[i] for i in sort_idx]

    # Classify: specialist (detects ≤2 techniques) vs generalist (detects ≥4)
    detects_count = (heatmap > 0).sum(axis=1)
    specialists = [(geom_names_sorted[i], heatmap[i])
                   for i in range(len(geom_names_sorted))
                   if 0 < detects_count[i] <= 2]
    generalists = [(geom_names_sorted[i], heatmap[i])
                   for i in range(len(geom_names_sorted))
                   if detects_count[i] >= 4]

    print(f"\n  Generalists (detect ≥4 techniques): {len(generalists)}")
    for name, row in generalists:
        print(f"    {name:<35} total={row.sum()}")
    print(f"\n  Specialists (detect ≤2 techniques): {len(specialists)}")
    for name, row in specialists:
        techs_detected = [tech_names[j] for j in range(len(tech_names)) if row[j] > 0]
        print(f"    {name:<35} → {', '.join(techs_detected)}")

    return {
        'heatmap': heatmap,
        'geom_names': geom_names_sorted,
        'tech_names': tech_names,
        'per_metric_by_tech': per_metric_by_tech,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("ADVANCED STEGANOGRAPHY DETECTION VIA EXOTIC GEOMETRIES")
    print("=" * 78)
    print(f"\nParameters: N_TRIALS={N_TRIALS}, N_TRIALS_RATE={N_TRIALS_RATE}, "
          f"DATA_SIZE={DATA_SIZE}")
    print(f"6 techniques, 2 carriers, 5 directions\n")

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get metric names
    dummy = np.random.RandomState(0).randint(0, 256, DATA_SIZE, dtype=np.uint8)
    dummy_results = analyzer.analyze(dummy)
    metric_names = []
    for r in dummy_results.results:
        for mn in r.metrics:
            metric_names.append(f"{r.geometry_name}:{mn}")
    metric_names = sorted(set(metric_names))
    print(f"  {len(metric_names)} metrics across {len(dummy_results.results)} geometries\n")

    d1 = direction1_raw(analyzer, metric_names)
    d2 = direction2_bitplane(analyzer, metric_names, d1['clean'])
    d3 = direction3_rate_sensitivity(analyzer, metric_names, d1['clean'])
    d4 = direction4_composition(analyzer, metric_names, d1['clean'])
    d5 = direction5_per_geometry(metric_names, d1['stego_raw_metrics'], d1['clean'])

    # ── SUMMARY ──
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"\n  D1 — Raw Byte Detection (100% rate):")
    for tech in TECHNIQUES:
        vals = [d1['results'].get((tech, cn), 0) for cn in CARRIERS]
        print(f"    {tech:<20} photo={vals[0]:>3}  texture={vals[1]:>3}")

    print(f"\n  D2 — Bitplane Detection (plane 0, 100% rate):")
    for tech in TECHNIQUES:
        vals = [d2['results_bp0'].get((tech, cn), 0) for cn in CARRIERS]
        print(f"    {tech:<20} photo={vals[0]:>3}  texture={vals[1]:>3}")
    print(f"    PVD on plane 1:    photo={d2['results_bp1_pvd'].get('photo_blocks', 0):>3}  "
          f"texture={d2['results_bp1_pvd'].get('natural_texture', 0):>3}")

    d4_transforms = ['DE2', 'BP0', 'BP0→DE2', 'DE2→BP0']

    print(f"\n  D3 — Rate Sensitivity (raw bytes):")
    rates_all = d3['rates'] + [1.0]
    for cn in CARRIERS:
        print(f"    {cn}:")
        print(f"      {'Technique':<20}", end="")
        for r in rates_all:
            print(f"  {r:>5.0%}", end="")
        print()
        for tech in TECHNIQUES:
            print(f"      {tech:<20}", end="")
            for r in rates_all:
                if r == 1.0:
                    n = d1['results'].get((tech, cn), 0)
                else:
                    n = d3['results'].get((tech, r, cn), 0)
                print(f"  {n:>5}", end="")
            print()

    print(f"\n  D4 — Preprocessing Transforms (photo_blocks, 50%):")
    for tech in TECHNIQUES:
        parts = [f"{tr}={d4['results'].get((tech, tr), 0):>2}"
                 for tr in d4_transforms]
        print(f"    {tech:<20} {' '.join(parts)}")

    print(f"\n  D5 — Per-Geometry Sensitivity (texture, raw):")
    tech_short = [t[:8] for t in list(TECHNIQUES.keys())]
    print(f"    {'Geometry':<35} {' '.join(f'{s:>8}' for s in tech_short)}  total")
    for i in range(min(10, len(d5['geom_names']))):
        name = d5['geom_names'][i]
        row = d5['heatmap'][i]
        if row.sum() == 0:
            break
        vals = ' '.join(f'{int(v):>8}' for v in row)
        print(f"    {name:<35} {vals}  {int(row.sum()):>5}")

    print(f"\n[Investigation complete]")

    return d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION — 6-panel layout (22×14, dark theme)
# =============================================================================

def make_figure(d1, d2, d3, d4, d5):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'
    tech_names = list(TECHNIQUES.keys())
    tech_colors = [TECH_COLORS[t] for t in tech_names]

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ── Top-left: D1 — Grouped bar chart: sig metrics at raw byte level ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)

    x = np.arange(len(tech_names))
    bar_width = 0.35
    for ci, carrier_name in enumerate(CARRIERS):
        vals = [d1['results'].get((t, carrier_name), 0) for t in tech_names]
        offset = (ci - 0.5) * bar_width
        bars = ax1.bar(x + offset, vals, bar_width, alpha=0.85,
                       color=[TECH_COLORS[t] for t in tech_names],
                       edgecolor='#333',
                       label=carrier_name if ci == 0 else None)
        if ci == 1:
            # Second carrier: darker shade
            for bar in bars:
                bar.set_alpha(0.5)
                bar.set_hatch('//')

    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace(' ', '\n') for t in tech_names],
                        fontsize=7, color=FG)
    ax1.set_ylabel('Significant metrics vs clean', color=FG, fontsize=9)
    ax1.set_title('D1: Raw Byte Detection (100% rate)', fontsize=11,
                  fontweight='bold', color=FG)
    # Manual legend for carriers
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor='#888', alpha=0.85, label='photo_blocks'),
        Patch(facecolor='#888', alpha=0.5, hatch='//', label='natural_texture'),
    ], fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    # ── Top-center: D2 — Bitplane detection with D1 overlay ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)

    carrier_name = 'photo_blocks'  # Primary display
    vals_d1 = [d1['results'].get((t, carrier_name), 0) for t in tech_names]
    vals_d2 = [d2['results_bp0'].get((t, carrier_name), 0) for t in tech_names]

    bars_d1 = ax2.bar(x - 0.18, vals_d1, 0.35, alpha=0.3,
                      color=tech_colors, edgecolor='#555', linestyle='--',
                      label='Raw (D1)')
    bars_d2 = ax2.bar(x + 0.18, vals_d2, 0.35, alpha=0.85,
                      color=tech_colors, edgecolor='#333',
                      label='Bitplane 0 (D2)')

    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace(' ', '\n') for t in tech_names],
                        fontsize=7, color=FG)
    ax2.set_ylabel('Significant metrics vs clean', color=FG, fontsize=9)
    ax2.set_title('D2: Bitplane Detection (photo_blocks)', fontsize=11,
                  fontweight='bold', color=FG)
    ax2.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)
    for i, v in enumerate(vals_d2):
        ax2.text(i + 0.18, v + 0.5, str(v), ha='center', color=FG, fontsize=7,
                 fontweight='bold')

    # ── Top-right: D3 — Rate sensitivity heatmap (natural_texture, raw bytes) ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    d3_carrier = 'natural_texture'  # strongest detection carrier
    rates_all = d3['rates'] + [1.0]
    hm_data = np.zeros((len(tech_names), len(rates_all)))
    for ti, tech in enumerate(tech_names):
        for ri, rate in enumerate(rates_all):
            if rate == 1.0:
                hm_data[ti, ri] = d1['results'].get((tech, d3_carrier), 0)
            else:
                hm_data[ti, ri] = d3['results'].get((tech, rate, d3_carrier), 0)

    im3 = ax3.imshow(hm_data, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                     vmin=0)
    ax3.set_xticks(range(len(rates_all)))
    ax3.set_xticklabels([f'{r:.0%}' for r in rates_all], fontsize=8, color=FG)
    ax3.set_yticks(range(len(tech_names)))
    ax3.set_yticklabels(tech_names, fontsize=8, color=FG)
    ax3.set_xlabel('Embedding rate', color=FG, fontsize=9)
    ax3.set_title('D3: Rate Sensitivity (texture, raw)', fontsize=11,
                  fontweight='bold', color=FG)

    # Annotate cells and mark minimum detectable rate
    for ti in range(len(tech_names)):
        min_rate = None
        for ri in range(len(rates_all)):
            val = int(hm_data[ti, ri])
            ax3.text(ri, ti, str(val), ha='center', va='center',
                     color='white' if val > hm_data.max() * 0.5 else FG,
                     fontsize=8, fontweight='bold')
            if val > 0 and min_rate is None:
                min_rate = ri
        if min_rate is not None:
            ax3.text(min_rate, ti + 0.35, '★', ha='center', va='center',
                     color='#FFD700', fontsize=10)

    cb3 = fig.colorbar(im3, ax=ax3, shrink=0.7, pad=0.02)
    cb3.set_label('Sig metrics', color=FG, fontsize=8)
    cb3.ax.tick_params(colors=FG, labelsize=7)

    # ── Bottom-left: D4 — Transform comparison ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    carrier_name = 'photo_blocks'
    d4_transforms = ['DE2', 'BP0', 'BP0→DE2', 'DE2→BP0']
    n_groups = len(d4_transforms)
    bar_w = 0.8 / n_groups
    x4 = np.arange(len(tech_names))
    group_colors = ['#E91E63', '#4CAF50', '#2196F3', '#FF9800']

    for gi, tr in enumerate(d4_transforms):
        vals = [d4['results'].get((tech, tr), 0) for tech in tech_names]
        offset = (gi - n_groups / 2 + 0.5) * bar_w
        ax4.bar(x4 + offset, vals, bar_w, alpha=0.85,
                color=group_colors[gi], edgecolor='#333', label=tr)

    ax4.set_xticks(x4)
    ax4.set_xticklabels([t.replace(' ', '\n') for t in tech_names],
                        fontsize=7, color=FG)
    ax4.set_ylabel('Significant metrics vs clean', color=FG, fontsize=9)
    ax4.set_title('D4: Preprocessing Transforms (photo, 50%)', fontsize=11,
                  fontweight='bold', color=FG)
    ax4.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG,
               ncol=2)

    # ── Bottom-center: D5 — Geometry × Technique heatmap ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    hm5 = d5['heatmap']
    geom_names = d5['geom_names']
    # Limit to geometries with at least 1 detection
    mask = hm5.sum(axis=1) > 0
    hm5_filtered = hm5[mask]
    geom_filtered = [geom_names[i] for i in range(len(geom_names)) if mask[i]]

    if len(hm5_filtered) > 0:
        im5 = ax5.imshow(hm5_filtered, aspect='auto', cmap='YlOrRd',
                         interpolation='nearest', vmin=0)
        ax5.set_xticks(range(len(tech_names)))
        ax5.set_xticklabels([t.replace(' ', '\n') for t in tech_names],
                            fontsize=6, color=FG)
        ax5.set_yticks(range(len(geom_filtered)))
        ax5.set_yticklabels(geom_filtered, fontsize=6, color=FG)

        for gi in range(len(geom_filtered)):
            for ti in range(len(tech_names)):
                val = int(hm5_filtered[gi, ti])
                if val > 0:
                    ax5.text(ti, gi, str(val), ha='center', va='center',
                             color='white' if val > hm5_filtered.max() * 0.5 else FG,
                             fontsize=7)

        cb5 = fig.colorbar(im5, ax=ax5, shrink=0.7, pad=0.02)
        cb5.set_label('Sig metrics', color=FG, fontsize=8)
        cb5.ax.tick_params(colors=FG, labelsize=7)

    ax5.set_title('D5: Geometry × Technique (texture, raw)', fontsize=11,
                  fontweight='bold', color=FG)

    # ── Bottom-right: Summary text panel ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    d4_transforms = ['DE2', 'BP0', 'BP0→DE2', 'DE2→BP0']
    lines = [
        "Stego Deep — Key Findings",
        "",
        "D1: Raw Byte (texture, 100%)",
    ]
    for tech in tech_names:
        v = d1['results'].get((tech, 'natural_texture'), 0)
        lines.append(f"  {tech:<18} {v:>3} sig")

    lines.append("")
    lines.append("D2: Bitplane 0 (all techniques)")
    bp0_total = sum(d2['results_bp0'].get((t, c), 0)
                    for t in tech_names for c in CARRIERS)
    lines.append(f"  Total: {bp0_total} (bitplane blind)")

    lines.append("")
    lines.append("D3: Min Rate (texture, raw)")
    rates_all = d3['rates'] + [1.0]
    for tech in tech_names:
        min_rate = None
        for r in rates_all:
            if r == 1.0:
                v = d1['results'].get((tech, 'natural_texture'), 0)
            else:
                v = d3['results'].get((tech, r, 'natural_texture'), 0)
            if v > 0 and min_rate is None:
                min_rate = r
        if min_rate is not None:
            lines.append(f"  {tech:<18} {min_rate:>5.0%}")
        else:
            lines.append(f"  {tech:<18}  n/a")

    lines.append("")
    lines.append("D4: Best Transform (photo, 50%)")
    for tech in tech_names:
        best_tr = None
        best_val = 0
        for tr in d4_transforms:
            v = d4['results'].get((tech, tr), 0)
            if v > best_val:
                best_val = v
                best_tr = tr
        if best_tr:
            lines.append(f"  {tech:<18} {best_tr}={best_val}")
        else:
            lines.append(f"  {tech:<18} none")

    lines.append("")
    lines.append("D5: Top Geometries (texture)")
    for i in range(min(5, len(d5['geom_names']))):
        name = d5['geom_names'][i]
        total = d5['heatmap'][i].sum()
        if total > 0:
            lines.append(f"  {name:<28} {total:>2}")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor='#444'))

    fig.suptitle('Advanced Steganography Detection via Exotic Geometries',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'stego_deep.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    d1, d2, d3, d4, d5 = main()
    make_figure(d1, d2, d3, d4, d5)
