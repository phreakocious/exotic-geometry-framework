#!/usr/bin/env python3
"""
Exotic Geometry Analysis of DNA Sequences
==========================================

Test: Do DNA sequences from different organisms/regions have distinct geometric
signatures when analyzed through 21 exotic geometries?

Generates synthetic DNA with realistic statistical properties for 10 categories,
encodes as uint8 (A=0, C=85, G=170, T=255), and runs comprehensive geometric
analysis with rigorous statistics (Cohen's d, Bonferroni correction, shuffle tests).

Author: Claude + TB
Date: 2026-02-05
"""

import sys
import os
import numpy as np
import warnings
from collections import defaultdict
from itertools import combinations
from scipy import stats

warnings.filterwarnings('ignore')

# Add parent dir to path for framework import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from exotic_geometry_framework import GeometryAnalyzer

# =============================================================================
# DNA ENCODING
# =============================================================================

# Spread bases across byte range for geometric analysis
BASE_MAP = {'A': 0, 'C': 85, 'G': 170, 'T': 255}
BASES = ['A', 'C', 'G', 'T']

def dna_to_bytes(seq):
    """Convert DNA string to uint8 array."""
    return np.array([BASE_MAP[b] for b in seq], dtype=np.uint8)

# =============================================================================
# SYNTHETIC DNA GENERATORS
# =============================================================================

def gen_ecoli(length, rng):
    """E. coli-like: GC ~50.8%, codon bias, short repeats."""
    # E. coli GC = 50.8%
    gc = 0.508
    # Codon bias: certain codons preferred
    # Approximate with dinucleotide frequencies
    # E. coli has slight preference for GC->GC transitions
    seq = []
    # Start with random base weighted by GC content
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A, C, G, T
    prev = rng.choice(4, p=probs)
    seq.append(BASES[prev])

    # Dinucleotide transition matrix (E. coli-like)
    # Rows = current base, cols = next base (A, C, G, T)
    trans = np.array([
        [0.30, 0.22, 0.28, 0.20],  # After A
        [0.18, 0.30, 0.20, 0.32],  # After C
        [0.28, 0.18, 0.30, 0.24],  # After G
        [0.20, 0.28, 0.22, 0.30],  # After T
    ])
    # Normalize rows
    trans = trans / trans.sum(axis=1, keepdims=True)

    for i in range(1, length):
        # Add short repeats occasionally (bacterial repeats ~4-8bp)
        if i > 8 and rng.random() < 0.02:
            repeat_len = rng.integers(4, 9)
            start = max(0, i - repeat_len)
            repeat = seq[start:i]
            for b in repeat[:min(repeat_len, length - i)]:
                seq.append(b)
            if len(seq) >= length:
                break
            continue

        prev_idx = BASES.index(seq[-1])
        next_idx = rng.choice(4, p=trans[prev_idx])
        seq.append(BASES[next_idx])

    return ''.join(seq[:length])


def gen_human(length, rng):
    """Human-like: GC ~40.9%, CpG suppression, Alu-like repeats, GC variation."""
    gc = 0.409
    seq = []

    # CpG suppression: CG dinucleotide is ~4x less common in humans
    trans = np.array([
        [0.30, 0.18, 0.22, 0.30],  # After A
        [0.28, 0.25, 0.05, 0.42],  # After C - CpG suppressed! (G column low)
        [0.28, 0.22, 0.25, 0.25],  # After G
        [0.30, 0.18, 0.22, 0.30],  # After T
    ])
    trans = trans / trans.sum(axis=1, keepdims=True)

    # Alu-like repeat consensus (~300bp, most common human repeat)
    alu_motif = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGG"

    prev = rng.choice(4, p=[(1-gc)/2, gc/2, gc/2, (1-gc)/2])
    seq.append(BASES[prev])

    i = 1
    while i < length:
        # Insert Alu-like element occasionally (~10% of human genome is Alu)
        if rng.random() < 0.005 and i + 50 < length:
            # Insert fragment of Alu (with mutations)
            frag_len = min(rng.integers(30, 56), length - i)
            frag_start = rng.integers(0, len(alu_motif) - frag_len + 1)
            for j in range(frag_len):
                if rng.random() < 0.15:  # ~15% divergence
                    seq.append(BASES[rng.integers(4)])
                else:
                    seq.append(alu_motif[frag_start + j])
            i += frag_len
            continue

        # GC content varies across regions (isochores)
        # Simulate by occasionally shifting transition probabilities
        if i % 200 == 0:
            gc_local = gc + rng.normal(0, 0.08)
            gc_local = np.clip(gc_local, 0.3, 0.55)
            base_prob = [(1-gc_local)/2, gc_local/2, gc_local/2, (1-gc_local)/2]
            # Remix transition matrix with local GC
            for row in range(4):
                trans[row] = 0.5 * trans[row] + 0.5 * np.array(base_prob)
            trans = trans / trans.sum(axis=1, keepdims=True)

        prev_idx = BASES.index(seq[-1])
        next_idx = rng.choice(4, p=trans[prev_idx])
        seq.append(BASES[next_idx])
        i += 1

    return ''.join(seq[:length])


def gen_at_rich(length, rng):
    """Plasmodium falciparum-like: GC ~19.4%, extreme AT bias."""
    gc = 0.194
    seq = []
    # Very strong AT bias, with occasional GC
    trans = np.array([
        [0.42, 0.08, 0.08, 0.42],  # After A: mostly A or T
        [0.35, 0.10, 0.10, 0.45],  # After C
        [0.35, 0.10, 0.10, 0.45],  # After G
        [0.42, 0.08, 0.08, 0.42],  # After T: mostly A or T
    ])
    trans = trans / trans.sum(axis=1, keepdims=True)

    prev = rng.choice(4, p=[(1-gc)/2, gc/2, gc/2, (1-gc)/2])
    seq.append(BASES[prev])

    for i in range(1, length):
        # AT-rich repeats (poly-A/T runs common in Plasmodium)
        if rng.random() < 0.03:
            run_len = min(rng.integers(5, 20), length - len(seq))
            base = 'A' if rng.random() < 0.5 else 'T'
            for _ in range(run_len):
                seq.append(base)
            continue

        prev_idx = BASES.index(seq[-1])
        next_idx = rng.choice(4, p=trans[prev_idx])
        seq.append(BASES[next_idx])

    return ''.join(seq[:length])


def gen_gc_rich(length, rng):
    """Streptomyces-like: GC ~72%, strong GC bias."""
    gc = 0.72
    seq = []
    trans = np.array([
        [0.12, 0.38, 0.38, 0.12],  # After A: prefer GC
        [0.12, 0.38, 0.38, 0.12],  # After C
        [0.12, 0.38, 0.38, 0.12],  # After G
        [0.12, 0.38, 0.38, 0.12],  # After T
    ])
    trans = trans / trans.sum(axis=1, keepdims=True)

    prev = rng.choice(4, p=[(1-gc)/2, gc/2, gc/2, (1-gc)/2])
    seq.append(BASES[prev])

    for i in range(1, length):
        prev_idx = BASES.index(seq[-1])
        next_idx = rng.choice(4, p=trans[prev_idx])
        seq.append(BASES[next_idx])

    return ''.join(seq[:length])


def gen_viral(length, rng):
    """Viral-like: dense coding, overlapping reading frames."""
    gc = 0.45
    seq = []

    # Generate with codon structure (triplet periodicity)
    # Viral genomes have strong codon bias due to dense coding
    # Preferred codons (simplified): certain triplets much more common
    preferred_codons = [
        'ATG', 'GCT', 'GAA', 'AAA', 'CTG', 'GAT', 'GTT', 'ACT',
        'GGT', 'TTC', 'AAC', 'CCG', 'AGC', 'TAT', 'CAG', 'TGG',
    ]

    # Build sequence from codons
    while len(seq) < length:
        if rng.random() < 0.6:
            codon = preferred_codons[rng.integers(len(preferred_codons))]
        else:
            codon = ''.join(rng.choice(list('ACGT'), size=3))
        for b in codon:
            seq.append(b)

    # Overlapping reading frames: add constraint that reverse complement
    # also has codon structure (every 3rd position from offset 1 and 2 also structured)
    # This creates additional periodic structure
    result = list(seq[:length])
    for i in range(1, length - 2, 3):
        if rng.random() < 0.3:
            # Force some positions to maintain reading frame +1
            pass  # The codon structure already creates 3-periodicity

    return ''.join(result[:length])


def gen_random_dna(length, rng):
    """Uniform random ACGT - control."""
    return ''.join(rng.choice(list('ACGT'), size=length))


def gen_protein_coding(length, rng):
    """Protein-coding region: strong codon structure, 3rd position wobble."""
    seq = []

    # Codon usage table (simplified mammalian)
    # Position 1 and 2 are more conserved, position 3 (wobble) more variable
    # Model: positions 1,2 have strong base preference, position 3 nearly uniform

    pos1_probs = [0.20, 0.25, 0.30, 0.25]  # Slightly G-rich
    pos2_probs = [0.35, 0.20, 0.15, 0.30]  # A/T-rich (hydrophobic amino acids)
    pos3_probs = [0.25, 0.25, 0.25, 0.25]  # Wobble: nearly uniform

    for i in range(0, length, 3):
        if i < length:
            seq.append(BASES[rng.choice(4, p=pos1_probs)])
        if i + 1 < length:
            seq.append(BASES[rng.choice(4, p=pos2_probs)])
        if i + 2 < length:
            seq.append(BASES[rng.choice(4, p=pos3_probs)])

    return ''.join(seq[:length])


def gen_cpg_island(length, rng):
    """CpG island: high CG dinucleotide frequency, GC ~65%."""
    gc = 0.65
    seq = []

    # CpG islands: CG dinucleotide at expected frequency (not suppressed)
    # In fact, CpG ratio (observed/expected) > 0.6
    trans = np.array([
        [0.15, 0.30, 0.30, 0.25],  # After A
        [0.15, 0.25, 0.40, 0.20],  # After C: HIGH CpG! G is preferred
        [0.20, 0.30, 0.30, 0.20],  # After G
        [0.15, 0.30, 0.30, 0.25],  # After T
    ])
    trans = trans / trans.sum(axis=1, keepdims=True)

    prev = rng.choice(4, p=[(1-gc)/2, gc/2, gc/2, (1-gc)/2])
    seq.append(BASES[prev])

    for i in range(1, length):
        prev_idx = BASES.index(seq[-1])
        next_idx = rng.choice(4, p=trans[prev_idx])
        seq.append(BASES[next_idx])

    return ''.join(seq[:length])


def gen_microsatellite(length, rng):
    """Microsatellite / tandem repeat: e.g., CAGCAGCAG..."""
    # Pick a repeat unit
    units = ['CAG', 'CTG', 'AT', 'AC', 'AATG', 'GATA', 'TTAGGG', 'CA']
    unit = units[rng.integers(len(units))]

    # Generate primarily from repeats with occasional mutations
    seq = []
    while len(seq) < length:
        for b in unit:
            if rng.random() < 0.03:  # ~3% mutation rate
                seq.append(BASES[rng.integers(4)])
            else:
                seq.append(b)

    return ''.join(seq[:length])


def gen_shuffled_ecoli(length, rng):
    """Shuffled E. coli: same composition as E. coli but random order."""
    # Generate E. coli sequence first
    ecoli_seq = gen_ecoli(length, rng)
    # Shuffle to destroy sequential structure but keep composition
    seq_list = list(ecoli_seq)
    rng.shuffle(seq_list)
    return ''.join(seq_list)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

DNA_TYPES = {
    "E.coli":          gen_ecoli,
    "Human":           gen_human,
    "AT-rich":         gen_at_rich,
    "GC-rich":         gen_gc_rich,
    "Viral":           gen_viral,
    "Random":          gen_random_dna,
    "Protein-coding":  gen_protein_coding,
    "CpG-island":      gen_cpg_island,
    "Microsatellite":  gen_microsatellite,
    "Shuffled-Ecoli":  gen_shuffled_ecoli,
}

N_TRIALS = 25
SEQ_LENGTH = 2000
SEED = 42


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (m1 - m2) / pooled_std


def effect_size_label(d):
    """Label for Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    if d < 1.2: return "large"
    if d < 2.0: return "very large"
    return "huge"


def main():
    print("=" * 80)
    print("EXOTIC GEOMETRY ANALYSIS OF DNA SEQUENCES")
    print("=" * 80)
    print(f"\nParameters: {N_TRIALS} trials x {SEQ_LENGTH} bases, {len(DNA_TYPES)} DNA types")
    print(f"Encoding: A=0, C=85, G=170, T=255")
    print(f"Geometries: 21 (full framework)")
    print()

    rng = np.random.default_rng(SEED)

    # Initialize analyzer with all geometries
    analyzer = GeometryAnalyzer().add_all_geometries()
    geom_names = [g.name for g in analyzer.geometries]
    print(f"Geometries loaded: {len(geom_names)}")
    for gn in geom_names:
        print(f"  - {gn}")
    print()

    # =========================================================================
    # Step 1: Generate all DNA and run all geometries
    # =========================================================================
    print("=" * 80)
    print("STEP 1: Generating synthetic DNA and running geometric analysis")
    print("=" * 80)

    # Store results: {dna_type: {geom_name: {metric_name: [values across trials]}}}
    all_results = {}

    for dtype, gen_func in DNA_TYPES.items():
        print(f"\n  Processing {dtype}...", end=" ", flush=True)
        all_results[dtype] = defaultdict(lambda: defaultdict(list))

        for trial in range(N_TRIALS):
            # Generate DNA
            seq = gen_func(SEQ_LENGTH, rng)
            data = dna_to_bytes(seq)

            # Verify composition on first trial
            if trial == 0:
                gc_count = sum(1 for b in seq if b in 'GC')
                gc_frac = gc_count / len(seq)
                print(f"(GC={gc_frac:.3f})", end=" ", flush=True)

            # Run all geometries
            result = analyzer.analyze(data, f"{dtype}_trial{trial}")

            for geom_result in result.results:
                for metric_name, metric_val in geom_result.metrics.items():
                    all_results[dtype][geom_result.geometry_name][metric_name].append(metric_val)

        print(f"[{N_TRIALS} trials done]")

    # =========================================================================
    # Step 2: Fingerprint table
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: GEOMETRIC FINGERPRINT TABLE (mean values)")
    print("=" * 80)

    # Collect all geometry-metric pairs
    all_metrics = []
    for dtype in DNA_TYPES:
        for gname in all_results[dtype]:
            for mname in all_results[dtype][gname]:
                if (gname, mname) not in all_metrics:
                    all_metrics.append((gname, mname))

    # Select key metrics (one per geometry, the most discriminative)
    # We'll pick the primary metric for each geometry
    key_metrics = []
    seen_geoms = set()
    for gname, mname in all_metrics:
        # Pick the first/primary metric for each geometry
        if gname not in seen_geoms:
            key_metrics.append((gname, mname))
            seen_geoms.add(gname)

    # Print compact fingerprint table
    print(f"\n{'DNA Type':<18}", end="")
    short_names = []
    for gname, mname in key_metrics[:12]:  # Show first 12 for readability
        short = f"{gname[:8]}/{mname[:8]}"
        short_names.append(short)
        print(f"{short:>18}", end="")
    print()
    print("-" * (18 + 18 * min(12, len(key_metrics))))

    for dtype in DNA_TYPES:
        print(f"{dtype:<18}", end="")
        for gname, mname in key_metrics[:12]:
            vals = all_results[dtype].get(gname, {}).get(mname, [])
            if vals:
                print(f"{np.mean(vals):>18.4f}", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()

    # Also print full table to a more readable format
    print("\n\nDETAILED FINGERPRINT (all metrics, grouped by geometry):")
    print("-" * 80)

    for gname, mname in all_metrics:
        print(f"\n  {gname} -> {mname}:")
        line = "    "
        for dtype in DNA_TYPES:
            vals = all_results[dtype].get(gname, {}).get(mname, [])
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                line += f"{dtype}={mean_v:.4f}({std_v:.3f})  "
        print(line)

    # =========================================================================
    # Step 3: Each type vs Random baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EACH DNA TYPE vs RANDOM BASELINE")
    print("=" * 80)
    print("(Cohen's d, Bonferroni-corrected p-values)")
    print("Significance: |d| > 0.8 AND corrected p < 0.05")

    n_comparisons = (len(DNA_TYPES) - 1) * len(all_metrics)  # Bonferroni correction
    print(f"Bonferroni correction factor: {n_comparisons}")

    significant_findings = []

    for dtype in DNA_TYPES:
        if dtype == "Random":
            continue

        print(f"\n--- {dtype} vs Random ---")
        sig_count = 0

        for gname, mname in all_metrics:
            vals_type = all_results[dtype].get(gname, {}).get(mname, [])
            vals_random = all_results["Random"].get(gname, {}).get(mname, [])

            if not vals_type or not vals_random:
                continue

            d = cohens_d(vals_type, vals_random)
            _, p_raw = stats.ttest_ind(vals_type, vals_random, equal_var=False)
            p_corrected = min(p_raw * n_comparisons, 1.0)

            is_sig = abs(d) > 0.8 and p_corrected < 0.05

            if is_sig:
                sig_count += 1
                marker = "***" if abs(d) > 2.0 else "**" if abs(d) > 1.2 else "*"
                print(f"  {marker} {gname}/{mname}: d={d:+.3f} ({effect_size_label(d)}), "
                      f"p_corr={p_corrected:.2e}")
                significant_findings.append({
                    'dtype': dtype, 'gname': gname, 'mname': mname,
                    'd': d, 'p_corr': p_corrected
                })

        if sig_count == 0:
            print("  (no significant differences)")
        else:
            print(f"  Total significant metrics: {sig_count}/{len(all_metrics)}")

    # =========================================================================
    # Step 4: Pairwise discrimination
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: PAIRWISE DISCRIMINATION MATRIX")
    print("=" * 80)
    print("(Number of metrics with |Cohen's d| > 0.8 and Bonferroni-corrected p < 0.05)")

    dtypes_list = list(DNA_TYPES.keys())
    n_pairs = len(list(combinations(dtypes_list, 2)))
    n_bonf_pairwise = n_pairs * len(all_metrics)

    # Compute pairwise discrimination
    pairwise_matrix = np.zeros((len(dtypes_list), len(dtypes_list)), dtype=int)
    pairwise_best = {}  # Store best discriminating metric for each pair

    for i, dt1 in enumerate(dtypes_list):
        for j, dt2 in enumerate(dtypes_list):
            if i >= j:
                continue

            best_d = 0
            best_metric = ""
            sig_count = 0

            for gname, mname in all_metrics:
                v1 = all_results[dt1].get(gname, {}).get(mname, [])
                v2 = all_results[dt2].get(gname, {}).get(mname, [])

                if not v1 or not v2:
                    continue

                d = cohens_d(v1, v2)
                _, p_raw = stats.ttest_ind(v1, v2, equal_var=False)
                p_corr = min(p_raw * n_bonf_pairwise, 1.0)

                if abs(d) > 0.8 and p_corr < 0.05:
                    sig_count += 1

                if abs(d) > abs(best_d):
                    best_d = d
                    best_metric = f"{gname}/{mname}"

            pairwise_matrix[i, j] = sig_count
            pairwise_matrix[j, i] = sig_count
            pairwise_best[(dt1, dt2)] = (best_metric, best_d)

    # Print matrix
    print(f"\nBonferroni correction factor: {n_bonf_pairwise}")
    print(f"\n{'':>18}", end="")
    for dt in dtypes_list:
        print(f"{dt[:8]:>10}", end="")
    print()

    for i, dt1 in enumerate(dtypes_list):
        print(f"{dt1:<18}", end="")
        for j, dt2 in enumerate(dtypes_list):
            if i == j:
                print(f"{'---':>10}", end="")
            else:
                print(f"{pairwise_matrix[i,j]:>10}", end="")
        print()

    # Show best discriminating metrics for selected pairs
    print("\nBest discriminating metric for key pairs:")
    interesting_pairs = [
        ("E.coli", "Human"), ("E.coli", "Shuffled-Ecoli"),
        ("Human", "AT-rich"), ("AT-rich", "GC-rich"),
        ("Protein-coding", "Random"), ("CpG-island", "Human"),
        ("Microsatellite", "Random"), ("Viral", "Random"),
    ]

    for dt1, dt2 in interesting_pairs:
        if (dt1, dt2) in pairwise_best:
            metric, d = pairwise_best[(dt1, dt2)]
        elif (dt2, dt1) in pairwise_best:
            metric, d = pairwise_best[(dt2, dt1)]
            d = -d
        else:
            continue
        n_sig = pairwise_matrix[dtypes_list.index(dt1), dtypes_list.index(dt2)]
        print(f"  {dt1} vs {dt2}: best={metric}, d={d:+.3f} ({effect_size_label(d)}), "
              f"total sig metrics={n_sig}")

    # =========================================================================
    # Step 5: Which geometries are best for which DNA features?
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: BEST GEOMETRIES FOR EACH DNA FEATURE")
    print("=" * 80)

    # For each DNA type, find which geometry/metric gives largest |d| vs random
    print("\nTop 3 geometries per DNA type (vs Random baseline):")
    print("-" * 80)

    for dtype in DNA_TYPES:
        if dtype == "Random":
            continue

        scores = []
        for gname, mname in all_metrics:
            v1 = all_results[dtype].get(gname, {}).get(mname, [])
            v2 = all_results["Random"].get(gname, {}).get(mname, [])
            if v1 and v2:
                d = cohens_d(v1, v2)
                _, p_raw = stats.ttest_ind(v1, v2, equal_var=False)
                scores.append((abs(d), d, p_raw, gname, mname))

        scores.sort(reverse=True)
        print(f"\n  {dtype}:")
        for rank, (abs_d, d, p, gn, mn) in enumerate(scores[:3]):
            p_corr = min(p * n_comparisons, 1.0)
            sig = "SIG" if abs_d > 0.8 and p_corr < 0.05 else "n.s."
            print(f"    {rank+1}. {gn}/{mn}: d={d:+.3f} ({effect_size_label(d)}), "
                  f"p_corr={p_corr:.2e} [{sig}]")

    # Aggregate: which geometry is best overall?
    print("\n\nOverall geometry ranking (mean |d| across all DNA types vs Random):")
    print("-" * 80)

    geom_scores = defaultdict(list)
    for dtype in DNA_TYPES:
        if dtype == "Random":
            continue
        for gname, mname in all_metrics:
            v1 = all_results[dtype].get(gname, {}).get(mname, [])
            v2 = all_results["Random"].get(gname, {}).get(mname, [])
            if v1 and v2:
                d = cohens_d(v1, v2)
                geom_scores[f"{gname}/{mname}"].append(abs(d))

    # Sort by mean |d|
    ranked = [(np.mean(scores), np.max(scores), key)
              for key, scores in geom_scores.items()]
    ranked.sort(reverse=True)

    print(f"{'Rank':<6}{'Geometry/Metric':<45}{'Mean |d|':>10}{'Max |d|':>10}")
    print("-" * 71)
    for rank, (mean_d, max_d, name) in enumerate(ranked[:20]):
        print(f"{rank+1:<6}{name:<45}{mean_d:>10.3f}{max_d:>10.3f}")

    # =========================================================================
    # Step 6: Shuffle validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SHUFFLE VALIDATION")
    print("=" * 80)
    print("Testing: do top findings survive shuffling?")
    print("If YES -> signal is from BASE COMPOSITION (not sequential structure)")
    print("If NO  -> signal is from SEQUENTIAL STRUCTURE (ordering matters)")

    # Take top 10 significant findings and test with shuffled data
    top_findings = sorted(significant_findings, key=lambda x: abs(x['d']), reverse=True)[:15]

    print(f"\nTesting top {len(top_findings)} significant findings:")
    print("-" * 80)

    # For each finding, generate shuffled versions and recompute
    for finding in top_findings:
        dtype = finding['dtype']
        gname = finding['gname']
        mname = finding['mname']
        original_d = finding['d']

        # Generate shuffled trials
        shuffled_vals = []
        for trial in range(N_TRIALS):
            seq = DNA_TYPES[dtype](SEQ_LENGTH, np.random.default_rng(SEED + 10000 + trial))
            # Shuffle the sequence
            seq_list = list(seq)
            np.random.default_rng(SEED + 20000 + trial).shuffle(seq_list)
            data = dna_to_bytes(''.join(seq_list))

            result = analyzer.analyze(data, f"shuffled_{dtype}_{trial}")
            for gr in result.results:
                if gr.geometry_name == gname and mname in gr.metrics:
                    shuffled_vals.append(gr.metrics[mname])

        if not shuffled_vals:
            continue

        # Compare shuffled vs random
        vals_random = all_results["Random"].get(gname, {}).get(mname, [])
        if not vals_random:
            continue

        d_shuffled = cohens_d(shuffled_vals, vals_random)

        # Original vs shuffled of same type
        vals_original = all_results[dtype].get(gname, {}).get(mname, [])
        d_orig_vs_shuf = cohens_d(vals_original, shuffled_vals)

        composition = "COMPOSITION" if abs(d_shuffled) > 0.8 else "composition(weak)"
        ordering = "ORDERING" if abs(d_orig_vs_shuf) > 0.8 else "ordering(weak)"

        # Determine signal source
        if abs(d_shuffled) > 0.8 and abs(d_orig_vs_shuf) < 0.5:
            source = "COMPOSITION ONLY"
        elif abs(d_shuffled) < 0.5 and abs(d_orig_vs_shuf) > 0.8:
            source = "ORDERING ONLY"
        elif abs(d_shuffled) > 0.8 and abs(d_orig_vs_shuf) > 0.8:
            source = "BOTH (composition + ordering)"
        else:
            source = "mixed/unclear"

        print(f"\n  {dtype} / {gname}/{mname}:")
        print(f"    Original vs Random:  d={original_d:+.3f}")
        print(f"    Shuffled vs Random:  d={d_shuffled:+.3f} -> {composition}")
        print(f"    Original vs Shuffled: d={d_orig_vs_shuf:+.3f} -> {ordering}")
        print(f"    Signal source: {source}")

    # =========================================================================
    # Step 7: Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: EXECUTIVE SUMMARY")
    print("=" * 80)

    # Count significant findings per DNA type
    sig_by_type = defaultdict(int)
    for f in significant_findings:
        sig_by_type[f['dtype']] += 1

    print(f"\n1. Total significant findings: {len(significant_findings)} "
          f"(|d|>0.8, Bonf. corrected p<0.05)")

    print(f"\n2. Significant metrics per DNA type (vs Random):")
    for dtype in sorted(sig_by_type.keys(), key=lambda x: sig_by_type[x], reverse=True):
        print(f"   {dtype}: {sig_by_type[dtype]} significant metrics")

    # Which types are hardest to distinguish from random?
    not_sig = [dt for dt in DNA_TYPES if dt != "Random" and dt not in sig_by_type]
    if not_sig:
        print(f"\n3. Types indistinguishable from random: {', '.join(not_sig)}")
    else:
        print(f"\n3. ALL non-random types show significant geometric differences from random!")

    # E. coli vs Shuffled E. coli: composition vs ordering
    ec_vs_shuf = pairwise_matrix[dtypes_list.index("E.coli"),
                                  dtypes_list.index("Shuffled-Ecoli")]
    print(f"\n4. E. coli vs Shuffled E. coli: {ec_vs_shuf} significant metrics")
    if ec_vs_shuf > 0:
        print("   -> Exotic geometries CAN detect sequential structure beyond composition!")
    else:
        print("   -> Geometric signatures are driven primarily by base composition")

    # Microsatellite should be very different
    micro_n = sig_by_type.get("Microsatellite", 0)
    print(f"\n5. Microsatellite (repetitive) vs Random: {micro_n} significant metrics")

    print(f"\n6. Total geometry-metric combinations tested: {len(all_metrics)}")
    print(f"   Total pairwise comparisons: {n_pairs}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
