#!/usr/bin/env python3
"""
Investigation: 2D Bit-Matrix Steganalysis (Deep).

The hypothesis: byte sequences have internal 2D structure that steganographic
embedding disrupts.  By choosing the right 2D representation, SpatialFieldGeometry
can detect changes invisible to 1D analysis.

The original prototype (concatenated bitplanes → SpatialField) detected nothing.
Three problems: (a) binary 0/1 grids kill continuous metric variance, (b) only
15 spatial metrics vs 131+ from the full analyzer, (c) the tiled layout creates
artificial spatial discontinuities.

Five directions:
  D1: Representation Survey — which 2D layout makes stego visible?
      (byte grid, N×8 bit matrix, difference grid, binned co-occurrence)
  D2: Co-occurrence Deep Dive — byte transition matrices as a 2D field
  D3: Spatial Fingerprints — per-metric Cohen's d for each technique
  D4: Rate Sensitivity — minimum detectable rate via co-occurrence
  D5: Matrix Embed Challenge — full 24-geometry analyzer on best 2D representations

Note: Bitplane tiled layout (original approach) dropped — binary 0/1 grids
cause SpatialField basin computation to degrade (flat gradient → 200 iters/cell)
and detected nothing in initial testing.

Total budget: ~1600 SpatialField calls + ~200 full analyzer calls, est. 3–4 min.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer, SpatialFieldGeometry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import importlib.util

# Load generators from stego_deep
_spec = importlib.util.spec_from_file_location(
    "stego", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '1d', 'stego_deep.py'))
_stego = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stego)

generate_carrier = _stego.generate_carrier
TECHNIQUES = _stego.TECHNIQUES
TECH_COLORS = _stego.TECH_COLORS

N_TRIALS = 25
DATA_SIZE = 4096   # clean 64×64 byte grid


# =============================================================================
# 2D REPRESENTATIONS
# =============================================================================

def repr_byte_grid(data):
    """Reshape uint8 bytes into a square grid (preserves full dynamic range)."""
    side = int(np.ceil(np.sqrt(len(data))))
    padded = np.full(side * side, float(np.mean(data)))
    padded[:len(data)] = data.astype(np.float64)
    return padded.reshape(side, side)


def repr_bit_matrix(data):
    """N×8 bit matrix: rows=bytes, columns=bit positions (MSB→LSB)."""
    return np.unpackbits(data).reshape(-1, 8).astype(np.float64)


def repr_byte_diff_grid(data):
    """Absolute consecutive-byte differences reshaped into a square grid."""
    diffs = np.abs(np.diff(data.astype(np.int16))).astype(np.float64)
    side = int(np.ceil(np.sqrt(len(diffs))))
    padded = np.full(side * side, float(np.mean(diffs)))
    padded[:len(diffs)] = diffs
    return padded.reshape(side, side)


def repr_cooccurrence(data, n_bins=32):
    """Binned byte co-occurrence matrix (normalized transition counts).
    256 values are binned into n_bins buckets so the matrix is dense enough
    for meaningful spatial analysis with typical DATA_SIZE (~4096 bytes)."""
    binned = (data.astype(np.float64) * n_bins / 256).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    mat = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(mat, (binned[:-1], binned[1:]), 1)
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


REPRESENTATIONS = {
    'Byte Grid':       repr_byte_grid,
    'Bit Matrix N×8':  repr_bit_matrix,
    'Diff Grid':       repr_byte_diff_grid,
    'Co-occurrence':   repr_cooccurrence,
}


# =============================================================================
# STATISTICS
# =============================================================================

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled


def count_sig(clean_metrics, stego_metrics, n_metrics):
    """Count significant metrics (|d|>0.8 and Bonferroni-corrected p<0.05)."""
    alpha = 0.05 / n_metrics
    sig = 0
    details = []
    for mn in clean_metrics:
        if mn not in stego_metrics:
            continue
        d = cohens_d(stego_metrics[mn], clean_metrics[mn])
        _, p = stats.ttest_ind(stego_metrics[mn], clean_metrics[mn])
        is_sig = p < alpha and abs(d) > 0.8
        if is_sig:
            sig += 1
        details.append((mn, d, p, is_sig))
    return sig, details


# =============================================================================
# D1: REPRESENTATION SURVEY
# =============================================================================

def direction1_representations(geom):
    """Test all 5 representations across all 6 techniques."""
    print("\n" + "─" * 70)
    print("D1: Representation Survey — Which 2D layout makes stego visible?")
    print("─" * 70)

    carrier_type = 'natural_texture'
    tech_names = list(TECHNIQUES.keys())
    rep_names = list(REPRESENTATIONS.keys())

    results = {}       # (rep_name, tech_name) → sig_count
    all_details = {}   # (rep_name, tech_name) → detail list

    for rep_name, rep_fn in REPRESENTATIONS.items():
        # Clean baseline for this representation
        clean = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_carrier(carrier_type, trial, size=DATA_SIZE)
            grid = rep_fn(data)
            res = geom.compute_metrics(grid)
            for mn, mv in res.metrics.items():
                clean[mn].append(mv)

        n_metrics = len(clean)

        for tech_name in tech_names:
            embed_fn = TECHNIQUES[tech_name]
            stego_m = defaultdict(list)
            for trial in range(N_TRIALS):
                carrier = generate_carrier(carrier_type, trial, size=DATA_SIZE)
                stego_data = embed_fn(carrier, rate=1.0, seed=trial + 1000)
                grid = rep_fn(stego_data)
                res = geom.compute_metrics(grid)
                for mn, mv in res.metrics.items():
                    stego_m[mn].append(mv)

            sig, details = count_sig(clean, stego_m, n_metrics)
            results[(rep_name, tech_name)] = sig
            all_details[(rep_name, tech_name)] = details

        # Report
        print(f"\n  {rep_name} ({n_metrics} metrics):")
        for tech in tech_names:
            s = results[(rep_name, tech)]
            marker = "★" if s > 0 else "·"
            print(f"    {marker} {tech:<20} {s:>2}/{n_metrics}")

    # Summary table
    print(f"\n  Summary (sig metrics, representation × technique):")
    header = f"  {'Representation':<20}" + "".join(f" {t[:8]:>8}" for t in tech_names)
    print(header)
    for rep in rep_names:
        vals = " ".join(f" {results[(rep, t)]:>8}" for t in tech_names)
        total = sum(results[(rep, t)] for t in tech_names)
        print(f"  {rep:<20}{vals}  Σ={total}")

    return {'results': results, 'details': all_details}


# =============================================================================
# D2: CO-OCCURRENCE DEEP DIVE
# =============================================================================

def direction2_cooccurrence(geom):
    """Byte co-occurrence matrices on both carriers, detailed metric report."""
    print("\n" + "─" * 70)
    print("D2: Co-occurrence Deep Dive — Byte transitions as a 2D field")
    print("─" * 70)

    carriers = ['photo_blocks', 'natural_texture']
    tech_names = list(TECHNIQUES.keys())

    results = {}
    top_metrics = {}

    for carrier_type in carriers:
        # Clean baseline
        clean = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_carrier(carrier_type, trial, size=DATA_SIZE)
            cooc = repr_cooccurrence(data)
            res = geom.compute_metrics(cooc)
            for mn, mv in res.metrics.items():
                clean[mn].append(mv)

        n_metrics = len(clean)

        for tech_name in tech_names:
            embed_fn = TECHNIQUES[tech_name]
            stego_m = defaultdict(list)
            for trial in range(N_TRIALS):
                carrier = generate_carrier(carrier_type, trial, size=DATA_SIZE)
                stego_data = embed_fn(carrier, rate=1.0, seed=trial + 1000)
                cooc = repr_cooccurrence(stego_data)
                res = geom.compute_metrics(cooc)
                for mn, mv in res.metrics.items():
                    stego_m[mn].append(mv)

            sig, details = count_sig(clean, stego_m, n_metrics)
            results[(carrier_type, tech_name)] = sig

            # Top 3 by |d|
            sig_details = [(mn, d, p) for mn, d, p, is_sig in details if is_sig]
            sig_details.sort(key=lambda x: -abs(x[1]))
            top_metrics[(carrier_type, tech_name)] = sig_details[:3]

    # Report
    for carrier_type in carriers:
        print(f"\n  {carrier_type}:")
        for tech in tech_names:
            s = results[(carrier_type, tech)]
            if s > 0:
                tops = top_metrics[(carrier_type, tech)]
                detail_str = ", ".join(f"{mn}(d={d:+.1f})" for mn, d, _ in tops[:2])
                print(f"    ★ {tech:<20} {s:>2} sig  [{detail_str}]")
            else:
                print(f"    · {tech:<20}  0 sig")

    return {'results': results, 'top_metrics': top_metrics}


# =============================================================================
# D3: SPATIAL FINGERPRINTS
# =============================================================================

def direction3_fingerprints(geom):
    """Per-metric Cohen's d heatmap: which spatial metrics detect which techniques?"""
    print("\n" + "─" * 70)
    print("D3: Spatial Fingerprints — Cohen's d per technique × metric")
    print("─" * 70)

    carrier_type = 'natural_texture'
    tech_names = list(TECHNIQUES.keys())

    # Use two most interesting representations
    reps_to_test = {
        'Co-occurrence': repr_cooccurrence,
        'Byte Grid': repr_byte_grid,
    }

    fingerprints = {}   # rep_name → {tech_name → {metric_name → d}}

    for rep_name, rep_fn in reps_to_test.items():
        # Clean baseline
        clean = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_carrier(carrier_type, trial, size=DATA_SIZE)
            grid = rep_fn(data)
            res = geom.compute_metrics(grid)
            for mn, mv in res.metrics.items():
                clean[mn].append(mv)

        fingerprints[rep_name] = {}

        for tech_name in tech_names:
            embed_fn = TECHNIQUES[tech_name]
            stego_m = defaultdict(list)
            for trial in range(N_TRIALS):
                carrier = generate_carrier(carrier_type, trial, size=DATA_SIZE)
                stego_data = embed_fn(carrier, rate=1.0, seed=trial + 1000)
                grid = rep_fn(stego_data)
                res = geom.compute_metrics(grid)
                for mn, mv in res.metrics.items():
                    stego_m[mn].append(mv)

            metric_ds = {}
            for mn in clean:
                metric_ds[mn] = cohens_d(stego_m[mn], clean[mn])
            fingerprints[rep_name][tech_name] = metric_ds

    # Report
    for rep_name in reps_to_test:
        fp = fingerprints[rep_name]
        metric_names = sorted(next(iter(fp.values())).keys())
        print(f"\n  {rep_name} — Cohen's d per technique × metric:")
        header = f"    {'Metric':<28}" + "".join(f" {t[:8]:>8}" for t in tech_names)
        print(header)
        for mn in metric_names:
            vals = " ".join(
                f" {fp[t].get(mn, 0):>+8.2f}" for t in tech_names
            )
            any_big = any(abs(fp[t].get(mn, 0)) > 0.8 for t in tech_names)
            marker = "★" if any_big else " "
            print(f"  {marker} {mn:<28}{vals}")

    return {'fingerprints': fingerprints}


# =============================================================================
# D4: RATE SENSITIVITY
# =============================================================================

def direction4_rate(geom):
    """Rate sensitivity for co-occurrence representation."""
    print("\n" + "─" * 70)
    print("D4: Rate Sensitivity — Minimum detectable embedding rate")
    print("─" * 70)

    carrier_type = 'natural_texture'
    tech_names = list(TECHNIQUES.keys())
    rates = [0.05, 0.10, 0.25, 0.50, 1.00]

    # Clean baseline (co-occurrence)
    clean = defaultdict(list)
    for trial in range(N_TRIALS):
        data = generate_carrier(carrier_type, trial, size=DATA_SIZE)
        cooc = repr_cooccurrence(data)
        res = geom.compute_metrics(cooc)
        for mn, mv in res.metrics.items():
            clean[mn].append(mv)

    n_metrics = len(clean)
    results = {}   # (tech_name, rate) → sig_count

    for tech_name in tech_names:
        embed_fn = TECHNIQUES[tech_name]
        print(f"  {tech_name}...", end=" ", flush=True)
        for rate in rates:
            stego_m = defaultdict(list)
            for trial in range(N_TRIALS):
                carrier = generate_carrier(carrier_type, trial, size=DATA_SIZE)
                stego_data = embed_fn(carrier, rate=rate, seed=trial + 1000)
                cooc = repr_cooccurrence(stego_data)
                res = geom.compute_metrics(cooc)
                for mn, mv in res.metrics.items():
                    stego_m[mn].append(mv)

            sig, _ = count_sig(clean, stego_m, n_metrics)
            results[(tech_name, rate)] = sig
        vals = " ".join(f"{results[(tech_name, r)]:>3}" for r in rates)
        print(vals)

    # Report
    print(f"\n  Co-occurrence — sig metrics by technique × rate:")
    header = f"  {'Technique':<20}" + "".join(f" {r:>6.0%}" for r in rates)
    print(header)
    for tech in tech_names:
        vals = " ".join(f" {results[(tech, r)]:>6}" for r in rates)
        print(f"  {tech:<20}{vals}")

    return {'results': results, 'rates': rates}


# =============================================================================
# D5: MATRIX EMBED CHALLENGE — FULL ANALYZER
# =============================================================================

def direction5_matrix_challenge():
    """Run the full 24-geometry analyzer on byte grid and diff grid.
    Co-occurrence (256×256 = 65536 elements) is too large for O(N²) geometries,
    so we only test it via SpatialField (done in D1–D4).
    Target: crack Matrix Embed (invisible in 1D) and boost LSBMR."""
    print("\n" + "─" * 70)
    print("D5: Matrix Embed Challenge — Full analyzer on 2D representations")
    print("─" * 70)

    analyzer = GeometryAnalyzer().add_all_geometries()
    carrier_type = 'natural_texture'

    # Get metric names from a dummy run
    dummy = np.random.RandomState(0).randint(0, 256, DATA_SIZE, dtype=np.uint8)
    dummy_res = analyzer.analyze(dummy)
    all_metric_names = []
    for r in dummy_res.results:
        for mn in r.metrics:
            all_metric_names.append(f"{r.geometry_name}:{mn}")
    all_metric_names = sorted(set(all_metric_names))
    print(f"  Full analyzer: {len(all_metric_names)} metrics")

    # Only small representations (4096 elements) — co-occurrence is 65536
    # and causes O(N²) blowup in PersistentHomology / Wasserstein
    reps_to_test = {
        'Byte Grid':     repr_byte_grid,
        'Diff Grid':     repr_byte_diff_grid,
    }

    target_techs = ['Matrix Embed', 'LSBMR', 'PVD']
    results = {}
    top_metrics_out = {}

    for rep_name, rep_fn in reps_to_test.items():
        print(f"\n  {rep_name}:")

        # Clean baseline
        clean = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_carrier(carrier_type, trial, size=DATA_SIZE)
            grid = rep_fn(data)
            res = analyzer.analyze(grid)
            for r in res.results:
                for mn, mv in r.metrics.items():
                    clean[f"{r.geometry_name}:{mn}"].append(mv)

        n_metrics = len(clean)

        for tech_name in target_techs:
            embed_fn = TECHNIQUES[tech_name]
            stego_m = defaultdict(list)
            for trial in range(N_TRIALS):
                carrier = generate_carrier(carrier_type, trial, size=DATA_SIZE)
                stego_data = embed_fn(carrier, rate=1.0, seed=trial + 1000)
                grid = rep_fn(stego_data)
                res = analyzer.analyze(grid)
                for r in res.results:
                    for mn, mv in r.metrics.items():
                        stego_m[f"{r.geometry_name}:{mn}"].append(mv)

            sig, details = count_sig(clean, stego_m, n_metrics)
            results[(rep_name, tech_name)] = sig

            sig_details = [(mn, d, p) for mn, d, p, is_sig in details if is_sig]
            sig_details.sort(key=lambda x: -abs(x[1]))
            top_metrics_out[(rep_name, tech_name)] = sig_details[:5]

            marker = "★★★" if sig > 5 else ("★" if sig > 0 else "·")
            print(f"    {marker} {tech_name:<20} {sig:>3}/{n_metrics}")
            for mn, d, p in sig_details[:3]:
                print(f"        {mn:<45} d={d:+.2f}")

    return {'results': results, 'top_metrics': top_metrics_out}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("2D BIT-MATRIX STEGANALYSIS — DEEP INVESTIGATION")
    print("=" * 70)
    print(f"\nParameters: N_TRIALS={N_TRIALS}, DATA_SIZE={DATA_SIZE}")
    print(f"6 techniques, 4 representations, 5 directions\n")

    geom = SpatialFieldGeometry()

    d1 = direction1_representations(geom)
    d2 = direction2_cooccurrence(geom)
    d3 = direction3_fingerprints(geom)
    d4 = direction4_rate(geom)
    d5 = direction5_matrix_challenge()

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    tech_names = list(TECHNIQUES.keys())
    rep_names = list(REPRESENTATIONS.keys())

    print(f"\n  D1 — Representation Survey (SpatialField only, texture, 100%):")
    header = f"  {'Representation':<20}" + "".join(f" {t[:8]:>8}" for t in tech_names) + "    Σ"
    print(header)
    for rep in rep_names:
        vals = " ".join(f" {d1['results'][(rep, t)]:>8}" for t in tech_names)
        total = sum(d1['results'][(rep, t)] for t in tech_names)
        print(f"  {rep:<20}{vals} {total:>4}")

    print(f"\n  D2 — Co-occurrence (both carriers, 100%):")
    for carrier in ['photo_blocks', 'natural_texture']:
        vals = " ".join(f"{d2['results'][(carrier, t)]:>3}" for t in tech_names)
        print(f"    {carrier:<20} {vals}")

    print(f"\n  D4 — Rate Sensitivity (co-occurrence, texture):")
    for tech in tech_names:
        min_rate = "never"
        for r in d4['rates']:
            if d4['results'][(tech, r)] > 0:
                min_rate = f"{r:.0%}"
                break
        print(f"    {tech:<20} first detectable at {min_rate}")

    print(f"\n  D5 — Matrix Embed Challenge (full analyzer, texture, 100%):")
    d5_reps = ['Byte Grid', 'Diff Grid']
    for rep in d5_reps:
        for tech in ['Matrix Embed', 'LSBMR', 'PVD']:
            key = (rep, tech)
            if key in d5['results']:
                print(f"    {rep:<20} {tech:<20} {d5['results'][key]:>3} sig")

    print(f"\n[Investigation complete]")
    return d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION — 6-panel dark-theme figure (22×14)
# =============================================================================

def make_figure(d1, d2, d3, d4, d5):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'
    tech_names = list(TECHNIQUES.keys())
    rep_names = list(REPRESENTATIONS.keys())

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ── Panel A: D1 — Representation × Technique heatmap ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)

    heatmap = np.array([[d1['results'][(r, t)] for t in tech_names]
                        for r in rep_names])

    im1 = ax1.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0)
    ax1.set_xticks(range(len(tech_names)))
    ax1.set_xticklabels([t.replace(' ', '\n') for t in tech_names],
                        fontsize=7, color=FG)
    ax1.set_yticks(range(len(rep_names)))
    ax1.set_yticklabels(rep_names, fontsize=8, color=FG)

    for i in range(len(rep_names)):
        for j in range(len(tech_names)):
            val = int(heatmap[i, j])
            color = 'white' if val > heatmap.max() * 0.6 else FG
            ax1.text(j, i, str(val), ha='center', va='center',
                     fontsize=10, fontweight='bold', color=color)

    ax1.set_title('D1: Representation Survey\n(SpatialField, texture, 100%)',
                  fontsize=10, fontweight='bold', color=FG)

    # ── Panel B: D2 — Co-occurrence by carrier ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)

    x = np.arange(len(tech_names))
    bar_w = 0.35
    carriers = ['photo_blocks', 'natural_texture']
    carrier_colors = ['#64B5F6', '#FF8A65']

    for ci, carrier in enumerate(carriers):
        vals = [d2['results'].get((carrier, t), 0) for t in tech_names]
        offset = (ci - 0.5) * bar_w
        ax2.bar(x + offset, vals, bar_w, color=carrier_colors[ci],
                alpha=0.85, edgecolor='#333',
                label=carrier.replace('_', ' '))

    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace(' ', '\n') for t in tech_names],
                        fontsize=7, color=FG)
    ax2.set_ylabel('Significant metrics', color=FG, fontsize=9)
    ax2.legend(fontsize=7, loc='upper right', facecolor='#333',
               edgecolor='#555', labelcolor=FG)
    ax2.set_title('D2: Co-occurrence Detection\n(both carriers, 100%)',
                  fontsize=10, fontweight='bold', color=FG)

    # ── Panel C: D3 — Spatial Fingerprints (co-occurrence) ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    fp = d3['fingerprints'].get('Co-occurrence', {})
    if fp:
        metric_names_sorted = sorted(next(iter(fp.values())).keys())
        fp_matrix = np.array([[fp[t].get(mn, 0) for t in tech_names]
                              for mn in metric_names_sorted])

        vmax = max(abs(fp_matrix.min()), abs(fp_matrix.max()), 0.1)
        im3 = ax3.imshow(fp_matrix, cmap='RdBu_r', aspect='auto',
                         vmin=-vmax, vmax=vmax)
        ax3.set_xticks(range(len(tech_names)))
        ax3.set_xticklabels([t.replace(' ', '\n') for t in tech_names],
                            fontsize=7, color=FG)
        ax3.set_yticks(range(len(metric_names_sorted)))
        ax3.set_yticklabels([mn[:22] for mn in metric_names_sorted],
                            fontsize=6, color=FG)
        cb = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cb.set_label("Cohen's d", color=FG, fontsize=8)
        cb.ax.tick_params(colors=FG, labelsize=7)

    ax3.set_title("D3: Spatial Fingerprints\n(Co-occurrence, Cohen's d)",
                  fontsize=10, fontweight='bold', color=FG)

    # ── Panel D: D4 — Rate Sensitivity curves ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    rates = d4['rates']
    for tech in tech_names:
        vals = [d4['results'].get((tech, r), 0) for r in rates]
        ax4.plot(rates, vals, 'o-', label=tech, color=TECH_COLORS[tech],
                 linewidth=2, markersize=6, alpha=0.85)

    ax4.axhline(y=1, color='#666', linestyle='--', linewidth=1)
    ax4.set_xlabel('Embedding rate', color=FG, fontsize=9)
    ax4.set_ylabel('Significant metrics', color=FG, fontsize=9)
    ax4.set_xticks(rates)
    ax4.set_xticklabels([f'{r:.0%}' for r in rates], color=FG)
    ax4.legend(fontsize=6, loc='upper left', facecolor='#333',
               edgecolor='#555', labelcolor=FG)
    ax4.set_title('D4: Rate Sensitivity\n(Co-occurrence, texture)',
                  fontsize=10, fontweight='bold', color=FG)

    # ── Panel E: D5 — Matrix Embed Challenge ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    d5_techs = ['Matrix Embed', 'LSBMR', 'PVD']
    d5_reps = ['Byte Grid', 'Diff Grid']
    d5_matrix = np.array([
        [d5['results'].get((rep, tech), 0) for tech in d5_techs]
        for rep in d5_reps
    ])

    im5 = ax5.imshow(d5_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    ax5.set_xticks(range(len(d5_techs)))
    ax5.set_xticklabels(d5_techs, fontsize=8, color=FG)
    ax5.set_yticks(range(len(d5_reps)))
    ax5.set_yticklabels(d5_reps, fontsize=8, color=FG)

    for i in range(len(d5_reps)):
        for j in range(len(d5_techs)):
            val = int(d5_matrix[i, j])
            color = 'white' if val > d5_matrix.max() * 0.4 else FG
            ax5.text(j, i, str(val), ha='center', va='center',
                     fontsize=13, fontweight='bold', color=color)

    ax5.set_title('D5: Full Analyzer on 2D Reps\n(texture, 100%)',
                  fontsize=10, fontweight='bold', color=FG)

    # ── Panel F: Summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    _dark_ax(ax6)
    ax6.axis('off')

    # Best representation from D1
    rep_totals = {r: sum(d1['results'][(r, t)] for t in tech_names) for r in rep_names}
    best_rep = max(rep_totals, key=rep_totals.get)

    # Matrix Embed status
    me_detected = any(d5['results'].get((rep, 'Matrix Embed'), 0) > 0
                      for rep in ['Byte Grid', 'Diff Grid'])

    # Best rate for co-occurrence
    best_rates = {}
    for tech in tech_names:
        best_rates[tech] = 'never'
        for r in rates:
            if d4['results'].get((tech, r), 0) > 0:
                best_rates[tech] = f'{r:.0%}'
                break

    summary_lines = [
        "Key Findings:",
        "",
        f"Best 2D repr: {best_rep} ({rep_totals[best_rep]} sig)",
        f"Matrix Embed: {'CRACKED' if me_detected else 'still invisible'}",
        "",
        "Min detectable rate (co-occurrence):",
    ] + [
        f"  {tech[:16]:<18} {best_rates[tech]}"
        for tech in tech_names
    ] + [
        "",
        "Binary bit grids kill metric variance.",
        "Co-occurrence preserves byte-pair",
        "transition statistics as a smooth 2D",
        "field amenable to spatial analysis.",
    ]

    for i, line in enumerate(summary_lines):
        weight = 'bold' if i == 0 else 'normal'
        size = 11 if i == 0 else 9
        ax6.text(0.05, 0.97 - i * 0.058, line, transform=ax6.transAxes,
                 fontsize=size, fontweight=weight, color=FG,
                 verticalalignment='top', fontfamily='monospace')

    fig.suptitle('2D Bit-Matrix Steganalysis — Deep Investigation',
                 fontsize=14, fontweight='bold', color=FG, y=0.98)

    fig.savefig('figures/stego_bitmatrix.png', dpi=180, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print("  → figures/stego_bitmatrix.png")
    plt.close(fig)


if __name__ == "__main__":
    d1, d2, d3, d4, d5 = main()
    make_figure(d1, d2, d3, d4, d5)
