#!/usr/bin/env python3
"""
Investigation: 2D Site Percolation Phase Transition via SpatialFieldGeometry

Site percolation on a square lattice has a critical threshold p_c ≈ 0.5927.
Below p_c: isolated clusters. Above p_c: a spanning cluster emerges.
At p_c: fractal structure, power-law cluster sizes, scale-free geometry.

Can SpatialFieldGeometry detect the phase transition and distinguish
different occupation probabilities?

Methodology: N_TRIALS=25, shuffled baselines, Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats, ndimage
from exotic_geometry_framework import SpatialFieldGeometry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
ALPHA = 0.05
GRID_SIZE = 128

METRIC_NAMES = [
    'tension_mean', 'tension_std', 'curvature_mean', 'curvature_std',
    'anisotropy_mean', 'criticality_saddle_frac', 'criticality_extrema_frac',
    'n_basins', 'basin_size_entropy', 'basin_depth_cv',
    'coherence_score', 'multiscale_coherence_1',
    'multiscale_coherence_2', 'multiscale_coherence_4',
    'multiscale_coherence_8',
]

# Occupation probabilities spanning the transition
P_C = 0.5927
PROBABILITIES = [0.20, 0.35, 0.50, 0.55, P_C, 0.65, 0.75, 0.90]


# =============================================================================
# PERCOLATION HELPERS
# =============================================================================

def site_percolation(H, W, p, rng):
    """Generate a site percolation configuration. Returns float64 field."""
    return (rng.random((H, W)) < p).astype(np.float64)


def cluster_stats(field):
    """Compute cluster statistics for a binary percolation field."""
    binary = (field > 0.5).astype(int)
    labeled, n_clusters = ndimage.label(binary)
    if n_clusters == 0:
        return 0, 0, 0, False
    sizes = ndimage.sum(binary, labeled, range(1, n_clusters + 1))
    largest = np.max(sizes) if len(sizes) > 0 else 0
    spans = _check_spanning(labeled, field.shape)
    return n_clusters, largest, np.mean(sizes), spans


def _check_spanning(labeled, shape):
    """Check if any cluster spans top-to-bottom or left-to-right."""
    H, W = shape
    top = set(labeled[0, :]) - {0}
    bottom = set(labeled[H-1, :]) - {0}
    left = set(labeled[:, 0]) - {0}
    right = set(labeled[:, W-1]) - {0}
    return bool(top & bottom) or bool(left & right)


# =============================================================================
# HELPERS
# =============================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    ps = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def shuffle_field(field, rng):
    flat = field.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(field.shape)


def collect_metrics(geom, fields):
    out = {m: [] for m in METRIC_NAMES}
    for f in fields:
        res = geom.compute_metrics(f)
        for m in METRIC_NAMES:
            v = res.metrics.get(m, float('nan'))
            if not np.isnan(v):
                out[m].append(v)
    return out


# =============================================================================
# INVESTIGATION
# =============================================================================

def run_investigation():
    geom = SpatialFieldGeometry()

    print("=" * 78)
    print("2D SITE PERCOLATION PHASE TRANSITION")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, p_c≈{P_C}, N={N_TRIALS}")
    print("=" * 78)

    prob_data = {}
    shuffled_data = {}
    example_fields = {}
    cluster_info = {}  # {p: (n_clusters, largest, mean_size, spanning_frac)}

    for p in PROBABILITIES:
        label = f"p={p:.4f}" if p == P_C else f"p={p:.2f}"
        print(f"  {label:12s}...", end=" ", flush=True)
        fields, shuf_fields = [], []
        nc_list, largest_list, span_list = [], [], []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            field = site_percolation(GRID_SIZE, GRID_SIZE, p, rng)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))

            nc, largest, mean_sz, spans = cluster_stats(field)
            nc_list.append(nc)
            largest_list.append(largest)
            span_list.append(int(spans))

            if trial == 0:
                example_fields[p] = field.copy()

        prob_data[p] = collect_metrics(geom, fields)
        shuffled_data[p] = collect_metrics(geom, shuf_fields)
        cluster_info[p] = (np.mean(nc_list), np.mean(largest_list),
                           np.mean(span_list))
        print(f"clusters={np.mean(nc_list):.0f}, "
              f"largest={np.mean(largest_list):.0f}, "
              f"spanning={np.mean(span_list):.0%}")

    probs = PROBABILITIES

    # Each vs shuffled
    bonf_s = ALPHA / (len(METRIC_NAMES) * len(probs))
    print(f"\n{'─' * 78}")
    print(f"  Each p vs SHUFFLED baseline (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for p in probs:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(prob_data[p][m])
            b = np.array(shuffled_data[p][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, pv = stats.ttest_ind(a, b, equal_var=False)
            if pv < bonf_s:
                sig += 1
                findings.append((m, d, pv))
        label = f"p={p:.4f}" if p == P_C else f"p={p:.2f}"
        print(f"\n  {label}: {sig} significant metrics")
        findings.sort(key=lambda x: -abs(x[1]))
        for m, d, pv in findings[:5]:
            print(f"    {m:30s}  d={d:+8.2f}  p={pv:.2e}")

    # Pairwise: adjacent probabilities
    print(f"\n{'─' * 78}")
    print(f"  ADJACENT pairwise comparisons")
    print(f"{'─' * 78}")

    bonf_a = ALPHA / (len(METRIC_NAMES) * (len(probs) - 1))
    pair_results = []
    for i in range(len(probs) - 1):
        p1, p2 = probs[i], probs[i+1]
        sig = 0
        best_d, best_m = 0, ""
        for m in METRIC_NAMES:
            a = np.array(prob_data[p1][m])
            b = np.array(prob_data[p2][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, pv = stats.ttest_ind(a, b, equal_var=False)
            if pv < bonf_a:
                sig += 1
            if abs(d) > abs(best_d):
                best_d, best_m = d, m
        l1 = f"p={p1:.4f}" if p1 == P_C else f"p={p1:.2f}"
        l2 = f"p={p2:.4f}" if p2 == P_C else f"p={p2:.2f}"
        pair_results.append((p1, p2, sig, best_m, best_d))
        print(f"  {l1:12s} vs {l2:12s}: {sig:2d} sig  "
              f"best: {best_m:25s} d={best_d:+8.2f}")

    # All pairwise
    n_pairs = len(probs) * (len(probs) - 1) // 2
    bonf_p = ALPHA / (len(METRIC_NAMES) * n_pairs)
    print(f"\n{'─' * 78}")
    print(f"  ALL pairwise (Bonferroni α={bonf_p:.2e})")
    print(f"{'─' * 78}")

    all_pairs = []
    for i, p1 in enumerate(probs):
        for p2 in probs[i+1:]:
            sig = 0
            best_d, best_m = 0, ""
            for m in METRIC_NAMES:
                a = np.array(prob_data[p1][m])
                b = np.array(prob_data[p2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, pv = stats.ttest_ind(a, b, equal_var=False)
                if pv < bonf_p:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            all_pairs.append((p1, p2, sig, best_m, best_d))
            l1 = f"{p1:.4f}" if p1 == P_C else f"{p1:.2f}"
            l2 = f"{p2:.4f}" if p2 == P_C else f"{p2:.2f}"
            print(f"  p={l1:6s} vs p={l2:6s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")

    return prob_data, shuffled_data, example_fields, cluster_info, all_pairs


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(prob_data, example_fields, cluster_info, all_pairs):
    print("\nGenerating figure...", flush=True)

    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': '#111111',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    probs = PROBABILITIES
    n = len(probs)

    fig = plt.figure(figsize=(20, 14), facecolor='black')
    gs = gridspec.GridSpec(4, n, figure=fig, height_ratios=[1.2, 0.8, 1.0, 1.0],
                           hspace=0.5, wspace=0.35)

    # Row 0: Example percolation fields
    for i, p in enumerate(probs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[p], cmap='inferno', interpolation='nearest',
                  vmin=0, vmax=1)
        label = f"p={p:.4f}" if p == P_C else f"p={p:.2f}"
        color = '#00FF00' if p == P_C else 'white'
        ax.set_title(label, fontsize=9, fontweight='bold', color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        if p == P_C:
            for spine in ax.spines.values():
                spine.set_edgecolor('#00FF00')
                spine.set_linewidth(2)

    # Row 1: Cluster statistics vs p
    ax_nc = fig.add_subplot(gs[1, :3])
    nc_vals = [cluster_info[p][0] for p in probs]
    ax_nc.plot(probs, nc_vals, 'o-', color='#FF9800', markersize=6, linewidth=2)
    ax_nc.axvline(P_C, color='#00FF00', linestyle='--', alpha=0.7, linewidth=1)
    ax_nc.set_xlabel('Occupation probability p', fontsize=9)
    ax_nc.set_ylabel('# Clusters', fontsize=9)
    ax_nc.set_title('Cluster count vs p', fontsize=10, fontweight='bold')
    ax_nc.tick_params(labelsize=8)

    ax_span = fig.add_subplot(gs[1, 3:6])
    span_vals = [cluster_info[p][2] for p in probs]
    ax_span.plot(probs, span_vals, 's-', color='#E91E63', markersize=6, linewidth=2)
    ax_span.axvline(P_C, color='#00FF00', linestyle='--', alpha=0.7, linewidth=1)
    ax_span.set_xlabel('Occupation probability p', fontsize=9)
    ax_span.set_ylabel('Spanning fraction', fontsize=9)
    ax_span.set_title('Spanning cluster probability vs p', fontsize=10, fontweight='bold')
    ax_span.tick_params(labelsize=8)

    ax_lg = fig.add_subplot(gs[1, 6:])
    lg_vals = [cluster_info[p][1] for p in probs]
    ax_lg.plot(probs, lg_vals, 'D-', color='#4CAF50', markersize=6, linewidth=2)
    ax_lg.axvline(P_C, color='#00FF00', linestyle='--', alpha=0.7, linewidth=1)
    ax_lg.set_xlabel('Occupation probability p', fontsize=9)
    ax_lg.set_ylabel('Largest cluster', fontsize=9)
    ax_lg.set_title('Largest cluster size vs p', fontsize=10, fontweight='bold')
    ax_lg.tick_params(labelsize=8)

    # Row 2: Key SpatialFieldGeometry metrics vs p
    plot_metrics = ['coherence_score', 'n_basins', 'curvature_mean', 'anisotropy_mean',
                    'tension_mean', 'basin_size_entropy', 'multiscale_coherence_4',
                    'criticality_extrema_frac']
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3',
              '#9C27B0', '#00BCD4', '#FFEB3B', '#FF5722']

    for j in range(min(n, len(plot_metrics))):
        metric = plot_metrics[j]
        ax = fig.add_subplot(gs[2, j])
        means = [np.mean(prob_data[p][metric]) for p in probs]
        stds = [np.std(prob_data[p][metric]) for p in probs]
        ax.errorbar(probs, means, yerr=stds, fmt='o-', color=colors[j],
                    markersize=4, capsize=2, linewidth=1.5)
        ax.axvline(P_C, color='#00FF00', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(metric.replace('_', ' '), fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel('Value', fontsize=7)

    # Row 3: Pairwise distinguishability heatmap + summary
    ax_mat = fig.add_subplot(gs[3, :4])
    mat = np.zeros((n, n))
    for p1, p2, sig, _, _ in all_pairs:
        i1, i2 = probs.index(p1), probs.index(p2)
        mat[i1, i2] = sig
        mat[i2, i1] = sig
    im = ax_mat.imshow(mat, cmap='magma', interpolation='nearest', vmin=0)
    labels = [f"{p:.4f}" if p == P_C else f"{p:.2f}" for p in probs]
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax_mat.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax_mat.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color='white' if mat[i,j] > 8 else '#aaaaaa')
    ax_mat.set_title('Pairwise significant metrics', fontsize=10, fontweight='bold')
    cb = plt.colorbar(im, ax=ax_mat, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    # Summary text
    ax_txt = fig.add_subplot(gs[3, 4:])
    ax_txt.axis('off')
    pairs_sorted = sorted(all_pairs, key=lambda x: -x[2])
    lines = ['Pairwise results (sorted by sig count):\n']
    for p1, p2, sig, bm, bd in pairs_sorted[:15]:
        l1 = f"{p1:.4f}" if p1 == P_C else f"{p1:.2f}"
        l2 = f"{p2:.4f}" if p2 == P_C else f"{p2:.2f}"
        lines.append(f'p={l1} vs p={l2}: {sig:2d} sig  {bm}')
    ax_txt.text(0.05, 0.95, '\n'.join(lines), transform=ax_txt.transAxes,
               fontsize=7, fontfamily='monospace', va='top', color='#cccccc')

    fig.suptitle(f'Site Percolation Phase Transition (p_c ≈ {P_C})',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'percolation.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved percolation.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    prob_data, shuffled_data, example_fields, cluster_info, all_pairs = run_investigation()
    make_figure(prob_data, example_fields, cluster_info, all_pairs)

    probs = PROBABILITIES
    n_pairs = len(probs) * (len(probs) - 1) // 2
    pairs_ok = sum(1 for _, _, s, _, _ in all_pairs if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Probabilities tested: {len(probs)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
