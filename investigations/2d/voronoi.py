#!/usr/bin/env python3
"""
Investigation: Voronoi Tessellation Fingerprinting via SpatialFieldGeometry

Voronoi tessellations generated from different point processes produce
different spatial structures:

- Poisson (uniform random): standard Voronoi, baseline
- Clustered (Thomas process): tight clusters → highly variable cell sizes
- Regular (jittered grid): nearly hexagonal cells, low variance
- Halton (quasi-random): very uniform coverage, anti-clustering
- Poisson disk: minimum-distance constraint, blue noise
- Clustered + regular mix: bimodal cell sizes

The tessellation is encoded as a distance-to-boundary field: each pixel gets
the distance to the nearest Voronoi edge. This creates a smooth 2D field
that preserves the spatial structure of the tessellation.

Methodology: N_TRIALS=25, shuffled baselines, Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from scipy.spatial import Voronoi
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
ALPHA = 0.05
GRID_SIZE = 128
N_POINTS = 100  # approximate number of Voronoi cells

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn


# =============================================================================
# POINT PROCESS GENERATORS
# =============================================================================

def poisson_points(n, H, W, rng):
    """Uniform random (Poisson process)."""
    return rng.random((n, 2)) * np.array([W, H])


def clustered_points(n, H, W, rng, n_clusters=10, cluster_std=8.0):
    """Thomas cluster process: clusters of points around parent locations."""
    parents = rng.random((n_clusters, 2)) * np.array([W, H])
    points_per = max(1, n // n_clusters)
    pts = []
    for px, py in parents:
        offsets = rng.normal(0, cluster_std, (points_per, 2))
        cluster_pts = np.column_stack([px + offsets[:, 0], py + offsets[:, 1]])
        pts.append(cluster_pts)
    pts = np.vstack(pts)[:n]
    pts[:, 0] = np.clip(pts[:, 0], 0, W)
    pts[:, 1] = np.clip(pts[:, 1], 0, H)
    return pts


def regular_points(n, H, W, rng, jitter=2.0):
    """Jittered grid: nearly regular with small perturbation."""
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(W / (2 * side), W * (1 - 1 / (2 * side)), side)
    ys = np.linspace(H / (2 * side), H * (1 - 1 / (2 * side)), side)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])[:n]
    pts += rng.normal(0, jitter, pts.shape)
    pts[:, 0] = np.clip(pts[:, 0], 0, W)
    pts[:, 1] = np.clip(pts[:, 1], 0, H)
    return pts


def halton_points(n, H, W, rng):
    """Halton quasi-random sequence (bases 2, 3)."""
    def halton_seq(index, base):
        result = 0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result

    # Start from random offset to get different sequences per trial
    offset = rng.integers(0, 1000)
    pts = np.array([[halton_seq(i + offset, 2) * W,
                     halton_seq(i + offset, 3) * H]
                    for i in range(1, n + 1)])
    return pts


def poisson_disk_points(n, H, W, rng, min_dist=None):
    """Poisson disk sampling: minimum distance constraint (blue noise)."""
    if min_dist is None:
        min_dist = np.sqrt(H * W / n) * 0.7

    pts = [rng.random(2) * np.array([W, H])]
    active = [0]
    k = 30  # rejection samples per point

    while len(pts) < n and active:
        idx = rng.integers(len(active))
        center = pts[active[idx]]
        found = False

        for _ in range(k):
            angle = rng.random() * 2 * np.pi
            r = min_dist + rng.random() * min_dist
            candidate = center + np.array([r * np.cos(angle), r * np.sin(angle)])

            if (0 <= candidate[0] <= W and 0 <= candidate[1] <= H):
                ok = True
                for p in pts:
                    if np.sqrt((p[0] - candidate[0])**2 + (p[1] - candidate[1])**2) < min_dist:
                        ok = False
                        break
                if ok:
                    pts.append(candidate)
                    active.append(len(pts) - 1)
                    found = True
                    break

        if not found:
            active.pop(idx)

    return np.array(pts[:n])


POINT_PROCESSES = {
    'poisson':     poisson_points,
    'clustered':   clustered_points,
    'regular':     regular_points,
    'halton':      halton_points,
    'poisson_disk': poisson_disk_points,
}


# =============================================================================
# VORONOI → DISTANCE FIELD
# =============================================================================

def voronoi_distance_field(points, H, W):
    """
    Create distance-to-nearest-generator field.
    Each pixel value = distance to its nearest Voronoi generator.
    This naturally encodes the cell structure.
    """
    y, x = np.mgrid[:H, :W]
    coords = np.column_stack([x.ravel(), y.ravel()])  # (H*W, 2)

    # Nearest-neighbor distances
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(coords)
    field = dists.reshape(H, W)

    # Normalize
    if field.max() > field.min():
        field = (field - field.min()) / (field.max() - field.min())

    return field


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


def collect_metrics(analyzer, fields):
    out = {m: [] for m in METRIC_NAMES}
    for f in fields:
        res = analyzer.analyze(f)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


# =============================================================================
# INVESTIGATION
# =============================================================================

def run_investigation():
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    print("=" * 78)
    print("VORONOI TESSELLATION FINGERPRINTING")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, ~{N_POINTS} generators, N={N_TRIALS}")
    print("=" * 78)

    proc_data = {}
    shuffled_data = {}
    example_fields = {}
    example_points = {}

    for name, gen_func in POINT_PROCESSES.items():
        print(f"  {name:14s}...", end=" ", flush=True)
        fields, shuf_fields = [], []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            pts = gen_func(N_POINTS, GRID_SIZE, GRID_SIZE, rng)
            field = voronoi_distance_field(pts, GRID_SIZE, GRID_SIZE)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))

            if trial == 0:
                example_fields[name] = field.copy()
                example_points[name] = pts.copy()

            if (trial + 1) % 5 == 0:
                print(f"{trial+1}", end=" ", flush=True)

        proc_data[name] = collect_metrics(analyzer, fields)
        shuffled_data[name] = collect_metrics(analyzer, shuf_fields)
        print()

    names = list(POINT_PROCESSES.keys())

    # Each vs shuffled
    bonf_s = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  Each process vs SHUFFLED (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for name in names:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(proc_data[name][m])
            b = np.array(shuffled_data[name][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf_s:
                sig += 1
                findings.append((m, d, p))
        print(f"\n  {name}: {sig} significant metrics")
        findings.sort(key=lambda x: -abs(x[1]))
        for m, d, p in findings[:5]:
            print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")

    # Pairwise
    n_pairs = len(names) * (len(names) - 1) // 2
    bonf_p = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  PAIRWISE comparisons (Bonferroni α={bonf_p:.2e})")
    print(f"{'─' * 78}")

    pair_results = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            sig = 0
            best_d, best_m = 0, ""
            for m in METRIC_NAMES:
                a = np.array(proc_data[n1][m])
                b = np.array(proc_data[n2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_p:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            pair_results.append((n1, n2, sig, best_m, best_d))
            print(f"  {n1:14s} vs {n2:14s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")

    return proc_data, shuffled_data, example_fields, example_points, pair_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(proc_data, example_fields, example_points, pair_results):
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

    names = list(POINT_PROCESSES.keys())
    n = len(names)
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

    fig = plt.figure(figsize=(16, 24), facecolor='black')
    gs = gridspec.GridSpec(4, n, figure=fig, height_ratios=[1.3, 1.0, 1.0, 1.0],
                           hspace=0.45, wspace=0.3)

    # Row 0: Distance fields with overlaid points
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[name], cmap='viridis', interpolation='bilinear',
                  extent=[0, GRID_SIZE, GRID_SIZE, 0])
        pts = example_points[name]
        ax.scatter(pts[:, 0], pts[:, 1], c='white', s=3, alpha=0.7)
        ax.set_title(name.replace('_', ' '), fontsize=9, fontweight='bold',
                     color=colors[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: Key metrics
    compare_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins',
                       'Surface:gaussian_curvature_mean', 'PersistentHomology2D:persistence_entropy',
                       'SpectralPower:spectral_slope']

    for j in range(min(n, len(compare_metrics))):
        metric = compare_metrics[j]
        ax = fig.add_subplot(gs[1, j])
        means = [np.mean(proc_data[nm][metric]) for nm in names]
        stds = [np.std(proc_data[nm][metric]) for nm in names]
        ax.bar(range(n), means, yerr=stds, capsize=3,
               color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(n))
        ax.set_xticklabels([nm.replace('_', '\n') for nm in names],
                           fontsize=6)
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    # Row 2: Pairwise matrix
    ax_mat = fig.add_subplot(gs[2, :])
    mat = np.zeros((n, n))
    for n1, n2, sig, _, _ in pair_results:
        i1, i2 = names.index(n1), names.index(n2)
        mat[i1, i2] = sig
        mat[i2, i1] = sig
    im = ax_mat.imshow(mat, cmap='magma', interpolation='nearest', vmin=0)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(names, fontsize=7, rotation=35, ha='right')
    ax_mat.set_yticklabels(names, fontsize=5)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax_mat.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white' if mat[i,j] > 8 else '#aaaaaa')
    ax_mat.set_title('Pairwise significant metrics', fontsize=10, fontweight='bold')
    cb = plt.colorbar(im, ax=ax_mat, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    # Row 3: Multiscale coherence profiles
    ax_ms = fig.add_subplot(gs[3, :3])
    scales = [1, 2, 4, 8]
    for i, name in enumerate(names):
        means = [np.mean(proc_data[name][f'SpatialField:multiscale_coherence_{s}']) for s in scales]
        ax_ms.plot(scales, means, 'o-', color=colors[i], label=name.replace('_', ' '),
                   markersize=5, linewidth=1.5)
    ax_ms.set_xlabel('Scale', fontsize=9)
    ax_ms.set_ylabel('Coherence', fontsize=9)
    ax_ms.set_title('Multiscale coherence profiles', fontsize=10, fontweight='bold')
    ax_ms.legend(fontsize=7)
    ax_ms.tick_params(labelsize=8)

    ax_tens = fig.add_subplot(gs[3, 3:])
    for i, name in enumerate(names):
        vals = proc_data[name].get('SpatialField:tension_mean', [])
        if vals:
            ax_tens.hist(vals, bins=12, alpha=0.5, color=colors[i],
                        label=name.replace('_', ' '), density=True)
    ax_tens.set_xlabel('Tension mean', fontsize=9)
    ax_tens.set_ylabel('Density', fontsize=9)
    ax_tens.set_title('Tension distribution', fontsize=10, fontweight='bold')
    ax_tens.legend(fontsize=7)
    ax_tens.tick_params(labelsize=8)

    fig.suptitle('Voronoi Tessellation Fingerprinting',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'voronoi.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved voronoi.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    proc_data, shuffled_data, example_fields, example_points, pair_results = run_investigation()
    make_figure(proc_data, example_fields, example_points, pair_results)

    names = list(POINT_PROCESSES.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    pairs_ok = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Point processes tested: {len(names)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
