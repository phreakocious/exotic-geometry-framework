#!/usr/bin/env python3
"""
Investigation: Growth Model Fingerprinting via SpatialFieldGeometry

Three fundamentally different 2D growth processes:
- DLA (Diffusion-Limited Aggregation): fractal, branching, diffusion-driven
- Eden model: compact, roughly circular, surface-growth
- Random clusters: uniform random occupation (no growth dynamics)

All produce clusters of similar total mass but with radically different
spatial structure. Can SpatialFieldGeometry distinguish them?

Methodology: N_TRIALS=25, shuffled baselines, Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from exotic_geometry_framework import SpatialFieldGeometry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
ALPHA = 0.05
GRID_SIZE = 128
TARGET_MASS = 3000  # ~18% fill of 128x128

METRIC_NAMES = [
    'tension_mean', 'tension_std', 'curvature_mean', 'curvature_std',
    'anisotropy_mean', 'criticality_saddle_frac', 'criticality_extrema_frac',
    'n_basins', 'basin_size_entropy', 'basin_depth_cv',
    'coherence_score', 'multiscale_coherence_1',
    'multiscale_coherence_2', 'multiscale_coherence_4',
    'multiscale_coherence_8',
]


# =============================================================================
# GROWTH MODELS
# =============================================================================

def dla_cluster(H, W, n_particles, rng, max_walk=5000):
    """
    Diffusion-Limited Aggregation on a grid.
    Particles random-walk from boundary until they stick to existing cluster.
    """
    grid = np.zeros((H, W), dtype=np.float64)
    cx, cy = W // 2, H // 2
    grid[cy, cx] = 1.0  # seed

    placed = 1
    launch_r = 5  # initial launch radius
    dirs = np.array([[-1,0],[1,0],[0,-1],[0,1]])

    while placed < n_particles:
        # Launch from circle
        angle = rng.random() * 2 * np.pi
        r = min(launch_r + 5, min(H, W) // 2 - 2)
        py = int(cy + r * np.sin(angle)) % H
        px = int(cx + r * np.cos(angle)) % W

        for step in range(max_walk):
            d = dirs[rng.integers(4)]
            ny, nx = (py + d[0]) % H, (px + d[1]) % W

            # Check if any neighbor is occupied
            stuck = False
            for dy, dx in dirs:
                ck_y, ck_x = (ny + dy) % H, (nx + dx) % W
                if grid[ck_y, ck_x] > 0:
                    stuck = True
                    break

            if stuck:
                grid[ny, nx] = 1.0
                placed += 1
                # Update launch radius
                dist = np.sqrt((ny - cy)**2 + (nx - cx)**2)
                if dist + 5 > launch_r:
                    launch_r = dist + 5
                break

            py, px = ny, nx

            # Kill if too far
            dist = np.sqrt((py - cy)**2 + (px - cx)**2)
            if dist > min(H, W) // 2:
                break

    return grid


def eden_cluster(H, W, n_particles, rng):
    """
    Eden model: pick a random boundary site and fill it.
    Produces compact, roughly circular clusters.
    """
    grid = np.zeros((H, W), dtype=np.float64)
    cx, cy = W // 2, H // 2
    grid[cy, cx] = 1.0

    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Maintain boundary set
    boundary = set()
    for dy, dx in dirs:
        ny, nx = (cy + dy) % H, (cx + dx) % W
        if grid[ny, nx] == 0:
            boundary.add((ny, nx))

    placed = 1
    while placed < n_particles and boundary:
        # Pick random boundary site
        site = list(boundary)
        idx = rng.integers(len(site))
        y, x = site[idx]
        boundary.discard((y, x))

        grid[y, x] = 1.0
        placed += 1

        # Add new boundary neighbors
        for dy, dx in dirs:
            ny, nx = (y + dy) % H, (x + dx) % W
            if grid[ny, nx] == 0 and (ny, nx) not in boundary:
                boundary.add((ny, nx))

    return grid


def random_cluster(H, W, n_particles, rng):
    """
    Random placement: uniformly scatter particles (no growth dynamics).
    """
    grid = np.zeros((H, W), dtype=np.float64)
    total = H * W
    indices = rng.choice(total, size=min(n_particles, total), replace=False)
    rows = indices // W
    cols = indices % W
    grid[rows, cols] = 1.0
    return grid


MODELS = {
    'DLA':    dla_cluster,
    'Eden':   eden_cluster,
    'Random': random_cluster,
}


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
    print("GROWTH MODEL FINGERPRINTING: DLA vs Eden vs Random")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, target_mass={TARGET_MASS}, N={N_TRIALS}")
    print("=" * 78)

    model_data = {}
    shuffled_model_data = {}
    example_fields = {}

    for name, gen_func in MODELS.items():
        print(f"\n  {name:8s}:", end=" ", flush=True)
        fields, shuf_fields = [], []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            field = gen_func(GRID_SIZE, GRID_SIZE, TARGET_MASS, rng)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))

            if trial == 0:
                example_fields[name] = field.copy()

            if (trial + 1) % 5 == 0:
                print(f"{trial+1}", end=" ", flush=True)

        model_data[name] = collect_metrics(geom, fields)
        shuffled_model_data[name] = collect_metrics(geom, shuf_fields)
        densities = [np.mean(f) for f in fields]
        print(f" density={np.mean(densities):.3f}")

    names = list(MODELS.keys())

    # Each vs shuffled
    bonf_s = ALPHA / (len(METRIC_NAMES) * len(names))
    print(f"\n{'─' * 78}")
    print(f"  Each model vs SHUFFLED baseline (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for name in names:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(model_data[name][m])
            b = np.array(shuffled_model_data[name][m])
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
    bonf_p = ALPHA / (len(METRIC_NAMES) * n_pairs)
    print(f"\n{'─' * 78}")
    print(f"  PAIRWISE comparisons (Bonferroni α={bonf_p:.2e})")
    print(f"{'─' * 78}")

    pair_results = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            sig = 0
            best_d, best_m = 0, ""
            findings = []
            for m in METRIC_NAMES:
                a = np.array(model_data[n1][m])
                b = np.array(model_data[n2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_p:
                    sig += 1
                    findings.append((m, d, p))
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            pair_results.append((n1, n2, sig, best_m, best_d))
            print(f"\n  {n1:8s} vs {n2:8s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")
            findings.sort(key=lambda x: -abs(x[1]))
            for m, d, p in findings[:3]:
                print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")

    return model_data, shuffled_model_data, example_fields, pair_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(model_data, example_fields, pair_results):
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

    names = list(MODELS.keys())
    n = len(names)
    model_colors = {'DLA': '#FF9800', 'Eden': '#4CAF50', 'Random': '#2196F3'}

    fig = plt.figure(figsize=(16, 12), facecolor='black')
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.3, 1.0, 1.0],
                           hspace=0.45, wspace=0.35)

    # Row 0: Example fields
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[name], cmap='hot', interpolation='nearest')
        ax.set_title(name, fontsize=12, fontweight='bold',
                     color=model_colors[name])
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: Key metrics as grouped bars
    compare_metrics = ['coherence_score', 'n_basins', 'curvature_mean',
                       'anisotropy_mean', 'tension_mean', 'basin_size_entropy']

    for j, metric in enumerate(compare_metrics[:3]):
        ax = fig.add_subplot(gs[1, j])
        means = [np.mean(model_data[nm][metric]) for nm in names]
        stds = [np.std(model_data[nm][metric]) for nm in names]
        colors = [model_colors[nm] for nm in names]
        ax.bar(range(n), means, yerr=stds, capsize=4,
               color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(metric.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)

    # Row 2: More metrics + summary text
    for j, metric in enumerate(compare_metrics[3:]):
        ax = fig.add_subplot(gs[2, j])
        means = [np.mean(model_data[nm][metric]) for nm in names]
        stds = [np.std(model_data[nm][metric]) for nm in names]
        colors = [model_colors[nm] for nm in names]
        ax.bar(range(n), means, yerr=stds, capsize=4,
               color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(metric.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)

    ax_txt = fig.add_subplot(gs[2, 2])
    ax_txt.axis('off')
    lines = ['Pairwise Results:\n']
    for n1, n2, sig, bm, bd in pair_results:
        lines.append(f'{n1:8s} vs {n2:8s}: {sig:2d} sig')
        lines.append(f'  best: {bm} d={bd:+.1f}\n')
    ax_txt.text(0.05, 0.95, '\n'.join(lines), transform=ax_txt.transAxes,
               fontsize=8, fontfamily='monospace', va='top', color='#cccccc')

    fig.suptitle('Growth Model Fingerprinting: DLA vs Eden vs Random',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'growth_models.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved growth_models.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    model_data, shuffled_data, example_fields, pair_results = run_investigation()
    make_figure(model_data, example_fields, pair_results)

    names = list(MODELS.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    pairs_ok = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Models tested: {len(names)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
