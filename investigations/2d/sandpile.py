#!/usr/bin/env python3
"""
Investigation: Abelian Sandpile (Self-Organized Criticality) via SpatialFieldGeometry

The Abelian sandpile model exhibits self-organized criticality (SOC):
- Drop sand grains one at a time onto a pile
- When a site reaches threshold (≥4), it topples, distributing to neighbors
- The system organizes itself to a critical state with power-law avalanches

Compare sandpile configurations against:
1. Random fields with same value distribution (shuffled baseline)
2. Different pile sizes (early = subcritical, late = critical)
3. Relaxed vs active (mid-avalanche) configurations

Methodology: N_TRIALS=25, shuffled baselines, Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
ALPHA = 0.05
GRID_SIZE = 64  # Smaller grid — sandpile simulation is iterative

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn

DROP_COUNTS = [500, 2000, 10000, 50000]


# =============================================================================
# ABELIAN SANDPILE
# =============================================================================

def sandpile(H, W, n_drops, rng):
    """
    Run Abelian sandpile model. Drop grains at random positions.
    Open boundary conditions (grains fall off edges).
    Returns the relaxed height field as float64.
    """
    grid = np.zeros((H, W), dtype=np.int32)

    for _ in range(n_drops):
        # Drop at random position
        y, x = rng.integers(0, H), rng.integers(0, W)
        grid[y, x] += 1

        # Topple until stable
        while True:
            unstable = grid >= 4
            if not np.any(unstable):
                break
            # Topple all unstable sites simultaneously
            topple_count = grid // 4
            grid -= 4 * topple_count

            # Distribute to neighbors
            if np.any(topple_count > 0):
                # Up
                grid[1:, :] += topple_count[:-1, :]
                # Down
                grid[:-1, :] += topple_count[1:, :]
                # Left
                grid[:, 1:] += topple_count[:, :-1]
                # Right
                grid[:, :-1] += topple_count[:, 1:]
                # Grains at boundary fall off (open BC) — already handled
                # since we don't wrap around

    return grid.astype(np.float64)


def sandpile_identity(H, W):
    """
    Compute the identity element of the sandpile group.
    This is the unique recurrent configuration c such that c + c = c.
    Beautiful fractal-like pattern.
    """
    # Start with max stable config (all 3s), add it to itself, relax
    grid = np.full((H, W), 6, dtype=np.int32)

    while True:
        unstable = grid >= 4
        if not np.any(unstable):
            break
        topple_count = grid // 4
        grid -= 4 * topple_count
        if np.any(topple_count > 0):
            grid[1:, :] += topple_count[:-1, :]
            grid[:-1, :] += topple_count[1:, :]
            grid[:, 1:] += topple_count[:, :-1]
            grid[:, :-1] += topple_count[:, 1:]

    return grid.astype(np.float64)


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
    print("ABELIAN SANDPILE — SELF-ORGANIZED CRITICALITY")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, N={N_TRIALS}")
    print("=" * 78)

    # Part 1: Different drop counts (subcritical → critical)
    drop_data = {}
    shuffled_drop_data = {}
    example_fields = {}

    for n_drops in DROP_COUNTS:
        print(f"\n  {n_drops:6d} drops:", end=" ", flush=True)
        fields, shuf_fields = [], []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            field = sandpile(GRID_SIZE, GRID_SIZE, n_drops, rng)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))

            if trial == 0:
                example_fields[n_drops] = field.copy()

            if (trial + 1) % 5 == 0:
                print(f"{trial+1}", end=" ", flush=True)

        drop_data[n_drops] = collect_metrics(analyzer, fields)
        shuffled_drop_data[n_drops] = collect_metrics(analyzer, shuf_fields)
        means = [np.mean(f) for f in fields]
        print(f" mean_height={np.mean(means):.2f}")

    # Part 2: Sandpile identity (single deterministic field, compare to shuffled)
    print("\n  Identity element...", end=" ", flush=True)
    id_field = sandpile_identity(GRID_SIZE, GRID_SIZE)
    example_fields['identity'] = id_field.copy()
    # Generate shuffled variants for comparison
    id_fields = [id_field.copy() for _ in range(N_TRIALS)]  # identical
    id_shuf_fields = [shuffle_field(id_field, np.random.default_rng(2000 + t))
                      for t in range(N_TRIALS)]
    id_data = collect_metrics(analyzer, id_fields)
    id_shuf_data = collect_metrics(analyzer, id_shuf_fields)
    print("done")

    # Each drop count vs shuffled
    bonf_s = ALPHA / (len(METRIC_NAMES) * (len(DROP_COUNTS) + 1))
    print(f"\n{'─' * 78}")
    print(f"  Each configuration vs SHUFFLED (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for n_drops in DROP_COUNTS:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(drop_data[n_drops][m])
            b = np.array(shuffled_drop_data[n_drops][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf_s:
                sig += 1
                findings.append((m, d, p))
        print(f"\n  {n_drops:6d} drops: {sig} significant metrics")
        findings.sort(key=lambda x: -abs(x[1]))
        for m, d, p in findings[:5]:
            print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")

    # Identity vs shuffled
    sig = 0
    findings = []
    for m in METRIC_NAMES:
        a = np.array(id_data[m])
        b = np.array(id_shuf_data[m])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < bonf_s:
            sig += 1
            findings.append((m, d, p))
    print(f"\n  Identity: {sig} significant metrics")
    findings.sort(key=lambda x: -abs(x[1]))
    for m, d, p in findings[:5]:
        print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")

    # Pairwise drop counts
    n_pairs = len(DROP_COUNTS) * (len(DROP_COUNTS) - 1) // 2
    bonf_p = ALPHA / (len(METRIC_NAMES) * n_pairs)
    print(f"\n{'─' * 78}")
    print(f"  PAIRWISE drop counts (Bonferroni α={bonf_p:.2e})")
    print(f"{'─' * 78}")

    pair_results = []
    for i, d1 in enumerate(DROP_COUNTS):
        for d2 in DROP_COUNTS[i+1:]:
            sig = 0
            best_d, best_m = 0, ""
            for m in METRIC_NAMES:
                a = np.array(drop_data[d1][m])
                b = np.array(drop_data[d2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_p:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            pair_results.append((d1, d2, sig, best_m, best_d))
            print(f"  {d1:6d} vs {d2:6d}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")

    return drop_data, shuffled_drop_data, example_fields, pair_results, id_data, id_shuf_data


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(drop_data, example_fields, pair_results):
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

    n_drops_list = DROP_COUNTS
    n_cols = len(n_drops_list) + 1  # +1 for identity

    fig = plt.figure(figsize=(18, 30), facecolor='black')
    gs = gridspec.GridSpec(3, n_cols, figure=fig, height_ratios=[1.3, 1.0, 1.0],
                           hspace=0.45, wspace=0.35)

    # Row 0: Example fields
    for i, nd in enumerate(n_drops_list):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[nd], cmap='YlOrRd', interpolation='nearest')
        ax.set_title(f'{nd} drops', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[0, len(n_drops_list)])
    ax.imshow(example_fields['identity'], cmap='YlOrRd', interpolation='nearest')
    ax.set_title('Identity', fontsize=10, fontweight='bold', color='#00FF00')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#00FF00')
        spine.set_linewidth(2)

    # Row 1: Metrics vs drop count
    plot_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins',
                    'Surface:gaussian_curvature_mean', 'PersistentHomology2D:persistence_entropy',
                    'SpectralPower:spectral_slope']
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

    for j, metric in enumerate(plot_metrics):
        ax = fig.add_subplot(gs[1, j])
        means = [np.mean(drop_data[nd][metric]) for nd in n_drops_list]
        stds = [np.std(drop_data[nd][metric]) for nd in n_drops_list]
        ax.errorbar(range(len(n_drops_list)), means, yerr=stds, fmt='o-',
                    color=colors[j], markersize=5, capsize=3, linewidth=1.5)
        ax.set_xticks(range(len(n_drops_list)))
        ax.set_xticklabels([str(nd) for nd in n_drops_list], fontsize=7,
                           rotation=35, ha='right')
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    # Row 2: Pairwise heatmap + summary
    n = len(n_drops_list)
    ax_mat = fig.add_subplot(gs[2, :3])
    mat = np.zeros((n, n))
    for d1, d2, sig, _, _ in pair_results:
        i1 = n_drops_list.index(d1)
        i2 = n_drops_list.index(d2)
        mat[i1, i2] = sig
        mat[i2, i1] = sig
    im = ax_mat.imshow(mat, cmap='magma', interpolation='nearest', vmin=0)
    labels = [str(nd) for nd in n_drops_list]
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(labels, fontsize=8)
    ax_mat.set_yticklabels(labels, fontsize=6)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax_mat.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color='white' if mat[i,j] > 6 else '#aaaaaa')
    ax_mat.set_title('Pairwise significant metrics', fontsize=10, fontweight='bold')
    cb = plt.colorbar(im, ax=ax_mat, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    ax_txt = fig.add_subplot(gs[2, 3:])
    ax_txt.axis('off')
    lines = ['Pairwise results:\n']
    for d1, d2, sig, bm, bd in pair_results:
        lines.append(f'{d1:6d} vs {d2:6d}: {sig:2d} sig')
        lines.append(f'  best: {bm} d={bd:+.1f}')
    ax_txt.text(0.05, 0.95, '\n'.join(lines), transform=ax_txt.transAxes,
               fontsize=8, fontfamily='monospace', va='top', color='#cccccc')

    fig.suptitle('Abelian Sandpile: Self-Organized Criticality',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'sandpile.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved sandpile.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_investigation()
    drop_data, shuffled_data, example_fields, pair_results = results[:4]
    make_figure(drop_data, example_fields, pair_results)

    n_pairs = len(DROP_COUNTS) * (len(DROP_COUNTS) - 1) // 2
    pairs_ok = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Drop counts tested: {len(DROP_COUNTS)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
