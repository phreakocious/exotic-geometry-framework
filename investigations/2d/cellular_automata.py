#!/usr/bin/env python3
"""
Investigation: 2D Cellular Automata Fingerprinting via SpatialFieldGeometry

Can we distinguish different CA rule sets by their spatial geometry?
- Game of Life (B3/S23): gliders, oscillators, still lifes
- Seeds (B2/S): explosive ephemeral growth
- Day & Night (B3678/S34678): symmetric blob-like patterns
- HighLife (B36/S23): like GoL but with replicators
- Diamoeba (B35678/S5678): growing amoeba-like blobs
- Anneal (B4678/S35678): expanding/shrinking regions

Also: GoL at different epochs (t=10, 50, 200, 1000) — spatial evolution.

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
GRID_SIZE = 128

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn

RULES = {
    'GoL':        (frozenset({3}), frozenset({2, 3})),
    'Seeds':      (frozenset({2}), frozenset()),
    'DayNight':   (frozenset({3,6,7,8}), frozenset({3,4,6,7,8})),
    'HighLife':   (frozenset({3,6}), frozenset({2,3})),
    'Diamoeba':   (frozenset({3,5,6,7,8}), frozenset({5,6,7,8})),
    'Anneal':     (frozenset({4,6,7,8}), frozenset({3,5,6,7,8})),
}


def ca_step(grid, birth, survival):
    n = (np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
         np.roll(grid, 1, 1) + np.roll(grid, -1, 1) +
         np.roll(np.roll(grid, 1, 0), 1, 1) +
         np.roll(np.roll(grid, 1, 0), -1, 1) +
         np.roll(np.roll(grid, -1, 0), 1, 1) +
         np.roll(np.roll(grid, -1, 0), -1, 1))
    birth_mask = np.zeros_like(grid, dtype=bool)
    survive_mask = np.zeros_like(grid, dtype=bool)
    for b in birth:
        birth_mask |= (n == b)
    for s in survival:
        survive_mask |= (n == s)
    return np.where(grid == 0, np.where(birth_mask, 1, 0),
                                np.where(survive_mask, 1, 0))


def run_ca(H, W, birth, survival, n_steps, rng, init_density=0.5):
    grid = (rng.random((H, W)) < init_density).astype(int)
    for _ in range(n_steps):
        grid = ca_step(grid, birth, survival)
    return grid.astype(np.float64)


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


def run_investigation():
    analyzer = GeometryAnalyzer().add_spatial_geometries()
    N_STEPS = 200

    print("=" * 78)
    print("2D CELLULAR AUTOMATA FINGERPRINTING")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, {N_STEPS} steps, N={N_TRIALS}")
    print("=" * 78)

    rule_data = {}
    shuffled_rule_data = {}
    example_fields = {}

    for name, (birth, survival) in RULES.items():
        print(f"  {name:12s}...", end=" ", flush=True)
        fields, shuf_fields = [], []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            field = run_ca(GRID_SIZE, GRID_SIZE, birth, survival, N_STEPS, rng)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))
            if trial == 0:
                example_fields[name] = field.copy()
        rule_data[name] = collect_metrics(analyzer, fields)
        shuffled_rule_data[name] = collect_metrics(analyzer, shuf_fields)
        densities = [np.mean(f) for f in fields]
        print(f"density={np.mean(densities):.3f}")

    names = list(RULES.keys())

    # Each vs shuffled
    bonf_s = ALPHA / (len(METRIC_NAMES) * len(names))
    print(f"\n{'─' * 78}")
    print(f"  Each rule vs SHUFFLED baseline (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for name in names:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(rule_data[name][m])
            b = np.array(shuffled_rule_data[name][m])
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
            for m in METRIC_NAMES:
                a = np.array(rule_data[n1][m])
                b = np.array(rule_data[n2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_p:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            pair_results.append((n1, n2, sig, best_m, best_d))
            print(f"  {n1:12s} vs {n2:12s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")

    # GoL temporal evolution
    print(f"\n{'─' * 78}")
    print(f"  GoL TEMPORAL EVOLUTION")
    print(f"{'─' * 78}")

    epochs = [10, 50, 200, 1000]
    epoch_data = {}
    epoch_examples = {}
    birth, survival = RULES['GoL']
    for ep in epochs:
        fields = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            f = run_ca(GRID_SIZE, GRID_SIZE, birth, survival, ep, rng)
            fields.append(f)
            if trial == 0:
                epoch_examples[ep] = f.copy()
        epoch_data[ep] = collect_metrics(analyzer, fields)
        print(f"  t={ep:4d}: density={np.mean([np.mean(f) for f in fields]):.3f}")

    bonf_e = ALPHA / (len(METRIC_NAMES) * (len(epochs) - 1))
    for i in range(len(epochs) - 1):
        e1, e2 = epochs[i], epochs[i+1]
        sig = 0
        best_d, best_m = 0, ""
        for m in METRIC_NAMES:
            a = np.array(epoch_data[e1][m])
            b = np.array(epoch_data[e2][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf_e:
                sig += 1
            if abs(d) > abs(best_d):
                best_d, best_m = d, m
        print(f"  t={e1:4d} vs t={e2:4d}: {sig:2d} sig  "
              f"best: {best_m:25s} d={best_d:+8.2f}")

    return rule_data, example_fields, pair_results, epoch_data, epoch_examples


def make_figure(rule_data, example_fields, pair_results, epoch_data, epoch_examples):
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

    names = list(RULES.keys())
    n = len(names)

    fig = plt.figure(figsize=(18, 35), facecolor='black')
    gs = gridspec.GridSpec(4, n, figure=fig, height_ratios=[1.2, 1.0, 1.0, 1.0],
                           hspace=0.45, wspace=0.3)

    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[name], cmap='hot', interpolation='nearest')
        ax.set_title(name, fontsize=10, fontweight='bold', color='white')
        ax.set_xticks([])
        ax.set_yticks([])

    compare_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins',
                       'Surface:gaussian_curvature_mean', 'PersistentHomology2D:persistence_entropy',
                       'Conformal2D:structure_isotropy', 'SpectralPower:spectral_slope']
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#00BCD4']

    for j in range(min(n, len(compare_metrics))):
        metric = compare_metrics[j]
        ax = fig.add_subplot(gs[1, j])
        means = [np.mean(rule_data[nm][metric]) for nm in names]
        stds = [np.std(rule_data[nm][metric]) for nm in names]
        ax.bar(range(n), means, yerr=stds, capsize=3,
               color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=7, rotation=35, ha='right')
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    ax_mat = fig.add_subplot(gs[2, :3])
    mat = np.zeros((n, n))
    for n1, n2, sig, _, _ in pair_results:
        i1, i2 = names.index(n1), names.index(n2)
        mat[i1, i2] = sig
        mat[i2, i1] = sig
    im = ax_mat.imshow(mat, cmap='magma', interpolation='nearest', vmin=0)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(names, fontsize=8, rotation=35, ha='right')
    ax_mat.set_yticklabels(names, fontsize=6)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax_mat.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white' if mat[i,j] > 8 else '#cccccc')
    ax_mat.set_title('Pairwise significant metrics', fontsize=10, fontweight='bold')
    cb = plt.colorbar(im, ax=ax_mat, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    ax_txt = fig.add_subplot(gs[2, 3:])
    ax_txt.axis('off')
    lines = ['Best discriminator per pair:\n']
    pair_sorted = sorted(pair_results, key=lambda x: -x[2])
    for n1, n2, sig, bm, bd in pair_sorted:
        lines.append(f'{n1:12s} vs {n2:12s}: {sig:2d}  {bm}')
    ax_txt.text(0.05, 0.95, '\n'.join(lines), transform=ax_txt.transAxes,
               fontsize=7, fontfamily='monospace', va='top', color='#cccccc')

    epochs = [10, 50, 200, 1000]
    for i, ep in enumerate(epochs):
        ax = fig.add_subplot(gs[3, i])
        ax.imshow(epoch_examples[ep], cmap='hot', interpolation='nearest')
        ax.set_title(f'GoL t={ep}', fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    evo_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins']
    for j, metric in enumerate(evo_metrics):
        ax = fig.add_subplot(gs[3, 4 + j])
        means = [np.mean(epoch_data[ep][metric]) for ep in epochs]
        stds = [np.std(epoch_data[ep][metric]) for ep in epochs]
        ax.errorbar(epochs, means, yerr=stds, fmt='o-', color='#FF9800',
                    markersize=5, capsize=3, linewidth=1.5)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_title(f'GoL {metric.replace("_"," ")}', fontsize=9, fontweight='bold')
        ax.set_xscale('log')
        ax.tick_params(labelsize=7)

    fig.suptitle('Cellular Automata Fingerprinting via SpatialFieldGeometry',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'cellular_automata.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved cellular_automata.png")
    plt.close(fig)


if __name__ == "__main__":
    results = run_investigation()
    make_figure(*results)

    names = list(RULES.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    pair_results = results[2]
    pairs_ok = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Rules tested: {len(names)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
