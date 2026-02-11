#!/usr/bin/env python3
"""
Investigation: Gray-Scott Reaction-Diffusion Pattern Fingerprinting

The Gray-Scott model produces radically different spatial patterns (spots, stripes,
worms, coral, mazes, chaos) from just two parameters (F, k). All share the same
underlying PDE — only the parameter regime differs.

Can SpatialFieldGeometry fingerprint each morphology class?

Methodology: N_TRIALS=25 (different initial noise), Cohen's d, Bonferroni correction.
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
N_STEPS = 10000
Du, Dv = 1.0, 0.5

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn

# Gray-Scott parameter regimes (Pearson/Karl Sims classification)
REGIMES = {
    'spots':   (0.034, 0.063),   # Stable circular spots (type η)
    'stripes': (0.042, 0.059),   # Parallel stripes / fingerprints (type θ)
    'worms':   (0.054, 0.063),   # Thin winding worms
    'coral':   (0.026, 0.061),   # Branching coral-like growth (type λ)
    'mazes':   (0.029, 0.057),   # Dense labyrinthine pattern
    'chaos':   (0.014, 0.054),   # Turbulent, unstable patterns (type β)
}


# =============================================================================
# GRAY-SCOTT SIMULATION
# =============================================================================

def laplacian_9pt(field):
    """9-point discrete Laplacian with periodic BCs (Karl Sims weights)."""
    p = np.pad(field, 1, mode='wrap')
    H, W = field.shape
    return (0.2 * (p[0:H, 1:W+1] + p[2:H+2, 1:W+1] +
                   p[1:H+1, 0:W] + p[1:H+1, 2:W+2]) +
            0.05 * (p[0:H, 0:W] + p[0:H, 2:W+2] +
                    p[2:H+2, 0:W] + p[2:H+2, 2:W+2]) - field)


def gray_scott(H, W, F, k, n_steps, rng):
    """
    Run Gray-Scott reaction-diffusion.

    du/dt = Du * ∇²u - uv² + F(1-u)
    dv/dt = Dv * ∇²v + uv² - (F+k)v

    Returns the v-field (activator, shows the pattern).
    """
    u = np.ones((H, W))
    v = np.zeros((H, W))

    # Seed: perturbed square in center + random spots
    cx, cy = W // 2, H // 2
    sz = max(10, min(H, W) // 6)
    u[cy-sz:cy+sz, cx-sz:cx+sz] = 0.50 + 0.02 * rng.standard_normal((2*sz, 2*sz))
    v[cy-sz:cy+sz, cx-sz:cx+sz] = 0.25 + 0.02 * rng.standard_normal((2*sz, 2*sz))

    # A few random spots to break symmetry
    n_spots = 5
    for _ in range(n_spots):
        sy, sx = rng.integers(0, H), rng.integers(0, W)
        r = rng.integers(3, 8)
        y0, y1 = max(0, sy-r), min(H, sy+r)
        x0, x1 = max(0, sx-r), min(W, sx+r)
        u[y0:y1, x0:x1] = 0.50 + 0.02 * rng.standard_normal((y1-y0, x1-x0))
        v[y0:y1, x0:x1] = 0.25 + 0.02 * rng.standard_normal((y1-y0, x1-x0))

    for step in range(n_steps):
        Lu = laplacian_9pt(u)
        Lv = laplacian_9pt(v)
        uvv = u * v * v
        u += Du * Lu - uvv + F * (1.0 - u)
        v += Dv * Lv + uvv - (F + k) * v
        # Clamp to [0, 1]
        np.clip(u, 0, 1, out=u)
        np.clip(v, 0, 1, out=v)

    return v


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


# =============================================================================
# INVESTIGATION
# =============================================================================

def run_investigation():
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    print("=" * 78)
    print("GRAY-SCOTT REACTION-DIFFUSION FINGERPRINTING")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, {N_STEPS} steps, N_TRIALS={N_TRIALS}")
    print("=" * 78)

    regime_data = {}     # {name: {metric: [values]}}
    shuffled_data = {}   # {name: {metric: [values]}}
    example_fields = {}  # {name: field} for visualization

    for name, (F, k) in REGIMES.items():
        print(f"\n  {name:8s} (F={F:.3f}, k={k:.3f}):", end=" ", flush=True)
        regime_data[name] = {m: [] for m in METRIC_NAMES}
        shuffled_data[name] = {m: [] for m in METRIC_NAMES}

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            field = gray_scott(GRID_SIZE, GRID_SIZE, F, k, N_STEPS, rng)

            if trial == 0:
                example_fields[name] = field.copy()

            shuf = shuffle_field(field, np.random.default_rng(1000 + trial))

            res = analyzer.analyze(field)
            res_shuf = analyzer.analyze(shuf)

            for r in res.results:
                for mn, mv in r.metrics.items():
                    key = f"{r.geometry_name}:{mn}"
                    if key in regime_data[name] and np.isfinite(mv):
                        regime_data[name][key].append(mv)
            for r in res_shuf.results:
                for mn, mv in r.metrics.items():
                    key = f"{r.geometry_name}:{mn}"
                    if key in shuffled_data[name] and np.isfinite(mv):
                        shuffled_data[name][key].append(mv)

            if (trial + 1) % 5 == 0:
                print(f"{trial+1}", end=" ", flush=True)
        print()

    names = list(REGIMES.keys())

    # Each type vs shuffled baseline
    bonf_s = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  Each regime vs SHUFFLED baseline (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for name in names:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(regime_data[name][m])
            b = np.array(shuffled_data[name][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf_s:
                sig += 1
                findings.append((m, d, p))
        F, k = REGIMES[name]
        print(f"\n  {name.upper()} (F={F:.3f}, k={k:.3f}): {sig} significant metrics")
        findings.sort(key=lambda x: -abs(x[1]))
        for m, d, p in findings[:5]:
            print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")

    # Pairwise comparisons
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
                a = np.array(regime_data[n1][m])
                b = np.array(regime_data[n2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_p:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            pair_results.append((n1, n2, sig, best_m, best_d))
            print(f"  {n1:8s} vs {n2:8s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")

    return regime_data, shuffled_data, example_fields, pair_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(regime_data, example_fields, pair_results):
    print("\nGenerating figure...", flush=True)

    names = list(REGIMES.keys())
    n = len(names)

    BG = '#181818'
    FG = '#e0e0e0'
    fig = plt.figure(figsize=(18, 20), facecolor=BG)
    gs = gridspec.GridSpec(3, n, figure=fig, height_ratios=[1.3, 1.0, 1.0],
                           hspace=0.4, wspace=0.3)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # Row 0: Example patterns
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        _dark_ax(ax)
        ax.imshow(example_fields[name], cmap='inferno', interpolation='bilinear')
        F, k = REGIMES[name]
        ax.set_title(f'{name}\nF={F:.3f}, k={k:.3f}', fontsize=9, fontweight='bold', color=FG)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: Key metrics as bar charts
    compare_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins',
                       'Surface:gaussian_curvature_mean', 'PersistentHomology2D:persistence_entropy',
                       'Conformal2D:structure_isotropy', 'SpectralPower:spectral_slope']
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#795548']

    for j in range(min(n, len(compare_metrics))):
        metric = compare_metrics[j]
        ax = fig.add_subplot(gs[1, j])
        _dark_ax(ax)
        means = [np.mean(regime_data[nm][metric]) for nm in names]
        stds = [np.std(regime_data[nm][metric]) for nm in names]
        bars = ax.bar(range(n), means, yerr=stds, capsize=3,
                      color=colors, alpha=0.85, edgecolor='#333')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right', color=FG)
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=9, fontweight='bold', color=FG)

    # Row 2: Pairwise distinguishability matrix
    ax_mat = fig.add_subplot(gs[2, :])
    _dark_ax(ax_mat)
    mat = np.zeros((n, n))
    for n1, n2, sig, _, _ in pair_results:
        i1, i2 = names.index(n1), names.index(n2)
        mat[i1, i2] = sig
        mat[i2, i1] = sig
    im = ax_mat.imshow(mat, cmap='YlOrRd', interpolation='nearest', vmin=0)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(names, fontsize=8, rotation=30, ha='right', color=FG)
    ax_mat.set_yticklabels(names, fontsize=6, color=FG)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax_mat.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white' if mat[i,j] > 8 else 'black')
    ax_mat.set_title('Pairwise significant metrics', fontsize=10, fontweight='bold', color=FG)
    plt.colorbar(im, ax=ax_mat, shrink=0.8)

    fig.suptitle('Gray-Scott Reaction-Diffusion: Morphology Fingerprinting',
                 fontsize=14, fontweight='bold', y=0.98, color=FG)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'reaction_diffusion.png'),
                dpi=180, bbox_inches='tight', facecolor=BG)
    print("  Saved reaction_diffusion.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    regime_data, shuffled_data, example_fields, pair_results = run_investigation()
    make_figure(regime_data, example_fields, pair_results)

    # Summary
    names = list(REGIMES.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    pairs_distinguished = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Regimes tested: {len(names)}")
    print(f"  Pairs distinguished: {pairs_distinguished}/{n_pairs}")
    all_distinguished = pairs_distinguished == n_pairs
    print(f"  All pairs distinguished: {all_distinguished}")
