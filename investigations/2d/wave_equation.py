#!/usr/bin/env python3
"""
Investigation: Wave Equation Interference Patterns via SpatialFieldGeometry

2D wave equation with different source configurations:
- Single point source: concentric rings
- Two-source interference: classic double-slit pattern
- Multi-source (5): complex interference
- Plane wave: uniform wavefront
- Random sources (20): speckle-like pattern
- Cavity mode: standing wave in rectangular box

All share the same wave equation PDE. Only source geometry differs.

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
N_STEPS = 300
DT = 0.3
C = 1.0  # wave speed

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn


# =============================================================================
# WAVE EQUATION SOLVER
# =============================================================================

def laplacian(u):
    """5-point discrete Laplacian with absorbing boundaries."""
    L = -4 * u
    L[1:, :] += u[:-1, :]
    L[:-1, :] += u[1:, :]
    L[:, 1:] += u[:, :-1]
    L[:, :-1] += u[:, 1:]
    return L


def run_wave(H, W, sources, n_steps, dt, c, freq, rng, damping=0.002):
    """
    Run 2D wave equation: u_tt = c^2 * ∇²u + sources - damping*u_t

    sources: list of (y, x) positions that oscillate at given frequency
    Returns time-averaged |u|² (energy density pattern).
    """
    u = np.zeros((H, W))
    u_prev = np.zeros((H, W))

    c2dt2 = (c * dt) ** 2
    energy = np.zeros((H, W))

    for step in range(n_steps):
        L = laplacian(u)
        u_next = 2 * u - u_prev + c2dt2 * L - damping * dt * (u - u_prev)

        # Drive sources
        t = step * dt
        for (sy, sx) in sources:
            u_next[sy, sx] += dt * np.sin(2 * np.pi * freq * t)

        # Absorbing boundary (simple: zero edges)
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0

        u_prev = u
        u = u_next

        # Accumulate energy after transient
        if step > n_steps // 3:
            energy += u ** 2

    # Normalize to [0, 1] for analysis
    energy /= (n_steps - n_steps // 3)
    if energy.max() > energy.min():
        energy = (energy - energy.min()) / (energy.max() - energy.min())

    return energy


# =============================================================================
# SOURCE CONFIGURATIONS
# =============================================================================

def single_source(H, W, rng):
    """Single point source at center."""
    cy, cx = H // 2, W // 2
    # Small random offset
    dy, dx = rng.integers(-3, 4), rng.integers(-3, 4)
    return [(cy + dy, cx + dx)]


def double_source(H, W, rng):
    """Two sources separated horizontally — interference pattern."""
    cy, cx = H // 2, W // 2
    sep = 20 + rng.integers(-3, 4)
    return [(cy, cx - sep // 2), (cy, cx + sep // 2)]


def multi_source(H, W, rng):
    """5 sources in a cross pattern."""
    cy, cx = H // 2, W // 2
    r = 15 + rng.integers(-3, 4)
    return [
        (cy, cx),
        (cy - r, cx),
        (cy + r, cx),
        (cy, cx - r),
        (cy, cx + r),
    ]


def plane_wave(H, W, rng):
    """Line of sources along left edge — approximates plane wave."""
    x = 5
    sources = [(y, x) for y in range(10, H - 10, 3)]
    return sources


def random_sources(H, W, rng):
    """20 randomly placed sources — speckle-like."""
    n = 20
    ys = rng.integers(10, H - 10, n)
    xs = rng.integers(10, W - 10, n)
    return list(zip(ys.tolist(), xs.tolist()))


def cavity_mode(H, W, rng):
    """Single source near corner — excites box modes."""
    margin = 10 + rng.integers(0, 5)
    return [(margin, margin)]


SOURCE_CONFIGS = {
    'single':    single_source,
    'double':    double_source,
    'multi5':    multi_source,
    'plane':     plane_wave,
    'random20':  random_sources,
    'cavity':    cavity_mode,
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
    freq = 0.15  # wave frequency

    print("=" * 78)
    print("WAVE EQUATION INTERFERENCE PATTERNS")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, {N_STEPS} steps, freq={freq}, N={N_TRIALS}")
    print("=" * 78)

    config_data = {}
    shuffled_data = {}
    example_fields = {}

    for name, src_func in SOURCE_CONFIGS.items():
        print(f"  {name:10s}...", end=" ", flush=True)
        fields, shuf_fields = [], []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            sources = src_func(GRID_SIZE, GRID_SIZE, rng)
            field = run_wave(GRID_SIZE, GRID_SIZE, sources, N_STEPS, DT, C, freq, rng)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))

            if trial == 0:
                example_fields[name] = field.copy()

            if (trial + 1) % 5 == 0:
                print(f"{trial+1}", end=" ", flush=True)

        config_data[name] = collect_metrics(analyzer, fields)
        shuffled_data[name] = collect_metrics(analyzer, shuf_fields)
        print()

    names = list(SOURCE_CONFIGS.keys())

    # Each vs shuffled
    bonf_s = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  Each source config vs SHUFFLED (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for name in names:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(config_data[name][m])
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
                a = np.array(config_data[n1][m])
                b = np.array(config_data[n2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_p:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            pair_results.append((n1, n2, sig, best_m, best_d))
            print(f"  {n1:10s} vs {n2:10s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")

    return config_data, shuffled_data, example_fields, pair_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(config_data, example_fields, pair_results):
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

    names = list(SOURCE_CONFIGS.keys())
    n = len(names)
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#00BCD4']

    fig = plt.figure(figsize=(18, 20), facecolor='black')
    gs = gridspec.GridSpec(3, n, figure=fig, height_ratios=[1.3, 1.0, 1.0],
                           hspace=0.45, wspace=0.3)

    # Row 0: Energy density patterns
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[name], cmap='inferno', interpolation='bilinear')
        ax.set_title(name, fontsize=10, fontweight='bold', color=colors[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: Key metrics
    compare_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins',
                       'Surface:gaussian_curvature_mean', 'PersistentHomology2D:persistence_entropy',
                       'Conformal2D:structure_isotropy', 'SpectralPower:spectral_slope']

    for j in range(n):
        metric = compare_metrics[j]
        ax = fig.add_subplot(gs[1, j])
        means = [np.mean(config_data[nm][metric]) for nm in names]
        stds = [np.std(config_data[nm][metric]) for nm in names]
        ax.bar(range(n), means, yerr=stds, capsize=3,
               color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=6, rotation=40, ha='right')
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    # Row 2: Pairwise matrix + summary
    ax_mat = fig.add_subplot(gs[2, :])
    mat = np.zeros((n, n))
    for n1, n2, sig, _, _ in pair_results:
        i1, i2 = names.index(n1), names.index(n2)
        mat[i1, i2] = sig
        mat[i2, i1] = sig
    im = ax_mat.imshow(mat, cmap='magma', interpolation='nearest', vmin=0)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(names, fontsize=7, rotation=40, ha='right')
    ax_mat.set_yticklabels(names, fontsize=5)
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

    fig.suptitle('Wave Equation: Source Configuration Fingerprinting',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'wave_equation.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved wave_equation.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config_data, shuffled_data, example_fields, pair_results = run_investigation()
    make_figure(config_data, example_fields, pair_results)

    names = list(SOURCE_CONFIGS.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    pairs_ok = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Source configs tested: {len(names)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
