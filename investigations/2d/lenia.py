#!/usr/bin/env python3
"""
Investigation: Lenia (Continuous Cellular Automata) Fingerprinting

Lenia is a continuous generalization of Game of Life:
- Continuous states [0,1] instead of binary
- Smooth convolution kernel instead of discrete neighbor count
- Growth function maps kernel output to state change

Different kernel parameters produce strikingly different organisms:
- Orbium: smooth gliders (R=13, peaks=[0.15,0.33])
- Geminium: self-replicating (R=10, peaks=[0.12,0.28])
- Hydrogeminium: aquatic-looking blobs
- Scutium: crawling shield shapes
- Random soup: initial random state with default kernel

We compare different kernel radii and growth function parameters.

Methodology: N_TRIALS=25, shuffled baselines, Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from scipy.signal import fftconvolve
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
ALPHA = 0.05
GRID_SIZE = 128
N_STEPS = 200
DT = 0.1

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn


# =============================================================================
# LENIA ENGINE
# =============================================================================

def bell(x, mu, sigma):
    """Smooth bump function (normalized Gaussian-like)."""
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def kernel_ring(R, peaks, sigma=0.15):
    """
    Concentric ring kernel for Lenia.
    peaks: list of (relative) radii at which rings peak.
    """
    size = 2 * R + 1
    y, x = np.mgrid[-R:R+1, -R:R+1]
    r = np.sqrt(x**2 + y**2) / R
    K = np.zeros((size, size))
    for p in peaks:
        K += bell(r, p, sigma)
    K[R, R] = 0  # exclude center
    total = K.sum()
    if total > 0:
        K /= total
    return K


def growth_function(u, mu, sigma):
    """Growth mapping: bell-shaped around mu."""
    return 2 * bell(u, mu, sigma) - 1


def lenia_step(field, kernel, growth_mu, growth_sigma, dt):
    """One step of Lenia."""
    # Convolution with periodic padding via FFT
    H, W = field.shape
    kH, kW = kernel.shape
    padded = np.pad(field, ((kH//2, kH//2), (kW//2, kW//2)), mode='wrap')
    potential = fftconvolve(padded, kernel, mode='valid')[:H, :W]
    growth = growth_function(potential, growth_mu, growth_sigma)
    field = field + dt * growth
    return np.clip(field, 0, 1)


def run_lenia(H, W, kernel, growth_mu, growth_sigma, n_steps, dt, rng,
              init_density=0.3, init_radius=None):
    """Run Lenia simulation."""
    field = np.zeros((H, W))

    if init_radius is None:
        init_radius = min(H, W) // 4

    # Initialize with random blob in center
    cy, cx = H // 2, W // 2
    y, x = np.mgrid[:H, :W]
    mask = ((x - cx)**2 + (y - cy)**2) < init_radius**2
    field[mask] = rng.random(mask.sum()) * init_density + 0.2

    for _ in range(n_steps):
        field = lenia_step(field, kernel, growth_mu, growth_sigma, dt)

    return field


# =============================================================================
# LENIA CONFIGURATIONS
# =============================================================================

# Each config: (kernel_radius, kernel_peaks, growth_mu, growth_sigma, label)
# Note: growth_sigma must be >= ~0.020 for random-blob initialization to sustain life.
CONFIGS = {
    'orbium':    (13, [0.5],       0.15, 0.025),  # smooth gliders
    'smooth':    (10, [0.3, 0.7],  0.12, 0.020),  # double-ring, blobs
    'spiky':     (7,  [0.5],       0.15, 0.030),  # sharp-peaked growth
    'diffuse':   (20, [0.4],       0.10, 0.035),  # large radius, diffuse
    'multi':     (12, [0.2,0.5,0.8], 0.20, 0.025), # triple ring
    'tight':     (5,  [0.5],       0.22, 0.024),  # small radius, GoL-like
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

    print("=" * 78)
    print("LENIA (CONTINUOUS CA) FINGERPRINTING")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, {N_STEPS} steps, dt={DT}, N={N_TRIALS}")
    print("=" * 78)

    config_data = {}
    shuffled_data = {}
    example_fields = {}

    for name, (R, peaks, g_mu, g_sigma) in CONFIGS.items():
        print(f"\n  {name:10s} (R={R:2d}):", end=" ", flush=True)
        kernel = kernel_ring(R, peaks)
        fields, shuf_fields = [], []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            field = run_lenia(GRID_SIZE, GRID_SIZE, kernel, g_mu, g_sigma,
                            N_STEPS, DT, rng)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))

            if trial == 0:
                example_fields[name] = field.copy()

            if (trial + 1) % 5 == 0:
                print(f"{trial+1}", end=" ", flush=True)

        config_data[name] = collect_metrics(analyzer, fields)
        shuffled_data[name] = collect_metrics(analyzer, shuf_fields)
        densities = [np.mean(f) for f in fields]
        print(f" density={np.mean(densities):.3f}")

    names = list(CONFIGS.keys())

    # Each vs shuffled
    bonf_s = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  Each config vs SHUFFLED (Bonferroni α={bonf_s:.2e})")
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

    names = list(CONFIGS.keys())
    n = len(names)
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#00BCD4']

    fig = plt.figure(figsize=(18, 24), facecolor='black')
    gs = gridspec.GridSpec(4, n, figure=fig, height_ratios=[1.3, 0.8, 1.0, 1.0],
                           hspace=0.45, wspace=0.3)

    # Row 0: Lenia patterns
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[name], cmap='magma', interpolation='bilinear',
                  vmin=0, vmax=1)
        R, peaks, g_mu, g_sigma = CONFIGS[name]
        ax.set_title(f'{name}\nR={R}', fontsize=9, fontweight='bold',
                     color=colors[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: Kernel visualizations
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[1, i])
        R, peaks, _, _ = CONFIGS[name]
        K = kernel_ring(R, peaks)
        ax.imshow(K, cmap='hot', interpolation='bilinear')
        ax.set_title('Kernel', fontsize=8, color='#888888')
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2: Key metrics
    compare_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins',
                       'Surface:gaussian_curvature_mean', 'PersistentHomology2D:persistence_entropy',
                       'Conformal2D:structure_isotropy', 'SpectralPower:spectral_slope']

    for j in range(n):
        metric = compare_metrics[j]
        ax = fig.add_subplot(gs[2, j])
        means = [np.mean(config_data[nm][metric]) for nm in names]
        stds = [np.std(config_data[nm][metric]) for nm in names]
        ax.bar(range(n), means, yerr=stds, capsize=3,
               color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=6, rotation=35, ha='right')
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    # Row 3: Pairwise matrix + summary
    ax_mat = fig.add_subplot(gs[3, :])
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
                           fontsize=7, fontweight='bold',
                           color='white' if mat[i,j] > 8 else '#aaaaaa')
    ax_mat.set_title('Pairwise significant metrics', fontsize=10, fontweight='bold')
    cb = plt.colorbar(im, ax=ax_mat, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    fig.suptitle('Lenia: Continuous Cellular Automata Fingerprinting',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'lenia.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved lenia.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config_data, shuffled_data, example_fields, pair_results = run_investigation()
    make_figure(config_data, example_fields, pair_results)

    names = list(CONFIGS.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    pairs_ok = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Configs tested: {len(names)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
