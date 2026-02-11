#!/usr/bin/env python3
"""
Investigation: 2D Ising Phase Transition & Procedural Noise Fingerprinting

1. ISING MODEL: Can SpatialFieldGeometry detect the phase transition at T_c ≈ 2.269?
   - Temperature scan from ordered (T=1) to disordered (T=∞)
   - Expected: sharp metric changes near T_c, all T distinguishable from random

2. PROCEDURAL NOISE: Can we fingerprint white / value / Perlin / pink (1/f) noise?
   - All "look random" in histograms but have different spatial correlations
   - Expected: all pairs distinguishable, white ≈ shuffled (no false positives)

Methodology: N_TRIALS=25, Cohen's d, Bonferroni correction, shuffled baselines.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer

N_TRIALS = 25
ALPHA = 0.05
FIELD_SIZE = 64
T_C = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.26919


# =============================================================================
# ISING MODEL (vectorized checkerboard Metropolis)
# =============================================================================

def ising_equilibrate(H, W, T, n_sweeps, rng):
    """2D Ising model via checkerboard Metropolis. Periodic BCs."""
    spins = rng.choice([-1.0, 1.0], size=(H, W))
    rows, cols = np.mgrid[:H, :W]
    masks = [(rows + cols) % 2 == 0, (rows + cols) % 2 == 1]

    for _ in range(n_sweeps):
        for mask in masks:
            nn = (np.roll(spins, 1, 0) + np.roll(spins, -1, 0) +
                  np.roll(spins, 1, 1) + np.roll(spins, -1, 1))
            dE = 2.0 * spins * nn
            prob = np.where(dE <= 0, 1.0, np.exp(np.clip(-dE / T, -500, 0)))
            flip = mask & (rng.random((H, W)) < prob)
            spins[flip] *= -1

    return spins


# =============================================================================
# NOISE GENERATORS
# =============================================================================

def white_noise(H, W, rng):
    return rng.uniform(0, 1, (H, W))


def value_noise(H, W, scale=8, rng=None):
    """Grid random values + smoothstep interpolation."""
    rng = rng or np.random.default_rng()
    gx_n, gy_n = W // scale + 2, H // scale + 2
    vals = rng.uniform(0, 1, (gy_n, gx_n))
    y, x = np.mgrid[:H, :W]
    gx0, gy0 = x // scale, y // scale
    fx, fy = (x % scale) / scale, (y % scale) / scale
    u, v = fx*fx*(3 - 2*fx), fy*fy*(3 - 2*fy)
    r0 = vals[gy0, gx0] + u * (vals[gy0, gx0+1] - vals[gy0, gx0])
    r1 = vals[gy0+1, gx0] + u * (vals[gy0+1, gx0+1] - vals[gy0+1, gx0])
    return r0 + v * (r1 - r0)


def perlin_noise(H, W, scale=8, rng=None):
    """Gradient noise with smoothstep interpolation."""
    rng = rng or np.random.default_rng()
    gx_n, gy_n = W // scale + 2, H // scale + 2
    ang = rng.uniform(0, 2*np.pi, (gy_n, gx_n))
    gx_v, gy_v = np.cos(ang), np.sin(ang)
    y, x = np.mgrid[:H, :W]
    gx0, gy0 = x // scale, y // scale
    fx, fy = (x % scale) / scale, (y % scale) / scale
    d00 = gx_v[gy0, gx0]*fx       + gy_v[gy0, gx0]*fy
    d10 = gx_v[gy0, gx0+1]*(fx-1) + gy_v[gy0, gx0+1]*fy
    d01 = gx_v[gy0+1, gx0]*fx     + gy_v[gy0+1, gx0]*(fy-1)
    d11 = gx_v[gy0+1, gx0+1]*(fx-1) + gy_v[gy0+1, gx0+1]*(fy-1)
    u, v = fx*fx*(3 - 2*fx), fy*fy*(3 - 2*fy)
    return (d00 + u*(d10-d00)) + v * ((d01 + u*(d11-d01)) - (d00 + u*(d10-d00)))


def pink_noise_2d(H, W, rng=None):
    """1/f noise via spectral filtering."""
    rng = rng or np.random.default_rng()
    white = rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    fmag = np.sqrt(fx**2 + fy**2)
    fmag[0, 0] = 1.0
    filt = white / fmag
    filt[0, 0] = 0
    r = np.real(np.fft.ifft2(filt))
    return (r - r.min()) / (r.max() - r.min() + 1e-15)


# =============================================================================
# HELPERS
# =============================================================================

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn


def cohens_d(a, b):
    na, nb = len(a), len(b)
    ps = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def collect_metrics(analyzer, fields):
    """Run all 8 spatial geometries on list of fields, return {metric: [values]}."""
    out = {m: [] for m in METRIC_NAMES}
    for f in fields:
        res = analyzer.analyze(f)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def shuffle_field(field, rng):
    flat = field.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(field.shape)


# =============================================================================
# PART 1: ISING MODEL
# =============================================================================

def run_ising(analyzer):
    print("=" * 78)
    print("PART 1: ISING MODEL PHASE TRANSITION")
    print(f"T_c = {T_C:.5f}, field={FIELD_SIZE}x{FIELD_SIZE}, "
          f"N={N_TRIALS}, 3000 sweeps")
    print("=" * 78)

    temperatures = [1.0, 1.5, 2.0, 2.15, T_C, 2.4, 2.8, 5.0]
    N_SWEEPS = 3000

    ising_data = {}
    for T in temperatures:
        print(f"  T={T:.3f}...", end=" ", flush=True)
        fields = []
        mags = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            spins = ising_equilibrate(FIELD_SIZE, FIELD_SIZE, T, N_SWEEPS, rng)
            fields.append(spins.copy())
            mags.append(abs(np.mean(spins)))
        ising_data[T] = (collect_metrics(analyzer, fields), np.mean(mags))
        print(f"|m|={np.mean(mags):.3f}")

    # T=∞ baseline
    print(f"  T=∞...", end=" ", flush=True)
    fields_inf = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(42 + trial)
        fields_inf.append(rng.choice([-1.0, 1.0], size=(FIELD_SIZE, FIELD_SIZE)))
    ising_data['inf'] = (collect_metrics(analyzer, fields_inf), 0.0)
    print("|m|=0.000")

    # Temperature scan table
    key_metrics = ['SpatialField:tension_mean', 'SpatialField:curvature_mean',
                   'SpatialField:anisotropy_mean', 'SpatialField:n_basins',
                   'SpatialField:coherence_score', 'SpatialField:multiscale_coherence_4']
    all_T = temperatures + ['inf']

    print(f"\n{'Metric':>25s}", end="")
    for T in all_T:
        label = f"T={T:.2f}" if isinstance(T, float) else "T=∞"
        print(f" {label:>9s}", end="")
    print()
    print("─" * (25 + 10 * len(all_T)))

    for m in key_metrics:
        print(f"{m:>25s}", end="")
        for T in all_T:
            vals = ising_data[T][0][m]
            print(f" {np.mean(vals):9.3f}" if vals else f" {'NaN':>9s}", end="")
        print()

    # Magnetization row
    print(f"{'|magnetization|':>25s}", end="")
    for T in all_T:
        mag = ising_data[T][1]
        print(f" {mag:9.3f}", end="")
    print()

    # Each T vs T=∞
    bonf = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  Each T vs T=∞  (Bonferroni α={bonf:.2e})")
    print(f"{'─' * 78}")

    for T in temperatures:
        sig = 0
        best_d, best_m = 0, ""
        for m in METRIC_NAMES:
            a = np.array(ising_data[T][0][m])
            b = np.array(ising_data['inf'][0][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf:
                sig += 1
            if abs(d) > abs(best_d):
                best_d, best_m = d, m
        tc_mark = " ← T_c" if abs(T - T_C) < 0.001 else ""
        print(f"  T={T:5.3f}: {sig:2d} sig metrics  "
              f"best: {best_m:25s} d={best_d:+8.2f}{tc_mark}")

    # Adjacent temperature comparisons
    print(f"\n{'─' * 78}")
    print(f"  ADJACENT temperature pairs")
    print(f"{'─' * 78}")
    bonf_adj = ALPHA / len(METRIC_NAMES)
    for i in range(len(temperatures) - 1):
        T1, T2 = temperatures[i], temperatures[i+1]
        sig = 0
        best_d, best_m = 0, ""
        for m in METRIC_NAMES:
            a = np.array(ising_data[T1][0][m])
            b = np.array(ising_data[T2][0][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf_adj:
                sig += 1
            if abs(d) > abs(best_d):
                best_d, best_m = d, m
        print(f"  T={T1:.3f} vs T={T2:.3f}: {sig:2d} sig  "
              f"best: {best_m:25s} d={best_d:+8.2f}")

    return ising_data


# =============================================================================
# PART 2: PROCEDURAL NOISE
# =============================================================================

def run_noise(analyzer):
    print(f"\n{'=' * 78}")
    print("PART 2: PROCEDURAL NOISE FINGERPRINTING")
    print(f"Field={FIELD_SIZE}x{FIELD_SIZE}, N={N_TRIALS}")
    print("=" * 78)

    generators = {
        'white':  lambda rng: white_noise(FIELD_SIZE, FIELD_SIZE, rng),
        'value':  lambda rng: value_noise(FIELD_SIZE, FIELD_SIZE, 8, rng),
        'perlin': lambda rng: perlin_noise(FIELD_SIZE, FIELD_SIZE, 8, rng),
        'pink':   lambda rng: pink_noise_2d(FIELD_SIZE, FIELD_SIZE, rng),
    }

    noise_data = {}
    shuf_data = {}
    for name, gen in generators.items():
        print(f"  {name}...", end=" ", flush=True)
        fields, shuf_fields = [], []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            f = gen(rng)
            fields.append(f)
            shuf_fields.append(shuffle_field(f, np.random.default_rng(1000 + trial)))
        noise_data[name] = collect_metrics(analyzer, fields)
        shuf_data[name] = collect_metrics(analyzer, shuf_fields)
        print("done")

    names = list(generators.keys())

    # Each vs shuffled baseline
    bonf_s = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  Each type vs SHUFFLED baseline (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for name in names:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(noise_data[name][m])
            b = np.array(shuf_data[name][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf_s:
                sig += 1
                findings.append((m, d, p))
        print(f"\n  {name.upper()}: {sig} significant metrics")
        findings.sort(key=lambda x: -abs(x[1]))
        for m, d, p in findings[:5]:
            print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")
        if name == 'white' and sig == 0:
            print("    (expected: white noise ≈ shuffled white noise)")
        elif name == 'white' and sig > 0:
            print("    ⚠ FALSE POSITIVE: white should ≈ shuffled!")

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
            findings = []
            for m in METRIC_NAMES:
                a = np.array(noise_data[n1][m])
                b = np.array(noise_data[n2][m])
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
            print(f"  {n1:8s} vs {n2:8s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")
            if sig > 0:
                findings.sort(key=lambda x: -abs(x[1]))
                for m, d, p in findings[:3]:
                    print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")

    return noise_data, pair_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    ising_data = run_ising(analyzer)
    noise_data, pair_results = run_noise(analyzer)

    # Final summary
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")

    print("\nIsing model phase transition:")
    print(f"  T_c = {T_C:.5f}")
    print("  → Geometry should detect ordered phase (T < T_c) as distinct from random")
    print("  → Critical regime (T ≈ T_c) should show unique multi-scale signatures")

    print("\nNoise fingerprinting:")
    all_distinguished = all(r[2] > 0 for r in pair_results)
    print(f"  All pairs distinguished: {all_distinguished}")
    for n1, n2, sc, bm, bd in pair_results:
        print(f"    {n1:8s} vs {n2:8s}: {sc:2d} metrics  (best d={bd:+.2f})")

    make_figure(ising_data)


def make_figure(ising_data):
    """Generate Ising phase transition figure with dark background."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    BG = '#181818'
    FG = '#e0e0e0'

    temperatures = [T for T in ising_data if isinstance(T, float)]
    temperatures.sort()
    all_T = temperatures + ['inf']

    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # Show 6 ising fields at different T
    show_temps = [1.0, 1.5, 2.0, T_C, 2.8, 5.0]
    show_temps = [T for T in show_temps if T in ising_data]

    curve_metrics = [
        ('|Magnetization|', None),
        ('SpatialField:tension_mean', 'Tension (gradient energy)'),
        ('SpatialField:multiscale_coherence_4', 'Multi-scale coherence (4x)'),
        ('SpatialField:n_basins', 'Basin count'),
        ('Surface:gaussian_curvature_mean', 'Gaussian curvature'),
        ('SpectralPower:spectral_slope', 'Spectral slope'),
    ]

    T_arr = np.array(temperatures)
    for idx, (metric, title) in enumerate(curve_metrics):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        _dark_ax(ax)

        if metric == '|Magnetization|':
            means = [ising_data[T][1] for T in temperatures]
            ax.plot(T_arr, means, 'o-', color='#e0e0e0', markersize=4, linewidth=1.5)
            ax.set_ylabel('|m|', fontsize=9, color=FG)
            title = '|Magnetization|'
        else:
            means = [np.mean(ising_data[T][0][metric]) for T in temperatures]
            stds = [np.std(ising_data[T][0][metric]) for T in temperatures]
            ax.errorbar(T_arr, means, yerr=stds, fmt='o-', markersize=4,
                        linewidth=1.5, capsize=2, color='steelblue')
            # T=∞ reference
            inf_mean = np.mean(ising_data['inf'][0][metric])
            ax.axhline(inf_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(T_arr[-1] + 0.1, inf_mean, 'T=∞', fontsize=7, color='#999',
                    va='center')
            ax.set_ylabel(metric.split(':')[-1].replace('_', ' '), fontsize=8, color=FG)

        ax.axvline(T_C, color='#ff6600', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(T_C, 0.97, ' $T_c$', color='#ff6600', fontsize=9,
                fontweight='bold', va='top', transform=ax.get_xaxis_transform())
        ax.set_xlabel('Temperature', fontsize=9, color=FG)
        ax.set_title(title, fontsize=9, fontweight='bold', color=FG)
        ax.set_xlim(0.2, 5.5)

    fig.suptitle('Ising Model Phase Transition via SpatialFieldGeometry',
                 fontsize=14, fontweight='bold', y=0.98, color=FG)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'ising.png')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
