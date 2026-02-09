#!/usr/bin/env python3
"""
Investigation: Real-World Time Series vs Mathematical Models.

Can exotic geometry distinguish real-world signals from their standard
mathematical models? This tests the framework on empirical phenomena:

8 signal types:
  1. brownian_motion  — cumulative sum of Gaussian noise (random walk)
  2. pink_noise_1f    — 1/f spectrum, generated via Voss-McCartney
  3. fractional_bm    — fBm with H=0.7 (persistent) vs H=0.3 (antipersistent)
  4. arma_process     — ARMA(2,1) model, common in econometrics
  5. mean_reverting   — Ornstein-Uhlenbeck process (e.g. interest rates)
  6. regime_switch    — switching between two AR(1) processes (market states)
  7. heartbeat_model  — synthetic ECG-like signal (QRS complex + T wave)
  8. network_traffic  — self-similar with Hurst H≈0.8 (LRD)

Five directions:
  D1: Each signal vs iid random (is there detectable structure?)
  D2: Brownian vs fractional Brownian (can geometry detect H?)
  D3: AR vs mean-reverting vs regime-switching (can it see process class?)
  D4: Shuffle validation (ordering vs distribution)
  D5: Sample rate invariance (same process, different temporal resolution)

Methodology: N_TRIALS=25, DATA_SIZE=2000, Cohen's d > 0.8, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats, signal
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05

# --- Discover metric names ---
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_ALPHA = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn

print(f"1D metrics: {N_METRICS}, Bonferroni α={BONF_ALPHA:.2e}")


# =========================================================================
# SIGNAL GENERATORS
# =========================================================================

def to_uint8(x):
    """Normalize signal to uint8 range."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-10:
        return np.full(len(x), 128, dtype=np.uint8)
    scaled = (x - lo) / (hi - lo)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def gen_random(trial, size):
    return np.random.default_rng(9000 + trial).integers(0, 256, size, dtype=np.uint8)


def gen_brownian(trial, size):
    """Standard Brownian motion (cumulative sum of iid Gaussian)."""
    rng = np.random.default_rng(42 + trial)
    increments = rng.standard_normal(size)
    return to_uint8(np.cumsum(increments))


def gen_pink_noise(trial, size):
    """1/f pink noise via Voss-McCartney algorithm."""
    rng = np.random.default_rng(42 + trial)
    n_sources = 8
    sources = rng.standard_normal((n_sources, size))
    out = np.zeros(size)
    for k in range(n_sources):
        period = 2 ** k
        held = np.repeat(sources[k, ::period], period)[:size]
        out += held
    return to_uint8(out)


def gen_fbm(trial, size, hurst=0.7):
    """Fractional Brownian motion via Cholesky (exact but O(n²) for small n).
    For larger n, use circulant embedding approximation."""
    rng = np.random.default_rng(42 + trial)
    n = min(size, 1000)  # Cholesky is O(n³), limit size
    # Covariance: C(i,j) = 0.5 * (|i|^2H + |j|^2H - |i-j|^2H)
    idx = np.arange(n, dtype=np.float64)
    i_grid, j_grid = np.meshgrid(idx, idx)
    H2 = 2 * hurst
    cov = 0.5 * (np.abs(i_grid)**H2 + np.abs(j_grid)**H2 - np.abs(i_grid - j_grid)**H2)
    cov += np.eye(n) * 1e-10  # regularize
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(n)
    fbm_vals = L @ z
    # Extend to full size if needed
    if n < size:
        fbm_vals = np.interp(np.linspace(0, n-1, size), np.arange(n), fbm_vals)
    return to_uint8(fbm_vals)


def gen_fbm_persistent(trial, size):
    return gen_fbm(trial, size, hurst=0.7)


def gen_fbm_antipersistent(trial, size):
    return gen_fbm(trial, size, hurst=0.3)


def gen_arma(trial, size):
    """ARMA(2,1) process: y[t] = 0.7*y[t-1] - 0.2*y[t-2] + e[t] + 0.3*e[t-1]."""
    rng = np.random.default_rng(42 + trial)
    e = rng.standard_normal(size + 500)
    y = np.zeros(size + 500)
    for t in range(2, size + 500):
        y[t] = 0.7 * y[t-1] - 0.2 * y[t-2] + e[t] + 0.3 * e[t-1]
    return to_uint8(y[500:])


def gen_ou(trial, size):
    """Ornstein-Uhlenbeck (mean-reverting) process.
    dX = θ(μ - X)dt + σdW, with θ=2.0, μ=0, σ=1."""
    rng = np.random.default_rng(42 + trial)
    theta, mu, sigma = 2.0, 0.0, 1.0
    dt = 0.01
    x = np.zeros(size)
    x[0] = rng.standard_normal()
    for t in range(1, size):
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return to_uint8(x)


def gen_regime(trial, size):
    """Regime-switching AR(1): alternates between bull (φ=0.95, σ=0.5)
    and bear (φ=0.8, σ=2.0) states with random switches."""
    rng = np.random.default_rng(42 + trial)
    y = np.zeros(size)
    state = 0  # 0=bull, 1=bear
    params = [(0.95, 0.5), (0.8, 2.0)]

    for t in range(1, size):
        phi, sigma = params[state]
        y[t] = phi * y[t-1] + sigma * rng.standard_normal()
        # State switch with prob 0.01
        if rng.random() < 0.01:
            state = 1 - state
    return to_uint8(y)


def gen_heartbeat(trial, size):
    """Synthetic ECG-like signal: periodic QRS complex + T wave + noise."""
    rng = np.random.default_rng(42 + trial)
    rate = 60 + rng.integers(-10, 10)  # beats per "minute"
    period = int(size / (rate / 60 * (size / 500)))  # samples per beat
    if period < 20:
        period = 20

    t = np.arange(size, dtype=np.float64)
    ecg = np.zeros(size)

    for beat_start in range(0, size, period):
        # QRS complex (sharp peak)
        qrs_center = beat_start + period * 0.4
        qrs_width = period * 0.03
        ecg += 3.0 * np.exp(-0.5 * ((t - qrs_center) / max(qrs_width, 1))**2)

        # T wave (broader, smaller)
        t_center = beat_start + period * 0.65
        t_width = period * 0.08
        ecg += 0.8 * np.exp(-0.5 * ((t - t_center) / max(t_width, 1))**2)

    # Add baseline wander and noise
    ecg += 0.3 * np.sin(2 * np.pi * t / (size / 2))
    ecg += 0.1 * rng.standard_normal(size)
    return to_uint8(ecg)


def gen_network(trial, size):
    """Self-similar network traffic with Hurst H≈0.8.
    Aggregated ON/OFF sources with heavy-tailed durations."""
    rng = np.random.default_rng(42 + trial)
    n_sources = 50
    traffic = np.zeros(size)

    for _ in range(n_sources):
        state = rng.integers(0, 2)
        t = 0
        while t < size:
            # Pareto-distributed duration (heavy tail → long-range dependence)
            duration = int(rng.pareto(1.2) * 5) + 1
            end = min(t + duration, size)
            if state == 1:
                rate = rng.uniform(0.5, 2.0)
                traffic[t:end] += rate
            state = 1 - state
            t = end

    return to_uint8(traffic)


SIGNALS = {
    'brownian':      gen_brownian,
    'pink_noise':    gen_pink_noise,
    'fbm_H07':       gen_fbm_persistent,
    'fbm_H03':       gen_fbm_antipersistent,
    'arma_21':       gen_arma,
    'ou_process':    gen_ou,
    'regime_switch': gen_regime,
    'heartbeat':     gen_heartbeat,
    'network':       gen_network,
}


# =========================================================================
# ANALYSIS UTILITIES
# =========================================================================

def collect_metrics(analyzer, data_arrays):
    out = {m: [] for m in METRIC_NAMES}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def compare(data_a, data_b):
    sig = 0
    findings = []
    for m in METRIC_NAMES:
        a = np.array(data_a[m])
        b = np.array(data_b[m])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < BONF_ALPHA and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# D1: Each signal vs iid random
# =========================================================================
def direction_1(analyzer):
    print("\n" + "=" * 78)
    print("D1: EACH SIGNAL VS IID RANDOM")
    print("=" * 78)

    random_arrays = [gen_random(t, DATA_SIZE) for t in range(N_TRIALS)]
    random_data = collect_metrics(analyzer, random_arrays)

    all_data = {}
    d1_results = {}

    for name, gen_fn in SIGNALS.items():
        print(f"  {name:18s}...", end=" ", flush=True)
        arrays = [gen_fn(t, DATA_SIZE) for t in range(N_TRIALS)]
        data = collect_metrics(analyzer, arrays)
        all_data[name] = data

        n_sig, findings = compare(data, random_data)
        d1_results[name] = n_sig

        sample = arrays[0]
        ent_vals = np.bincount(sample, minlength=256)
        ent_p = ent_vals[ent_vals > 0] / len(sample)
        entropy = float(-np.sum(ent_p * np.log2(ent_p)))
        print(f"{n_sig:3d} sig  (H={entropy:.1f} bits)")
        for m, d, p in findings[:3]:
            print(f"    {m:45s}  d={d:+8.2f}")

    return all_data, random_data, d1_results


# =========================================================================
# D2: Brownian vs fractional Brownian (Hurst parameter detection)
# =========================================================================
def direction_2(analyzer, all_data):
    print("\n" + "=" * 78)
    print("D2: HURST PARAMETER DETECTION")
    print("=" * 78)
    print("  Can geometry distinguish H=0.3 (antipersistent) vs H=0.5 (Brownian)")
    print("  vs H=0.7 (persistent)?")

    pairs = [
        ('brownian', 'fbm_H07', 'H=0.5 vs H=0.7'),
        ('brownian', 'fbm_H03', 'H=0.5 vs H=0.3'),
        ('fbm_H03', 'fbm_H07', 'H=0.3 vs H=0.7'),
    ]

    for n1, n2, label in pairs:
        n_sig, findings = compare(all_data[n1], all_data[n2])
        print(f"\n  {label}: {n_sig} sig metrics")
        for m, d, p in findings[:5]:
            print(f"    {m:45s}  d={d:+8.2f}")

    # Additional Hurst values
    print(f"\n  Hurst sweep (H=0.1 to H=0.9):")
    hurst_data = {}
    for h_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        arrays = [gen_fbm(t, DATA_SIZE, hurst=h_val) for t in range(N_TRIALS)]
        hurst_data[h_val] = collect_metrics(analyzer, arrays)

    # Compare each to H=0.5
    for h_val in [0.1, 0.3, 0.7, 0.9]:
        n_sig, _ = compare(hurst_data[h_val], hurst_data[0.5])
        print(f"    H={h_val} vs H=0.5: {n_sig:2d} sig")

    return hurst_data


# =========================================================================
# D3: Process class distinction
# =========================================================================
def direction_3(all_data):
    print("\n" + "=" * 78)
    print("D3: PROCESS CLASS DISTINCTION")
    print("=" * 78)
    print("  Can geometry tell apart: AR, mean-reverting, regime-switching?")

    process_names = ['arma_21', 'ou_process', 'regime_switch']
    for i, n1 in enumerate(process_names):
        for n2 in process_names[i+1:]:
            n_sig, findings = compare(all_data[n1], all_data[n2])
            print(f"\n  {n1:15s} vs {n2:15s}: {n_sig} sig")
            for m, d, p in findings[:3]:
                print(f"    {m:45s}  d={d:+8.2f}")


# =========================================================================
# D4: Shuffle validation
# =========================================================================
def direction_4(analyzer, all_data):
    print("\n" + "=" * 78)
    print("D4: SHUFFLE VALIDATION — ORDERING VS DISTRIBUTION")
    print("=" * 78)

    results = {}
    for name, gen_fn in SIGNALS.items():
        arrays = [gen_fn(t, DATA_SIZE) for t in range(N_TRIALS)]
        shuf_arrays = []
        for arr in arrays:
            s = arr.copy()
            np.random.default_rng(1000).shuffle(s)
            shuf_arrays.append(s)
        shuf_data = collect_metrics(analyzer, shuf_arrays)

        n_sig, _ = compare(all_data[name], shuf_data)
        results[name] = n_sig
        label = "ordering-dep" if n_sig > 5 else "distributional"
        print(f"  {name:18s}: {n_sig:2d} sig (orig vs shuf) → {label}")

    return results


# =========================================================================
# D5: Sample rate invariance
# =========================================================================
def direction_5(analyzer):
    print("\n" + "=" * 78)
    print("D5: SAMPLE RATE INVARIANCE")
    print("=" * 78)
    print("  Same underlying process at different temporal resolutions.")
    print("  Subsample by factors of 1, 2, 4, 8 — does the signature persist?")

    for sig_name, gen_fn in [('brownian', gen_brownian),
                              ('heartbeat', gen_heartbeat),
                              ('network', gen_network)]:
        print(f"\n  {sig_name}:")
        base_arrays = [gen_fn(t, DATA_SIZE * 8) for t in range(N_TRIALS)]
        base_data = collect_metrics(analyzer, [a[:DATA_SIZE] for a in base_arrays])

        for factor in [2, 4, 8]:
            sub_arrays = [a[::factor][:DATA_SIZE] for a in base_arrays]
            sub_data = collect_metrics(analyzer, sub_arrays)
            n_sig, findings = compare(base_data, sub_data)
            print(f"    1x vs {factor}x subsample: {n_sig:2d} sig", end="")
            if findings:
                print(f"  best: {findings[0][0]:30s} d={findings[0][1]:+.2f}", end="")
            print()


# =========================================================================
# FIGURE
# =========================================================================
def make_figure(all_data, random_data, d1_results, d4_results, hurst_data):
    print("\nGenerating figure...", flush=True)

    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 24), facecolor='#181818')
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3,
                           height_ratios=[1.0, 1.0, 1.0, 1.0])

    names = list(SIGNALS.keys())
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3',
              '#9C27B0', '#00BCD4', '#FFC107', '#8BC34A', '#FF5722']

    # D1: Detection bar chart
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    sigs = [d1_results[n] for n in names]
    ax.bar(range(len(names)), sigs, color=colors[:len(names)], alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('Sig metrics vs random', fontsize=9)
    ax.set_title('D1: Detection — each signal vs iid random', fontsize=11, fontweight='bold')

    # D4: Shuffle validation
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    shuf_sigs = [d4_results.get(n, 0) for n in names]
    ax.bar(range(len(names)), shuf_sigs, color=colors[:len(names)], alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('Sig metrics (orig vs shuf)', fontsize=9)
    ax.set_title('D4: Ordering dependence', fontsize=11, fontweight='bold')

    # D2: Hurst sweep
    ax = _dark_ax(fig.add_subplot(gs[1, 0]))
    h_vals = sorted(hurst_data.keys())
    ref_data = hurst_data[0.5]
    h_sigs = []
    for h in h_vals:
        if h == 0.5:
            h_sigs.append(0)
        else:
            n_sig, _ = compare(hurst_data[h], ref_data)
            h_sigs.append(n_sig)
    ax.plot(h_vals, h_sigs, 'o-', color='#E91E63', linewidth=2, markersize=8)
    ax.set_xlabel('Hurst parameter H', fontsize=9)
    ax.set_ylabel('Sig metrics vs H=0.5', fontsize=9)
    ax.set_title('D2: Hurst parameter detection', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='#444', linewidth=0.5)

    # Sample signals
    ax = _dark_ax(fig.add_subplot(gs[1, 1]))
    rng = np.random.default_rng(42)
    for i, (name, gen_fn) in enumerate(list(SIGNALS.items())[:5]):
        sig_data = gen_fn(0, 500)
        offset = i * 60
        ax.plot(sig_data.astype(float) / 255 * 50 + offset,
                color=colors[i], linewidth=0.5, alpha=0.8, label=name)
    ax.legend(fontsize=6, facecolor='#333', edgecolor='#666', loc='upper right')
    ax.set_title('Sample signals (first 500 pts)', fontsize=11, fontweight='bold')
    ax.set_yticks([])

    # Pairwise matrix
    ax = _dark_ax(fig.add_subplot(gs[2, :]))
    n = len(names)
    mat = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j > i:
                n_sig, _ = compare(all_data[n1], all_data[n2])
                mat[i, j] = n_sig
                mat[j, i] = n_sig
    im = ax.imshow(mat, cmap='magma', interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                       fontsize=6, fontweight='bold',
                       color='white' if mat[i,j] > mat.max()/2 else '#aaa')
    ax.set_title('All pairwise comparisons (sig metrics)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Key metrics across signals
    ax = _dark_ax(fig.add_subplot(gs[3, :]))
    key_metrics = ['Tropical:slope', 'Fisher Information:det_fisher',
                   'Heisenberg (Nil):path_length', 'Wasserstein:transport_cost']
    x = np.arange(len(names))
    width = 0.2
    metric_colors = ['#E91E63', '#4CAF50', '#2196F3', '#FF9800']
    for k, metric in enumerate(key_metrics):
        means = []
        errs = []
        for name in names:
            vals = all_data[name].get(metric, [0])
            means.append(np.mean(vals))
            errs.append(np.std(vals))
        ax.bar(x + k * width, means, width, yerr=errs, capsize=2,
               color=metric_colors[k], alpha=0.8, label=metric.split(':')[-1])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666', loc='upper right')
    ax.set_title('Key metrics across all signals', fontsize=11, fontweight='bold')

    fig.suptitle('Time Series: Geometric Signatures of Stochastic Processes',
                 fontsize=14, fontweight='bold', color='white', y=0.995)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'time_series.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved time_series.png")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    all_data, random_data, d1_results = direction_1(analyzer)
    hurst_data = direction_2(analyzer, all_data)
    direction_3(all_data)
    d4_results = direction_4(analyzer, all_data)
    direction_5(analyzer)
    make_figure(all_data, random_data, d1_results, d4_results, hurst_data)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    ranked = sorted(d1_results.items(), key=lambda x: -x[1])
    for name, n_sig in ranked:
        shuf = d4_results.get(name, 0)
        label = "ordering-dep" if shuf > 5 else "distributional"
        print(f"  {name:18s}: {n_sig:3d} sig vs random, {shuf:2d} ordering  [{label}]")
