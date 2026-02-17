#!/usr/bin/env python3
"""
Investigation: Nonlinearity Detection Under Noise
==================================================

The surrogate investigation proved exotic geometries detect nonlinear structure
that simple features cannot. But real-world signals are noisy. This investigation
asks: at what signal-to-noise ratio does detection fail?

We add increasing Gaussian noise to signals with known nonlinear structure,
generate IAAFT surrogates of the noisy signal, and measure how many framework
metrics still distinguish original from surrogate. We compare against the
time-reversal asymmetry statistic (Trev) — the standard nonlinear test
from the surrogate literature.

Directions:
  D1: SNR sweep — detection count vs noise level for 4 signals
  D2: Framework vs standard Trev test at each noise level
  D3: Which geometries are most noise-robust?
  D4: Sequence length dependence at moderate noise
  D5: Detection probability curves — fraction of trials with ≥1 detection

Methodology: N_TRIALS=20, DATA_SIZE=2000, Cohen's d > 0.8, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from collections import defaultdict
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 20
DATA_SIZE = 2000
ALPHA = 0.05
IAAFT_ITERATIONS = 100

# Noise levels: std(noise) / std(signal)
# 0=clean, 0.1=mild, 0.5=-6dB, 1.0=0dB, 2.0=+6dB noise, 5.0=noise dominated
NOISE_FRACTIONS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


# =========================================================================
# SURROGATE GENERATION (same as surrogate.py)
# =========================================================================

def iaaft_surrogate(data, rng, n_iter=IAAFT_ITERATIONS):
    """IAAFT surrogate: preserves spectrum AND distribution."""
    data_f = data.astype(np.float64)
    n = len(data_f)
    target_fft = np.fft.rfft(data_f)
    target_amplitudes = np.abs(target_fft)
    sorted_data = np.sort(data_f)
    surrogate = data_f.copy()
    rng.shuffle(surrogate)
    for _ in range(n_iter):
        surr_fft = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_fft)
        matched_fft = target_amplitudes * np.exp(1j * surr_phases)
        surrogate = np.fft.irfft(matched_fft, n=n)
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_data[rank_order]
    return np.clip(surrogate, 0, 255).astype(np.uint8)


# =========================================================================
# SIGNAL GENERATORS (float domain, noise added before quantization)
# =========================================================================

def gen_lorenz_float(trial, size):
    """Lorenz attractor x-component as float."""
    dt = 0.01
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rng = np.random.default_rng(42 + trial)
    x, y, z = rng.uniform(-1, 1, 3) + np.array([1.0, 1.0, 1.0])
    warmup = 5000
    vals = []
    for _ in range(warmup + size * 5):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx; y += dy; z += dz
        vals.append(x)
    return np.array(vals[warmup::5][:size], dtype=np.float64)


def gen_henon_float(trial, size):
    """Hénon map x-component as float."""
    rng = np.random.default_rng(42 + trial)
    x, y = 0.1 + 0.001 * rng.uniform(), 0.1
    warmup = 500
    vals = []
    for _ in range(warmup + size):
        x_new = 1 - 1.4 * x * x + y
        y = 0.3 * x
        x = x_new
        if abs(x) > 1e6:
            x, y = 0.1, 0.1
        vals.append(x)
    return np.array(vals[warmup:warmup + size], dtype=np.float64)


def gen_logistic_float(trial, size):
    """Logistic map r=3.99 as float."""
    rng = np.random.default_rng(42 + trial)
    x = 0.1 + 0.01 * rng.uniform()
    warmup = 500
    vals = []
    for _ in range(warmup + size):
        x = 3.99 * x * (1 - x)
        vals.append(x)
    return np.array(vals[warmup:warmup + size], dtype=np.float64)


def gen_heartbeat_float(trial, size):
    """Synthetic ECG as float."""
    rng = np.random.default_rng(42 + trial)
    period = 60 + rng.integers(-10, 10)
    t = np.arange(size, dtype=np.float64)
    ecg = np.zeros(size)
    for beat_start in range(0, size, period):
        qrs_c = beat_start + period * 0.4
        ecg += 3.0 * np.exp(-0.5 * ((t - qrs_c) / max(period * 0.03, 1)) ** 2)
        t_c = beat_start + period * 0.65
        ecg += 0.8 * np.exp(-0.5 * ((t - t_c) / max(period * 0.08, 1)) ** 2)
    ecg += 0.1 * rng.standard_normal(size)
    return ecg


SIGNALS = {
    'lorenz':    gen_lorenz_float,
    'henon':     gen_henon_float,
    'logistic':  gen_logistic_float,
    'heartbeat': gen_heartbeat_float,
}


def to_uint8(x):
    """Normalize float signal to uint8."""
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-10:
        return np.full(len(x), 128, dtype=np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def add_noise(signal_float, noise_fraction, rng):
    """Add Gaussian noise at specified fraction of signal std."""
    if noise_fraction <= 0:
        return signal_float.copy()
    sig_std = np.std(signal_float)
    if sig_std < 1e-15:
        sig_std = 1.0
    noise = rng.standard_normal(len(signal_float)) * noise_fraction * sig_std
    return signal_float + noise


# =========================================================================
# STANDARD NONLINEAR TEST STATISTICS
# =========================================================================

def time_reversal_asymmetry(data, tau=1):
    """Time-reversal asymmetry: zero for all stationary linear Gaussian processes.
    Non-zero indicates nonlinearity or non-Gaussianity.
    T_rev = mean((x[t+tau] - x[t-tau])^3) / mean((x[t+tau] - x[t-tau])^2)^{3/2}
    """
    x = data.astype(np.float64)
    n = len(x) - 2 * tau
    if n < 10:
        return 0.0
    diffs = x[2 * tau:] - x[:n]
    m2 = np.mean(diffs ** 2)
    if m2 < 1e-20:
        return 0.0
    return float(np.mean(diffs ** 3) / m2 ** 1.5)


def delayed_mutual_info_ratio(data, tau=1):
    """Ratio of nonlinear to linear dependence at lag tau.
    MI / (0.5 * log(1 / (1 - r^2))) where r = linear autocorrelation.
    Equals 1 for Gaussian linear processes. >1 indicates nonlinear dependence."""
    x = data.astype(np.float64)
    if len(x) - tau < 20:
        return 1.0
    x1 = x[:-tau]
    x2 = x[tau:]

    # Linear autocorrelation
    r = np.corrcoef(x1, x2)[0, 1]
    if abs(r) > 0.999:
        return 1.0
    linear_mi = -0.5 * np.log(1 - r ** 2)

    # Nonlinear MI via histogram estimator
    n_bins = max(int(np.sqrt(len(x1) / 5)), 10)
    hist2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    if linear_mi < 1e-10:
        return 1.0 if mi < 1e-10 else float('inf')
    return float(mi / linear_mi)


STANDARD_TESTS = {
    'time_reversal_tau1': lambda d: time_reversal_asymmetry(d, tau=1),
    'time_reversal_tau2': lambda d: time_reversal_asymmetry(d, tau=2),
    'mi_ratio_tau1': lambda d: delayed_mutual_info_ratio(d, tau=1),
}
N_STANDARD = len(STANDARD_TESTS)
STANDARD_NAMES = list(STANDARD_TESTS.keys())


# =========================================================================
# UTILITIES
# =========================================================================

_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_FRAMEWORK = ALPHA / N_METRICS
BONF_STANDARD = ALPHA / N_STANDARD
del _analyzer, _dummy, _r, _mn

print(f"Framework: {N_METRICS} metrics (Bonferroni α={BONF_FRAMEWORK:.2e})")
print(f"Standard tests: {N_STANDARD} (Bonferroni α={BONF_STANDARD:.2e})")
print(f"Noise levels: {NOISE_FRACTIONS}")


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def count_sig(data_a, data_b, feature_names, alpha):
    """Count significant metrics and return sorted findings."""
    sig = 0
    findings = []
    for f in feature_names:
        a = np.array(data_a.get(f, []))
        b = np.array(data_b.get(f, []))
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < alpha and abs(d) > 0.8:
            sig += 1
            findings.append((f, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


def collect_framework(analyzer, data_arrays):
    """Collect all framework metrics across trials."""
    out = {m: [] for m in METRIC_NAMES}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def collect_standard(data_arrays):
    """Collect standard nonlinear test statistics across trials."""
    out = {name: [] for name in STANDARD_NAMES}
    for arr in data_arrays:
        for name, fn in STANDARD_TESTS.items():
            v = fn(arr)
            if np.isfinite(v):
                out[name].append(v)
    return out


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# CORE: SWEEP NOISE LEVELS
# =========================================================================

def noise_sweep(analyzer, n_trials=N_TRIALS, data_size=DATA_SIZE):
    """For each signal × noise level, compute framework + standard test metrics
    for original vs IAAFT surrogates. Returns structured results."""

    results = {}

    for sig_name, gen_fn in SIGNALS.items():
        print(f"\n  {sig_name}:", flush=True)
        results[sig_name] = {}

        for nf in NOISE_FRACTIONS:
            snr_db = -20 * np.log10(nf) if nf > 0 else float('inf')
            print(f"    noise={nf:.2f} (SNR={snr_db:+.0f}dB)...", end=" ", flush=True)

            originals = []
            surrogates = []

            for t in range(n_trials):
                # Generate clean signal
                clean = gen_fn(t, data_size)
                # Add noise in float domain, then quantize
                rng_noise = np.random.default_rng(8000 + t)
                noisy = add_noise(clean, nf, rng_noise)
                noisy_uint8 = to_uint8(noisy)
                originals.append(noisy_uint8)

                # Generate IAAFT surrogate of the noisy signal
                rng_surr = np.random.default_rng(9000 + t)
                surr = iaaft_surrogate(noisy_uint8, rng_surr)
                surrogates.append(surr)

            # Collect metrics
            orig_fw = collect_framework(analyzer, originals)
            surr_fw = collect_framework(analyzer, surrogates)
            orig_std = collect_standard(originals)
            surr_std = collect_standard(surrogates)

            # Count significant
            fw_sig, fw_findings = count_sig(orig_fw, surr_fw, METRIC_NAMES, BONF_FRAMEWORK)
            std_sig, std_findings = count_sig(orig_std, surr_std, STANDARD_NAMES, BONF_STANDARD)

            results[sig_name][nf] = {
                'fw_sig': fw_sig,
                'fw_findings': fw_findings,
                'std_sig': std_sig,
                'std_findings': std_findings,
                'orig_fw': orig_fw,
                'surr_fw': surr_fw,
            }
            print(f"fw={fw_sig}, std={std_sig}")

    return results


# =========================================================================
# D1: SNR SWEEP
# =========================================================================

def direction_1(results):
    print("\n" + "=" * 78)
    print("D1: SNR SWEEP — DETECTION COUNT VS NOISE LEVEL")
    print("=" * 78)

    for sig_name in SIGNALS:
        print(f"\n  {sig_name}:")
        print(f"    {'Noise':>8s}  {'SNR dB':>8s}  {'Framework':>10s}  {'Standard':>10s}")
        print(f"    {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 10}")
        for nf in NOISE_FRACTIONS:
            r = results[sig_name][nf]
            snr_db = -20 * np.log10(nf) if nf > 0 else float('inf')
            snr_str = f"{snr_db:+.0f}" if np.isfinite(snr_db) else "  ∞"
            print(f"    {nf:8.2f}  {snr_str:>8s}  {r['fw_sig']:10d}  {r['std_sig']:10d}")


# =========================================================================
# D2: FRAMEWORK VS STANDARD TEST — THRESHOLD COMPARISON
# =========================================================================

def direction_2(results):
    print("\n" + "=" * 78)
    print("D2: DETECTION THRESHOLD — FRAMEWORK VS TIME-REVERSAL ASYMMETRY")
    print("=" * 78)
    print("  At what noise level does each method lose detection?")
    print("  Threshold = highest noise where ≥1 metric is significant.\n")

    thresholds = {}
    for sig_name in SIGNALS:
        fw_thresh = 0.0
        std_thresh = 0.0
        for nf in NOISE_FRACTIONS:
            r = results[sig_name][nf]
            if r['fw_sig'] > 0:
                fw_thresh = nf
            if r['std_sig'] > 0:
                std_thresh = nf

        thresholds[sig_name] = {'framework': fw_thresh, 'standard': std_thresh}

        fw_db = -20 * np.log10(fw_thresh) if fw_thresh > 0 else float('inf')
        std_db = -20 * np.log10(std_thresh) if std_thresh > 0 else float('inf')
        advantage = fw_thresh / std_thresh if std_thresh > 0 else float('inf')

        fw_str = f"{fw_thresh:.2f} ({fw_db:+.0f}dB)" if fw_thresh > 0 else "clean only"
        std_str = f"{std_thresh:.2f} ({std_db:+.0f}dB)" if std_thresh > 0 else "never"

        print(f"  {sig_name:15s}:")
        print(f"    Framework:  detects through noise={fw_str}")
        print(f"    Standard:   detects through noise={std_str}")
        if advantage > 1:
            print(f"    → Framework tolerates {advantage:.1f}x more noise")
        elif advantage < 1:
            print(f"    → Standard test is more robust here")
        elif std_thresh == 0 and fw_thresh > 0:
            print(f"    → Only framework detects (standard never does)")

    return thresholds


# =========================================================================
# D3: MOST NOISE-ROBUST GEOMETRIES
# =========================================================================

def direction_3(results):
    print("\n" + "=" * 78)
    print("D3: MOST NOISE-ROBUST GEOMETRIES")
    print("=" * 78)
    print("  Which geometries retain detection at the highest noise levels?")
    print("  Score = sum of significant metrics across signals × noise levels.\n")

    geometry_names = sorted(set(m.split(':')[0] for m in METRIC_NAMES))
    geo_to_metrics = {g: [m for m in METRIC_NAMES if m.startswith(g + ':')]
                      for g in geometry_names}

    # Score each geometry: sum significant metrics across all conditions
    geo_scores = {g: 0 for g in geometry_names}
    # Also track at which noise levels each geometry detects
    geo_max_noise = {g: 0.0 for g in geometry_names}

    for sig_name in SIGNALS:
        for nf in NOISE_FRACTIONS:
            r = results[sig_name][nf]
            for geo in geometry_names:
                metrics = geo_to_metrics[geo]
                bonf_geo = ALPHA / max(len(metrics), 1)
                n_sig, _ = count_sig(r['orig_fw'], r['surr_fw'], metrics, bonf_geo)
                geo_scores[geo] += n_sig
                if n_sig > 0:
                    geo_max_noise[geo] = max(geo_max_noise[geo], nf)

    ranked = sorted(geo_scores.items(), key=lambda x: -x[1])

    print(f"  {'Geometry':<40s} {'Total sig':>10s} {'Max noise':>10s}")
    print(f"  {'─' * 40} {'─' * 10} {'─' * 10}")
    for geo, score in ranked[:20]:
        max_nf = geo_max_noise[geo]
        max_str = f"{max_nf:.2f}" if max_nf > 0 else "clean"
        bar = '█' * min(score, 40)
        print(f"  {geo:<40s} {score:>10d} {max_str:>10s}  {bar}")

    return geo_scores, geo_max_noise


# =========================================================================
# D4: SEQUENCE LENGTH DEPENDENCE
# =========================================================================

def direction_4(analyzer):
    print("\n" + "=" * 78)
    print("D4: SEQUENCE LENGTH AT MODERATE NOISE (noise_fraction=0.5)")
    print("=" * 78)
    print("  Does longer data help the framework detect nonlinearity through noise?")

    noise_frac = 0.5
    lengths = [500, 1000, 2000, 4000, 8000]

    d4 = {}
    for sig_name in ['lorenz', 'henon', 'logistic']:
        gen_fn = SIGNALS[sig_name]
        print(f"\n  {sig_name}:")
        d4[sig_name] = {}

        for data_size in lengths:
            print(f"    N={data_size:5d}...", end=" ", flush=True)

            originals = []
            surrogates = []
            for t in range(N_TRIALS):
                clean = gen_fn(t, data_size)
                noisy = add_noise(clean, noise_frac, np.random.default_rng(8000 + t))
                noisy_uint8 = to_uint8(noisy)
                originals.append(noisy_uint8)
                surrogates.append(iaaft_surrogate(noisy_uint8, np.random.default_rng(9000 + t)))

            orig_fw = collect_framework(analyzer, originals)
            surr_fw = collect_framework(analyzer, surrogates)
            orig_std = collect_standard(originals)
            surr_std = collect_standard(surrogates)

            fw_sig, _ = count_sig(orig_fw, surr_fw, METRIC_NAMES, BONF_FRAMEWORK)
            std_sig, _ = count_sig(orig_std, surr_std, STANDARD_NAMES, BONF_STANDARD)

            d4[sig_name][data_size] = {'fw_sig': fw_sig, 'std_sig': std_sig}
            print(f"fw={fw_sig}, std={std_sig}")

    return d4


# =========================================================================
# D5: DETECTION PROBABILITY CURVES
# =========================================================================

def direction_5(results):
    print("\n" + "=" * 78)
    print("D5: DETECTION PROBABILITY — WHAT FRACTION OF METRICS SURVIVE?")
    print("=" * 78)
    print("  Framework detection rate = sig_metrics / total_metrics")
    print("  Provides a smooth degradation curve.\n")

    d5 = {}
    for sig_name in SIGNALS:
        print(f"  {sig_name}:")
        d5[sig_name] = {}
        for nf in NOISE_FRACTIONS:
            r = results[sig_name][nf]
            fw_rate = r['fw_sig'] / N_METRICS
            snr_db = -20 * np.log10(nf) if nf > 0 else float('inf')
            snr_str = f"{snr_db:+.0f}dB" if np.isfinite(snr_db) else "  ∞"
            d5[sig_name][nf] = fw_rate
            print(f"    noise={nf:.2f} ({snr_str:>6s}): "
                  f"{r['fw_sig']:3d}/{N_METRICS} = {fw_rate:.1%} detection rate")

    return d5


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(results, thresholds, geo_scores, geo_max_noise, d4, d5):
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

    fig = plt.figure(figsize=(20, 22), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.12, height_ratios=[1.0, 1.0, 1.0])

    signal_colors = {
        'lorenz': '#E91E63',
        'henon': '#2196F3',
        'logistic': '#4CAF50',
        'heartbeat': '#FF9800',
    }

    # D1: SNR sweep curves
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    for sig_name in SIGNALS:
        nfs = [nf for nf in NOISE_FRACTIONS if nf > 0]
        fw_vals = [results[sig_name][nf]['fw_sig'] for nf in nfs]
        # Add clean point at x=0.01 for log scale
        snr_dbs = [-20 * np.log10(nf) for nf in nfs]
        ax.plot(snr_dbs, fw_vals, 'o-', color=signal_colors[sig_name],
                label=sig_name, linewidth=2, markersize=5)
    # Add clean values as rightmost points
    clean_x = 40  # represent infinity as 40 dB
    for sig_name in SIGNALS:
        clean_val = results[sig_name][0.0]['fw_sig']
        ax.plot(clean_x, clean_val, 's', color=signal_colors[sig_name],
                markersize=8, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('SNR (dB) →', fontsize=9)
    ax.set_ylabel('Significant metrics vs IAAFT', fontsize=9)
    ax.set_title('D1: Framework detection vs noise level', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')
    ax.axhline(y=0, color='#666', linestyle='--', linewidth=0.5)
    ax.axvline(x=37, color='#555', linestyle=':', linewidth=0.5)
    ax.annotate('clean', (37.5, ax.get_ylim()[1] * 0.95), fontsize=7, color='#aaa',
                ha='left', va='top')

    # D2: Framework vs standard thresholds
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    sig_names = list(SIGNALS.keys())
    x = np.arange(len(sig_names))
    fw_thresh = [thresholds[n]['framework'] for n in sig_names]
    std_thresh = [thresholds[n]['standard'] for n in sig_names]
    ax.bar(x - 0.18, fw_thresh, 0.32, color='#2196F3', alpha=0.85, label='Framework')
    ax.bar(x + 0.18, std_thresh, 0.32, color='#FF9800', alpha=0.85, label='Standard (Trev)')
    ax.set_xticks(x)
    ax.set_xticklabels(sig_names, fontsize=8)
    ax.set_ylabel('Max noise-to-signal ratio tolerated', fontsize=9)
    ax.set_title('D2: Noise tolerance — framework vs standard test', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # D3: Geometry robustness ranking (top 15)
    ax = _dark_ax(fig.add_subplot(gs[1, 0]))
    ranked = sorted(geo_scores.items(), key=lambda x: -x[1])[:15]
    geo_names = [g[0] for g in ranked]
    geo_vals = [g[1] for g in ranked]
    colors = ['#2196F3' if v > 5 else '#4CAF50' if v > 0 else '#666' for v in geo_vals]
    ax.barh(range(len(geo_names)), geo_vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(geo_names)))
    ax.set_yticklabels(geo_names, fontsize=7.5)
    ax.set_xlabel('Total sig detections (across signals × noise levels)', fontsize=8)
    ax.set_title('D3: Most noise-robust geometries', fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    # D4: Sequence length dependence
    ax = _dark_ax(fig.add_subplot(gs[1, 1]))
    for sig_name in d4:
        lengths = sorted(d4[sig_name].keys())
        fw_vals = [d4[sig_name][n]['fw_sig'] for n in lengths]
        ax.plot(lengths, fw_vals, 'o-', color=signal_colors[sig_name],
                label=sig_name, linewidth=2, markersize=5)
    ax.set_xlabel('Sequence length N', fontsize=9)
    ax.set_ylabel('Significant metrics vs IAAFT', fontsize=9)
    ax.set_title('D4: Length dependence at noise=0.5 (SNR +6dB)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')
    ax.set_xscale('log')

    # D5: Detection rate curves
    ax = _dark_ax(fig.add_subplot(gs[2, :]))
    for sig_name in SIGNALS:
        nfs = NOISE_FRACTIONS[1:]  # skip 0 for log scale
        rates = [d5[sig_name][nf] for nf in nfs]
        snr_dbs = [-20 * np.log10(nf) for nf in nfs]
        ax.plot(snr_dbs, [r * 100 for r in rates], 'o-', color=signal_colors[sig_name],
                label=sig_name, linewidth=2.5, markersize=6)
    # Add clean
    for sig_name in SIGNALS:
        clean_rate = d5[sig_name][0.0] * 100
        ax.plot(40, clean_rate, 's', color=signal_colors[sig_name],
                markersize=9, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('SNR (dB) →', fontsize=10)
    ax.set_ylabel('Detection rate (%)', fontsize=10)
    ax.set_title('D5: What fraction of nonlinear structure survives noise?',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#333', edgecolor='#666')
    ax.axhline(y=0, color='#666', linestyle='--', linewidth=0.5)
    ax.axvline(x=37, color='#555', linestyle=':', linewidth=0.5)
    ax.annotate('clean', (37.5, ax.get_ylim()[1] * 0.95), fontsize=8, color='#aaa',
                ha='left', va='top')

    fig.suptitle('Nonlinearity Detection Under Noise',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    gs.update(top=0.95)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'noise_robustness.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved noise_robustness.png")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    print("\nPhase 1: SNR sweep (4 signals × 8 noise levels × 20 trials)...")
    results = noise_sweep(analyzer)

    direction_1(results)
    thresholds = direction_2(results)
    geo_scores, geo_max_noise = direction_3(results)

    print("\nPhase 2: Sequence length sweep (3 signals × 5 lengths × 20 trials)...")
    d4 = direction_4(analyzer)

    d5 = direction_5(results)

    make_figure(results, thresholds, geo_scores, geo_max_noise, d4, d5)

    # Summary
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")

    for sig_name in SIGNALS:
        clean = results[sig_name][0.0]['fw_sig']
        fw_t = thresholds[sig_name]['framework']
        std_t = thresholds[sig_name]['standard']
        fw_db = f"{-20 * np.log10(fw_t):+.0f}dB" if fw_t > 0 else "clean only"
        std_db = f"{-20 * np.log10(std_t):+.0f}dB" if std_t > 0 else "never"
        print(f"  {sig_name:15s}: clean={clean:3d} sig, "
              f"framework survives to {fw_db}, standard to {std_db}")

    # Key finding
    total_fw_advantage = 0
    for sig_name in SIGNALS:
        fw_t = thresholds[sig_name]['framework']
        std_t = thresholds[sig_name]['standard']
        if fw_t > std_t:
            total_fw_advantage += 1
        elif fw_t == std_t and fw_t > 0:
            pass

    n_signals = len(SIGNALS)
    print(f"\n  Framework more noise-robust than standard test: "
          f"{total_fw_advantage}/{n_signals} signals")
