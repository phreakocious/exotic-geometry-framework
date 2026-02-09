#!/usr/bin/env python3
"""
Investigation: Deep Noise Analysis — Exploring Anomalies
=========================================================

Follow-up to noise_robustness.py, which found several surprising results:

1. Spiral geometry detects nonlinearity at noise=5.0 (-14dB) — everything
   else is dead by noise=2.0. Is this real?
2. Heartbeat barely degrades through noise=0.5, then 7 sig at noise=2.0.
   Which specific metrics are the "last survivors"?
3. Heartbeat goes 26→28 at noise=0.05 — stochastic resonance?
4. Different geometries dominate at different SNR regimes.

Directions:
  D1: Last survivors — which specific metrics survive at each signal's
      highest detectable noise level?
  D2: Spiral anomaly — validate with more trials and examine what it measures
  D3: Stochastic resonance — fine-grained noise sweep near 0 for heartbeat
  D4: Optimal geometry per SNR regime — who's best at clean / moderate / extreme?
  D5: Metric survival curves — individual metric degradation profiles

Methodology: N_TRIALS=25 for anomaly validation, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05
IAAFT_ITERATIONS = 100


# =========================================================================
# INFRASTRUCTURE (shared with noise_robustness.py)
# =========================================================================

def iaaft_surrogate(data, rng, n_iter=IAAFT_ITERATIONS):
    data_f = data.astype(np.float64)
    n = len(data_f)
    target_amplitudes = np.abs(np.fft.rfft(data_f))
    sorted_data = np.sort(data_f)
    surrogate = data_f.copy()
    rng.shuffle(surrogate)
    for _ in range(n_iter):
        surr_fft = np.fft.rfft(surrogate)
        matched_fft = target_amplitudes * np.exp(1j * np.angle(surr_fft))
        surrogate = np.fft.irfft(matched_fft, n=n)
        surrogate = sorted_data[np.argsort(np.argsort(surrogate))]
    return np.clip(surrogate, 0, 255).astype(np.uint8)


def to_uint8(x):
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-10:
        return np.full(len(x), 128, dtype=np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def add_noise(signal_float, noise_fraction, rng):
    if noise_fraction <= 0:
        return signal_float.copy()
    sig_std = max(np.std(signal_float), 1e-15)
    return signal_float + rng.standard_normal(len(signal_float)) * noise_fraction * sig_std


def gen_lorenz_float(trial, size):
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
    rng = np.random.default_rng(42 + trial)
    x = 0.1 + 0.01 * rng.uniform()
    warmup = 500
    vals = []
    for _ in range(warmup + size):
        x = 3.99 * x * (1 - x)
        vals.append(x)
    return np.array(vals[warmup:warmup + size], dtype=np.float64)


def gen_heartbeat_float(trial, size):
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
    'lorenz': gen_lorenz_float,
    'henon': gen_henon_float,
    'logistic': gen_logistic_float,
    'heartbeat': gen_heartbeat_float,
}


# Framework setup
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def run_comparison(analyzer, gen_fn, noise_fraction, n_trials=N_TRIALS):
    """Run framework on original vs IAAFT at given noise level.
    Returns dict of {metric: (d_value, p_value, is_sig)} for all metrics."""
    originals = []
    surrogates = []
    for t in range(n_trials):
        clean = gen_fn(t, DATA_SIZE)
        noisy = add_noise(clean, noise_fraction, np.random.default_rng(8000 + t))
        noisy_u8 = to_uint8(noisy)
        originals.append(noisy_u8)
        surrogates.append(iaaft_surrogate(noisy_u8, np.random.default_rng(9000 + t)))

    # Collect all metric values
    orig_vals = {m: [] for m in METRIC_NAMES}
    surr_vals = {m: [] for m in METRIC_NAMES}

    for arr in originals:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in orig_vals and np.isfinite(mv):
                    orig_vals[key].append(mv)

    for arr in surrogates:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in surr_vals and np.isfinite(mv):
                    surr_vals[key].append(mv)

    results = {}
    n_sig = 0
    for m in METRIC_NAMES:
        a = np.array(orig_vals.get(m, []))
        b = np.array(surr_vals.get(m, []))
        if len(a) < 3 or len(b) < 3:
            results[m] = (0.0, 1.0, False)
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        is_sig = p < BONF and abs(d) > 0.8
        results[m] = (d, p, is_sig)
        if is_sig:
            n_sig += 1

    return results, n_sig


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# D1: LAST SURVIVORS — WHAT METRICS PERSIST AT EXTREME NOISE?
# =========================================================================

def direction_1(analyzer):
    print("=" * 78)
    print("D1: LAST SURVIVORS — WHICH METRICS PERSIST AT EXTREME NOISE?")
    print("=" * 78)
    print("  For each signal, find the highest noise level with ≥1 detection,")
    print("  then list every surviving metric.\n")

    # Test at the noise levels where we know things get interesting
    test_levels = {
        'lorenz':    [0.5, 1.0],
        'henon':     [0.5, 1.0],
        'logistic':  [0.5, 1.0],
        'heartbeat': [1.0, 2.0],
    }

    d1 = {}
    for sig_name, levels in test_levels.items():
        gen_fn = SIGNALS[sig_name]
        d1[sig_name] = {}
        for nf in levels:
            snr_db = -20 * np.log10(nf) if nf > 0 else float('inf')
            print(f"  {sig_name} @ noise={nf} ({snr_db:+.0f}dB)...", end=" ", flush=True)

            results, n_sig = run_comparison(analyzer, gen_fn, nf)
            survivors = [(m, d, p) for m, (d, p, sig) in results.items() if sig]
            survivors.sort(key=lambda x: -abs(x[1]))
            d1[sig_name][nf] = survivors

            print(f"{n_sig} sig")
            for m, d, p in survivors[:10]:
                geo = m.split(':')[0]
                metric = m.split(':', 1)[1]
                print(f"    {geo:35s} | {metric:30s}  d={d:+8.2f}")
            if len(survivors) > 10:
                print(f"    ... and {len(survivors) - 10} more")

    return d1


# =========================================================================
# D2: SPIRAL ANOMALY — IS noise=5.0 DETECTION REAL?
# =========================================================================

def direction_2(analyzer):
    print("\n" + "=" * 78)
    print("D2: SPIRAL ANOMALY — VALIDATING EXTREME NOISE DETECTION")
    print("=" * 78)
    print("  Spiral geometry reportedly detects nonlinearity at noise=5.0 (-14dB).")
    print("  Testing with N=30 trials for stronger statistics.\n")

    spiral_metrics = [m for m in METRIC_NAMES if m.startswith('Spiral')]
    print(f"  Spiral metrics: {spiral_metrics}\n")

    bonf_spiral = ALPHA / max(len(spiral_metrics), 1)

    d2 = {}
    for sig_name in SIGNALS:
        gen_fn = SIGNALS[sig_name]
        print(f"  {sig_name}:")

        for nf in [2.0, 5.0, 10.0]:
            print(f"    noise={nf:.1f}...", end=" ", flush=True)
            # Run with more trials for validation
            results, total_sig = run_comparison(analyzer, gen_fn, nf, n_trials=30)

            # Check Spiral specifically
            spiral_results = {m: results[m] for m in spiral_metrics}
            spiral_sig = [(m, d, p) for m, (d, p, sig) in spiral_results.items()
                          if p < bonf_spiral and abs(d) > 0.8]
            spiral_sig.sort(key=lambda x: -abs(x[1]))

            d2[(sig_name, nf)] = {
                'total_sig': total_sig,
                'spiral_sig': spiral_sig,
                'spiral_results': spiral_results,
            }

            # Also check what other geometries survive
            other_sig = [(m, d, p) for m, (d, p, sig) in results.items()
                         if sig and not m.startswith('Spiral')]

            print(f"total={total_sig}, spiral={len(spiral_sig)}, other={len(other_sig)}")

            for m, d, p in spiral_sig:
                metric = m.split(':', 1)[1]
                print(f"      Spiral:{metric:25s}  d={d:+8.2f}  p={p:.2e}")
            for m, d, p in other_sig[:3]:
                print(f"      {m:45s}  d={d:+8.2f}")

    return d2


# =========================================================================
# D3: STOCHASTIC RESONANCE — DOES A LITTLE NOISE HELP?
# =========================================================================

def direction_3(analyzer):
    print("\n" + "=" * 78)
    print("D3: STOCHASTIC RESONANCE — DOES NOISE EVER IMPROVE DETECTION?")
    print("=" * 78)
    print("  Heartbeat went 26→28 sig when noise=0.05 was added.")
    print("  Fine-grained sweep to test if this is real.\n")

    fine_noise = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]

    d3 = {}
    for sig_name in ['heartbeat', 'lorenz']:
        gen_fn = SIGNALS[sig_name]
        print(f"  {sig_name}:")
        d3[sig_name] = {}

        for nf in fine_noise:
            print(f"    noise={nf:.2f}...", end=" ", flush=True)
            results, n_sig = run_comparison(analyzer, gen_fn, nf)
            d3[sig_name][nf] = n_sig
            print(f"{n_sig} sig")

    # Check for stochastic resonance: peak > clean
    for sig_name in d3:
        clean = d3[sig_name][0.0]
        peak = max(d3[sig_name].values())
        peak_nf = max(d3[sig_name], key=d3[sig_name].get)
        if peak > clean:
            print(f"\n  ★ {sig_name}: STOCHASTIC RESONANCE detected!")
            print(f"    Clean={clean} sig, Peak={peak} sig at noise={peak_nf:.2f}")
            print(f"    Gain: +{peak - clean} metrics ({(peak-clean)/clean*100:.0f}% improvement)")
        else:
            print(f"\n  {sig_name}: No stochastic resonance (clean={clean} is already optimal)")

    return d3


# =========================================================================
# D4: OPTIMAL GEOMETRY PER SNR REGIME
# =========================================================================

def direction_4(analyzer):
    print("\n" + "=" * 78)
    print("D4: OPTIMAL GEOMETRY PER SNR REGIME")
    print("=" * 78)
    print("  Which geometry is best at: clean / moderate / extreme noise?")
    print("  Using Hénon (strongest nonlinear signal).\n")

    gen_fn = SIGNALS['henon']
    geometry_names = sorted(set(m.split(':')[0] for m in METRIC_NAMES))
    geo_to_metrics = {g: [m for m in METRIC_NAMES if m.startswith(g + ':')]
                      for g in geometry_names}

    regimes = {
        'clean':    0.0,
        'mild':     0.1,
        'moderate': 0.5,
        'heavy':    1.0,
        'extreme':  2.0,
    }

    d4 = {}
    for regime_name, nf in regimes.items():
        snr_db = -20 * np.log10(nf) if nf > 0 else float('inf')
        snr_str = f"{snr_db:+.0f}dB" if np.isfinite(snr_db) else "∞"
        print(f"  {regime_name} (noise={nf}, SNR={snr_str})...", end=" ", flush=True)

        results, total = run_comparison(analyzer, gen_fn, nf)

        # Score each geometry by detection count
        geo_counts = {}
        for geo in geometry_names:
            metrics = geo_to_metrics[geo]
            bonf_geo = ALPHA / max(len(metrics), 1)
            count = sum(1 for m in metrics
                        if results[m][1] < bonf_geo and abs(results[m][0]) > 0.8)
            geo_counts[geo] = count

        ranked = sorted(geo_counts.items(), key=lambda x: -x[1])
        d4[regime_name] = {'total': total, 'geo_counts': geo_counts, 'ranked': ranked}
        print(f"total={total}")

        # Top 5
        for geo, count in ranked[:5]:
            n_metrics = len(geo_to_metrics[geo])
            print(f"    {geo:40s}  {count}/{n_metrics}")

    return d4


# =========================================================================
# D5: METRIC SURVIVAL CURVES — PER-METRIC DEGRADATION
# =========================================================================

def direction_5(analyzer):
    print("\n" + "=" * 78)
    print("D5: METRIC SURVIVAL CURVES")
    print("=" * 78)
    print("  Track individual metrics across noise levels to identify")
    print("  the most robust nonlinear detectors.\n")

    noise_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    # Use Hénon (richest signal) for tracking
    gen_fn = SIGNALS['henon']
    d5_data = {}

    for nf in noise_levels:
        print(f"  noise={nf:.2f}...", end=" ", flush=True)
        results, n_sig = run_comparison(analyzer, gen_fn, nf)
        d5_data[nf] = results
        print(f"{n_sig} sig")

    # For each metric, compute its "survival profile"
    survival = {}
    for m in METRIC_NAMES:
        sig_levels = []
        for nf in noise_levels:
            d, p, is_sig = d5_data[nf][m]
            if is_sig:
                sig_levels.append(nf)
        if sig_levels:
            survival[m] = {
                'max_noise': max(sig_levels),
                'n_levels': len(sig_levels),
                'clean_d': d5_data[0.0][m][0],
            }

    # Rank by robustness
    ranked = sorted(survival.items(), key=lambda x: (-x[1]['max_noise'], -x[1]['n_levels']))

    print(f"\n  Metrics detected at clean: {sum(1 for m in METRIC_NAMES if d5_data[0.0][m][2])}")
    print(f"  Metrics surviving to noise=0.5: {sum(1 for m, s in survival.items() if s['max_noise'] >= 0.5)}")
    print(f"  Metrics surviving to noise=1.0: {sum(1 for m, s in survival.items() if s['max_noise'] >= 1.0)}")
    print(f"  Metrics surviving to noise=2.0: {sum(1 for m, s in survival.items() if s['max_noise'] >= 2.0)}")

    print(f"\n  Most robust metrics (Hénon map):")
    print(f"  {'Metric':<55s} {'Max noise':>10s} {'# levels':>8s} {'Clean d':>10s}")
    print(f"  {'─' * 55} {'─' * 10} {'─' * 8} {'─' * 10}")
    for m, s in ranked[:25]:
        geo = m.split(':')[0]
        metric = m.split(':', 1)[1]
        label = f"{geo}: {metric}"
        print(f"  {label:<55s} {s['max_noise']:>10.2f} {s['n_levels']:>8d} {s['clean_d']:>+10.2f}")

    # Group by geometry: what fraction of each geometry's metrics survive?
    geometry_names = sorted(set(m.split(':')[0] for m in METRIC_NAMES))
    print(f"\n  Geometry survival rates at noise=0.5:")
    geo_survival = []
    for geo in geometry_names:
        geo_metrics = [m for m in METRIC_NAMES if m.startswith(geo + ':')]
        n_total = len(geo_metrics)
        n_survive = sum(1 for m in geo_metrics
                        if m in survival and survival[m]['max_noise'] >= 0.5)
        if n_total > 0:
            geo_survival.append((geo, n_survive, n_total, n_survive / n_total))
    geo_survival.sort(key=lambda x: -x[3])
    for geo, n_surv, n_tot, rate in geo_survival:
        bar = '█' * int(rate * 20)
        print(f"    {geo:<40s}  {n_surv}/{n_tot:2d} = {rate:5.0%}  {bar}")

    return survival, ranked


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(d1, d3, d4, survival, ranked):
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

    fig = plt.figure(figsize=(20, 20), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # D1: Last survivors heatmap (heartbeat at noise=2.0)
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    # Show surviving metrics at highest noise for each signal
    all_survivors = []
    for sig_name in ['heartbeat', 'lorenz', 'henon', 'logistic']:
        levels = sorted(d1[sig_name].keys())
        for nf in reversed(levels):
            survivors = d1[sig_name][nf]
            if survivors:
                for m, d, p in survivors[:5]:
                    all_survivors.append((sig_name, nf, m, d))
                break

    if all_survivors:
        labels = [f"{s[0]}@{s[1]}: {s[2].split(':',1)[1][:25]}" for s in all_survivors]
        d_vals = [abs(s[3]) for s in all_survivors]
        colors = {'heartbeat': '#FF9800', 'lorenz': '#E91E63',
                  'henon': '#2196F3', 'logistic': '#4CAF50'}
        bar_colors = [colors.get(s[0], '#888') for s in all_survivors]
        ax.barh(range(len(labels)), d_vals, color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel('|Cohen\'s d|', fontsize=9)
        ax.set_title('D1: Last survivors at highest detectable noise', fontsize=10, fontweight='bold')
        ax.invert_yaxis()

    # D3: Stochastic resonance
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    for sig_name in d3:
        nfs = sorted(d3[sig_name].keys())
        vals = [d3[sig_name][nf] for nf in nfs]
        color = {'heartbeat': '#FF9800', 'lorenz': '#E91E63'}[sig_name]
        ax.plot(nfs, vals, 'o-', color=color, label=sig_name, linewidth=2, markersize=5)
        # Mark clean value
        ax.axhline(y=d3[sig_name][0.0], color=color, linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Noise fraction', fontsize=9)
    ax.set_ylabel('Significant metrics vs IAAFT', fontsize=9)
    ax.set_title('D3: Stochastic resonance test (fine noise sweep)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # D4: Geometry regime map
    ax = _dark_ax(fig.add_subplot(gs[1, :]))
    regime_names = list(d4.keys())
    # Top 10 geometries across all regimes
    all_geos = set()
    for rn in regime_names:
        for geo, count in d4[rn]['ranked'][:10]:
            if count > 0:
                all_geos.add(geo)
    all_geos = sorted(all_geos)

    if all_geos:
        data_matrix = np.zeros((len(all_geos), len(regime_names)))
        for j, rn in enumerate(regime_names):
            for i, geo in enumerate(all_geos):
                data_matrix[i, j] = d4[rn]['geo_counts'].get(geo, 0)

        im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest')
        ax.set_xticks(range(len(regime_names)))
        ax.set_xticklabels([f"{rn}" for rn in regime_names], fontsize=8)
        ax.set_yticks(range(len(all_geos)))
        ax.set_yticklabels([g[:30] for g in all_geos], fontsize=7)
        ax.set_title('D4: Geometry detection count by noise regime (Hénon)',
                     fontsize=10, fontweight='bold')
        # Add text annotations
        for i in range(len(all_geos)):
            for j in range(len(regime_names)):
                val = int(data_matrix[i, j])
                if val > 0:
                    color = 'white' if val > data_matrix.max() * 0.6 else '#ccc'
                    ax.text(j, i, str(val), ha='center', va='center',
                           fontsize=7, color=color)
        fig.colorbar(im, ax=ax, shrink=0.6, label='Sig metrics')

    # D5: Top 20 most robust metrics (survival profile)
    ax = _dark_ax(fig.add_subplot(gs[2, :]))
    top_metrics = ranked[:20]
    labels = [f"{m.split(':')[0][:20]}: {m.split(':',1)[1][:20]}" for m, s in top_metrics]
    max_noise_vals = [s['max_noise'] for m, s in top_metrics]
    n_levels_vals = [s['n_levels'] for m, s in top_metrics]

    x = np.arange(len(labels))
    ax.barh(x, max_noise_vals, color='#2196F3', alpha=0.85, label='Max surviving noise')
    # Overlay n_levels as dot
    ax2 = ax.twiny()
    ax2.plot(n_levels_vals, x, 'D', color='#FFD700', markersize=6, label='# noise levels sig')
    ax2.set_xlabel('# noise levels with detection', fontsize=8, color='#FFD700')
    ax2.tick_params(colors='#FFD700', labelsize=7)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel('Max noise fraction survived', fontsize=9)
    ax.set_title('D5: Most noise-robust metrics (Hénon map)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666', loc='lower right')
    ax.invert_yaxis()

    fig.suptitle('Deep Noise Analysis: Exploring Anomalies',
                 fontsize=14, fontweight='bold', color='white', y=0.995)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'noise_deep.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved noise_deep.png")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    print(f"Metrics: {N_METRICS}, Bonferroni α={BONF:.2e}\n")

    d1 = direction_1(analyzer)
    d2 = direction_2(analyzer)
    d3 = direction_3(analyzer)
    d4 = direction_4(analyzer)
    survival, ranked = direction_5(analyzer)

    make_figure(d1, d3, d4, survival, ranked)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")

    # Key findings
    # 1. Last survivors
    print("\n  D1 — Last survivors at extreme noise:")
    for sig_name in ['heartbeat', 'henon', 'logistic', 'lorenz']:
        levels = sorted(d1[sig_name].keys())
        for nf in reversed(levels):
            survivors = d1[sig_name][nf]
            if survivors:
                geos = set(m.split(':')[0] for m, _, _ in survivors)
                print(f"    {sig_name}@noise={nf}: {len(survivors)} metrics from "
                      f"{len(geos)} geometries")
                break

    # 2. Spiral anomaly
    print("\n  D2 — Spiral anomaly:")
    for (sig_name, nf), data in d2.items():
        if data['spiral_sig']:
            print(f"    {sig_name}@noise={nf}: {len(data['spiral_sig'])} spiral metrics sig "
                  f"(total={data['total_sig']})")

    # 3. Stochastic resonance
    print("\n  D3 — Stochastic resonance:")
    for sig_name in d3:
        clean = d3[sig_name][0.0]
        peak = max(d3[sig_name].values())
        peak_nf = max(d3[sig_name], key=d3[sig_name].get)
        if peak > clean:
            print(f"    ★ {sig_name}: peak={peak} at noise={peak_nf:.2f} "
                  f"(+{peak - clean} over clean={clean})")
        else:
            print(f"    {sig_name}: clean={clean} is optimal (no resonance)")

    # 4. Regime shifts
    print("\n  D4 — Best geometry per regime (Hénon):")
    for rn in d4:
        top_geo, top_count = d4[rn]['ranked'][0]
        print(f"    {rn:10s}: {top_geo} ({top_count})")

    # 5. Most robust metrics
    print(f"\n  D5 — Top 5 most noise-robust metrics (Hénon):")
    for m, s in ranked[:5]:
        print(f"    {m:50s}  survives to noise={s['max_noise']:.1f}")
