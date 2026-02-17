#!/usr/bin/env python3
"""
Investigation: Ablation Study — Do Exotic Geometries Actually Matter?

The framework uses 25 geometries producing ~131 metrics. But are the exotic
embeddings (Heisenberg, tropical, symplectic, etc.) doing anything that
simpler methods can't? This is the most important methodological question.

Five directions:
  D1: Simple baseline — can entropy + permutation entropy + spectral slope
      + autocorrelation replicate our key findings?
  D2: Geometry ablation — which individual geometries provide unique value?
  D3: Correlation structure — how many independent dimensions do 131 metrics
      actually span? (effective rank of the metric matrix)
  D4: Incremental value — greedy addition of geometries, marginal gain curve
  D5: Exotic necessity — for each previously-claimed detection, is the
      exotic geometry essential or redundant?

Test cases (spanning all investigation domains):
  - Prime gaps vs random (number theory)
  - Collatz 3n+1 vs random (dynamics)
  - fBm H=0.3 vs H=0.7 (Hurst detection)
  - Heartbeat vs random (time series)
  - Sine wave vs random (periodic signal)
  - Mach-O vs random text (content type)

Methodology: N_TRIALS=25, DATA_SIZE=2000, Cohen's d > 0.8, Bonferroni correction.
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


# =========================================================================
# DATA GENERATORS
# =========================================================================

def gen_random(trial, size):
    return np.random.default_rng(9000 + trial).integers(0, 256, size, dtype=np.uint8)

def _sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

_PRIMES = _sieve_primes(600_000)

def gen_prime_gaps(trial, size):
    """Prime gaps starting from random offset."""
    rng = np.random.default_rng(42 + trial)
    start_idx = int(rng.integers(100, len(_PRIMES) - size - 1))
    gaps = np.diff(_PRIMES[start_idx:start_idx + size + 1])
    return np.clip(gaps, 0, 255).astype(np.uint8)

def gen_collatz(trial, size):
    """Collatz 3n+1 sequence from random seed."""
    rng = np.random.default_rng(42 + trial)
    n = int(rng.integers(100000, 1000000))
    vals = []
    for _ in range(size * 3):
        vals.append(n & 0xFF)
        if n <= 1:
            n = int(rng.integers(100000, 1000000))
        elif n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
    return np.array(vals[:size], dtype=np.uint8)

def gen_fbm(trial, size, hurst=0.7):
    """Fractional Brownian motion."""
    rng = np.random.default_rng(42 + trial)
    n = min(size, 1000)
    idx = np.arange(n, dtype=np.float64)
    i_grid, j_grid = np.meshgrid(idx, idx)
    H2 = 2 * hurst
    cov = 0.5 * (np.abs(i_grid)**H2 + np.abs(j_grid)**H2 - np.abs(i_grid - j_grid)**H2)
    cov += np.eye(n) * 1e-10
    L = np.linalg.cholesky(cov)
    fbm_vals = L @ rng.standard_normal(n)
    if n < size:
        fbm_vals = np.interp(np.linspace(0, n-1, size), np.arange(n), fbm_vals)
    lo, hi = np.percentile(fbm_vals, [1, 99])
    scaled = np.clip((fbm_vals - lo) / (hi - lo + 1e-10) * 255, 0, 255)
    return scaled.astype(np.uint8)

def gen_fbm_03(trial, size):
    return gen_fbm(trial, size, hurst=0.3)

def gen_fbm_07(trial, size):
    return gen_fbm(trial, size, hurst=0.7)

def gen_heartbeat(trial, size):
    """Synthetic ECG."""
    rng = np.random.default_rng(42 + trial)
    period = 60 + rng.integers(-10, 10)
    t = np.arange(size, dtype=np.float64)
    ecg = np.zeros(size)
    for beat_start in range(0, size, period):
        qrs_c = beat_start + period * 0.4
        ecg += 3.0 * np.exp(-0.5 * ((t - qrs_c) / max(period * 0.03, 1))**2)
        t_c = beat_start + period * 0.65
        ecg += 0.8 * np.exp(-0.5 * ((t - t_c) / max(period * 0.08, 1))**2)
    ecg += 0.1 * rng.standard_normal(size)
    lo, hi = np.percentile(ecg, [1, 99])
    return np.clip((ecg - lo) / (hi - lo + 1e-10) * 255, 0, 255).astype(np.uint8)

def gen_sine(trial, size):
    rng = np.random.default_rng(42 + trial)
    freq = rng.uniform(5, 50)
    x = np.linspace(0, freq * 2 * np.pi, size)
    return ((np.sin(x) + 1) * 127.5).astype(np.uint8)


TEST_CASES = {
    'prime_gaps_vs_random':  (gen_prime_gaps, gen_random, 'Number theory'),
    'collatz_vs_random':     (gen_collatz, gen_random, 'Dynamics'),
    'fbm03_vs_fbm07':       (gen_fbm_03, gen_fbm_07, 'Hurst detection'),
    'heartbeat_vs_random':   (gen_heartbeat, gen_random, 'Time series'),
    'sine_vs_random':        (gen_sine, gen_random, 'Periodic'),
}


# =========================================================================
# SIMPLE BASELINE FEATURES
# =========================================================================

def compute_simple_features(data):
    """Standard features that don't need exotic geometry."""
    data_f = data.astype(np.float64)
    features = {}

    # Shannon entropy
    counts = np.bincount(data, minlength=256)
    p = counts[counts > 0] / len(data)
    features['entropy'] = float(-np.sum(p * np.log2(p)))

    # Basic statistics
    features['mean'] = float(np.mean(data_f))
    features['std'] = float(np.std(data_f))
    features['skewness'] = float(stats.skew(data_f))
    features['kurtosis'] = float(stats.kurtosis(data_f))

    # Distinct values
    features['n_distinct'] = len(np.unique(data))

    # Autocorrelation at lags 1, 5, 10
    dm = data_f - np.mean(data_f)
    var = np.var(data_f)
    if var > 1e-15:
        for lag in [1, 5, 10]:
            if lag < len(data):
                features[f'autocorr_lag{lag}'] = float(
                    np.mean(dm[:-lag] * dm[lag:]) / var)
            else:
                features[f'autocorr_lag{lag}'] = 0.0
    else:
        for lag in [1, 5, 10]:
            features[f'autocorr_lag{lag}'] = 0.0

    # Permutation entropy (order 3)
    order = 3
    perm_counts = {}
    for i in range(len(data) - order + 1):
        pattern = tuple(np.argsort(data_f[i:i+order]))
        perm_counts[pattern] = perm_counts.get(pattern, 0) + 1
    total = sum(perm_counts.values())
    perm_p = np.array([c / total for c in perm_counts.values()])
    features['perm_entropy'] = float(-np.sum(perm_p * np.log2(perm_p)))
    features['perm_n_patterns'] = len(perm_counts)

    # Spectral features (FFT)
    fft_vals = np.abs(np.fft.rfft(data_f))[1:]  # skip DC
    if len(fft_vals) > 1:
        freqs = np.arange(1, len(fft_vals) + 1, dtype=np.float64)
        power = fft_vals ** 2
        total_power = np.sum(power)
        if total_power > 0:
            features['spectral_centroid'] = float(np.sum(freqs * power) / total_power)
            p_norm = power / total_power
            p_norm = p_norm[p_norm > 0]
            features['spectral_entropy'] = float(-np.sum(p_norm * np.log2(p_norm)))
            # Spectral slope (log-log regression)
            log_f = np.log10(freqs)
            log_p = np.log10(np.maximum(power, 1e-20))
            slope, _, _, _, _ = stats.linregress(log_f, log_p)
            features['spectral_slope'] = float(slope)
        else:
            features['spectral_centroid'] = 0.0
            features['spectral_entropy'] = 0.0
            features['spectral_slope'] = 0.0
    else:
        features['spectral_centroid'] = 0.0
        features['spectral_entropy'] = 0.0
        features['spectral_slope'] = 0.0

    return features

SIMPLE_FEATURE_NAMES = list(compute_simple_features(np.zeros(100, dtype=np.uint8)).keys())
N_SIMPLE = len(SIMPLE_FEATURE_NAMES)
print(f"Simple baseline: {N_SIMPLE} features")


# =========================================================================
# UTILITIES
# =========================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def collect_simple(data_arrays):
    """Collect simple features across trials."""
    out = {f: [] for f in SIMPLE_FEATURE_NAMES}
    for arr in data_arrays:
        feats = compute_simple_features(arr)
        for f, v in feats.items():
            if np.isfinite(v):
                out[f].append(v)
    return out


def collect_framework(analyzer, data_arrays, metric_names):
    """Collect full framework metrics."""
    out = {m: [] for m in metric_names}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def count_sig(data_a, data_b, feature_names, alpha):
    """Count significant features between two metric dicts."""
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


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# D1: SIMPLE BASELINE VS FULL FRAMEWORK
# =========================================================================
def direction_1(analyzer, metric_names):
    print("\n" + "=" * 78)
    print("D1: SIMPLE BASELINE VS FULL FRAMEWORK")
    print("=" * 78)
    print(f"  Simple: {N_SIMPLE} features, Framework: {len(metric_names)} metrics")

    bonf_simple = ALPHA / N_SIMPLE
    bonf_full = ALPHA / len(metric_names)

    results = {}
    for case_name, (gen_a, gen_b, domain) in TEST_CASES.items():
        print(f"\n  {case_name} ({domain}):")

        arrays_a = [gen_a(t, DATA_SIZE) for t in range(N_TRIALS)]
        arrays_b = [gen_b(t, DATA_SIZE) for t in range(N_TRIALS)]

        # Simple baseline
        simple_a = collect_simple(arrays_a)
        simple_b = collect_simple(arrays_b)
        n_simple, simple_findings = count_sig(simple_a, simple_b, SIMPLE_FEATURE_NAMES, bonf_simple)

        # Full framework
        full_a = collect_framework(analyzer, arrays_a, metric_names)
        full_b = collect_framework(analyzer, arrays_b, metric_names)
        n_full, full_findings = count_sig(full_a, full_b, metric_names, bonf_full)

        results[case_name] = {
            'simple': n_simple, 'full': n_full,
            'simple_findings': simple_findings,
            'full_findings': full_findings,
            'full_a': full_a, 'full_b': full_b,
            'arrays_a': arrays_a, 'arrays_b': arrays_b,
        }

        ratio = n_simple / max(N_SIMPLE, 1) * 100
        ratio_full = n_full / max(len(metric_names), 1) * 100
        print(f"    Simple:    {n_simple:2d}/{N_SIMPLE} sig ({ratio:.0f}%)")
        print(f"    Framework: {n_full:3d}/{len(metric_names)} sig ({ratio_full:.0f}%)")

        if simple_findings:
            print(f"    Simple top: {simple_findings[0][0]:25s} d={simple_findings[0][1]:+.2f}")
        if full_findings:
            # Find first framework finding NOT in simple baseline
            exotic_only = [f for f in full_findings
                          if not any(f[0].endswith(sf) for sf in SIMPLE_FEATURE_NAMES)]
            if exotic_only:
                print(f"    Exotic top: {exotic_only[0][0]:45s} d={exotic_only[0][1]:+.2f}")

    return results


# =========================================================================
# D2: INDIVIDUAL GEOMETRY ABLATION
# =========================================================================
def direction_2(d1_results):
    print("\n" + "=" * 78)
    print("D2: INDIVIDUAL GEOMETRY ABLATION")
    print("=" * 78)
    print("  Which geometries contribute unique detections?")

    # Get all geometry names from metric names
    all_metrics_flat = set()
    for case_data in d1_results.values():
        all_metrics_flat.update(case_data['full_a'].keys())

    geometry_names = sorted(set(m.split(':')[0] for m in all_metrics_flat))
    print(f"  {len(geometry_names)} geometries found")

    geo_scores = {g: {'total_sig': 0, 'cases_detected': 0} for g in geometry_names}

    for case_name, case_data in d1_results.items():
        full_a = case_data['full_a']
        full_b = case_data['full_b']

        for geo in geometry_names:
            geo_metrics = [m for m in full_a.keys() if m.startswith(geo + ':')]
            if not geo_metrics:
                continue
            bonf_geo = ALPHA / len(geo_metrics)
            n_sig, _ = count_sig(full_a, full_b, geo_metrics, bonf_geo)
            geo_scores[geo]['total_sig'] += n_sig
            if n_sig > 0:
                geo_scores[geo]['cases_detected'] += 1

    # Rank by total detections
    ranked = sorted(geo_scores.items(), key=lambda x: -x[1]['total_sig'])
    print(f"\n  {'Geometry':<45s} {'Total sig':>10s} {'Cases':>6s}")
    print(f"  {'─'*45} {'─'*10} {'─'*6}")
    for geo, scores in ranked:
        print(f"  {geo:<45s} {scores['total_sig']:>10d} {scores['cases_detected']:>5d}/5")

    return geo_scores


# =========================================================================
# D3: CORRELATION STRUCTURE / EFFECTIVE DIMENSIONALITY
# =========================================================================
def direction_3(analyzer, metric_names):
    print("\n" + "=" * 78)
    print("D3: CORRELATION STRUCTURE — EFFECTIVE DIMENSIONALITY")
    print("=" * 78)
    print("  How many independent dimensions do 131 metrics span?")

    # Generate diverse data to sample the metric space
    generators = [gen_random, gen_prime_gaps, gen_collatz, gen_fbm_03,
                  gen_fbm_07, gen_heartbeat, gen_sine]
    all_vectors = []

    for gen_fn in generators:
        for trial in range(10):
            arr = gen_fn(trial, DATA_SIZE)
            res = analyzer.analyze(arr)
            vec = []
            for m in metric_names:
                geo_name, met_name = m.split(':', 1)
                found = False
                for r in res.results:
                    if r.geometry_name == geo_name and met_name in r.metrics:
                        v = r.metrics[met_name]
                        vec.append(v if np.isfinite(v) else 0.0)
                        found = True
                        break
                if not found:
                    vec.append(0.0)
            all_vectors.append(vec)

    X = np.array(all_vectors)
    print(f"  Data matrix: {X.shape[0]} samples × {X.shape[1]} metrics")

    # Remove constant columns
    stds = np.std(X, axis=0)
    active = stds > 1e-10
    X_active = X[:, active]
    n_active = np.sum(active)
    print(f"  Active metrics (std > 0): {n_active}")

    # Z-score normalize
    X_norm = (X_active - np.mean(X_active, axis=0)) / (np.std(X_active, axis=0) + 1e-15)

    # SVD for effective rank
    U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
    total_var = np.sum(S**2)
    cum_var = np.cumsum(S**2) / total_var

    rank_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    rank_95 = int(np.searchsorted(cum_var, 0.95) + 1)
    rank_99 = int(np.searchsorted(cum_var, 0.99) + 1)

    print(f"\n  Effective rank:")
    print(f"    90% variance: {rank_90} dimensions")
    print(f"    95% variance: {rank_95} dimensions")
    print(f"    99% variance: {rank_99} dimensions")
    print(f"    Total active:  {n_active} metrics")
    print(f"    Redundancy:    {n_active - rank_95} metrics are >95% redundant")

    # Correlation analysis
    corr = np.corrcoef(X_active.T)
    n_high_corr = np.sum(np.abs(corr) > 0.95) - n_active  # subtract diagonal
    n_high_corr //= 2  # symmetric
    print(f"\n  Highly correlated pairs (|r| > 0.95): {n_high_corr}")

    return S, cum_var, rank_90, rank_95, rank_99, n_active


# =========================================================================
# D4: INCREMENTAL VALUE — GREEDY GEOMETRY ADDITION
# =========================================================================
def direction_4(d1_results):
    print("\n" + "=" * 78)
    print("D4: INCREMENTAL VALUE — GREEDY GEOMETRY ADDITION")
    print("=" * 78)
    print("  Start with the best single geometry, greedily add more.")
    print("  At each step, add the geometry that increases total detections most.")

    # Get geometry names and their metrics
    all_metrics = set()
    for case_data in d1_results.values():
        all_metrics.update(case_data['full_a'].keys())

    geometry_names = sorted(set(m.split(':')[0] for m in all_metrics))
    geo_to_metrics = {g: [m for m in all_metrics if m.startswith(g + ':')]
                      for g in geometry_names}

    selected = []
    remaining = set(geometry_names)
    curve = []

    while remaining:
        best_geo = None
        best_total = -1

        for geo in remaining:
            candidate = selected + [geo]
            candidate_metrics = []
            for g in candidate:
                candidate_metrics.extend(geo_to_metrics[g])

            total_sig = 0
            for case_data in d1_results.values():
                bonf = ALPHA / max(len(candidate_metrics), 1)
                n_sig, _ = count_sig(case_data['full_a'], case_data['full_b'],
                                     candidate_metrics, bonf)
                total_sig += n_sig

            if total_sig > best_total:
                best_total = total_sig
                best_geo = geo

        selected.append(best_geo)
        remaining.remove(best_geo)
        n_metrics = sum(len(geo_to_metrics[g]) for g in selected)
        curve.append((len(selected), best_geo, n_metrics, best_total))

        if len(selected) <= 10 or len(selected) == len(geometry_names):
            print(f"  +{best_geo:40s} ({n_metrics:3d} metrics) → {best_total:3d} total sig")

    # Find knee point
    totals = [c[3] for c in curve]
    max_total = max(totals)
    for i, (n_geo, geo, n_met, total) in enumerate(curve):
        if total >= max_total * 0.90:
            print(f"\n  90% of max reached at {n_geo} geometries ({n_met} metrics)")
            break
    for i, (n_geo, geo, n_met, total) in enumerate(curve):
        if total >= max_total * 0.95:
            print(f"  95% of max reached at {n_geo} geometries ({n_met} metrics)")
            break

    return curve


# =========================================================================
# D5: EXOTIC NECESSITY — CASE-BY-CASE
# =========================================================================
def direction_5(d1_results):
    print("\n" + "=" * 78)
    print("D5: EXOTIC NECESSITY — WHAT REQUIRES EXOTIC GEOMETRY?")
    print("=" * 78)

    # Define "standard" geometries (things with clear non-exotic equivalents)
    STANDARD_GEOS = {
        'Higher-Order Statistics',    # permutation entropy = standard
        'Fisher Information',          # well-studied
        'Persistent Homology',         # TDA is established
    }

    for case_name, case_data in d1_results.items():
        full_a = case_data['full_a']
        full_b = case_data['full_b']
        bonf = ALPHA / len(full_a)

        standard_only = []
        exotic_only = []
        for m in full_a.keys():
            geo = m.split(':')[0]
            a = np.array(full_a.get(m, []))
            b = np.array(full_b.get(m, []))
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf and abs(d) > 0.8:
                if geo in STANDARD_GEOS:
                    standard_only.append((m, d))
                else:
                    exotic_only.append((m, d))

        n_std = len(standard_only)
        n_exotic = len(exotic_only)
        n_total = n_std + n_exotic
        simple_sig = case_data['simple']

        print(f"\n  {case_name}:")
        print(f"    Simple baseline:    {simple_sig:3d}/{N_SIMPLE}")
        print(f"    Standard geos:      {n_std:3d} sig metrics")
        print(f"    Exotic geos:        {n_exotic:3d} sig metrics")
        print(f"    Total framework:    {n_total:3d} sig metrics")
        if n_total > 0:
            print(f"    Exotic contribution: {n_exotic/n_total*100:.0f}% of all detections")

        # Top exotic-only findings
        exotic_only.sort(key=lambda x: -abs(x[1]))
        if exotic_only[:3]:
            print(f"    Top exotic-only:")
            for m, d in exotic_only[:3]:
                print(f"      {m:50s}  d={d:+.2f}")


# =========================================================================
# FIGURE
# =========================================================================
def make_figure(d1_results, geo_scores, svd_data, greedy_curve, metric_names=None):
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
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                           height_ratios=[1.0, 1.0, 1.0])

    # D1: Simple vs Framework bar chart
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    cases = list(d1_results.keys())
    short_names = [c.replace('_vs_random', '').replace('_vs_', ' vs ') for c in cases]
    simple_vals = [d1_results[c]['simple'] / N_SIMPLE * 100 for c in cases]
    n_framework = len(metric_names)
    full_vals = [d1_results[c]['full'] / n_framework * 100 for c in cases]
    x = np.arange(len(cases))
    ax.bar(x - 0.2, simple_vals, 0.35, color='#FF9800', alpha=0.85, label=f'Simple ({N_SIMPLE})')
    ax.bar(x + 0.2, full_vals, 0.35, color='#2196F3', alpha=0.85, label=f'Framework ({n_framework})')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('% metrics significant', fontsize=9)
    ax.set_title('D1: Simple baseline vs full framework', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # D2: Geometry ablation
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    ranked_geos = sorted(geo_scores.items(), key=lambda x: -x[1]['total_sig'])[:15]
    geo_names = [g[0][:25] for g in ranked_geos]
    geo_sigs = [g[1]['total_sig'] for g in ranked_geos]
    ax.barh(range(len(geo_names)), geo_sigs, color='#4CAF50', alpha=0.85)
    ax.set_yticks(range(len(geo_names)))
    ax.set_yticklabels(geo_names, fontsize=7)
    ax.set_xlabel('Total sig metrics (across 5 cases)', fontsize=9)
    ax.set_title('D2: Top 15 geometries by detection power', fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    # D3: Cumulative variance (SVD)
    S, cum_var, r90, r95, r99, n_active = svd_data
    ax = _dark_ax(fig.add_subplot(gs[1, 0]))
    ax.plot(range(1, len(cum_var)+1), cum_var * 100, color='#E91E63', linewidth=2)
    ax.axhline(y=90, color='#666', linestyle='--', linewidth=0.5)
    ax.axhline(y=95, color='#666', linestyle='--', linewidth=0.5)
    ax.axhline(y=99, color='#666', linestyle='--', linewidth=0.5)
    ax.axvline(x=r90, color='#FF9800', linestyle=':', linewidth=1, label=f'90%: {r90}d')
    ax.axvline(x=r95, color='#4CAF50', linestyle=':', linewidth=1, label=f'95%: {r95}d')
    ax.axvline(x=r99, color='#2196F3', linestyle=':', linewidth=1, label=f'99%: {r99}d')
    ax.set_xlabel('Dimensions', fontsize=9)
    ax.set_ylabel('Cumulative variance %', fontsize=9)
    ax.set_title(f'D3a: Effective dimensionality ({n_active} → {r95} at 95%)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # D3: Singular value spectrum
    ax = _dark_ax(fig.add_subplot(gs[1, 1]))
    ax.semilogy(range(1, min(51, len(S)+1)), S[:50]**2 / np.sum(S**2) * 100,
                'o-', color='#9C27B0', markersize=3, linewidth=1)
    ax.set_xlabel('Component', fontsize=9)
    ax.set_ylabel('% variance (log scale)', fontsize=9)
    ax.set_title('D3b: Singular value spectrum (first 50)', fontsize=11, fontweight='bold')

    # D4: Greedy addition curve
    ax = _dark_ax(fig.add_subplot(gs[2, 0]))
    n_geos = [c[0] for c in greedy_curve]
    totals = [c[3] for c in greedy_curve]
    ax.plot(n_geos, totals, 'o-', color='#00BCD4', markersize=3, linewidth=2)
    max_total = max(totals)
    ax.axhline(y=max_total * 0.90, color='#FF9800', linestyle='--', linewidth=0.5,
               label='90% of max')
    ax.axhline(y=max_total * 0.95, color='#4CAF50', linestyle='--', linewidth=0.5,
               label='95% of max')
    ax.set_xlabel('Number of geometries', fontsize=9)
    ax.set_ylabel('Total sig metrics (5 cases)', fontsize=9)
    ax.set_title('D4: Greedy geometry addition', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # D5: Simple vs standard vs exotic breakdown
    ax = _dark_ax(fig.add_subplot(gs[2, 1]))
    case_labels = [c.replace('_vs_random', '').replace('_vs_', '\nvs ') for c in cases]
    simple_pct = [d1_results[c]['simple'] for c in cases]
    full_pct = [d1_results[c]['full'] for c in cases]
    x = np.arange(len(cases))
    width = 0.35
    ax.bar(x - width/2, simple_pct, width, color='#FF9800', alpha=0.85, label='Simple')
    ax.bar(x + width/2, full_pct, width, color='#2196F3', alpha=0.85, label='Framework')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=7)
    ax.set_ylabel('Significant metrics (count)', fontsize=9)
    ax.set_title('D5: Raw detection counts', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    fig.suptitle('Ablation Study: Do Exotic Geometries Matter?',
                 fontsize=14, fontweight='bold', color='white', y=0.995)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'ablation.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved ablation.png")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    # Discover metric names
    _dummy = analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
    metric_names = []
    for _r in _dummy.results:
        for _mn in sorted(_r.metrics.keys()):
            metric_names.append(f"{_r.geometry_name}:{_mn}")
    print(f"Framework: {len(metric_names)} metrics")

    d1_results = direction_1(analyzer, metric_names)
    geo_scores = direction_2(d1_results)
    svd_data = direction_3(analyzer, metric_names)
    greedy_curve = direction_4(d1_results)
    direction_5(d1_results)
    make_figure(d1_results, geo_scores, svd_data, greedy_curve, metric_names)

    # Final verdict
    print(f"\n{'=' * 78}")
    print("VERDICT")
    print(f"{'=' * 78}")
    for case_name, case_data in d1_results.items():
        s = case_data['simple']
        f = case_data['full']
        gain = f - s
        print(f"  {case_name:30s}: Simple={s:2d}/{N_SIMPLE}, "
              f"Full={f:3d}/{len(metric_names)}, Gain=+{gain}")

    S, cum_var, r90, r95, r99, n_active = svd_data
    print(f"\n  Effective dimensionality: {r95} at 95% variance (from {n_active} active metrics)")
    print(f"  Redundancy: {n_active - r95} metrics are >95% redundant")
