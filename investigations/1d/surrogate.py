#!/usr/bin/env python3
"""
Investigation: Surrogate Testing — Nonlinear Structure Detection

Tests whether exotic geometric embeddings detect nonlinear structure in
time series beyond what standard nonlinear analysis tools can find.

Method: IAAFT surrogates (Schreiber & Schmitz, 1996) preserve both the power
spectrum AND the marginal distribution of a signal. By construction, linear
statistics fail to distinguish original from IAAFT surrogate.

Three-tier comparison:
  Tier 1 — Linear baseline (11 features): entropy, autocorrelation, spectral
           slope. Should detect 0 vs IAAFT (by construction).
  Tier 2 — Nonlinear baseline (11 features): permutation entropy, mutual
           information, Lempel-Ziv complexity, time reversal asymmetry.
           Standard tools a competent analyst would use before reaching for
           geometric methods.
  Tier 3 — Framework (233 metrics), partitioned into:
           (a) Standard nonlinear: geometries that repackage known nonlinear
               analysis (Information Theory, HOS, Hölder, Spectral,
               Predictability, Zipf, Multifractal, p-Variation, Attractor,
               Recurrence, Visibility Graph)
           (b) Geometric: actual manifold embeddings (Torus, Hyperbolic,
               root systems, quasicrystals, Thurston geometries, etc.)

7 test signals: Lorenz, Hénon, logistic map, synthetic ECG, prime gaps,
Collatz, coupled AR with bounded nonlinearity.

Directions:
  D1: Three-tier comparison — linear vs nonlinear vs framework
  D2: Framework decomposition — standard-nonlinear vs geometric metrics
  D3: Geometry-by-geometry ranking (geometric lenses only)
  D4: Structure hierarchy — shuffle vs FT vs IAAFT
  D5: The honest test — geometric detections beyond nonlinear baseline

Methodology: N_TRIALS=25, DATA_SIZE=2000, Cohen's d > 0.8, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from math import factorial
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05
IAAFT_ITERATIONS = 100

# Framework metric partition: geometries that are standard nonlinear analysis
# repackaged as geometry classes (NOT genuine geometric embeddings)
STANDARD_NONLINEAR_GEOS = {
    'Information Theory', 'Higher-Order Statistics', 'Hölder Regularity',
    'Spectral Analysis', 'Predictability', 'Zipf–Mandelbrot (8-bit)',
    'Zipf–Mandelbrot (16-bit)', 'Multifractal Spectrum', 'p-Variation',
    'Attractor Reconstruction', 'Recurrence Quantification', 'Visibility Graph',
}


# =========================================================================
# SURROGATE GENERATION
# =========================================================================

def shuffle_surrogate(data, rng):
    """Destroy all temporal structure. Preserves distribution only."""
    s = data.copy()
    rng.shuffle(s)
    return s


def ft_surrogate(data, rng):
    """Fourier Transform surrogate: preserves power spectrum, randomizes phases.
    Destroys nonlinear structure and may alter marginal distribution."""
    n = len(data)
    fft = np.fft.rfft(data.astype(np.float64))
    amplitudes = np.abs(fft)
    random_phases = rng.uniform(0, 2 * np.pi, len(fft))
    # Keep DC and Nyquist real
    random_phases[0] = 0
    if n % 2 == 0:
        random_phases[-1] = 0
    surrogate = np.fft.irfft(amplitudes * np.exp(1j * random_phases), n=n)
    # Rescale to original range
    surrogate = (surrogate - surrogate.min()) / (surrogate.max() - surrogate.min() + 1e-15)
    surrogate = surrogate * (data.max() - data.min()) + data.min()
    return np.clip(surrogate, 0, 255).astype(np.uint8)


def iaaft_surrogate(data, rng, n_iter=IAAFT_ITERATIONS):
    """Iterative Amplitude Adjusted Fourier Transform surrogate.
    Preserves BOTH the power spectrum AND the marginal distribution.
    This is the gold standard null hypothesis for nonlinear structure.

    Algorithm (Schreiber & Schmitz, 1996):
    1. Sort original data to get rank order
    2. Start with random shuffle
    3. Iterate:
       a. Match spectrum: FFT, replace amplitudes with original, IFFT
       b. Match distribution: rank-order to match original sorted values
    4. Converge to a surrogate with both properties preserved.
    """
    data_f = data.astype(np.float64)
    n = len(data_f)

    # Target spectrum (from original)
    target_fft = np.fft.rfft(data_f)
    target_amplitudes = np.abs(target_fft)

    # Target distribution (sorted original values)
    sorted_data = np.sort(data_f)

    # Initialize with random shuffle
    surrogate = data_f.copy()
    rng.shuffle(surrogate)

    for iteration in range(n_iter):
        # Step A: Match spectrum
        surr_fft = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_fft)
        # Replace amplitudes, keep phases
        matched_fft = target_amplitudes * np.exp(1j * surr_phases)
        surrogate = np.fft.irfft(matched_fft, n=n)

        # Step B: Match distribution (rank ordering)
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_data[rank_order]

    return np.clip(surrogate, 0, 255).astype(np.uint8)


# =========================================================================
# SIGNAL GENERATORS
# =========================================================================

def to_uint8(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-10:
        return np.full(len(x), 128, dtype=np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def gen_lorenz(trial, size):
    """Lorenz attractor x-component. Deterministic chaos, sensitive to IC."""
    dt = 0.01
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
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
    # Subsample to reduce autocorrelation
    vals = vals[warmup::5]
    return to_uint8(vals[:size])


def gen_henon(trial, size):
    """Hénon map x-component."""
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
    return to_uint8(vals[warmup:warmup + size])


def gen_logistic(trial, size):
    """Logistic map at r=3.99 (chaos)."""
    rng = np.random.default_rng(42 + trial)
    x = 0.1 + 0.01 * rng.uniform()
    warmup = 500
    vals = []
    for _ in range(warmup + size):
        x = 3.99 * x * (1 - x)
        vals.append(x)
    return to_uint8(vals[warmup:warmup + size])


def gen_heartbeat(trial, size):
    """Synthetic ECG with nonlinear QRS complex."""
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
    return to_uint8(ecg)


def _sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

_PRIMES = _sieve_primes(600_000)

def gen_prime_gaps(trial, size):
    """Prime gap sequence."""
    rng = np.random.default_rng(42 + trial)
    start_idx = int(rng.integers(100, len(_PRIMES) - size - 1))
    gaps = np.diff(_PRIMES[start_idx:start_idx + size + 1])
    return np.clip(gaps, 0, 255).astype(np.uint8)


def gen_collatz(trial, size):
    """Collatz 3n+1 sequence."""
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


def gen_coupled_ar(trial, size):
    """Coupled AR processes with bounded nonlinear interaction.
    x[t] = 0.6*x[t-1] + 0.4*tanh(y[t-1]) + noise
    y[t] = 0.6*y[t-1] + 0.4*x[t-1]^2/(1+x[t-1]^2) + noise
    Bounded nonlinear coupling (tanh, logistic) prevents divergence
    while creating structure invisible to linear methods.
    """
    rng = np.random.default_rng(42 + trial)
    x = np.zeros(size + 500)
    y = np.zeros(size + 500)
    for t in range(1, size + 500):
        x[t] = 0.6 * x[t-1] + 0.4 * np.tanh(y[t-1]) + 0.3 * rng.standard_normal()
        y[t] = 0.6 * y[t-1] + 0.4 * x[t-1]**2 / (1 + x[t-1]**2) + 0.3 * rng.standard_normal()
    return to_uint8(x[500:])


SIGNALS = {
    'lorenz':     gen_lorenz,
    'henon':      gen_henon,
    'logistic':   gen_logistic,
    'heartbeat':  gen_heartbeat,
    'prime_gaps': gen_prime_gaps,
    'collatz':    gen_collatz,
    'coupled_ar': gen_coupled_ar,
}


# =========================================================================
# TIER 1: LINEAR BASELINE
# =========================================================================

def compute_simple_features(data):
    """Standard linear features — should FAIL against IAAFT surrogates."""
    data_f = data.astype(np.float64)
    features = {}

    counts = np.bincount(data, minlength=256)
    p = counts[counts > 0] / len(data)
    features['entropy'] = float(-np.sum(p * np.log2(p)))

    features['mean'] = float(np.mean(data_f))
    features['std'] = float(np.std(data_f))
    features['skewness'] = float(stats.skew(data_f))
    features['kurtosis'] = float(stats.kurtosis(data_f))
    features['n_distinct'] = len(np.unique(data))

    dm = data_f - np.mean(data_f)
    var = np.var(data_f)
    for lag in [1, 5, 10]:
        if var > 1e-15 and lag < len(data):
            features[f'autocorr_lag{lag}'] = float(np.mean(dm[:-lag] * dm[lag:]) / var)
        else:
            features[f'autocorr_lag{lag}'] = 0.0

    # Spectral slope
    fft_vals = np.abs(np.fft.rfft(data_f))[1:]
    if len(fft_vals) > 1:
        freqs = np.arange(1, len(fft_vals) + 1, dtype=np.float64)
        power = fft_vals ** 2
        log_f = np.log10(freqs)
        log_p = np.log10(np.maximum(power, 1e-20))
        slope, _, _, _, _ = stats.linregress(log_f, log_p)
        features['spectral_slope'] = float(slope)
        total_power = np.sum(power)
        if total_power > 0:
            features['spectral_centroid'] = float(np.sum(freqs * power) / total_power)
        else:
            features['spectral_centroid'] = 0.0
    else:
        features['spectral_slope'] = 0.0
        features['spectral_centroid'] = 0.0

    return features

SIMPLE_NAMES = list(compute_simple_features(np.zeros(100, dtype=np.uint8)).keys())
N_SIMPLE = len(SIMPLE_NAMES)


# =========================================================================
# TIER 2: NONLINEAR BASELINE
# =========================================================================

def _permutation_entropy(x, order=3):
    """Bandt-Pompe permutation entropy."""
    n = len(x)
    perms = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i:i+order]).tolist())
        perms[pattern] = perms.get(pattern, 0) + 1
    total = sum(perms.values())
    probs = np.array(list(perms.values())) / total
    return float(-np.sum(probs * np.log2(probs)))


def _lempel_ziv_complexity(data):
    """Lempel-Ziv complexity on binarized sequence (normalized)."""
    binary = (data > np.median(data)).astype(int)
    s = ''.join(map(str, binary))
    n = len(s)
    if n == 0:
        return 0.0
    vocabulary = set()
    w = ''
    c = 0
    for char in s:
        w += char
        if w not in vocabulary:
            vocabulary.add(w)
            c += 1
            w = ''
    if w:
        c += 1
    return c / (n / max(np.log2(n), 1))


def _time_reversal_asymmetry(x, lag=1):
    """Time reversal asymmetry statistic (3rd order)."""
    n = len(x)
    if n <= lag:
        return 0.0
    return float(np.mean((x[lag:] - x[:-lag])**3))


def _histogram_mutual_info(x, lag, n_bins=16):
    """Histogram-based mutual information at given lag."""
    if lag >= len(x):
        return 0.0
    a = x[:-lag]
    b = x[lag:]
    hist_ab, _, _ = np.histogram2d(a, b, bins=n_bins)
    total = np.sum(hist_ab)
    if total == 0:
        return 0.0
    p_ab = hist_ab / total
    p_a = np.sum(p_ab, axis=1)
    p_b = np.sum(p_ab, axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_ab[i, j] > 0 and p_a[i] > 0 and p_b[j] > 0:
                mi += p_ab[i, j] * np.log2(p_ab[i, j] / (p_a[i] * p_b[j]))
    return mi


def compute_nonlinear_baseline(data):
    """Standard nonlinear features a competent analyst would compute
    before reaching for geometric methods. All O(n) or O(n*order)."""
    x = data.astype(np.float64)
    features = {}

    # Permutation entropy (orders 3, 5, 7) — ordinal pattern structure
    for order in [3, 5, 7]:
        features[f'perm_entropy_{order}'] = _permutation_entropy(x, order)

    # Forbidden permutation fraction (order 4) — determinism indicator
    n = len(x)
    order = 4
    possible = factorial(order)
    observed = len(set(tuple(np.argsort(x[i:i+order]).tolist())
                       for i in range(n - order + 1)))
    features['forbidden_frac_4'] = 1 - observed / possible

    # Lempel-Ziv complexity — compressibility
    features['lz_complexity'] = _lempel_ziv_complexity(data)

    # Time reversal asymmetry — irreversibility (nonlinearity indicator)
    features['trev_asym_1'] = _time_reversal_asymmetry(x, 1)
    features['trev_asym_3'] = _time_reversal_asymmetry(x, 3)

    # Mutual information (histogram-based, lags 1, 3, 5) — nonlinear dependence
    features['mi_lag1'] = _histogram_mutual_info(x, 1)
    features['mi_lag3'] = _histogram_mutual_info(x, 3)
    features['mi_lag5'] = _histogram_mutual_info(x, 5)

    # Turning point ratio — structural complexity
    diffs = np.diff(np.sign(np.diff(x)))
    features['turning_points'] = float(np.sum(diffs != 0)) / max(len(diffs), 1)

    return features


NONLINEAR_NAMES = list(compute_nonlinear_baseline(
    np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8)).keys())
N_NONLINEAR = len(NONLINEAR_NAMES)


# =========================================================================
# UTILITIES
# =========================================================================

# Discover framework metrics and partition into standard vs geometric
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
STANDARD_METRICS = []
GEOMETRIC_METRICS = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        key = f"{_r.geometry_name}:{_mn}"
        METRIC_NAMES.append(key)
        if _r.geometry_name in STANDARD_NONLINEAR_GEOS:
            STANDARD_METRICS.append(key)
        else:
            GEOMETRIC_METRICS.append(key)
N_METRICS = len(METRIC_NAMES)
N_STD = len(STANDARD_METRICS)
N_GEO = len(GEOMETRIC_METRICS)
BONF_SIMPLE = ALPHA / N_SIMPLE
BONF_NONLINEAR = ALPHA / N_NONLINEAR
BONF_FULL = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn

print(f"Linear baseline: {N_SIMPLE} features")
print(f"Nonlinear baseline: {N_NONLINEAR} features")
print(f"Framework: {N_METRICS} metrics ({N_STD} standard-nonlinear, {N_GEO} geometric)")


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def collect_simple(data_arrays):
    out = {f: [] for f in SIMPLE_NAMES}
    for arr in data_arrays:
        feats = compute_simple_features(arr)
        for f, v in feats.items():
            if np.isfinite(v):
                out[f].append(v)
    return out


def collect_nonlinear(data_arrays):
    out = {f: [] for f in NONLINEAR_NAMES}
    for arr in data_arrays:
        feats = compute_nonlinear_baseline(arr)
        for f, v in feats.items():
            if np.isfinite(v):
                out[f].append(v)
    return out


def collect_framework(analyzer, data_arrays):
    out = {m: [] for m in METRIC_NAMES}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def count_sig(data_a, data_b, feature_names, alpha):
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
# GENERATE ALL DATA
# =========================================================================
def generate_all(analyzer):
    """Generate originals + 3 surrogate types for all signals."""
    print("\nGenerating signals and surrogates...")
    all_data = {}

    for name, gen_fn in SIGNALS.items():
        print(f"  {name:15s}...", end=" ", flush=True)

        originals = [gen_fn(t, DATA_SIZE) for t in range(N_TRIALS)]

        shuffle_surrs, ft_surrs, iaaft_surrs = [], [], []
        for t, orig in enumerate(originals):
            rng = np.random.default_rng(5000 + t)
            shuffle_surrs.append(shuffle_surrogate(orig, rng))
            ft_surrs.append(ft_surrogate(orig, np.random.default_rng(6000 + t)))
            iaaft_surrs.append(iaaft_surrogate(orig, np.random.default_rng(7000 + t)))

        # Collect all three tiers
        all_data[name] = {
            'orig_simple': collect_simple(originals),
            'orig_nonlinear': collect_nonlinear(originals),
            'orig_full': collect_framework(analyzer, originals),
            'shuf_simple': collect_simple(shuffle_surrs),
            'shuf_nonlinear': collect_nonlinear(shuffle_surrs),
            'shuf_full': collect_framework(analyzer, shuffle_surrs),
            'ft_simple': collect_simple(ft_surrs),
            'ft_nonlinear': collect_nonlinear(ft_surrs),
            'ft_full': collect_framework(analyzer, ft_surrs),
            'iaaft_simple': collect_simple(iaaft_surrs),
            'iaaft_nonlinear': collect_nonlinear(iaaft_surrs),
            'iaaft_full': collect_framework(analyzer, iaaft_surrs),
        }
        print("done")

    return all_data


# =========================================================================
# D1: THREE-TIER COMPARISON
# =========================================================================
def direction_1(all_data):
    print("\n" + "=" * 78)
    print("D1: THREE-TIER COMPARISON — LINEAR VS NONLINEAR VS FRAMEWORK")
    print("=" * 78)
    print(f"  Linear: {N_SIMPLE} features (entropy, autocorrelation, spectral)")
    print(f"  Nonlinear: {N_NONLINEAR} features (permutation entropy, MI, LZ, TRA)")
    print(f"  Framework: {N_METRICS} metrics ({N_STD} standard + {N_GEO} geometric)")

    d1 = {}
    for name in SIGNALS:
        d = all_data[name]
        lin_sig, _ = count_sig(d['orig_simple'], d['iaaft_simple'],
                               SIMPLE_NAMES, BONF_SIMPLE)
        nl_sig, nl_findings = count_sig(d['orig_nonlinear'], d['iaaft_nonlinear'],
                                        NONLINEAR_NAMES, BONF_NONLINEAR)
        fw_sig, _ = count_sig(d['orig_full'], d['iaaft_full'],
                              METRIC_NAMES, BONF_FULL)
        d1[name] = {'linear': lin_sig, 'nonlinear': nl_sig, 'framework': fw_sig}
        print(f"\n  {name:15s}: linear={lin_sig:2d}  nonlinear={nl_sig:2d}  framework={fw_sig:3d}")
        for m, dval, p in nl_findings[:3]:
            print(f"    NL: {m:25s}  d={dval:+.2f}")

    return d1


# =========================================================================
# D2: FRAMEWORK DECOMPOSITION — STANDARD VS GEOMETRIC
# =========================================================================
def direction_2(all_data):
    print("\n" + "=" * 78)
    print("D2: FRAMEWORK DECOMPOSITION — STANDARD-NONLINEAR VS GEOMETRIC")
    print("=" * 78)
    print(f"  Standard nonlinear ({N_STD} metrics): {', '.join(sorted(STANDARD_NONLINEAR_GEOS))}")
    print(f"  Geometric ({N_GEO} metrics): actual manifold embeddings")

    d2 = {}
    for name in SIGNALS:
        d = all_data[name]
        # Use BONF_FULL (not partition-specific) so std+geo = framework total
        std_sig, std_findings = count_sig(d['orig_full'], d['iaaft_full'],
                                          STANDARD_METRICS, BONF_FULL)
        geo_sig, geo_findings = count_sig(d['orig_full'], d['iaaft_full'],
                                          GEOMETRIC_METRICS, BONF_FULL)
        d2[name] = {
            'standard': std_sig, 'geometric': geo_sig,
            'std_findings': std_findings, 'geo_findings': geo_findings,
        }
        print(f"\n  {name:15s}: standard={std_sig:3d}/{N_STD}  geometric={geo_sig:3d}/{N_GEO}")
        if geo_findings:
            print(f"    Top geometric:")
            for m, dval, p in geo_findings[:3]:
                print(f"      {m:50s}  d={dval:+.2f}")

    return d2


# =========================================================================
# D3: GEOMETRY-BY-GEOMETRY RANKING (GEOMETRIC ONLY)
# =========================================================================
def direction_3(all_data):
    print("\n" + "=" * 78)
    print("D3: GEOMETRIC LENSES — NONLINEAR DETECTION RANKING")
    print("  (Excluding standard nonlinear analysis repackaged as geometry)")
    print("=" * 78)

    # Only rank genuinely geometric lenses
    geo_names = sorted(set(m.split(':')[0] for m in GEOMETRIC_METRICS))
    geo_to_metrics = {g: [m for m in GEOMETRIC_METRICS if m.startswith(g + ':')]
                      for g in geo_names}

    geo_scores = {g: 0 for g in geo_names}
    for name in SIGNALS:
        d = all_data[name]
        for geo in geo_names:
            metrics = geo_to_metrics[geo]
            # Use BONF_FULL for consistency with D1/D2 (not per-geometry correction)
            n_sig, _ = count_sig(d['orig_full'], d['iaaft_full'], metrics, BONF_FULL)
            geo_scores[geo] += n_sig

    ranked = sorted(geo_scores.items(), key=lambda x: -x[1])
    print(f"\n  {'Geometry':<45s} {'Detections':>10s}")
    print(f"  {'─'*45} {'─'*10}")
    for geo, score in ranked:
        bar = '█' * min(score, 40)
        print(f"  {geo:<45s} {score:>5d}  {bar}")

    return geo_scores


# =========================================================================
# D4: STRUCTURE HIERARCHY — SHUFFLE VS FT VS IAAFT
# =========================================================================
def direction_4(all_data):
    print("\n" + "=" * 78)
    print("D4: STRUCTURE HIERARCHY — WHAT EACH SURROGATE DESTROYS")
    print("=" * 78)
    print("  Shuffle: destroys everything → detects ANY structure")
    print("  FT: preserves spectrum → detects nonlinear + distributional")
    print("  IAAFT: preserves spectrum + distribution → detects ONLY nonlinear")

    d4 = {}
    for name in SIGNALS:
        d = all_data[name]

        shuf_sig, _ = count_sig(d['orig_full'], d['shuf_full'], METRIC_NAMES, BONF_FULL)
        ft_sig, _ = count_sig(d['orig_full'], d['ft_full'], METRIC_NAMES, BONF_FULL)
        iaaft_sig, _ = count_sig(d['orig_full'], d['iaaft_full'], METRIC_NAMES, BONF_FULL)

        d4[name] = {'shuffle': shuf_sig, 'ft': ft_sig, 'iaaft': iaaft_sig}
        print(f"  {name:15s}:  shuffle={shuf_sig:3d}  FT={ft_sig:3d}  IAAFT={iaaft_sig:3d}")

    return d4


# =========================================================================
# D5: THE HONEST TEST — GEOMETRIC VALUE-ADD
# =========================================================================
def direction_5(all_data, d1, d2):
    print("\n" + "=" * 78)
    print("D5: THE HONEST TEST — DO GEOMETRIC EMBEDDINGS ADD VALUE?")
    print("  Comparing: nonlinear baseline vs geometric-only framework metrics")
    print("=" * 78)

    value_add = []
    for name in SIGNALS:
        nl_sig = d1[name]['nonlinear']
        geo_sig = d2[name]['geometric']
        geo_findings = d2[name]['geo_findings']
        has_value = geo_sig > 0

        print(f"\n  {name:15s}: nonlinear_baseline={nl_sig:2d}  geometric={geo_sig:3d}"
              f"{'  ← geometric adds value' if has_value and nl_sig == 0 else ''}")

        if has_value:
            value_add.append(name)
            for m, dval, p in geo_findings[:5]:
                if np.isfinite(dval):
                    print(f"    {m:50s}  d={dval:+.2f}")

    print(f"\n  Summary:")
    print(f"  Signals where geometric embeddings detect nonlinear structure: "
          f"{len(value_add)}/{len(SIGNALS)}")

    # The key question: do geometric lenses find anything the nonlinear
    # baseline misses?
    geo_only = [n for n in value_add if d1[n]['nonlinear'] == 0]
    if geo_only:
        print(f"  Signals where nonlinear baseline=0 but geometric>0: "
              f"{', '.join(geo_only)}")
        print(f"  These {len(geo_only)} signal(s) show genuinely geometric detection")
        print(f"  beyond standard nonlinear tools.")
    else:
        both = [n for n in value_add if d1[n]['nonlinear'] > 0]
        if both:
            print(f"  Nonlinear baseline also detects on the same signals.")
            print(f"  Geometric metrics provide ADDITIONAL detections but are")
            print(f"  not the ONLY path to detecting nonlinear structure.")
        else:
            print(f"  No signals with geometric-only detections found.")

    return value_add


# =========================================================================
# FIGURE
# =========================================================================
def make_figure(all_data, d1, d2, d4, geo_scores, value_add):
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
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                           height_ratios=[1.0, 1.0, 1.2])

    names = list(SIGNALS.keys())
    x = np.arange(len(names))

    # D1: Three-tier comparison
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    lin_vals = [d1[n]['linear'] for n in names]
    nl_vals = [d1[n]['nonlinear'] for n in names]
    fw_vals = [d1[n]['framework'] for n in names]
    width = 0.25
    ax.bar(x - width, lin_vals, width, color='#FF9800', alpha=0.85,
           label=f'Linear ({N_SIMPLE})')
    ax.bar(x, nl_vals, width, color='#4CAF50', alpha=0.85,
           label=f'Nonlinear ({N_NONLINEAR})')
    ax.bar(x + width, fw_vals, width, color='#2196F3', alpha=0.85,
           label=f'Framework ({N_METRICS})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Sig metrics vs IAAFT', fontsize=9)
    ax.set_title('D1: Three-tier comparison — linear vs nonlinear vs framework',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666')

    # D4: Structure hierarchy
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    shuf_vals = [d4[n]['shuffle'] for n in names]
    ft_vals = [d4[n]['ft'] for n in names]
    iaaft_vals = [d4[n]['iaaft'] for n in names]
    ax.bar(x - width, shuf_vals, width, color='#E91E63', alpha=0.85, label='Shuffle')
    ax.bar(x, ft_vals, width, color='#4CAF50', alpha=0.85, label='FT')
    ax.bar(x + width, iaaft_vals, width, color='#2196F3', alpha=0.85, label='IAAFT')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Sig metrics', fontsize=9)
    ax.set_title('D4: Structure hierarchy (what each surrogate preserves)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666')

    # D2: Framework decomposition — standard vs geometric
    ax = _dark_ax(fig.add_subplot(gs[1, 0]))
    std_vals = [d2[n]['standard'] for n in names]
    geo_vals = [d2[n]['geometric'] for n in names]
    ax.bar(x - 0.2, std_vals, 0.35, color='#FF9800', alpha=0.85,
           label=f'Std nonlinear ({N_STD})')
    ax.bar(x + 0.2, geo_vals, 0.35, color='#9C27B0', alpha=0.85,
           label=f'Geometric ({N_GEO})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Sig metrics vs IAAFT', fontsize=9)
    ax.set_title('D2: Framework decomposition — standard vs geometric',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666')

    # D3: Geometric geometry ranking (only geometric, score > 0)
    ax = _dark_ax(fig.add_subplot(gs[1, 1]))
    ranked = sorted(geo_scores.items(), key=lambda x: -x[1])
    ranked = [(g, s) for g, s in ranked if s > 0]
    n_zero = sum(1 for s in geo_scores.values() if s == 0)
    if ranked:
        geo_names_sorted = [g[0] for g in ranked]
        geo_vals_sorted = [g[1] for g in ranked]
        ax.barh(range(len(geo_names_sorted)), geo_vals_sorted,
                color='#9C27B0', alpha=0.85)
        ax.set_yticks(range(len(geo_names_sorted)))
        ax.set_yticklabels(geo_names_sorted, fontsize=6)
        ax.set_xlabel('Nonlinear detections (across 7 signals)', fontsize=8)
        ax.invert_yaxis()
    title_d3 = 'D3: Geometric lenses — nonlinear detection'
    if n_zero > 0:
        title_d3 += f'\n({n_zero} with 0 omitted)'
    ax.set_title(title_d3, fontsize=10, fontweight='bold')

    # D5: Top geometric detections (log scale, colored by signal)
    ax = _dark_ax(fig.add_subplot(gs[2, :]))
    if value_add:
        all_findings = []
        for name in value_add:
            for m, dval, p in d2[name]['geo_findings'][:5]:
                if np.isfinite(dval):
                    all_findings.append((name, m, dval))
        all_findings.sort(key=lambda x: -abs(x[2]))
        all_findings = all_findings[:20]

        if all_findings:
            labels = [f"{f[0]}: {f[1].split(':',1)[1][:30]}" for f in all_findings]
            d_vals = [abs(f[2]) for f in all_findings]
            signal_names_list = [f[0] for f in all_findings]

            # Color by signal
            signal_palette = {}
            palette = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0',
                       '#00BCD4', '#FFEB3B']
            for i, s in enumerate(SIGNALS):
                signal_palette[s] = palette[i % len(palette)]
            colors_bar = [signal_palette[s] for s in signal_names_list]

            ax.barh(range(len(labels)), d_vals, color=colors_bar, alpha=0.85)
            if max(d_vals) / max(min(d_vals), 0.1) > 20:
                ax.set_xscale('log')
                ax.set_xlabel("|Cohen's d| (log scale)", fontsize=9)
            else:
                ax.set_xlabel("|Cohen's d|", fontsize=9)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()

            # Legend
            from matplotlib.patches import Patch
            seen = []
            handles = []
            for s in signal_names_list:
                if s not in seen:
                    seen.append(s)
                    handles.append(Patch(facecolor=signal_palette[s], label=s))
            ax.legend(handles=handles, fontsize=7, facecolor='#333',
                      edgecolor='#666', loc='lower right')

    ax.set_title(f'D5: Top geometric detections — {len(value_add)} of '
                 f'{len(SIGNALS)} signals with geometric value-add '
                 f'(manifold embeddings only)',
                 fontsize=10, fontweight='bold')

    fig.suptitle('Surrogate Testing: Do Geometric Embeddings Detect Nonlinear Structure?',
                 fontsize=14, fontweight='bold', color='white', y=0.995)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'surrogate.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved surrogate.png")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    all_data = generate_all(analyzer)
    d1 = direction_1(all_data)
    d2 = direction_2(all_data)
    geo_scores = direction_3(all_data)
    d4 = direction_4(all_data)
    value_add = direction_5(all_data, d1, d2)
    make_figure(all_data, d1, d2, d4, geo_scores, value_add)

    print(f"\n{'=' * 78}")
    print("VERDICT")
    print(f"{'=' * 78}")
    for name in SIGNALS:
        t = d1[name]
        s = d2[name]
        h = d4[name]
        has_geo = " ◆" if name in value_add else ""
        print(f"  {name:15s}: linear={t['linear']:2d}  nonlinear={t['nonlinear']:2d}  "
              f"framework={t['framework']:3d}  "
              f"(std={s['standard']:3d} geo={s['geometric']:3d})"
              f"{has_geo}")

    total_fw = sum(d1[n]['framework'] for n in SIGNALS)
    total_nl = sum(d1[n]['nonlinear'] for n in SIGNALS)
    total_lin = sum(d1[n]['linear'] for n in SIGNALS)
    total_std = sum(d2[n]['standard'] for n in SIGNALS)
    total_geo = sum(d2[n]['geometric'] for n in SIGNALS)

    print(f"\n  Total detections vs IAAFT:")
    print(f"    Linear baseline:           {total_lin:4d}")
    print(f"    Nonlinear baseline:        {total_nl:4d}")
    print(f"    Framework (all):           {total_fw:4d}")
    print(f"      ├─ Standard nonlinear:   {total_std:4d}")
    print(f"      └─ Geometric:            {total_geo:4d}")

    if total_geo > 0:
        print(f"\n  Geometric embeddings contribute {total_geo} detections.")
        if total_nl > 0:
            print(f"  But nonlinear baseline also finds {total_nl} — geometry is")
            print(f"  not the ONLY path to detecting nonlinear structure.")
            print(f"  Geometric embeddings provide complementary detections")
            print(f"  via manifold topology, not uniquely indispensable ones.")
        else:
            print(f"  Nonlinear baseline finds 0 — geometric embeddings are")
            print(f"  the only tools here that detect nonlinear structure.")
    else:
        print(f"\n  Geometric embeddings contribute 0 detections.")
        print(f"  All framework value comes from standard nonlinear analysis")
        print(f"  repackaged as geometry classes.")
