#!/usr/bin/env python3
"""
Investigation: Surrogate Testing — Do Exotic Geometries Detect Nonlinear Structure?

The ablation study showed simple features detect the same phenomena as the full
framework. This investigation asks the harder question: can exotic geometries
detect structure that simple features PROVABLY CANNOT?

Method: IAAFT surrogates (Schreiber & Schmitz, 1996) preserve both the power
spectrum AND the marginal distribution of a signal. By construction, any method
based on linear statistics (entropy, autocorrelation, spectral slope) will
FAIL to distinguish original from surrogate. Only methods sensitive to nonlinear
or higher-order structure can succeed.

If exotic geometries distinguish original from IAAFT surrogate where simple
features cannot, that proves they capture genuinely new information.

7 test signals:
  1. lorenz_x     — Lorenz attractor x-component (deterministic chaos)
  2. henon_x      — Hénon map x-component (low-dimensional chaos)
  3. logistic     — Logistic map r=3.99 (1D chaos)
  4. heartbeat    — Synthetic ECG (periodic + nonlinear waveform)
  5. prime_gaps   — Prime gap sequence (number-theoretic structure)
  6. collatz      — Collatz 3n+1 sequence (arithmetic dynamics)
  7. coupled_ar   — Coupled AR processes (nonlinear interaction)

Three surrogate types:
  - Shuffle: destroys all temporal structure (sanity check)
  - FT surrogate: preserves spectrum, randomizes phases (linear null)
  - IAAFT surrogate: preserves spectrum AND distribution (strongest null)

Directions:
  D1: Simple baseline vs IAAFT surrogates (should fail — 0 sig expected)
  D2: Full framework vs IAAFT surrogates (the key test)
  D3: Which geometries detect nonlinear structure? (geometry-by-geometry)
  D4: Effect size comparison: shuffle vs FT vs IAAFT (structure hierarchy)
  D5: The smoking gun — signals where simple=0 but exotic>0

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
IAAFT_ITERATIONS = 100  # convergence iterations for IAAFT


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
    """Two coupled AR(1) processes with nonlinear interaction.
    x[t] = 0.8*x[t-1] + 0.3*y[t-1]^2 + noise
    y[t] = 0.6*y[t-1] + 0.2*x[t-1] + noise
    The x^2 coupling creates nonlinear structure invisible to linear methods.
    """
    rng = np.random.default_rng(42 + trial)
    x = np.zeros(size + 500)
    y = np.zeros(size + 500)
    for t in range(1, size + 500):
        x[t] = 0.8 * x[t-1] + 0.3 * y[t-1]**2 + 0.5 * rng.standard_normal()
        y[t] = 0.6 * y[t-1] + 0.2 * x[t-1] + 0.5 * rng.standard_normal()
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
# SIMPLE BASELINE
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
# UTILITIES
# =========================================================================

# Discover framework metrics
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_SIMPLE = ALPHA / N_SIMPLE
BONF_FULL = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn

print(f"Simple: {N_SIMPLE} features, Framework: {N_METRICS} metrics")
print(f"Bonferroni: simple α={BONF_SIMPLE:.2e}, framework α={BONF_FULL:.2e}")


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

        # Collect metrics
        orig_simple = collect_simple(originals)
        orig_full = collect_framework(analyzer, originals)

        shuf_simple = collect_simple(shuffle_surrs)
        shuf_full = collect_framework(analyzer, shuffle_surrs)

        ft_simple = collect_simple(ft_surrs)
        ft_full = collect_framework(analyzer, ft_surrs)

        iaaft_simple = collect_simple(iaaft_surrs)
        iaaft_full = collect_framework(analyzer, iaaft_surrs)

        all_data[name] = {
            'orig_simple': orig_simple, 'orig_full': orig_full,
            'shuf_simple': shuf_simple, 'shuf_full': shuf_full,
            'ft_simple': ft_simple, 'ft_full': ft_full,
            'iaaft_simple': iaaft_simple, 'iaaft_full': iaaft_full,
        }
        print("done")

    return all_data


# =========================================================================
# D1: SIMPLE BASELINE VS IAAFT (should fail)
# =========================================================================
def direction_1(all_data):
    print("\n" + "=" * 78)
    print("D1: SIMPLE BASELINE VS IAAFT SURROGATES")
    print("  (By construction, linear features should fail here)")
    print("=" * 78)

    d1 = {}
    for name in SIGNALS:
        d = all_data[name]
        n_sig, findings = count_sig(d['orig_simple'], d['iaaft_simple'],
                                     SIMPLE_NAMES, BONF_SIMPLE)
        d1[name] = n_sig
        status = "PASS (detected)" if n_sig > 0 else "FAIL (as expected)"
        print(f"  {name:15s}: {n_sig:2d}/{N_SIMPLE} sig — {status}")
        for m, dval, p in findings[:3]:
            print(f"    {m:25s}  d={dval:+.2f}")

    return d1


# =========================================================================
# D2: FULL FRAMEWORK VS IAAFT (the key test)
# =========================================================================
def direction_2(all_data):
    print("\n" + "=" * 78)
    print("D2: FULL FRAMEWORK VS IAAFT SURROGATES")
    print("  (This is the key test — can exotic geometry see nonlinear structure?)")
    print("=" * 78)

    d2 = {}
    for name in SIGNALS:
        d = all_data[name]
        n_sig, findings = count_sig(d['orig_full'], d['iaaft_full'],
                                     METRIC_NAMES, BONF_FULL)
        d2[name] = {'n_sig': n_sig, 'findings': findings}
        print(f"\n  {name:15s}: {n_sig:3d}/{N_METRICS} sig")
        for m, dval, p in findings[:5]:
            print(f"    {m:50s}  d={dval:+.2f}")

    return d2


# =========================================================================
# D3: GEOMETRY-BY-GEOMETRY NONLINEAR DETECTION
# =========================================================================
def direction_3(all_data):
    print("\n" + "=" * 78)
    print("D3: WHICH GEOMETRIES DETECT NONLINEAR STRUCTURE?")
    print("=" * 78)

    geometry_names = sorted(set(m.split(':')[0] for m in METRIC_NAMES))
    geo_to_metrics = {g: [m for m in METRIC_NAMES if m.startswith(g + ':')]
                      for g in geometry_names}

    geo_scores = {g: 0 for g in geometry_names}

    for name in SIGNALS:
        d = all_data[name]
        for geo in geometry_names:
            metrics = geo_to_metrics[geo]
            bonf_geo = ALPHA / max(len(metrics), 1)
            n_sig, _ = count_sig(d['orig_full'], d['iaaft_full'], metrics, bonf_geo)
            geo_scores[geo] += n_sig

    ranked = sorted(geo_scores.items(), key=lambda x: -x[1])
    print(f"\n  {'Geometry':<45s} {'Nonlinear detections':>20s}")
    print(f"  {'─'*45} {'─'*20}")
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
# D5: SMOKING GUN — WHERE SIMPLE=0 BUT EXOTIC>0
# =========================================================================
def direction_5(all_data, d1, d2):
    print("\n" + "=" * 78)
    print("D5: THE SMOKING GUN — SIMPLE=0, EXOTIC>0")
    print("=" * 78)

    smoking_guns = []
    for name in SIGNALS:
        simple_sig = d1[name]
        exotic_sig = d2[name]['n_sig']
        exotic_findings = d2[name]['findings']

        is_gun = simple_sig == 0 and exotic_sig > 0
        marker = " *** SMOKING GUN ***" if is_gun else ""
        print(f"\n  {name:15s}: simple={simple_sig:2d}, exotic={exotic_sig:3d}{marker}")

        if is_gun and exotic_findings:
            smoking_guns.append(name)
            print(f"  Nonlinear structure detected by:")
            for m, dval, p in exotic_findings[:10]:
                geo = m.split(':')[0]
                print(f"    {geo:35s} | {m.split(':',1)[1]:25s}  d={dval:+.2f}")

    if smoking_guns:
        print(f"\n  SMOKING GUNS FOUND: {len(smoking_guns)}")
        print(f"  Signals: {', '.join(smoking_guns)}")
        print(f"  These prove exotic geometry detects nonlinear structure")
        print(f"  that no combination of linear features can replicate.")
    else:
        print(f"\n  No pure smoking guns found.")
        print(f"  Simple baseline may have leaked some detections")
        print(f"  (IAAFT is approximate, not perfect).")

    return smoking_guns


# =========================================================================
# FIGURE
# =========================================================================
def make_figure(all_data, d1, d2, d4, geo_scores, smoking_guns):
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
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                           height_ratios=[1.0, 1.0, 1.2])

    names = list(SIGNALS.keys())
    x = np.arange(len(names))

    # D1+D2: Simple vs Exotic against IAAFT
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    simple_vals = [d1[n] for n in names]
    exotic_vals = [d2[n]['n_sig'] for n in names]
    ax.bar(x - 0.2, simple_vals, 0.35, color='#FF9800', alpha=0.85, label=f'Simple ({N_SIMPLE})')
    ax.bar(x + 0.2, exotic_vals, 0.35, color='#2196F3', alpha=0.85, label=f'Framework ({N_METRICS})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Sig metrics vs IAAFT', fontsize=9)
    ax.set_title('D1+D2: Simple vs exotic — IAAFT surrogates', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')
    # Mark smoking guns
    for i, n in enumerate(names):
        if n in smoking_guns:
            ax.annotate('★', (i + 0.2, exotic_vals[i] + 1), fontsize=14,
                       color='#FFD700', ha='center', fontweight='bold')

    # D4: Structure hierarchy
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    shuf_vals = [d4[n]['shuffle'] for n in names]
    ft_vals = [d4[n]['ft'] for n in names]
    iaaft_vals = [d4[n]['iaaft'] for n in names]
    width = 0.25
    ax.bar(x - width, shuf_vals, width, color='#E91E63', alpha=0.85, label='Shuffle')
    ax.bar(x, ft_vals, width, color='#4CAF50', alpha=0.85, label='FT')
    ax.bar(x + width, iaaft_vals, width, color='#2196F3', alpha=0.85, label='IAAFT')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Sig metrics', fontsize=9)
    ax.set_title('D4: Structure hierarchy (what each surrogate preserves)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # D3: Geometry ranking for nonlinear detection
    ax = _dark_ax(fig.add_subplot(gs[1, :]))
    ranked = sorted(geo_scores.items(), key=lambda x: -x[1])
    geo_names_sorted = [g[0][:30] for g in ranked]
    geo_vals_sorted = [g[1] for g in ranked]
    colors = ['#2196F3' if v > 0 else '#666' for v in geo_vals_sorted]
    ax.barh(range(len(geo_names_sorted)), geo_vals_sorted, color=colors, alpha=0.85)
    ax.set_yticks(range(len(geo_names_sorted)))
    ax.set_yticklabels(geo_names_sorted, fontsize=7)
    ax.set_xlabel('Total nonlinear detections (across 7 signals)', fontsize=9)
    ax.set_title('D3: Which geometries detect nonlinear structure?', fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    # D5: Smoking gun detail
    ax = _dark_ax(fig.add_subplot(gs[2, :]))
    if smoking_guns:
        # Show top 15 exotic findings across smoking gun signals
        all_findings = []
        for name in smoking_guns:
            for m, dval, p in d2[name]['findings'][:5]:
                all_findings.append((name, m, dval))
        all_findings.sort(key=lambda x: -abs(x[2]))
        all_findings = all_findings[:15]

        labels = [f"{f[0]}: {f[1].split(':',1)[1][:30]}" for f in all_findings]
        d_vals = [abs(f[2]) for f in all_findings]
        geos = [f[1].split(':')[0][:20] for f in all_findings]

        colors_bar = ['#E91E63' if d > 5 else '#FF9800' if d > 2 else '#4CAF50'
                      for d in d_vals]
        ax.barh(range(len(labels)), d_vals, color=colors_bar, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('|Cohen\'s d| (effect size)', fontsize=9)
        ax.set_title(f'D5: Smoking gun detections ({len(smoking_guns)} signals)',
                     fontsize=11, fontweight='bold')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'No pure smoking guns found\n(IAAFT approximation may leak)',
                transform=ax.transAxes, ha='center', va='center', fontsize=14,
                color='#666')
        ax.set_title('D5: No smoking guns', fontsize=11, fontweight='bold')

    fig.suptitle('Surrogate Testing: Do Exotic Geometries Detect Nonlinear Structure?',
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
    smoking_guns = direction_5(all_data, d1, d2)
    make_figure(all_data, d1, d2, d4, geo_scores, smoking_guns)

    print(f"\n{'=' * 78}")
    print("VERDICT")
    print(f"{'=' * 78}")
    for name in SIGNALS:
        s = d1[name]
        e = d2[name]['n_sig']
        h = d4[name]
        gun = " ★" if name in smoking_guns else ""
        print(f"  {name:15s}: simple={s:2d}  exotic={e:3d}  "
              f"(shuf={h['shuffle']:3d} FT={h['ft']:3d} IAAFT={h['iaaft']:3d}){gun}")

    n_guns = len(smoking_guns)
    total_exotic = sum(d2[n]['n_sig'] for n in SIGNALS)
    total_simple = sum(d1[n] for n in SIGNALS)
    print(f"\n  Smoking guns: {n_guns}/{len(SIGNALS)} signals")
    print(f"  Total simple detections (vs IAAFT): {total_simple}")
    print(f"  Total exotic detections (vs IAAFT): {total_exotic}")
    if total_exotic > 0 and total_simple == 0:
        print(f"\n  ★ FRAMEWORK DETECTS PURELY NONLINEAR STRUCTURE ★")
        print(f"  Simple features: 0 detections. Exotic geometries: {total_exotic}.")
        print(f"  This proves the geometric embeddings capture information")
        print(f"  that no combination of linear features can replicate.")
    elif total_exotic > total_simple:
        print(f"\n  Framework detects more nonlinear structure than simple baseline.")
        print(f"  Gain: +{total_exotic - total_simple} detections beyond simple features.")
    else:
        print(f"\n  No evidence that exotic geometries detect unique nonlinear structure.")
