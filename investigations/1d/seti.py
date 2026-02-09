#!/usr/bin/env python3
"""
Investigation: Geometric SETI — Detecting Artificial Signals in Cosmic Noise
=============================================================================

Current SETI uses matched filters: narrowband carriers, chirped signals,
specific pulse patterns. You only find what you look for. This investigation
asks: can exotic geometry detect ARBITRARY artificial structure in realistic
radio noise — including modulation schemes that spectral analysis would miss?

Signal types:
  Natural (should NOT trigger detection):
    thermal     — White Gaussian noise (the null hypothesis)
    colored     — 1/f^α noise (typical radio background)
    pulsar      — Periodic pulses in colored noise (natural but structured)
    frb         — Fast radio burst-like transient
    solar       — Broadband emission with intermittent flares

  Artificial (should trigger detection):
    carrier     — Narrowband sine wave (what SETI currently looks for)
    chirp       — Frequency-sweeping signal (another standard target)
    spread      — Spread-spectrum (energy distributed across all frequencies)
    chaotic     — Lorenz-modulated carrier (deterministic but unpredictable)
    digital     — BPSK-modulated random data (communication-like)
    math_seq    — Prime-gap-modulated signal (mathematical intelligence marker)
    compressed  — Encrypted/compressed bitstream as modulation

Directions:
  D1: Natural vs random — do natural sources trigger false positives?
  D2: Artificial vs noise — can each artificial signal type be detected?
  D3: SNR sensitivity — at what power level does each artificial signal
      become detectable? (The "minimum detectable signal" for each type)
  D4: What matched filters miss — signals invisible to spectral analysis
      but visible to geometric analysis
  D5: The cocktail test — artificial signals embedded in colored noise
      (realistic scenario)

Methodology: N_TRIALS=25, DATA_SIZE=2000, Bonferroni correction, Cohen's d > 0.8.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats, signal as sp_signal
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05
IAAFT_ITERATIONS = 100


# =========================================================================
# NOISE MODELS
# =========================================================================

def gen_thermal(trial, size):
    """White Gaussian noise — the null hypothesis."""
    rng = np.random.default_rng(42 + trial)
    return rng.standard_normal(size)


def gen_colored(trial, size, alpha=1.0):
    """1/f^alpha noise (alpha=1: pink, alpha=2: brown/red).
    Typical radio telescope background is pink-ish."""
    rng = np.random.default_rng(42 + trial)
    white = rng.standard_normal(size)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(size)
    freqs[0] = 1  # avoid division by zero
    fft *= 1.0 / (freqs ** (alpha / 2))
    colored = np.fft.irfft(fft, n=size)
    return colored / np.std(colored)  # unit variance


def gen_pulsar(trial, size):
    """Periodic pulses in colored noise — natural but structured."""
    rng = np.random.default_rng(42 + trial)
    noise = gen_colored(trial + 1000, size, alpha=0.8)
    period = 40 + rng.integers(-10, 10)
    pulse_width = max(2, period // 15)
    pulses = np.zeros(size)
    for i in range(0, size, period):
        for j in range(pulse_width):
            if i + j < size:
                pulses[i + j] = 2.0 * np.exp(-0.5 * (j / max(pulse_width / 3, 1)) ** 2)
    # Add jitter to pulse amplitude
    for i in range(0, size, period):
        amp = 1.5 + 0.5 * rng.standard_normal()
        for j in range(pulse_width):
            if i + j < size:
                pulses[i + j] *= max(amp, 0.3)
    return noise + pulses


def gen_frb(trial, size):
    """Fast radio burst: a single bright transient in noise."""
    rng = np.random.default_rng(42 + trial)
    noise = gen_colored(trial + 2000, size, alpha=0.5)
    # One or two bursts at random positions
    n_bursts = 1 + (trial % 2)
    for _ in range(n_bursts):
        pos = rng.integers(size // 4, 3 * size // 4)
        width = 5 + rng.integers(0, 10)
        amplitude = 5.0 + 2.0 * rng.standard_normal()
        t = np.arange(size)
        burst = abs(amplitude) * np.exp(-0.5 * ((t - pos) / width) ** 2)
        # Add frequency sweep (dispersion)
        burst *= np.cos(2 * np.pi * (0.1 + 0.001 * (t - pos)) * (t - pos))
        noise += burst
    return noise


def gen_solar(trial, size):
    """Broadband solar-type emission with intermittent flares."""
    rng = np.random.default_rng(42 + trial)
    noise = gen_colored(trial + 3000, size, alpha=1.2)
    # Add random flares (exponential rise, slow decay)
    n_flares = 2 + rng.integers(0, 4)
    for _ in range(n_flares):
        pos = rng.integers(0, size)
        amp = 3.0 + 2.0 * abs(rng.standard_normal())
        rise = 3 + rng.integers(0, 5)
        decay = 15 + rng.integers(0, 20)
        t = np.arange(size) - pos
        flare = np.where(t < 0, amp * np.exp(t / rise), amp * np.exp(-t / decay))
        noise += np.clip(flare, 0, None)
    return noise


# =========================================================================
# ARTIFICIAL SIGNAL GENERATORS
# =========================================================================

def gen_carrier(trial, size):
    """Pure narrowband sine wave — what SETI already looks for."""
    rng = np.random.default_rng(42 + trial)
    freq = 0.05 + 0.1 * rng.uniform()
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(size, dtype=np.float64)
    return np.sin(2 * np.pi * freq * t + phase)


def gen_chirp(trial, size):
    """Frequency-sweeping signal — another standard SETI target."""
    rng = np.random.default_rng(42 + trial)
    f0 = 0.01 + 0.05 * rng.uniform()
    f1 = 0.1 + 0.15 * rng.uniform()
    t = np.arange(size, dtype=np.float64) / size
    return np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t * size)


def gen_spread_spectrum(trial, size):
    """Spread-spectrum: energy distributed across all frequencies.
    Uses a pseudo-random chipping sequence to spread a narrowband signal.
    Hard for spectral analysis — looks like noise in any single frequency bin."""
    rng = np.random.default_rng(42 + trial)
    # Data bits (slow)
    n_bits = 20
    bits = rng.choice([-1.0, 1.0], size=n_bits)
    data = np.repeat(bits, size // n_bits + 1)[:size]
    # Chipping sequence (fast pseudo-random)
    chip_rate = 50
    chips = rng.choice([-1.0, 1.0], size=size * chip_rate // size)
    chip_signal = np.repeat(chips, size // len(chips) + 1)[:size]
    # Spread = data × chips
    return data * chip_signal


def gen_chaotic_modulation(trial, size):
    """Lorenz-modulated carrier: deterministic but unpredictable.
    The kind of signal that would defeat any matched filter."""
    rng = np.random.default_rng(42 + trial)
    dt = 0.01
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x, y, z = 1.0 + 0.1 * rng.standard_normal(3)
    warmup = 2000
    vals = []
    for _ in range(warmup + size * 3):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx; y += dy; z += dz
        vals.append(x)
    lorenz = np.array(vals[warmup::3][:size])
    lorenz = lorenz / np.max(np.abs(lorenz))
    # Modulate a carrier
    t = np.arange(size, dtype=np.float64)
    carrier_freq = 0.1 + 0.05 * rng.uniform()
    return lorenz * np.sin(2 * np.pi * carrier_freq * t)


def gen_digital(trial, size):
    """BPSK-modulated random data — realistic digital communication."""
    rng = np.random.default_rng(42 + trial)
    # Random bits at various rates
    symbol_rate = 10 + rng.integers(0, 30)
    bits = rng.choice([-1.0, 1.0], size=size // symbol_rate + 1)
    data = np.repeat(bits, symbol_rate)[:size]
    # BPSK: multiply by carrier
    t = np.arange(size, dtype=np.float64)
    carrier_freq = 0.15 + 0.1 * rng.uniform()
    return data * np.sin(2 * np.pi * carrier_freq * t)


def gen_math_sequence(trial, size):
    """Prime-gap-modulated signal — mathematical intelligence marker.
    The gaps between primes encode the signal. An alien wanting to be
    noticed might broadcast mathematical structure."""
    rng = np.random.default_rng(42 + trial)
    # Generate prime gaps
    def sieve(limit):
        is_p = np.ones(limit + 1, dtype=bool)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i * i::i] = False
        return np.where(is_p)[0]

    primes = sieve(50000)
    start = rng.integers(100, len(primes) - size - 1)
    gaps = np.diff(primes[start:start + size + 1]).astype(np.float64)
    # Normalize and use as amplitude modulation
    gaps = (gaps - gaps.mean()) / max(gaps.std(), 1e-10)
    t = np.arange(size, dtype=np.float64)
    carrier_freq = 0.12 + 0.06 * rng.uniform()
    return gaps * np.sin(2 * np.pi * carrier_freq * t)


def gen_compressed(trial, size):
    """Encrypted/compressed bitstream as modulation.
    Looks spectrally flat (like noise) but has subtle higher-order structure
    from the compression algorithm. The hardest case."""
    rng = np.random.default_rng(42 + trial)
    # Simulate compressed data: transform random text through a pseudo-compression
    # Use logistic map at r=3.99 as a "compression-like" deterministic sequence
    x = 0.1 + 0.01 * rng.uniform()
    vals = []
    for _ in range(500 + size):
        x = 3.99 * x * (1 - x)
        vals.append(x)
    bits = np.array(vals[500:500 + size])
    # Convert to ±1
    bits = 2.0 * (bits > 0.5).astype(float) - 1.0
    # BPSK modulate
    t = np.arange(size, dtype=np.float64)
    carrier_freq = 0.13 + 0.07 * rng.uniform()
    return bits * np.sin(2 * np.pi * carrier_freq * t)


NATURAL = {
    'thermal':  gen_thermal,
    'colored':  gen_colored,
    'pulsar':   gen_pulsar,
    'frb':      gen_frb,
    'solar':    gen_solar,
}

ARTIFICIAL = {
    'carrier':      gen_carrier,
    'chirp':        gen_chirp,
    'spread':       gen_spread_spectrum,
    'chaotic':      gen_chaotic_modulation,
    'digital':      gen_digital,
    'math_seq':     gen_math_sequence,
    'compressed':   gen_compressed,
}


# =========================================================================
# UTILITIES
# =========================================================================

def to_uint8(x):
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 1e-10:
        return np.full(len(x), 128, dtype=np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def embed_in_noise(signal_float, noise_float, snr_linear):
    """Embed signal in noise at given SNR (linear, not dB).
    SNR = signal_power / noise_power."""
    sig_power = np.var(signal_float)
    noise_power = np.var(noise_float)
    if noise_power < 1e-15:
        return signal_float.copy()
    # Scale signal to achieve desired SNR
    scale = np.sqrt(snr_linear * noise_power / max(sig_power, 1e-15))
    return noise_float + scale * signal_float


def snr_to_db(snr):
    if snr <= 0:
        return float('-inf')
    return 10 * np.log10(snr)


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


def compare_to_thermal(analyzer, gen_fn, n_trials=N_TRIALS, noise_gen=gen_thermal):
    """Compare a signal type against thermal noise baseline."""
    sig_arrays = [to_uint8(gen_fn(t, DATA_SIZE)) for t in range(n_trials)]
    noise_arrays = [to_uint8(noise_gen(t + 5000, DATA_SIZE)) for t in range(n_trials)]

    sig_metrics = {m: [] for m in METRIC_NAMES}
    noise_metrics = {m: [] for m in METRIC_NAMES}

    for arr in sig_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in sig_metrics and np.isfinite(mv):
                    sig_metrics[key].append(mv)

    for arr in noise_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in noise_metrics and np.isfinite(mv):
                    noise_metrics[key].append(mv)

    n_sig = 0
    findings = []
    for m in METRIC_NAMES:
        a = np.array(sig_metrics.get(m, []))
        b = np.array(noise_metrics.get(m, []))
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < BONF and abs(d) > 0.8:
            n_sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return n_sig, findings


def compare_embedded(analyzer, sig_gen, noise_gen, snr_linear, n_trials=N_TRIALS):
    """Compare signal-in-noise against pure noise."""
    embedded_arrays = []
    noise_arrays = []

    for t in range(n_trials):
        sig = sig_gen(t, DATA_SIZE)
        noise = noise_gen(t + 5000, DATA_SIZE)
        combined = embed_in_noise(sig, noise, snr_linear)
        embedded_arrays.append(to_uint8(combined))

        pure_noise = noise_gen(t + 6000, DATA_SIZE)
        noise_arrays.append(to_uint8(pure_noise))

    emb_metrics = {m: [] for m in METRIC_NAMES}
    noise_metrics = {m: [] for m in METRIC_NAMES}

    for arr in embedded_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in emb_metrics and np.isfinite(mv):
                    emb_metrics[key].append(mv)

    for arr in noise_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in noise_metrics and np.isfinite(mv):
                    noise_metrics[key].append(mv)

    n_sig = 0
    findings = []
    for m in METRIC_NAMES:
        a = np.array(emb_metrics.get(m, []))
        b = np.array(noise_metrics.get(m, []))
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < BONF and abs(d) > 0.8:
            n_sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return n_sig, findings


# Simple spectral detection (matched filter proxy)
def spectral_detect(sig_arrays, noise_arrays):
    """Simple spectral analysis: can the signal be distinguished by
    peak frequency, spectral flatness, and spectral entropy?"""
    def spectral_features(arr):
        x = arr.astype(np.float64)
        fft = np.abs(np.fft.rfft(x))[1:]
        power = fft ** 2
        total = np.sum(power) + 1e-20
        # Peak frequency
        peak_freq = np.argmax(power)
        # Spectral flatness (Wiener entropy)
        log_mean = np.mean(np.log(power + 1e-20))
        arith_mean = np.mean(power)
        flatness = np.exp(log_mean) / (arith_mean + 1e-20)
        # Spectral entropy
        p = power / total
        entropy = -np.sum(p * np.log2(p + 1e-20))
        return peak_freq, flatness, entropy

    sig_feats = [spectral_features(a) for a in sig_arrays]
    noise_feats = [spectral_features(a) for a in noise_arrays]

    n_detected = 0
    bonf_spectral = ALPHA / 3
    for i in range(3):
        a = np.array([f[i] for f in sig_feats])
        b = np.array([f[i] for f in noise_feats])
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < bonf_spectral and abs(d) > 0.8:
            n_detected += 1
    return n_detected


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# D1: NATURAL SOURCES VS THERMAL — FALSE POSITIVE CHECK
# =========================================================================

def direction_1(analyzer):
    print("=" * 78)
    print("D1: NATURAL SOURCES VS THERMAL NOISE — FALSE POSITIVE CHECK")
    print("=" * 78)
    print("  Do natural astrophysical signals trigger false detection?\n")

    d1 = {}
    for name, gen_fn in NATURAL.items():
        if name == 'thermal':
            continue
        print(f"  {name:12s} vs thermal...", end=" ", flush=True)
        n_sig, findings = compare_to_thermal(analyzer, gen_fn)
        d1[name] = {'n_sig': n_sig, 'findings': findings}
        print(f"{n_sig} sig")
        for m, d, p in findings[:3]:
            print(f"    {m:50s}  d={d:+.2f}")

    # Also thermal vs thermal (should be ~0)
    print(f"  {'thermal':12s} vs thermal...", end=" ", flush=True)
    n_sig, findings = compare_to_thermal(analyzer, gen_thermal)
    d1['thermal_self'] = {'n_sig': n_sig}
    print(f"{n_sig} sig (sanity: should be ~0)")

    return d1


# =========================================================================
# D2: ARTIFICIAL SIGNALS VS THERMAL — CLEAN DETECTION
# =========================================================================

def direction_2(analyzer):
    print("\n" + "=" * 78)
    print("D2: ARTIFICIAL SIGNALS VS THERMAL NOISE — CLEAN DETECTION")
    print("=" * 78)
    print("  Can each artificial signal type be detected in isolation?\n")

    d2 = {}
    for name, gen_fn in ARTIFICIAL.items():
        print(f"  {name:12s} vs thermal...", end=" ", flush=True)
        n_sig, findings = compare_to_thermal(analyzer, gen_fn)
        d2[name] = {'n_sig': n_sig, 'findings': findings}
        print(f"{n_sig} sig")
        for m, d, p in findings[:3]:
            print(f"    {m:50s}  d={d:+.2f}")

    return d2


# =========================================================================
# D3: SNR SENSITIVITY — MINIMUM DETECTABLE SIGNAL
# =========================================================================

def direction_3(analyzer):
    print("\n" + "=" * 78)
    print("D3: SNR SENSITIVITY — MINIMUM DETECTABLE SIGNAL IN COLORED NOISE")
    print("=" * 78)
    print("  At what SNR does each artificial signal become detectable?")
    print("  Noise = 1/f colored (realistic radio background).\n")

    # SNR levels (linear): 0.001 to 10
    snr_levels = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    d3 = {}
    for sig_name in ['carrier', 'spread', 'chaotic', 'math_seq', 'compressed']:
        gen_fn = ARTIFICIAL[sig_name]
        print(f"  {sig_name}:")
        d3[sig_name] = {}

        for snr in snr_levels:
            print(f"    SNR={snr:.3f} ({snr_to_db(snr):+.0f}dB)...", end=" ", flush=True)
            n_sig, findings = compare_embedded(analyzer, gen_fn, gen_colored, snr)

            # Also check spectral detection
            embedded = [to_uint8(embed_in_noise(gen_fn(t, DATA_SIZE),
                       gen_colored(t + 5000, DATA_SIZE), snr)) for t in range(N_TRIALS)]
            noise_only = [to_uint8(gen_colored(t + 6000, DATA_SIZE)) for t in range(N_TRIALS)]
            spectral = spectral_detect(embedded, noise_only)

            d3[sig_name][snr] = {'fw': n_sig, 'spectral': spectral}
            print(f"fw={n_sig:3d}, spectral={spectral}/3")

    return d3


# =========================================================================
# D4: WHAT MATCHED FILTERS MISS
# =========================================================================

def direction_4(analyzer):
    print("\n" + "=" * 78)
    print("D4: WHAT MATCHED FILTERS MISS")
    print("=" * 78)
    print("  Signals embedded in colored noise at SNR=0.5 (~-3dB).")
    print("  Compare: framework detection vs spectral detection.\n")

    target_snr = 0.5
    d4 = {}
    for sig_name, gen_fn in ARTIFICIAL.items():
        print(f"  {sig_name:12s} @ SNR=0.5...", end=" ", flush=True)

        n_sig, findings = compare_embedded(analyzer, gen_fn, gen_colored, target_snr)

        embedded = [to_uint8(embed_in_noise(gen_fn(t, DATA_SIZE),
                   gen_colored(t + 5000, DATA_SIZE), target_snr)) for t in range(N_TRIALS)]
        noise_only = [to_uint8(gen_colored(t + 6000, DATA_SIZE)) for t in range(N_TRIALS)]
        spectral = spectral_detect(embedded, noise_only)

        d4[sig_name] = {
            'fw': n_sig,
            'spectral': spectral,
            'findings': findings,
            'fw_only': n_sig > 0 and spectral == 0,
        }
        marker = " ★ GEOMETRY-ONLY" if d4[sig_name]['fw_only'] else ""
        print(f"fw={n_sig:3d}, spectral={spectral}/3{marker}")
        if d4[sig_name]['fw_only'] and findings:
            for m, d, p in findings[:3]:
                print(f"    {m:50s}  d={d:+.2f}")

    return d4


# =========================================================================
# D5: COCKTAIL TEST — REALISTIC SCENARIO
# =========================================================================

def direction_5(analyzer):
    print("\n" + "=" * 78)
    print("D5: THE COCKTAIL TEST — REALISTIC DETECTION SCENARIO")
    print("=" * 78)
    print("  Can we detect artificial signals in a realistic mix of")
    print("  colored noise + RFI (radio frequency interference)?")
    print("  Noise = colored + intermittent broadband bursts.\n")

    def gen_realistic_noise(trial, size):
        """Colored noise with occasional RFI spikes."""
        noise = gen_colored(trial + 7000, size, alpha=0.8)
        rng = np.random.default_rng(trial + 8000)
        # Add RFI bursts
        n_rfi = rng.integers(2, 6)
        for _ in range(n_rfi):
            pos = rng.integers(0, size)
            width = rng.integers(5, 30)
            amp = 3.0 + 2.0 * abs(rng.standard_normal())
            t = np.arange(size)
            rfi = amp * np.exp(-0.5 * ((t - pos) / width) ** 2)
            noise += rfi
        return noise

    snr_levels = [0.01, 0.05, 0.1, 0.5, 1.0]

    d5 = {}
    for sig_name in ['spread', 'chaotic', 'math_seq']:
        gen_fn = ARTIFICIAL[sig_name]
        print(f"  {sig_name}:")
        d5[sig_name] = {}

        for snr in snr_levels:
            print(f"    SNR={snr:.2f} ({snr_to_db(snr):+.0f}dB)...", end=" ", flush=True)
            n_sig, findings = compare_embedded(analyzer, gen_fn, gen_realistic_noise, snr)
            d5[sig_name][snr] = n_sig
            print(f"{n_sig} sig")

    return d5


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(d1, d2, d3, d4, d5):
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
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # D1+D2: Natural vs Artificial detection
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    natural_names = [n for n in d1 if n != 'thermal_self']
    natural_vals = [d1[n]['n_sig'] for n in natural_names]
    art_names = list(d2.keys())
    art_vals = [d2[n]['n_sig'] for n in art_names]

    all_names = natural_names + [''] + art_names
    all_vals = natural_vals + [0] + art_vals
    colors = (['#4CAF50'] * len(natural_names) + ['#181818'] +
              ['#2196F3'] * len(art_names))

    y = np.arange(len(all_names))
    ax.barh(y, all_vals, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(all_names, fontsize=7)
    ax.set_xlabel('Significant metrics vs thermal noise', fontsize=9)
    ax.set_title('D1+D2: Natural (green) vs Artificial (blue)', fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='#666', linewidth=0.5)
    ax.invert_yaxis()

    # D3: SNR sensitivity curves
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    sig_colors = {
        'carrier': '#FFD700', 'spread': '#E91E63',
        'chaotic': '#2196F3', 'math_seq': '#4CAF50',
        'compressed': '#FF9800',
    }
    for sig_name in d3:
        snrs = sorted(d3[sig_name].keys())
        fw_vals = [d3[sig_name][s]['fw'] for s in snrs]
        snr_dbs = [snr_to_db(s) for s in snrs]
        ax.plot(snr_dbs, fw_vals, 'o-', color=sig_colors.get(sig_name, '#888'),
                label=sig_name, linewidth=2, markersize=4)
    ax.set_xlabel('SNR (dB)', fontsize=9)
    ax.set_ylabel('Significant metrics', fontsize=9)
    ax.set_title('D3: Detection sensitivity in colored noise', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666')
    ax.axhline(y=0, color='#666', linestyle='--', linewidth=0.5)

    # D4: Framework vs spectral (what matched filters miss)
    ax = _dark_ax(fig.add_subplot(gs[1, :]))
    art_names = list(d4.keys())
    fw_vals = [d4[n]['fw'] for n in art_names]
    spec_vals = [d4[n]['spectral'] for n in art_names]
    x = np.arange(len(art_names))
    ax.bar(x - 0.18, fw_vals, 0.32, color='#2196F3', alpha=0.85,
           label=f'Framework ({N_METRICS} metrics)')
    ax.bar(x + 0.18, spec_vals, 0.32, color='#FF9800', alpha=0.85,
           label='Spectral (3 features)')
    for i, n in enumerate(art_names):
        if d4[n]['fw_only']:
            ax.annotate('★', (i - 0.18, fw_vals[i] + 0.5), fontsize=14,
                        color='#FFD700', ha='center', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(art_names, fontsize=8)
    ax.set_ylabel('Detections', fontsize=9)
    ax.set_title('D4: Framework vs matched filter @ SNR=0.5 in colored noise '
                 '(★ = geometry detects, spectral misses)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # D5: Cocktail test (realistic scenario)
    ax = _dark_ax(fig.add_subplot(gs[2, :]))
    for sig_name in d5:
        snrs = sorted(d5[sig_name].keys())
        vals = [d5[sig_name][s] for s in snrs]
        snr_dbs = [snr_to_db(s) for s in snrs]
        ax.plot(snr_dbs, vals, 'o-', color=sig_colors.get(sig_name, '#888'),
                label=sig_name, linewidth=2.5, markersize=6)
    ax.set_xlabel('SNR (dB)', fontsize=10)
    ax.set_ylabel('Significant metrics', fontsize=10)
    ax.set_title('D5: Detection in realistic noise (colored + RFI)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#333', edgecolor='#666')
    ax.axhline(y=0, color='#666', linestyle='--', linewidth=0.5)

    fig.suptitle('Geometric SETI: Detecting Artificial Signals in Cosmic Noise',
                 fontsize=14, fontweight='bold', color='white', y=0.995)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'seti.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved seti.png")
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
    d5 = direction_5(analyzer)

    make_figure(d1, d2, d3, d4, d5)

    # Summary
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")

    print("\n  D1 — False positives on natural sources:")
    for name in d1:
        if name != 'thermal_self':
            n = d1[name]['n_sig']
            status = "✓ expected (structured)" if n > 0 else "✓ clean"
            print(f"    {name:12s}: {n:3d} sig  {status}")

    print("\n  D2 — Clean artificial detection:")
    for name in d2:
        print(f"    {name:12s}: {d2[name]['n_sig']:3d} sig")

    print("\n  D3 — Minimum detectable SNR (≥1 sig metric):")
    for sig_name in d3:
        for snr in sorted(d3[sig_name].keys()):
            if d3[sig_name][snr]['fw'] > 0:
                print(f"    {sig_name:12s}: SNR={snr:.3f} ({snr_to_db(snr):+.0f}dB)")
                break
        else:
            print(f"    {sig_name:12s}: not detected at any SNR")

    print("\n  D4 — What matched filters miss @ SNR=0.5:")
    for name in d4:
        if d4[name]['fw_only']:
            print(f"    ★ {name:12s}: fw={d4[name]['fw']}, spectral=0 — GEOMETRY ONLY")
        else:
            print(f"      {name:12s}: fw={d4[name]['fw']}, spectral={d4[name]['spectral']}")

    n_geo_only = sum(1 for n in d4 if d4[n]['fw_only'])
    print(f"\n  Geometry-only detections: {n_geo_only}/{len(ARTIFICIAL)} signal types")
    print(f"  These signals would be MISSED by standard spectral SETI.")
