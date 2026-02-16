#!/usr/bin/env python3
"""
Investigation: Signature Space — PC Interpretation + 7D Breaker Hunt
=====================================================================

The meta-investigation established that the framework's 41 classifier
signatures live in a ~7-dimensional subspace (participation ratio = 7.0,
8 singular values above Marchenko-Pastur). Two open questions:

1. What do the 7 axes mean? (PC interpretation via pole analysis)
2. Is 7 the true limit, or an artifact of the 41 signatures chosen?
   (Probe with structurally orthogonal new sources)

DIRECTIONS:
D1: PC Interpretation — label the 7 axes by their poles and loadings
D2: PC Loadings Heatmap — V^T matrix grouped by geometry family
D3: New Candidate Generators — 8 sources designed to probe 7D gaps
D4: Projection Test — do new sources break 7D?
D5: Updated Dimensionality — re-run SVD with new sources included
"""

import sys
import os
import json
import glob
import time
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.investigation_runner import Runner

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONSTANTS
# =============================================================================

CATEGORIES = {
    'chaos': [
        'Henon Chaos', 'Logistic Chaos', 'Logistic Edge-of-Chaos',
        'Lorenz Attractor (x)', 'Rossler Attractor', 'Baker Map',
        'Standard Map (Chirikov)', 'Tent Map',
    ],
    'number_theory': [
        'Prime Gaps', 'Prime Gap Pairs', 'Prime Gap Diffs',
        'Divisor Count d(n)', 'Totient Ratio', 'Mertens Function',
        'Moebius Function',
    ],
    'collatz': [
        'Collatz (Hailstone)', 'Collatz (Parity)',
        'Collatz High Bits', 'Collatz Stopping Times',
    ],
    'noise': [
        'Gaussian White Noise', 'Pink Noise', 'Perlin Noise', 'Random',
    ],
    'waveform': ['Sine Wave', 'Sawtooth Wave', 'Square Wave'],
    'binary': [
        'Mach-O Binary', 'x86-64 Architecture', 'ARM64 Architecture',
        'Java Bytecode', 'WASM Bytecode',
    ],
    'bio': [
        'Bacterial DNA (E. coli)', 'Eukaryotic DNA (Human)', 'Viral DNA',
    ],
    'other': [
        'AES-ECB (Structured)', 'glibc LCG', 'RANDU',
        'LZ4 Compressed', 'NN Trained Dense', 'NN Pruned 90%', 'ASCII Text',
    ],
}

CATEGORY_COLORS = {
    'chaos': '#e74c3c',
    'number_theory': '#3498db',
    'collatz': '#e67e22',
    'noise': '#95a5a6',
    'waveform': '#2ecc71',
    'binary': '#9b59b6',
    'bio': '#1abc9c',
    'other': '#f1c40f',
}

# Colors for the 8 new sources
NEW_COLORS = {
    'fBm H=0.1': '#ff6b6b',
    'fBm H=0.9': '#c0392b',
    'Multiplicative Cascade': '#e84393',
    'Thue-Morse': '#6c5ce7',
    'Rudin-Shapiro': '#a29bfe',
    'GF(256) Accumulate': '#00b894',
    'Poisson Spike Train': '#fdcb6e',
    'iid 4-Symbol': '#636e72',
}


# =============================================================================
# LOAD EXISTING SIGNATURES
# =============================================================================

def load_signatures():
    """Load all signature JSONs, extract tau1 metrics only.

    Handles signatures with different metric counts by aligning on metric
    names. Uses the intersection of tau1 metrics present in ALL signatures.
    """
    sig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'signatures')
    files = sorted(glob.glob(os.path.join(sig_dir, '*.json')))

    # First pass: find tau1 metrics common to all signatures
    sigs = []
    for f in files:
        with open(f) as fh:
            sig = json.load(fh)
        tau1_metrics = [m for m in sig['metrics'] if m.endswith(':tau1')]
        sig['_tau1_set'] = set(tau1_metrics)
        sig['_metric_to_idx'] = {m: i for i, m in enumerate(sig['metrics'])}
        sigs.append(sig)

    # Intersection of tau1 metrics across all signatures
    common_tau1 = sigs[0]['_tau1_set']
    for sig in sigs[1:]:
        common_tau1 &= sig['_tau1_set']
    metric_names = sorted(common_tau1)

    # Build aligned matrices
    names = []
    means_matrix = np.zeros((len(sigs), len(metric_names)), dtype=np.float64)
    stds_matrix = np.zeros((len(sigs), len(metric_names)), dtype=np.float64)

    for i, sig in enumerate(sigs):
        names.append(sig['name'])
        for j, m in enumerate(metric_names):
            idx = sig['_metric_to_idx'][m]
            means_matrix[i, j] = sig['means'][idx]
            stds_matrix[i, j] = sig['stds'][idx]

    # Build name -> category
    name_to_cat = {}
    for cat, cat_names in CATEGORIES.items():
        for n in cat_names:
            name_to_cat[n] = cat
    categories = [name_to_cat.get(n, 'other') for n in names]

    print(f"Loaded {len(names)} signatures, {len(metric_names)} tau1 metrics each")
    return names, categories, metric_names, means_matrix, stds_matrix


# =============================================================================
# D3: NEW CANDIDATE GENERATORS
# =============================================================================

def gen_fbm_01(rng, size):
    """Fractional Brownian motion H=0.1 (anti-persistent)."""
    freqs = np.fft.rfftfreq(size, d=1.0)
    freqs[0] = 1.0
    H = 0.1
    psd = freqs ** (-(2 * H + 1))
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    fft_vals = np.sqrt(psd) * np.exp(1j * phases)
    fft_vals[0] = 0
    signal = np.fft.irfft(fft_vals, n=size)
    signal = np.cumsum(signal)
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((signal - lo) / (hi - lo) * 255).astype(np.uint8)


def gen_fbm_09(rng, size):
    """Fractional Brownian motion H=0.9 (persistent)."""
    freqs = np.fft.rfftfreq(size, d=1.0)
    freqs[0] = 1.0
    H = 0.9
    psd = freqs ** (-(2 * H + 1))
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    fft_vals = np.sqrt(psd) * np.exp(1j * phases)
    fft_vals[0] = 0
    signal = np.fft.irfft(fft_vals, n=size)
    signal = np.cumsum(signal)
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((signal - lo) / (hi - lo) * 255).astype(np.uint8)


def gen_multiplicative_cascade(rng, size):
    """Log-normal multiplicative cascade (multifractal)."""
    # Start with uniform mass, iteratively bisect with random multipliers
    n_levels = int(np.ceil(np.log2(size)))
    n_bins = 2 ** n_levels
    mass = np.ones(n_bins, dtype=np.float64)

    sigma = 0.5  # log-normal volatility
    for level in range(n_levels):
        step = n_bins >> (level + 1)
        for i in range(0, n_bins, 2 * step):
            # Random log-normal multiplier pair that conserves mass
            w = rng.lognormal(0, sigma)
            total = mass[i:i + 2 * step].sum()
            left_frac = w / (w + 1.0)
            mass[i:i + step] *= left_frac * 2
            mass[i + step:i + 2 * step] *= (1 - left_frac) * 2

    # Truncate to size, normalize to uint8
    signal = mass[:size]
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((signal - lo) / (hi - lo) * 255).astype(np.uint8)


def gen_thue_morse(rng, size):
    """Thue-Morse sequence — perfectly balanced non-periodic substitution."""
    # T(n) = number of 1s in binary representation of n, mod 2
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        # popcount mod 2
        n = i
        bits = 0
        while n:
            bits += n & 1
            n >>= 1
        vals[i] = 255 if (bits % 2) else 0
    # Add slight jitter from rng to allow trial variation
    jitter = rng.integers(-10, 11, size, dtype=np.int16)
    return np.clip(vals.astype(np.int16) + jitter, 0, 255).astype(np.uint8)


def gen_rudin_shapiro(rng, size):
    """Rudin-Shapiro sequence — flat power spectrum but deterministic."""
    # RS(n) = (-1)^f(n) where f(n) counts "11" patterns in binary of n
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        n = i
        prev_bit = 0
        count = 0
        while n:
            bit = n & 1
            if bit and prev_bit:
                count += 1
            prev_bit = bit
            n >>= 1
        vals[i] = 255 if (count % 2 == 0) else 0
    jitter = rng.integers(-10, 11, size, dtype=np.int16)
    return np.clip(vals.astype(np.int16) + jitter, 0, 255).astype(np.uint8)


def gen_gf256_accumulate(rng, size):
    """GF(256) multiply-accumulate using AES field arithmetic."""
    # GF(256) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11B)
    def gf_mul(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi_bit = a & 0x80
            a = (a << 1) & 0xFF
            if hi_bit:
                a ^= 0x1B  # reduction
            b >>= 1
        return p

    # Start with random seed, multiply-accumulate
    acc = int(rng.integers(1, 256))
    mul = int(rng.integers(2, 256))
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        acc = gf_mul(acc, mul) ^ (i & 0xFF)
        vals[i] = acc
    return vals


def gen_poisson_spike(rng, size):
    """Poisson spike train with refractory period."""
    rate = 0.05  # spikes per sample
    refractory = 15  # minimum inter-spike interval
    vals = np.zeros(size, dtype=np.uint8)
    t = 0
    while t < size:
        # Inter-spike interval: exponential + refractory
        isi = int(rng.exponential(1.0 / rate)) + refractory
        t += isi
        if t < size:
            # Spike: brief pulse
            width = min(3, size - t)
            vals[t:t + width] = 255
    return vals


def gen_iid_4symbol(rng, size):
    """iid uniform over {0, 64, 128, 192} — alphabet-size control."""
    symbols = np.array([0, 64, 128, 192], dtype=np.uint8)
    indices = rng.integers(0, 4, size)
    return symbols[indices]


NEW_SOURCES = [
    ('fBm H=0.1', gen_fbm_01),
    ('fBm H=0.9', gen_fbm_09),
    ('Multiplicative Cascade', gen_multiplicative_cascade),
    ('Thue-Morse', gen_thue_morse),
    ('Rudin-Shapiro', gen_rudin_shapiro),
    ('GF(256) Accumulate', gen_gf256_accumulate),
    ('Poisson Spike Train', gen_poisson_spike),
    ('iid 4-Symbol', gen_iid_4symbol),
]


# =============================================================================
# ROUND 2: ADVERSARIAL GENERATORS (targeted at unexplained variance)
# =============================================================================
# Metrics with most variance unexplained by 7D:
#   Wasserstein:self_similarity  (84%), Clifford:path_regularity (78%),
#   Spiral:growth_rate (72%), E8:alignment_std (71%),
#   HOS:skew_max (44%), HOS:kurt_max (43%), Lorentzian:lightlike_fraction (69%)
# Strategy: design sources to activate these specific metrics in novel ways.

def gen_devils_staircase(rng, size):
    """Devil's staircase (Cantor function) — singular continuous, exactly
    self-similar at all scales. Targets Wasserstein:self_similarity."""
    # Build by iterative removal of middle thirds
    t = np.linspace(0, 1, size)
    vals = np.zeros(size, dtype=np.float64)
    for i, x in enumerate(t):
        # Compute Cantor function via ternary expansion
        result = 0.0
        power = 0.5
        # Random starting offset for trial variation
        x = (x + rng.uniform(0, 0.01)) % 1.0
        for _ in range(30):  # enough precision
            digit = int(x * 3)
            if digit == 1:
                result += power
                break
            elif digit == 0:
                x = x * 3
            else:  # digit == 2
                result += power
                x = x * 3 - 2
            power *= 0.5
        vals[i] = result
    return (vals * 255).astype(np.uint8)


def gen_levy_flight(rng, size):
    """Lévy flight (α=1.0, Cauchy jumps) — infinite mean, infinite variance.
    Targets HOS:skew_max, HOS:kurt_max, Fisher:effective_dimension."""
    # Cauchy-distributed increments
    increments = rng.standard_cauchy(size)
    walk = np.cumsum(increments)
    # Robust normalization (use percentiles, not min/max which would be extreme)
    lo, hi = np.percentile(walk, [1, 99])
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return np.clip((walk - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def gen_weierstrass(rng, size):
    """Weierstrass function — continuous everywhere, differentiable nowhere.
    Fractal dimension > 1. Targets Clifford:path_regularity, Spiral:growth_rate."""
    a = 0.5
    b = 7  # must be odd integer, ab > 1 + 3π/2
    t = np.linspace(0, 4 * np.pi, size)
    phase = rng.uniform(0, 2 * np.pi)
    vals = np.zeros(size, dtype=np.float64)
    for n in range(50):  # sum converges quickly
        vals += a**n * np.cos(b**n * np.pi * (t + phase))
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((vals - lo) / (hi - lo) * 255).astype(np.uint8)


def gen_chirp_exp(rng, size):
    """Exponential chirp — frequency sweeps from 1 Hz to ~100 Hz.
    Non-stationary, breaks self-similarity at every scale.
    Targets Wasserstein:self_similarity, Ammann-Beenker:silver_ratio_score."""
    t = np.linspace(0, 1, size)
    f0 = 1.0 + rng.uniform(0, 2)
    f1 = 80.0 + rng.uniform(0, 40)
    phase = rng.uniform(0, 2 * np.pi)
    # Exponential sweep: phase = 2π f0 (k^t - 1) / ln(k), k = f1/f0
    k = f1 / f0
    instantaneous_phase = 2 * np.pi * f0 * (k**t - 1) / np.log(k) + phase
    vals = np.sin(instantaneous_phase)
    return ((vals + 1) / 2 * 255).astype(np.uint8)


def gen_half_cauchy(rng, size):
    """iid half-Cauchy samples — extreme right skew, very heavy right tail.
    Targets HOS:skew_max, HOS:kurt_max directly."""
    vals = np.abs(rng.standard_cauchy(size))
    lo, hi = 0, np.percentile(vals, 98)
    if hi < 1e-15:
        hi = 1.0
    return np.clip(vals / hi * 255, 0, 255).astype(np.uint8)


def gen_step_function(rng, size):
    """Random piecewise-constant (staircase) — long flat regions with jumps.
    Targets path_regularity (extremely regular within segments),
    growth_rate (zero then sudden), lightlike_fraction (jumps are 'lightlike')."""
    n_steps = rng.integers(5, 20)
    boundaries = np.sort(rng.integers(0, size, n_steps))
    boundaries = np.concatenate([[0], boundaries, [size]])
    levels = rng.integers(0, 256, len(boundaries) - 1)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(len(levels)):
        vals[boundaries[i]:boundaries[i + 1]] = levels[i]
    return vals


def gen_fibonacci_word(rng, size):
    """Fibonacci word — Sturmian sequence with golden-ratio quasi-periodicity.
    The canonical 1D quasicrystal. Targets all aperiodic tiling metrics
    (Einstein, Penrose, Ammann-Beenker) differently from Thue-Morse."""
    # Build by substitution: a -> ab, b -> a
    # Start with 'a', iterate until long enough
    a, b = '1', '0'
    word = a
    while len(word) < size + 100:
        word_new = ''
        for c in word:
            if c == '1':
                word_new += '10'
            else:
                word_new += '1'
        word = word_new
    word = word[:size]
    vals = np.array([255 if c == '1' else 0 for c in word], dtype=np.uint8)
    jitter = rng.integers(-8, 9, size, dtype=np.int16)
    return np.clip(vals.astype(np.int16) + jitter, 0, 255).astype(np.uint8)


def gen_hilbert_walk(rng, size):
    """1D trace of Hilbert space-filling curve — maximally space-filling,
    preserves locality. Unusual spiral/growth structure.
    Targets Spiral:growth_rate, Clifford:path_regularity."""
    # Hilbert curve in 2D, take x-coordinate as 1D signal
    order = int(np.ceil(np.log2(np.sqrt(size)))) + 1
    n_points = 4**order

    def d2xy(n, d):
        """Convert Hilbert curve index to (x,y) coordinates."""
        x = y = 0
        s = 1
        while s < n:
            rx = 1 if (d & 2) else 0
            ry = 1 if ((d & 1) ^ rx) else 0
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y

    n_grid = 2**order
    # Sample evenly along the curve
    indices = np.linspace(0, n_points - 1, size, dtype=int)
    offset = rng.integers(0, n_points // 4)
    xs = np.zeros(size)
    for i, idx in enumerate(indices):
        x, y = d2xy(n_grid, (idx + offset) % n_points)
        xs[i] = x
    lo, hi = xs.min(), xs.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((xs - lo) / (hi - lo) * 255).astype(np.uint8)


def gen_burst_process(rng, size):
    """Extreme ON/OFF burst process — long silence punctuated by violent bursts.
    Targets Lorentzian:lightlike_fraction (extreme velocity contrasts),
    Fisher:effective_dimension, Wasserstein:self_similarity."""
    vals = np.zeros(size, dtype=np.uint8)
    t = 0
    while t < size:
        # Long OFF period
        off_len = int(rng.exponential(200))
        t += off_len
        if t >= size:
            break
        # Short ON burst with extreme values
        burst_len = rng.integers(3, 15)
        end = min(t + burst_len, size)
        vals[t:end] = rng.integers(180, 256, end - t, dtype=np.uint8)
        t = end
    return vals


def gen_logistic_bifurcation(rng, size):
    """Logistic map at period-doubling accumulation point (r ≈ 3.56995).
    Critical point between order and chaos — edge of chaos with exact
    self-similarity (Feigenbaum universality). Targets effective_dimension,
    self_similarity at the critical ratio δ = 4.669..."""
    r = 3.56995  # Feigenbaum point
    x = rng.uniform(0.2, 0.8)
    for _ in range(2000):  # transient
        x = r * x * (1 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = r * x * (1 - x)
        vals[i] = int(x * 255)
    return vals


ROUND2_SOURCES = [
    ("Devil's Staircase", gen_devils_staircase),
    ('Lévy Flight (Cauchy)', gen_levy_flight),
    ('Weierstrass Function', gen_weierstrass),
    ('Exponential Chirp', gen_chirp_exp),
    ('Half-Cauchy iid', gen_half_cauchy),
    ('Random Step Function', gen_step_function),
    ('Fibonacci Word', gen_fibonacci_word),
    ('Hilbert Curve Walk', gen_hilbert_walk),
    ('Burst Process', gen_burst_process),
    ('Logistic Bifurcation', gen_logistic_bifurcation),
]

ROUND2_COLORS = {
    "Devil's Staircase": '#ff4757',
    'Lévy Flight (Cauchy)': '#ff6348',
    'Weierstrass Function': '#ffa502',
    'Exponential Chirp': '#2ed573',
    'Half-Cauchy iid': '#1e90ff',
    'Random Step Function': '#3742fa',
    'Fibonacci Word': '#a55eea',
    'Hilbert Curve Walk': '#f78fb3',
    'Burst Process': '#e056fd',
    'Logistic Bifurcation': '#7bed9f',
}


# =============================================================================
# ROUND 3: EXPANDED-METRIC BLIND SPOT CLOSURE TEST
# =============================================================================
# The 3 new geometries (Hölder Regularity, p-Variation, Multi-Scale Wasserstein)
# add 19 metrics targeting pathological local regularity.  But existing signatures
# don't include those metrics.  Round 3 works entirely in the Runner's full
# 160-metric space: regenerate representative existing-category sources alongside
# the 5 breakers and test whether the new metrics resolve the blind spot.

def gen_random_uniform(rng, size):
    """iid uniform bytes — the canonical null model."""
    return rng.integers(0, 256, size, dtype=np.uint8)

def gen_sine_wave(rng, size):
    """Sine wave with randomised phase and small frequency jitter."""
    freq = 0.02 + rng.uniform(-0.002, 0.002)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(size, dtype=np.float64)
    wave = 127.5 + 127.5 * np.sin(2 * np.pi * freq * t + phase)
    return np.clip(wave, 0, 255).astype(np.uint8)

def gen_lorenz_x(rng, size):
    """Lorenz attractor x-coordinate, perturbed initial conditions."""
    dt = 0.01
    x, y, z = rng.uniform(-0.5, 0.5, 3) + np.array([1.0, 1.0, 1.0])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for _ in range(5000):  # transient
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz
    vals = np.zeros(size)
    for i in range(size):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz
        vals[i] = x
    # Scale to uint8
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-10:
        return np.full(size, 128, dtype=np.uint8)
    vals = (vals - vmin) / (vmax - vmin) * 255
    return vals.astype(np.uint8)

def gen_logistic_chaos(rng, size):
    """Logistic map r=4 (full chaos)."""
    x = rng.uniform(0.1, 0.9)
    for _ in range(500):  # transient
        x = 4.0 * x * (1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 4.0 * x * (1.0 - x)
        vals[i] = int(x * 255)
    return vals

def gen_prime_gaps(rng, size):
    """Prime gaps via simple sieve from a randomised offset."""
    # Sieve of Eratosthenes for a window large enough to get `size` gaps
    # Each gap is typically < 200 for primes up to ~10^7
    limit = max(size * 40, 100000)
    offset = int(rng.integers(0, 100))  # small offset for trial variation
    is_prime = np.ones(limit, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    # Skip some primes based on offset for trial variation
    primes = primes[offset:]
    gaps = np.diff(primes)[:size]
    gaps = np.minimum(gaps, 255)
    if len(gaps) < size:
        gaps = np.pad(gaps, (0, size - len(gaps)), mode='wrap')
    return gaps.astype(np.uint8)

def gen_pink_noise(rng, size):
    """Pink (1/f) noise via spectral shaping."""
    white = rng.standard_normal(size)
    freqs = np.fft.rfftfreq(size, d=1.0)
    freqs[0] = 1.0
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    out = np.fft.irfft(spectrum, n=size)
    out = (out - out.min()) / (out.max() - out.min() + 1e-10) * 255
    return out.astype(np.uint8)

def gen_collatz_hailstone(rng, size):
    """Collatz hailstone bytes from randomised starting values."""
    start = int(rng.integers(10**8, 10**9))
    vals = []
    n = start
    while len(vals) < size * 3:  # overshoot then slice
        if n == 1:
            n = int(rng.integers(10**8, 10**9))
        vals.append(n & 0xFF)
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    return np.array(vals[:size], dtype=np.uint8)

# Representative sample of existing-category sources (8 sources spanning the
# diversity of the original 41 signatures)
REPRESENTATIVE_SOURCES = [
    ('Random (repr.)', gen_random_uniform),
    ('Sine (repr.)', gen_sine_wave),
    ('Lorenz (repr.)', gen_lorenz_x),
    ('Logistic r=4 (repr.)', gen_logistic_chaos),
    ('Prime Gaps (repr.)', gen_prime_gaps),
    ('Pink Noise (repr.)', gen_pink_noise),
    ('Collatz (repr.)', gen_collatz_hailstone),
    ('iid 4-Sym (repr.)', gen_iid_4symbol),
]

# The 5 breakers from Round 2
BREAKER_SOURCES = [
    ("Devil's Staircase", gen_devils_staircase),
    ('Lévy Flight (Cauchy)', gen_levy_flight),
    ('Random Step Function', gen_step_function),
    ('Fibonacci Word', gen_fibonacci_word),
    ('Hilbert Curve Walk', gen_hilbert_walk),
]


def _collect_full(sources, runner):
    """Collect profiles in the Runner's full metric space (no alignment)."""
    profiles = {}
    for name, gen_fn in sources:
        rngs = runner.trial_rngs()
        chunks = [gen_fn(rng, runner.data_size) for rng in rngs]
        metrics = runner.collect(chunks)
        profile = np.zeros(len(runner.metric_names))
        for j, m in enumerate(runner.metric_names):
            vals = metrics.get(m, [])
            if len(vals) > 0:
                profile[j] = np.mean(vals)
        profiles[name] = profile
        n_active = np.sum(np.abs(profile) > 1e-15)
        print(f"  {name:25s}  {n_active}/{len(runner.metric_names)} active metrics")
    return profiles


def direction_6(runner, repr_profiles, breaker_profiles):
    """D6: Expanded-metric blind spot closure test.

    Measure discriminative power: for each metric, how well does it separate
    breakers from representative sources?  Compare old metrics vs new metrics.
    Also use runner.compare() per breaker vs representative pool for
    statistical significance counts.
    """
    print("\n" + "=" * 78)
    print("D6: Expanded-Metric Blind Spot Closure Test")
    print("=" * 78)

    # Identify old vs new metrics
    new_geo_prefixes = ('Hölder Regularity:', 'p-Variation:', 'Multi-Scale Wasserstein:')
    old_idx = [j for j, m in enumerate(runner.metric_names)
               if not m.startswith(new_geo_prefixes)]
    new_idx = [j for j, m in enumerate(runner.metric_names)
               if m.startswith(new_geo_prefixes)]
    new_metric_names = [runner.metric_names[j] for j in new_idx]
    print(f"  Old metrics: {len(old_idx)}, New metrics: {len(new_idx)}")

    # Build representative distribution (mean/std across repr sources)
    repr_vecs = np.array(list(repr_profiles.values()))
    repr_mean = np.mean(repr_vecs, axis=0)
    repr_std = np.std(repr_vecs, axis=0)
    repr_std[repr_std < 1e-10] = 1.0

    # Z-score each breaker relative to the representative distribution
    print("\n  Per-metric z-scores of breakers (relative to representative pool):")
    breaker_zscores = {}  # {breaker_name: array of z-scores}
    for name, vec in breaker_profiles.items():
        z = (vec - repr_mean) / repr_std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        breaker_zscores[name] = z

    # For each breaker, find top-5 discriminating metrics (by |z|)
    # and whether they come from old or new geometries
    print("\n  Top-5 discriminating metrics per breaker:")
    per_breaker_top = {}
    for name, z in breaker_zscores.items():
        top_idx = np.argsort(np.abs(z))[-5:][::-1]
        top = [(runner.metric_names[j], z[j], j in new_idx) for j in top_idx]
        per_breaker_top[name] = top
        print(f"\n    {name}:")
        for mname, zval, is_new in top:
            tag = "[NEW]" if is_new else "[old]"
            print(f"      {mname:45s}  z={zval:+6.2f}  {tag}")

    # Aggregate: mean |z| of old metrics vs new metrics per breaker
    print("\n  Mean |z| per metric group:")
    group_stats = {}
    for name, z in breaker_zscores.items():
        old_z = np.mean(np.abs(z[old_idx]))
        new_z = np.mean(np.abs(z[new_idx]))
        group_stats[name] = (old_z, new_z)
        ratio = new_z / old_z if old_z > 1e-10 else float('inf')
        print(f"    {name:25s}  old={old_z:.2f}  new={new_z:.2f}  ratio={ratio:.2f}x")

    # Count how many new metrics have |z| > 2 for each breaker
    print("\n  Metrics with |z| > 2 per breaker:")
    outlier_counts = {}
    for name, z in breaker_zscores.items():
        n_old_outlier = int(np.sum(np.abs(z[old_idx]) > 2))
        n_new_outlier = int(np.sum(np.abs(z[new_idx]) > 2))
        outlier_counts[name] = (n_old_outlier, n_new_outlier)
        print(f"    {name:25s}  old: {n_old_outlier}/{len(old_idx)}  "
              f"new: {n_new_outlier}/{len(new_idx)}")

    # Pairwise distance matrix: old metrics vs all metrics
    all_names = list(repr_profiles.keys()) + list(breaker_profiles.keys())
    all_vecs = np.array([repr_profiles.get(n, breaker_profiles.get(n))
                         for n in all_names])

    # Compute pairwise Euclidean distances in z-scored space
    dist_matrices = {}
    for label, col_idx in [('old', old_idx), ('new', new_idx), ('all', list(range(len(runner.metric_names))))]:
        M = all_vecs[:, col_idx].copy()
        col_m = np.mean(M, axis=0)
        col_s = np.std(M, axis=0)
        col_s[col_s < 1e-10] = 1.0
        Z = (M - col_m) / col_s
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        # Pairwise distance
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(Z, 'euclidean'))
        dist_matrices[label] = D

    # Mean distance between breaker set and repr set
    n_repr = len(repr_profiles)
    breaker_idx = list(range(n_repr, len(all_names)))
    repr_idx_set = list(range(n_repr))

    print("\n  Mean inter-group distance (breaker ↔ repr):")
    for label in ['old', 'new', 'all']:
        D = dist_matrices[label]
        cross = D[np.ix_(breaker_idx, repr_idx_set)]
        within_repr = D[np.ix_(repr_idx_set, repr_idx_set)]
        within_repr_mean = np.mean(within_repr[np.triu_indices(n_repr, k=1)])
        cross_mean = np.mean(cross)
        separation = cross_mean / (within_repr_mean + 1e-10)
        print(f"    [{label:3s}] cross={cross_mean:.2f}  within_repr={within_repr_mean:.2f}  "
              f"separation={separation:.2f}")

    # New-metric profile heatmap data: breaker values on new metrics
    new_metric_data = {}
    for name in list(repr_profiles.keys()) + list(breaker_profiles.keys()):
        vec = repr_profiles.get(name, breaker_profiles.get(name))
        new_metric_data[name] = vec[new_idx]

    return {
        'breaker_zscores': breaker_zscores,
        'per_breaker_top': per_breaker_top,
        'group_stats': group_stats,
        'outlier_counts': outlier_counts,
        'dist_matrices': dist_matrices,
        'all_names': all_names,
        'n_repr': n_repr,
        'new_metric_names': new_metric_names,
        'new_metric_data': new_metric_data,
        'old_idx': old_idx,
        'new_idx': new_idx,
    }


def make_figure_round3(runner, d6, repr_profiles, breaker_profiles):
    """Third figure: Round 3 blind spot closure (4 panels, 2x2)."""
    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(22, 16), facecolor='#181818')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                          left=0.07, right=0.96, top=0.93, bottom=0.10)
    fig.suptitle("Round 3: Blind Spot Closure — New Geometries vs 7D Breakers",
                 fontsize=15, fontweight='bold', color='white')

    breaker_names = list(breaker_profiles.keys())
    repr_names_list = list(repr_profiles.keys())

    # ---- Top-left: New-metric heatmap (sources × 19 new metrics) ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#181818')
    for spine in ax1.spines.values():
        spine.set_color('#444444')

    all_source_names = repr_names_list + breaker_names
    heatmap_data = np.array([d6['new_metric_data'][n] for n in all_source_names])
    # Z-score across sources for display
    col_m = np.mean(heatmap_data, axis=0)
    col_s = np.std(heatmap_data, axis=0)
    col_s[col_s < 1e-10] = 1.0
    heatmap_z = (heatmap_data - col_m) / col_s

    im = ax1.imshow(heatmap_z, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    ax1.set_yticks(range(len(all_source_names)))
    short_names = [n.replace(' (repr.)', '') for n in all_source_names]
    ax1.set_yticklabels(short_names, fontsize=7)
    # Highlight breakers
    for i in range(len(repr_names_list), len(all_source_names)):
        ax1.get_yticklabels()[i].set_color('#ff4757')
        ax1.get_yticklabels()[i].set_fontweight('bold')

    # Metric labels
    short_metrics = [m.split(':')[1] if ':' in m else m for m in d6['new_metric_names']]
    ax1.set_xticks(range(len(short_metrics)))
    ax1.set_xticklabels(short_metrics, rotation=60, ha='right', fontsize=6)
    ax1.set_title('New Geometry Metrics (z-scored)', fontsize=11)
    plt.colorbar(im, ax=ax1, shrink=0.8, label='z-score')

    # ---- Top-right: Mean |z| old vs new per breaker ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#181818')
    for spine in ax2.spines.values():
        spine.set_color('#444444')

    x = np.arange(len(breaker_names))
    w = 0.35
    old_zs = [d6['group_stats'][n][0] for n in breaker_names]
    new_zs = [d6['group_stats'][n][1] for n in breaker_names]

    ax2.bar(x - w/2, old_zs, w, color='#636e72', alpha=0.85,
            label=f"Old ({len(d6['old_idx'])} metrics)")
    ax2.bar(x + w/2, new_zs, w, color='#e74c3c', alpha=0.85,
            label=f"New ({len(d6['new_idx'])} metrics)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(breaker_names, rotation=25, ha='right', fontsize=8)
    ax2.set_ylabel('Mean |z-score|', fontsize=10)
    ax2.set_title('Discriminative Power: Old vs New Metrics', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#222222', edgecolor='#444444', labelcolor='#cccccc')
    ax2.tick_params(colors='#cccccc', labelsize=7)
    for i, (oz, nz) in enumerate(zip(old_zs, new_zs)):
        ax2.text(i - w/2, oz + 0.05, f"{oz:.1f}", ha='center', fontsize=8, color='#cccccc')
        ax2.text(i + w/2, nz + 0.05, f"{nz:.1f}", ha='center', fontsize=8, color='white')

    # ---- Bottom-left: Outlier count (|z|>2) old vs new ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#181818')
    for spine in ax3.spines.values():
        spine.set_color('#444444')

    old_outliers = [d6['outlier_counts'][n][0] for n in breaker_names]
    new_outliers = [d6['outlier_counts'][n][1] for n in breaker_names]
    # Normalize by group size for fair comparison
    n_old = len(d6['old_idx'])
    n_new = len(d6['new_idx'])
    old_frac = [c / n_old * 100 for c in old_outliers]
    new_frac = [c / n_new * 100 for c in new_outliers]

    ax3.bar(x - w/2, old_frac, w, color='#636e72', alpha=0.85,
            label=f"Old ({n_old} met)")
    ax3.bar(x + w/2, new_frac, w, color='#e74c3c', alpha=0.85,
            label=f"New ({n_new} met)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(breaker_names, rotation=25, ha='right', fontsize=8)
    ax3.set_ylabel('% metrics with |z| > 2', fontsize=10)
    ax3.set_title('Outlier Rate: Old vs New Metrics', fontsize=11)
    ax3.legend(fontsize=8, facecolor='#222222', edgecolor='#444444', labelcolor='#cccccc')
    ax3.tick_params(colors='#cccccc', labelsize=7)
    for i, (of, nf) in enumerate(zip(old_frac, new_frac)):
        ax3.text(i - w/2, of + 1, f"{of:.0f}%", ha='center', fontsize=8, color='#cccccc')
        ax3.text(i + w/2, nf + 1, f"{nf:.0f}%", ha='center', fontsize=8, color='white')

    # ---- Bottom-right: Inter-group separation ratio ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#181818')
    for spine in ax4.spines.values():
        spine.set_color('#444444')

    labels_sep = ['Old Metrics', 'New Metrics', 'All Metrics']
    n_repr = d6['n_repr']
    breaker_set = list(range(n_repr, len(d6['all_names'])))
    repr_set = list(range(n_repr))
    separations = []
    for label_key in ['old', 'new', 'all']:
        D = d6['dist_matrices'][label_key]
        cross = np.mean(D[np.ix_(breaker_set, repr_set)])
        within = np.mean(D[np.ix_(repr_set, repr_set)][np.triu_indices(n_repr, k=1)])
        separations.append(cross / (within + 1e-10))

    bars = ax4.bar(range(3), separations, color=['#636e72', '#e74c3c', '#3498db'],
                   alpha=0.85)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(labels_sep, fontsize=10)
    ax4.set_ylabel('Separation Ratio (cross / within)', fontsize=10)
    ax4.set_title('Breaker–Representative Separation', fontsize=11)
    ax4.axhline(1.0, color='#666666', linewidth=0.8, linestyle='--', alpha=0.6)
    ax4.tick_params(colors='#cccccc', labelsize=7)
    for bar, s in zip(bars, separations):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{s:.2f}", ha='center', fontsize=10, color='white')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'figures', 'signature_space_r3.png')
    fig.savefig(out, dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# =============================================================================
# DIRECTIONS
# =============================================================================

def direction_1(names, categories, metric_names, means_matrix):
    """D1: PC Interpretation — label the 7 axes by poles and loadings."""
    print("\n" + "=" * 78)
    print("D1: PC Interpretation — Label the 7 Axes")
    print("=" * 78)

    n_sigs, n_met = means_matrix.shape

    # Z-score columns
    col_means = np.nanmean(means_matrix, axis=0)
    col_stds = np.nanstd(means_matrix, axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    Z = (means_matrix - col_means) / col_stds
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # SVD
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    scores = U * s  # (n_sigs, n_components) — PC scores

    pc_info = []
    for k in range(7):
        pc_scores = scores[:, k]

        # Top/bottom 3 signatures (poles)
        top_idx = np.argsort(pc_scores)[-3:][::-1]
        bot_idx = np.argsort(pc_scores)[:3]

        top_poles = [(names[i], categories[i], pc_scores[i]) for i in top_idx]
        bot_poles = [(names[i], categories[i], pc_scores[i]) for i in bot_idx]

        # Top 5 loadings (absolute value)
        loadings = Vt[k, :]
        top_load_idx = np.argsort(np.abs(loadings))[-5:][::-1]
        top_loadings = [(metric_names[i], loadings[i]) for i in top_load_idx]

        # Generate semantic label from top loading's geometry family + pole nature
        top_geom = top_loadings[0][0].split(':')[0]  # dominant geometry
        top_cats = [cat for _, cat, _ in top_poles]
        bot_cats = [cat for _, cat, _ in bot_poles]

        # Describe the axis by what separates the poles
        def _describe_pole(cats):
            """Most common category in the pole set."""
            from collections import Counter
            c = Counter(cats)
            return c.most_common(1)[0][0]

        pos_desc = _describe_pole(top_cats)
        neg_desc = _describe_pole(bot_cats)
        label = f"{pos_desc} vs {neg_desc} ({top_geom})"

        info = {
            'pc': k + 1,
            'var_explained': (s[k]**2 / np.sum(s**2)) * 100,
            'top_poles': top_poles,
            'bot_poles': bot_poles,
            'top_loadings': top_loadings,
            'label': label,
        }
        pc_info.append(info)

        print(f"\n  PC{k+1} ({info['var_explained']:.1f}% variance): {label}")
        print(f"    + poles: {', '.join(f'{n} [{c}]' for n, c, _ in top_poles)}")
        print(f"    - poles: {', '.join(f'{n} [{c}]' for n, c, _ in bot_poles)}")
        print(f"    loadings: {', '.join(f'{m.split(':')[0]}:{m.split(':')[1]}={v:+.3f}' for m, v in top_loadings)}")

    return {
        'pc_info': pc_info,
        'Z': Z,
        'U': U,
        's': s,
        'Vt': Vt,
        'scores': scores,
        'col_means': col_means,
        'col_stds': col_stds,
    }


def direction_2(metric_names, Vt):
    """D2: PC Loadings Heatmap — V^T grouped by geometry family."""
    print("\n" + "=" * 78)
    print("D2: PC Loadings Heatmap")
    print("=" * 78)

    # Group metrics by geometry family
    families = []
    boundaries = []
    prev_fam = None
    for i, m in enumerate(metric_names):
        fam = m.split(':')[0]
        if fam != prev_fam:
            boundaries.append(i)
            families.append(fam)
            prev_fam = fam

    # Loadings matrix: top 7 PCs × metrics
    loadings = Vt[:7, :]  # (7, n_met)

    print(f"  {len(families)} geometry families, {len(metric_names)} metrics")
    print(f"  Max |loading|: {np.max(np.abs(loadings)):.3f}")

    # Per-family mean absolute loading
    bounds_ext = boundaries + [len(metric_names)]
    for k in range(7):
        top_fam_idx = -1
        top_fam_val = 0
        for fi in range(len(families)):
            fam_loads = np.abs(loadings[k, bounds_ext[fi]:bounds_ext[fi + 1]])
            mean_load = np.mean(fam_loads)
            if mean_load > top_fam_val:
                top_fam_val = mean_load
                top_fam_idx = fi
        print(f"  PC{k+1}: dominant family = {families[top_fam_idx]} "
              f"(mean |loading| = {top_fam_val:.3f})")

    return {
        'loadings': loadings,
        'families': families,
        'boundaries': boundaries,
    }


def direction_4(names, categories, metric_names, means_matrix, col_means,
                col_stds, Vt, s, new_profiles):
    """D4: Projection test — do new sources break 7D?

    Uses empirical percentile test (not Gaussian z-score) because the
    residual distribution is heavily right-skewed with N=41.
    """
    print("\n" + "=" * 78)
    print("D4: Projection Test — Do New Sources Break 7D?")
    print("=" * 78)

    n_sigs = len(names)

    # Z-score the existing means matrix
    Z = (means_matrix - col_means) / col_stds
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Existing residuals in 7D
    Vt7 = Vt[:7, :]  # (7, n_met)
    existing_residuals = []
    for i in range(n_sigs):
        z_i = Z[i, :]
        proj = Vt7.T @ (Vt7 @ z_i)  # project and reconstruct
        residual = np.sum((z_i - proj)**2) / np.sum(z_i**2) if np.sum(z_i**2) > 1e-15 else 0.0
        existing_residuals.append(residual)

    existing_residuals = np.array(existing_residuals)
    sorted_existing = np.sort(existing_residuals)

    print(f"  Existing 7D residuals: median={np.median(existing_residuals):.4f}, "
          f"95th={np.percentile(existing_residuals, 95):.4f}, "
          f"max={np.max(existing_residuals):.4f}")

    # Show the 3 hardest-to-explain existing signatures
    worst_idx = np.argsort(existing_residuals)[-3:][::-1]
    print(f"  Highest-residual existing signatures:")
    for idx in worst_idx:
        print(f"    {names[idx]:25s} [{categories[idx]}]  "
              f"residual={existing_residuals[idx]:.4f}")

    # New source residuals — empirical percentile test
    new_residuals = {}
    for name, profile in new_profiles.items():
        # Z-score using existing column stats
        z_new = (profile - col_means) / col_stds
        z_new = np.nan_to_num(z_new, nan=0.0, posinf=0.0, neginf=0.0)

        proj = Vt7.T @ (Vt7 @ z_new)
        norm_sq = np.sum(z_new**2)
        residual = np.sum((z_new - proj)**2) / norm_sq if norm_sq > 1e-15 else 0.0

        # Empirical one-sided p-value: fraction of existing residuals >= this one
        # Use (n_exceed + 1) / (N + 1) for continuity correction
        n_exceed = np.sum(existing_residuals >= residual)
        p_value = (n_exceed + 1) / (n_sigs + 1)

        new_residuals[name] = {
            'residual': residual,
            'rank': n_sigs - n_exceed + 1,  # rank among existing (1 = lowest)
            'percentile': (1 - n_exceed / n_sigs) * 100,
            'p_value': p_value,
            'breaks_7d': p_value < 0.05,
        }

        status = "BREAKS 7D" if p_value < 0.05 else "within 7D"
        print(f"  {name:25s}  residual={residual:.4f}  "
              f"rank={n_sigs - n_exceed + 1}/{n_sigs}  "
              f"p={p_value:.3f}  [{status}]")

    return {
        'existing_residuals': existing_residuals,
        'new_residuals': new_residuals,
    }


def direction_5(names, categories, metric_names, means_matrix, col_means,
                col_stds, s_old, new_profiles):
    """D5: Updated dimensionality — SVD with new sources appended."""
    print("\n" + "=" * 78)
    print("D5: Updated Dimensionality")
    print("=" * 78)

    # Build augmented matrix
    new_names = list(new_profiles.keys())
    new_matrix = np.array([new_profiles[n] for n in new_names])
    aug_matrix = np.vstack([means_matrix, new_matrix])
    aug_names = names + new_names
    aug_categories = categories + ['new'] * len(new_names)

    n_total = len(aug_names)

    # Z-score the augmented matrix (using its OWN statistics for fair comparison)
    aug_col_means = np.nanmean(aug_matrix, axis=0)
    aug_col_stds = np.nanstd(aug_matrix, axis=0)
    aug_col_stds[aug_col_stds < 1e-10] = 1.0
    Z_aug = (aug_matrix - aug_col_means) / aug_col_stds
    Z_aug = np.nan_to_num(Z_aug, nan=0.0, posinf=0.0, neginf=0.0)

    # SVD of augmented
    U_aug, s_aug, Vt_aug = np.linalg.svd(Z_aug, full_matrices=False)
    varexp_aug = s_aug**2 / np.sum(s_aug**2)
    cumvar_aug = np.cumsum(varexp_aug)

    # Participation ratio
    pr_aug = np.sum(s_aug**2)**2 / np.sum(s_aug**4)
    pr_old = np.sum(s_old**2)**2 / np.sum(s_old**4)

    print(f"  Old participation ratio: {pr_old:.1f}")
    print(f"  New participation ratio: {pr_aug:.1f}")
    print(f"  Change: {pr_aug - pr_old:+.1f}")

    # Marchenko-Pastur for new matrix
    n_new = Z_aug.shape[0]
    n_cols = Z_aug.shape[1]
    rng = np.random.default_rng(99)
    mp_spectra = []
    for _ in range(25):
        rand_mat = rng.standard_normal((n_new, n_cols))
        _, s_rand, _ = np.linalg.svd(rand_mat, full_matrices=False)
        mp_spectra.append(s_rand)

    mp_spectra = np.array(mp_spectra)
    mp_mean = np.mean(mp_spectra, axis=0)
    mp_upper = np.percentile(mp_spectra, 97.5, axis=0)

    n_above_mp = np.sum(s_aug > mp_upper[:len(s_aug)])
    print(f"  Singular values above MP 97.5%: {n_above_mp} / {len(s_aug)}")
    print(f"  90% variance at dim: {np.searchsorted(cumvar_aug, 0.90) + 1}")

    # PC scores for atlas plot
    scores_aug = U_aug * s_aug

    # Also compute old MP for comparison
    rng2 = np.random.default_rng(99)
    mp_old_spectra = []
    n_old = means_matrix.shape[0]
    for _ in range(25):
        rand_mat = rng2.standard_normal((n_old, n_cols))
        _, s_r, _ = np.linalg.svd(rand_mat, full_matrices=False)
        mp_old_spectra.append(s_r)
    mp_old_upper = np.percentile(np.array(mp_old_spectra), 97.5, axis=0)

    return {
        's_aug': s_aug,
        's_old': s_old,
        'varexp_aug': varexp_aug,
        'cumvar_aug': cumvar_aug,
        'pr_old': pr_old,
        'pr_aug': pr_aug,
        'mp_mean': mp_mean,
        'mp_upper': mp_upper,
        'mp_old_upper': mp_old_upper,
        'n_above_mp': n_above_mp,
        'scores_aug': scores_aug,
        'aug_names': aug_names,
        'aug_categories': aug_categories,
        'n_old': n_old,
    }


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(d1, d2, d3_profiles, d4, d5, names, categories, metric_names):
    """Create the 6-panel figure (3x2)."""
    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(22, 24), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30,
                          left=0.07, right=0.96, top=0.95, bottom=0.04)
    fig.suptitle("Signature Space: PC Interpretation + 7D Breaker Hunt",
                 fontsize=16, fontweight='bold', color='white')

    # ---- Top-left: D1 PC Pole Chart -----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#181818')
    for spine in ax1.spines.values():
        spine.set_color('#444444')

    pc_info = d1['pc_info']
    n_pcs = len(pc_info)
    y_positions = np.arange(n_pcs)

    for k, info in enumerate(pc_info):
        y = n_pcs - 1 - k  # top to bottom

        # Draw axis line
        ax1.axhline(y, color='#333333', linewidth=0.5, zorder=1)

        # Plot top poles (positive side)
        for rank, (name, cat, score) in enumerate(info['top_poles']):
            x = 0.55 + rank * 0.15
            color = CATEGORY_COLORS.get(cat, '#888888')
            ax1.scatter(x, y, c=color, s=60, edgecolors='white',
                       linewidths=0.5, zorder=3)
            short = name[:14]
            ax1.text(x, y + 0.15, short, fontsize=5, color=color,
                    ha='center', va='bottom', rotation=30)

        # Plot bottom poles (negative side)
        for rank, (name, cat, score) in enumerate(info['bot_poles']):
            x = -(0.55 + rank * 0.15)
            color = CATEGORY_COLORS.get(cat, '#888888')
            ax1.scatter(x, y, c=color, s=60, edgecolors='white',
                       linewidths=0.5, zorder=3)
            short = name[:14]
            ax1.text(x, y + 0.15, short, fontsize=5, color=color,
                    ha='center', va='bottom', rotation=-30)

    ax1.set_yticks(range(n_pcs))
    ax1.set_yticklabels([f"PC{info['pc']} ({info['var_explained']:.1f}%)"
                         for info in reversed(pc_info)], fontsize=8)
    ax1.set_xlim(-1.1, 1.1)
    ax1.axvline(0, color='#555555', linewidth=1, linestyle='--')
    ax1.set_xlabel('\u2190 negative poles          positive poles \u2192',
                   fontsize=8)
    ax1.set_title("D1: Principal Component Poles (Top/Bottom 3 Signatures)",
                  fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', labelbottom=False)

    # ---- Top-right: D2 Loadings Heatmap --------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#181818')
    for spine in ax2.spines.values():
        spine.set_color('#444444')

    loadings = d2['loadings']  # (7, n_met)
    im2 = ax2.imshow(loadings, cmap='RdBu_r', vmin=-0.3, vmax=0.3,
                     aspect='auto', interpolation='nearest')

    # Family dividers on x-axis
    boundaries = d2['boundaries']
    families = d2['families']
    for b in boundaries[1:]:
        ax2.axvline(b - 0.5, color='#666666', linewidth=0.5, alpha=0.7)

    bounds_ext = boundaries + [len(metric_names)]
    mid_ticks = [(bounds_ext[i] + bounds_ext[i + 1]) / 2
                 for i in range(len(families))]
    ax2.set_xticks(mid_ticks)
    ax2.set_xticklabels([f[:8] for f in families], rotation=90, fontsize=5)
    ax2.set_yticks(range(7))
    ax2.set_yticklabels([f"PC{k+1}" for k in range(7)], fontsize=8)
    ax2.set_title("D2: PC Loadings by Geometry Family",
                  fontsize=11, fontweight='bold')

    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02)
    cb2.set_label('Loading', fontsize=8, color='#cccccc')
    cb2.ax.tick_params(labelsize=6, colors='#cccccc')

    # ---- Mid-left: D3 New source profiles (overlaid) -------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#181818')
    for spine in ax3.spines.values():
        spine.set_color('#444444')

    # Stack profiles and z-score across sources for visibility
    profile_names = list(d3_profiles.keys())
    profile_matrix = np.array([d3_profiles[n] for n in profile_names])
    # Z-score each metric (column) across the 8 sources
    pm_mean = np.mean(profile_matrix, axis=0)
    pm_std = np.std(profile_matrix, axis=0)
    pm_std[pm_std < 1e-15] = 1.0
    z_profiles = (profile_matrix - pm_mean) / pm_std

    for i, name in enumerate(profile_names):
        color = NEW_COLORS.get(name, '#888888')
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(z_profiles[i], kernel, mode='same')
        ax3.plot(smoothed, color=color, linewidth=0.8, alpha=0.8, label=name)

    ax3.set_xlabel('Metric index (131 tau1 metrics)', fontsize=8)
    ax3.set_ylabel('Z-scored value (across 8 sources)', fontsize=8)
    ax3.set_title("D3: New Source Metric Profiles (8 Candidates)",
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=6, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc', loc='upper right', ncol=2)
    ax3.tick_params(labelsize=7)

    # ---- Mid-right: D4 Residual bar chart ------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#181818')
    for spine in ax4.spines.values():
        spine.set_color('#444444')

    # Combine existing and new residuals, sorted
    all_residuals = []
    for i, name in enumerate(names):
        all_residuals.append((name, d4['existing_residuals'][i], 'existing',
                             categories[i]))
    for name, info in d4['new_residuals'].items():
        all_residuals.append((name, info['residual'], 'new', 'new'))

    all_residuals.sort(key=lambda x: x[1])

    bar_names = [r[0] for r in all_residuals]
    bar_vals = [r[1] for r in all_residuals]
    bar_colors = []
    for r in all_residuals:
        if r[2] == 'new':
            bar_colors.append(NEW_COLORS.get(r[0], '#ff6b6b'))
        else:
            bar_colors.append(CATEGORY_COLORS.get(r[3], '#888888'))

    y_pos = np.arange(len(bar_names))
    ax4.barh(y_pos, bar_vals, color=bar_colors, alpha=0.85, height=0.7)

    # Threshold line: empirical 95th percentile of existing residuals
    thresh = np.percentile(d4['existing_residuals'], 95)
    ax4.axvline(thresh, color='#e74c3c', linestyle='--', linewidth=1.5,
                alpha=0.8, label=f'95th pctile (existing)')

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([n[:18] for n in bar_names], fontsize=4.5)
    ax4.set_xlabel('7D Reconstruction Residual (fraction)', fontsize=8)
    ax4.set_title("D4: 7D Residuals — Existing + New Sources",
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc')
    ax4.tick_params(labelsize=6)

    # ---- Bot-left: D5 Singular value spectrum --------------------------------
    ax5a = fig.add_subplot(gs[2, 0])
    ax5a.set_facecolor('#181818')
    for spine in ax5a.spines.values():
        spine.set_color('#444444')

    k_old = np.arange(1, len(d5['s_old']) + 1)
    k_new = np.arange(1, len(d5['s_aug']) + 1)

    ax5a.semilogy(k_old, d5['s_old'], 's-', color='#95a5a6', markersize=4,
                  linewidth=1.2, label=f'Original 41 sigs (PR={d5["pr_old"]:.1f})',
                  alpha=0.7)
    ax5a.semilogy(k_new, d5['s_aug'], 'o-', color='#e74c3c', markersize=4,
                  linewidth=1.5, label=f'Augmented 49 sigs (PR={d5["pr_aug"]:.1f})',
                  zorder=3)

    # MP envelope for new
    mp_k = np.arange(1, len(d5['mp_upper']) + 1)
    ax5a.fill_between(mp_k[:len(d5['s_aug'])],
                      d5['mp_mean'][:len(d5['s_aug'])] * 0.7,
                      d5['mp_upper'][:len(d5['s_aug'])],
                      color='#3498db', alpha=0.2, label='MP 97.5% envelope')
    ax5a.semilogy(mp_k[:len(d5['s_aug'])],
                  d5['mp_mean'][:len(d5['s_aug'])],
                  '--', color='#3498db', linewidth=1, alpha=0.6)

    ax5a.set_xlabel('Component', fontsize=9)
    ax5a.set_ylabel('Singular value', fontsize=9)
    ax5a.set_title(f"D5: Singular Value Spectrum ({d5['n_above_mp']} above MP)",
                   fontsize=11, fontweight='bold')
    ax5a.legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
                labelcolor='#cccccc')
    ax5a.tick_params(labelsize=7)

    # ---- Bot-right: D5 Updated Atlas (PC1 vs PC2) ---------------------------
    ax5b = fig.add_subplot(gs[2, 1])
    ax5b.set_facecolor('#181818')
    for spine in ax5b.spines.values():
        spine.set_color('#444444')

    scores = d5['scores_aug']
    n_old = d5['n_old']
    aug_names = d5['aug_names']
    aug_cats = d5['aug_categories']

    # Old signatures as circles
    for i in range(n_old):
        cat = aug_cats[i]
        color = CATEGORY_COLORS.get(cat, '#888888')
        ax5b.scatter(scores[i, 0], scores[i, 1], c=color, s=50,
                    marker='o', edgecolors='white', linewidths=0.4,
                    alpha=0.7, zorder=2)

    # New signatures as stars
    for i in range(n_old, len(aug_names)):
        name = aug_names[i]
        color = NEW_COLORS.get(name, '#ff6b6b')
        ax5b.scatter(scores[i, 0], scores[i, 1], c=color, s=150,
                    marker='*', edgecolors='white', linewidths=0.5,
                    zorder=4)
        ax5b.annotate(name, (scores[i, 0], scores[i, 1]),
                     fontsize=6, color=color, ha='center', va='bottom',
                     xytext=(0, 8), textcoords='offset points',
                     fontweight='bold')

    varexp = d5['varexp_aug']
    ax5b.set_xlabel(f'PC1 ({varexp[0]*100:.1f}%)', fontsize=9)
    ax5b.set_ylabel(f'PC2 ({varexp[1]*100:.1f}%)', fontsize=9)
    ax5b.set_title("D5: Updated Atlas — Old (circles) + New (stars)",
                   fontsize=11, fontweight='bold')
    ax5b.axhline(0, color='#333333', linewidth=0.5)
    ax5b.axvline(0, color='#333333', linewidth=0.5)
    ax5b.tick_params(labelsize=7)

    # Category legend
    handles = []
    for cat in sorted(CATEGORY_COLORS.keys()):
        handles.append(plt.Line2D([0], [0], marker='o', color='none',
                                  markerfacecolor=CATEGORY_COLORS[cat],
                                  markersize=6, label=cat))
    handles.append(plt.Line2D([0], [0], marker='*', color='none',
                              markerfacecolor='#ff6b6b',
                              markersize=10, label='new source'))
    fig.legend(handles=handles, loc='lower center', ncol=9, fontsize=7,
               frameon=True, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc', bbox_to_anchor=(0.5, 0.005))

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'figures', 'signature_space.png')
    fig.savefig(outpath, dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")


# =============================================================================
# MAIN
# =============================================================================

def _collect_and_align(sources, runner, runner_to_sig_idx, metric_names):
    """Collect profiles for a list of sources and align to signature space."""
    profiles_raw = {}
    for name, gen_fn in sources:
        rngs = runner.trial_rngs()
        chunks = [gen_fn(rng, runner.data_size) for rng in rngs]
        metrics = runner.collect(chunks)
        profile = np.zeros(len(runner.metric_names))
        for j, m in enumerate(runner.metric_names):
            vals = metrics.get(m, [])
            if len(vals) > 0:
                profile[j] = np.mean(vals)
        profiles_raw[name] = profile
        n_active = np.sum(np.abs(profile) > 1e-15)
        print(f"  {name:25s}  {n_active}/{len(runner.metric_names)} active metrics")

    # Align to signature metric space
    profiles = {}
    for name, raw in profiles_raw.items():
        aligned = np.zeros(len(metric_names))
        for runner_j, sig_j in runner_to_sig_idx.items():
            aligned[sig_j] = raw[runner_j]
        profiles[name] = aligned
    return profiles


def make_figure_round2(d4_r2, d5_r2, names, categories, metric_names):
    """Second figure: Round 2 adversarial results (4 panels, 2x2)."""
    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(22, 16), facecolor='#181818')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30,
                          left=0.07, right=0.96, top=0.93, bottom=0.06)
    fig.suptitle("Round 2: Adversarial 7D Breaker Hunt",
                 fontsize=16, fontweight='bold', color='white')

    # ---- Top-left: Residual bar chart (R2 sources + existing) ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#181818')
    for spine in ax1.spines.values():
        spine.set_color('#444444')

    all_residuals = []
    for i, name in enumerate(names):
        all_residuals.append((name, d4_r2['existing_residuals'][i], 'existing',
                             categories[i]))
    for name, info in d4_r2['new_residuals'].items():
        all_residuals.append((name, info['residual'], 'r2', 'r2'))

    all_residuals.sort(key=lambda x: x[1])
    bar_names = [r[0] for r in all_residuals]
    bar_vals = [r[1] for r in all_residuals]
    bar_colors = []
    for r in all_residuals:
        if r[2] == 'r2':
            bar_colors.append(ROUND2_COLORS.get(r[0], '#ff4757'))
        else:
            bar_colors.append(CATEGORY_COLORS.get(r[3], '#888888'))

    y_pos = np.arange(len(bar_names))
    ax1.barh(y_pos, bar_vals, color=bar_colors, alpha=0.85, height=0.7)

    thresh = np.percentile(d4_r2['existing_residuals'], 95)
    ax1.axvline(thresh, color='#e74c3c', linestyle='--', linewidth=1.5,
                alpha=0.8, label='95th pctile (existing)')

    # Mark breakers
    n_breakers = sum(1 for v in d4_r2['new_residuals'].values() if v['breaks_7d'])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([n[:20] for n in bar_names], fontsize=4.5)
    ax1.set_xlabel('7D Reconstruction Residual', fontsize=8)
    ax1.set_title(f"Round 2: 7D Residuals ({n_breakers}/10 break)",
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc')
    ax1.tick_params(labelsize=6)

    # ---- Top-right: Singular value spectrum comparison ------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#181818')
    for spine in ax2.spines.values():
        spine.set_color('#444444')

    k_old = np.arange(1, len(d5_r2['s_old']) + 1)
    k_new = np.arange(1, len(d5_r2['s_aug']) + 1)

    ax2.semilogy(k_old, d5_r2['s_old'], 's-', color='#95a5a6', markersize=4,
                  linewidth=1.2, label=f'Original 41 (PR={d5_r2["pr_old"]:.1f})',
                  alpha=0.7)
    ax2.semilogy(k_new, d5_r2['s_aug'], 'o-', color='#ff4757', markersize=4,
                  linewidth=1.5, label=f'+R2 adversarial (PR={d5_r2["pr_aug"]:.1f})',
                  zorder=3)

    mp_k = np.arange(1, len(d5_r2['mp_upper']) + 1)
    n_plot = min(len(d5_r2['s_aug']), len(d5_r2['mp_upper']))
    ax2.fill_between(mp_k[:n_plot],
                      d5_r2['mp_mean'][:n_plot] * 0.7,
                      d5_r2['mp_upper'][:n_plot],
                      color='#3498db', alpha=0.2, label='MP 97.5% envelope')
    ax2.semilogy(mp_k[:n_plot],
                  d5_r2['mp_mean'][:n_plot],
                  '--', color='#3498db', linewidth=1, alpha=0.6)

    ax2.set_xlabel('Component', fontsize=9)
    ax2.set_ylabel('Singular value', fontsize=9)
    ax2.set_title(f"Singular Value Spectrum ({d5_r2['n_above_mp']} above MP)",
                   fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
                labelcolor='#cccccc')
    ax2.tick_params(labelsize=7)

    # ---- Bot-left: Atlas with R2 sources --------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#181818')
    for spine in ax3.spines.values():
        spine.set_color('#444444')

    scores = d5_r2['scores_aug']
    n_old = d5_r2['n_old']
    aug_names = d5_r2['aug_names']
    aug_cats = d5_r2['aug_categories']
    varexp = d5_r2['varexp_aug']

    for i in range(n_old):
        cat = aug_cats[i]
        color = CATEGORY_COLORS.get(cat, '#888888')
        ax3.scatter(scores[i, 0], scores[i, 1], c=color, s=40,
                    marker='o', edgecolors='white', linewidths=0.3,
                    alpha=0.6, zorder=2)

    for i in range(n_old, len(aug_names)):
        name = aug_names[i]
        color = ROUND2_COLORS.get(name, '#ff4757')
        ax3.scatter(scores[i, 0], scores[i, 1], c=color, s=180,
                    marker='*', edgecolors='white', linewidths=0.5, zorder=4)
        ax3.annotate(name, (scores[i, 0], scores[i, 1]),
                     fontsize=5.5, color=color, ha='center', va='bottom',
                     xytext=(0, 8), textcoords='offset points',
                     fontweight='bold')

    ax3.set_xlabel(f'PC1 ({varexp[0]*100:.1f}%)', fontsize=9)
    ax3.set_ylabel(f'PC2 ({varexp[1]*100:.1f}%)', fontsize=9)
    ax3.set_title("Updated Atlas — Old (circles) + R2 Adversarial (stars)",
                   fontsize=11, fontweight='bold')
    ax3.axhline(0, color='#333333', linewidth=0.5)
    ax3.axvline(0, color='#333333', linewidth=0.5)
    ax3.tick_params(labelsize=7)

    # ---- Bot-right: Per-source residual breakdown by leverage metrics --------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#181818')
    for spine in ax4.spines.values():
        spine.set_color('#444444')

    # Show residual decomposition: which metrics contribute most to each
    # R2 source's residual?
    r2_names = list(d4_r2['new_residuals'].keys())
    r2_residuals = [d4_r2['new_residuals'][n]['residual'] for n in r2_names]
    sort_idx = np.argsort(r2_residuals)[::-1]

    y_pos = np.arange(len(r2_names))
    sorted_names = [r2_names[i] for i in sort_idx]
    sorted_vals = [r2_residuals[i] for i in sort_idx]
    sorted_colors = [ROUND2_COLORS.get(n, '#ff4757') for n in sorted_names]

    bars = ax4.barh(y_pos, sorted_vals, color=sorted_colors, alpha=0.85,
                    height=0.7, edgecolor='white', linewidth=0.3)

    ax4.axvline(thresh, color='#e74c3c', linestyle='--', linewidth=1.5,
                alpha=0.8, label='95th pctile')

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(sorted_names, fontsize=8)
    ax4.set_xlabel('7D Reconstruction Residual', fontsize=9)
    ax4.set_title("Round 2 Sources Ranked by Residual",
                   fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc')
    ax4.tick_params(labelsize=7)

    # Add p-value annotations
    for i, (name, val) in enumerate(zip(sorted_names, sorted_vals)):
        info = d4_r2['new_residuals'][name]
        p_str = f"p={info['p_value']:.3f}"
        if info['breaks_7d']:
            p_str += " *"
        ax4.text(val + 0.01, i, p_str, va='center', fontsize=7,
                 color='#e74c3c' if info['breaks_7d'] else '#888888')

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'figures', 'signature_space_r2.png')
    fig.savefig(outpath, dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")


def run_investigation():
    print("=" * 78)
    print("SIGNATURE SPACE: PC Interpretation + 7D Breaker Hunt")
    print("=" * 78)

    # Load existing signatures
    names, categories, metric_names, means_matrix, stds_matrix = load_signatures()

    # Set up Runner for new source collection (same metrics as signatures)
    runner = Runner("Signature Space", mode="1d", data_size=16384, n_trials=25)

    # Build alignment map: runner metrics (no tau suffix) -> signature metrics (:tau1 suffix)
    sig_metric_set = set(metric_names)
    runner_to_sig_idx = {}
    for j, m in enumerate(runner.metric_names):
        sig_name = m + ':tau1'
        if sig_name in sig_metric_set:
            runner_to_sig_idx[j] = metric_names.index(sig_name)

    n_aligned = len(runner_to_sig_idx)
    print(f"Metric alignment: {n_aligned}/{len(runner.metric_names)} runner metrics "
          f"matched to {len(metric_names)} signature metrics")

    # D1: PC Interpretation
    d1 = direction_1(names, categories, metric_names, means_matrix)

    # D2: PC Loadings Heatmap
    d2 = direction_2(metric_names, d1['Vt'])

    # D3: Collect Round 1 source profiles
    print("\n" + "=" * 78)
    print("D3: Round 1 — 8 Candidate Sources")
    print("=" * 78)
    r1_profiles = _collect_and_align(NEW_SOURCES, runner, runner_to_sig_idx,
                                     metric_names)

    # D4: Round 1 projection test
    d4 = direction_4(names, categories, metric_names, means_matrix,
                     d1['col_means'], d1['col_stds'],
                     d1['Vt'], d1['s'], r1_profiles)

    # D5: Round 1 updated dimensionality
    d5 = direction_5(names, categories, metric_names, means_matrix,
                     d1['col_means'], d1['col_stds'], d1['s'], r1_profiles)

    # Round 1 figure
    make_figure(d1, d2, r1_profiles, d4, d5, names, categories, metric_names)

    # =====================================================================
    # ROUND 2: ADVERSARIAL SOURCES
    # =====================================================================
    print("\n" + "=" * 78)
    print("ROUND 2: ADVERSARIAL 7D BREAKER HUNT")
    print("=" * 78)
    print("Targeting metrics with most unexplained variance:")
    print("  Wasserstein:self_similarity, Clifford:path_regularity,")
    print("  Spiral:growth_rate, HOS:skew_max/kurt_max,")
    print("  Lorentzian:lightlike_fraction, Fisher:effective_dimension")

    print("\nCollecting adversarial source profiles...")
    r2_profiles = _collect_and_align(ROUND2_SOURCES, runner, runner_to_sig_idx,
                                     metric_names)

    # D4 Round 2
    d4_r2 = direction_4(names, categories, metric_names, means_matrix,
                        d1['col_means'], d1['col_stds'],
                        d1['Vt'], d1['s'], r2_profiles)

    # D5 Round 2: combine R1 + R2 + existing for final dimensionality
    all_new = {}
    all_new.update(r1_profiles)
    all_new.update(r2_profiles)
    d5_all = direction_5(names, categories, metric_names, means_matrix,
                         d1['col_means'], d1['col_stds'], d1['s'], all_new)

    # D5 Round 2 standalone (just R2)
    d5_r2 = direction_5(names, categories, metric_names, means_matrix,
                        d1['col_means'], d1['col_stds'], d1['s'], r2_profiles)

    # Round 2 figure
    make_figure_round2(d4_r2, d5_r2, names, categories, metric_names)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"\nD1: 7 PCs interpreted — top axis: {d1['pc_info'][0]['label']}")
    print(f"D2: {len(d2['families'])} geometry families mapped to 7 PCs")

    n_r1 = sum(1 for v in d4['new_residuals'].values() if v['breaks_7d'])
    r1_breakers = [n for n, v in d4['new_residuals'].items() if v['breaks_7d']]
    print(f"\nRound 1: {n_r1}/8 sources break 7D"
          + (f" — {', '.join(r1_breakers)}" if r1_breakers else ""))
    print(f"  PR: {d5['pr_old']:.1f} -> {d5['pr_aug']:.1f}")

    n_r2 = sum(1 for v in d4_r2['new_residuals'].values() if v['breaks_7d'])
    r2_breakers = [n for n, v in d4_r2['new_residuals'].items() if v['breaks_7d']]
    print(f"\nRound 2 (adversarial): {n_r2}/10 sources break 7D"
          + (f" — {', '.join(r2_breakers)}" if r2_breakers else ""))
    print(f"  PR: {d5_r2['pr_old']:.1f} -> {d5_r2['pr_aug']:.1f}")

    print(f"\nCombined (R1+R2): PR = {d5_all['pr_aug']:.1f} "
          f"({d5_all['n_above_mp']} above MP)")

    # Top 5 highest-residual R2 sources
    r2_ranked = sorted(d4_r2['new_residuals'].items(),
                       key=lambda x: -x[1]['residual'])
    print(f"\nTop 5 highest-residual adversarial sources:")
    for name, info in r2_ranked[:5]:
        print(f"  {name:25s}  residual={info['residual']:.4f}  p={info['p_value']:.3f}")

    # =====================================================================
    # ROUND 3: EXPANDED-METRIC BLIND SPOT CLOSURE TEST
    # =====================================================================
    print("\n" + "=" * 78)
    print("ROUND 3: BLIND SPOT CLOSURE — New Geometries Test")
    print("=" * 78)
    print("Testing whether Hölder Regularity, p-Variation, and Multi-Scale Wasserstein")
    print("resolve the 5 breakers from Round 2.")
    print(f"Runner metric space: {len(runner.metric_names)} metrics "
          f"(vs {len(metric_names)} in old signatures)")

    print("\nCollecting representative existing-category sources (full metric space)...")
    repr_profiles = _collect_full(REPRESENTATIVE_SOURCES, runner)

    print("\nCollecting breaker sources (full metric space)...")
    breaker_profiles = _collect_full(BREAKER_SOURCES, runner)

    d6 = direction_6(runner, repr_profiles, breaker_profiles)
    make_figure_round3(runner, d6, repr_profiles, breaker_profiles)

    # Round 3 summary
    print("\n" + "-" * 78)
    print("Round 3 Summary:")
    for name in breaker_profiles:
        old_z, new_z = d6['group_stats'][name]
        n_old_out, n_new_out = d6['outlier_counts'][name]
        print(f"  {name:25s}  mean|z| old={old_z:.2f} new={new_z:.2f}  "
              f"outliers old={n_old_out} new={n_new_out}/{len(d6['new_idx'])}")


if __name__ == "__main__":
    run_investigation()
